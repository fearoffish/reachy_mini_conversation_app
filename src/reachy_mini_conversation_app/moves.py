"""Movement system driving sequential primary moves.

Design overview
- Primary moves (emotions, dances, goto, idle) are mutually exclusive and run
  sequentially.
- There is a single control point to the robot: `ReachyMini.set_target`.
- The control loop runs near 100 Hz and is phase-aligned via a monotonic clock.
- Idle behaviour starts an infinite `IdleMove` after a short inactivity delay
  unless listening is active. The robot holds still and only makes occasional,
  subtle head movements (and far rarer antenna twitches).

Threading model
- A dedicated worker thread owns all real-time state and issues `set_target`
  commands.
- Other threads communicate via a command queue (enqueue moves, mark activity,
  toggle listening).

Units and frames
- Antennas and `body_yaw` are in radians.

Safety
- Listening freezes antennas, then blends them back on unfreeze.
- Interpolations and blends are used to avoid jumps at all times.
- `set_target` errors are rate-limited in logs.
"""

from __future__ import annotations
import time
import random
import logging
import threading
from queue import Empty, Queue
from typing import Any, Dict, Tuple
from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.motion.move import Move
from reachy_mini.utils.interpolation import linear_pose_interpolation


logger = logging.getLogger(__name__)

# Configuration constants
CONTROL_LOOP_FREQUENCY_HZ = 60.0  # Hz - Target frequency for the movement control loop

# Idle motion tuning. When idle the robot stays still and only occasionally makes a
# subtle head movement; antenna twitches are far rarer. All times are in seconds and
# refer to the idle phase (after the initial interpolation to neutral).
IDLE_HEAD_INTERVAL_S = (30.0, 60.0)  # random wait between subtle head gestures
IDLE_HEAD_RISE_S = 1.5  # ease into the offset
IDLE_HEAD_HOLD_S = 1.0  # dwell at the offset
IDLE_HEAD_FALL_S = 1.5  # ease back to neutral
IDLE_HEAD_ROLL_MAX_DEG = 3.0
IDLE_HEAD_PITCH_MAX_DEG = 5.0
IDLE_HEAD_YAW_MAX_DEG = 8.0

IDLE_ANTENNA_INTERVAL_S = (60.0, 120.0)  # rarer than head gestures
IDLE_ANTENNA_RISE_S = 0.25  # quick flick out
IDLE_ANTENNA_FALL_S = 0.25  # quick flick back
IDLE_ANTENNA_AMP_RAD = float(np.deg2rad(5.0))


def _ease(x: float) -> float:
    """Smooth 0->1 ease (raised cosine), clamped outside [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return float(0.5 - 0.5 * np.cos(np.pi * x))


def _gesture_envelope(local: float, rise: float, hold: float, fall: float) -> float:
    """Return a 0->1->0 envelope: ramp up over `rise`, hold at 1 for `hold`, ramp down over `fall`."""
    if local < 0.0:
        return 0.0
    if local < rise:
        return _ease(local / rise)
    if local < rise + hold:
        return 1.0
    if local < rise + hold + fall:
        return _ease(1.0 - (local - rise - hold) / fall)
    return 0.0

# Type definitions
FullBodyPose = Tuple[NDArray[np.float32], Tuple[float, float], float]  # (head_pose_4x4, antennas, body_yaw)


class IdleMove(Move):  # type: ignore
    """Idle behaviour: hold a still neutral pose with occasional subtle motion.

    After interpolating from the current pose to neutral, the robot stays still
    rather than continuously breathing. Every 30-60s it eases its head to a small
    random offset and back; far more rarely (60-120s) it does a brief, subtle
    antenna twitch. There is no continuous body or antenna motion.
    """

    def __init__(
        self,
        interpolation_start_pose: NDArray[np.float32],
        interpolation_start_antennas: Tuple[float, float],
        interpolation_duration: float = 1.0,
    ):
        """Initialize idle move.

        Args:
            interpolation_start_pose: 4x4 matrix of current head pose to interpolate from
            interpolation_start_antennas: Current antenna positions to interpolate from
            interpolation_duration: Duration of interpolation to neutral (seconds)

        """
        self.interpolation_start_pose = interpolation_start_pose
        self.interpolation_start_antennas = np.array(interpolation_start_antennas)
        self.interpolation_duration = interpolation_duration

        # Neutral base positions (held still between gestures)
        self.neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.neutral_antennas = np.array([-0.1745, 0.1745])  # ~10° offset to reduce shaking

        # Schedule the first head gesture and antenna twitch. Times are in idle-phase
        # seconds (i.e. measured from the end of the interpolation to neutral).
        self._head_start = random.uniform(*IDLE_HEAD_INTERVAL_S)
        self._head_target = self._random_head_target()
        self._antenna_start = random.uniform(*IDLE_ANTENNA_INTERVAL_S)
        self._antenna_sign = random.choice([-1.0, 1.0])

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return float("inf")  # Idle behaviour runs until interrupted

    @staticmethod
    def _random_head_target() -> NDArray[np.float64]:
        """Pick a small random (roll, pitch, yaw) head offset in degrees."""
        return np.array(
            [
                random.uniform(-IDLE_HEAD_ROLL_MAX_DEG, IDLE_HEAD_ROLL_MAX_DEG),
                random.uniform(-IDLE_HEAD_PITCH_MAX_DEG, IDLE_HEAD_PITCH_MAX_DEG),
                random.uniform(-IDLE_HEAD_YAW_MAX_DEG, IDLE_HEAD_YAW_MAX_DEG),
            ],
            dtype=np.float64,
        )

    def _head_offset_deg(self, idle_time: float) -> tuple[float, float, float]:
        """Return the (roll, pitch, yaw) head offset in degrees for the current idle time.

        Eases to a random target, dwells, then eases back. When a gesture finishes the
        next one is scheduled for 30-60s later.
        """
        span = IDLE_HEAD_RISE_S + IDLE_HEAD_HOLD_S + IDLE_HEAD_FALL_S
        local = idle_time - self._head_start
        if local >= span:
            # Gesture complete: rest at neutral and schedule the next one.
            self._head_start = idle_time + random.uniform(*IDLE_HEAD_INTERVAL_S)
            self._head_target = self._random_head_target()
            return 0.0, 0.0, 0.0
        factor = _gesture_envelope(local, IDLE_HEAD_RISE_S, IDLE_HEAD_HOLD_S, IDLE_HEAD_FALL_S)
        roll, pitch, yaw = self._head_target * factor
        return float(roll), float(pitch), float(yaw)

    def _antenna_delta_rad(self, idle_time: float) -> float:
        """Return the symmetric antenna offset in radians for the current idle time.

        A brief flick out and back; when it finishes the next twitch is scheduled for
        60-120s later (rarer than head gestures).
        """
        span = IDLE_ANTENNA_RISE_S + IDLE_ANTENNA_FALL_S
        local = idle_time - self._antenna_start
        if local >= span:
            self._antenna_start = idle_time + random.uniform(*IDLE_ANTENNA_INTERVAL_S)
            self._antenna_sign = random.choice([-1.0, 1.0])
            return 0.0
        factor = _gesture_envelope(local, IDLE_ANTENNA_RISE_S, 0.0, IDLE_ANTENNA_FALL_S)
        return self._antenna_sign * IDLE_ANTENNA_AMP_RAD * factor

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate idle move at time t."""
        if t < self.interpolation_duration:
            # Phase 1: Interpolate to neutral base position
            interpolation_t = t / self.interpolation_duration

            # Interpolate head pose
            head_pose = linear_pose_interpolation(
                self.interpolation_start_pose,
                self.neutral_head_pose,
                interpolation_t,
            )

            # Interpolate antennas
            antennas_interp = (
                1 - interpolation_t
            ) * self.interpolation_start_antennas + interpolation_t * self.neutral_antennas
            antennas = antennas_interp.astype(np.float64)

        else:
            # Phase 2: Hold still at neutral, with occasional subtle head movements
            # and far rarer antenna twitches. No continuous body or antenna motion.
            idle_time = t - self.interpolation_duration

            roll, pitch, yaw = self._head_offset_deg(idle_time)
            head_pose = create_head_pose(x=0, y=0, z=0, roll=roll, pitch=pitch, yaw=yaw, degrees=True, mm=False)

            antenna_delta = self._antenna_delta_rad(idle_time)
            antennas = (self.neutral_antennas + antenna_delta).astype(np.float64)

        # Return in official Move interface format: (head_pose, antennas_array, body_yaw)
        return (head_pose, antennas, 0.0)


def clone_full_body_pose(pose: FullBodyPose) -> FullBodyPose:
    """Create a deep copy of a full body pose tuple."""
    head, antennas, body_yaw = pose
    return (head.copy(), (float(antennas[0]), float(antennas[1])), float(body_yaw))


@dataclass
class MovementState:
    """State tracking for the movement system."""

    # Primary move state
    current_move: Move | None = None
    move_start_time: float | None = None
    last_activity_time: float = 0.0

    # Status flags
    last_primary_pose: FullBodyPose | None = None

    def update_activity(self) -> None:
        """Update the last activity time."""
        self.last_activity_time = time.monotonic()


@dataclass
class LoopFrequencyStats:
    """Track rolling loop frequency statistics."""

    mean: float = 0.0
    m2: float = 0.0
    min_freq: float = float("inf")
    count: int = 0
    last_freq: float = 0.0
    potential_freq: float = 0.0

    def reset(self) -> None:
        """Reset accumulators while keeping the last potential frequency."""
        self.mean = 0.0
        self.m2 = 0.0
        self.min_freq = float("inf")
        self.count = 0


class MovementManager:
    """Coordinate sequential moves and robot output at 100 Hz.

    Responsibilities:
    - Own a real-time loop that samples the current primary move (if any) and calls
      `set_target` exactly once per tick.
    - Start an idle `IdleMove` after `idle_inactivity_delay` when not
      listening and no moves are queued.
    - Expose thread-safe APIs so other threads can enqueue moves or mark activity
      without touching internal state.

    Timing:
    - All elapsed-time calculations rely on `time.monotonic()` through `self._now`
      to avoid wall-clock jumps.
    - The loop attempts 100 Hz

    Concurrency:
    - External threads communicate via `_command_queue` messages.
    """

    def __init__(
        self,
        current_robot: ReachyMini,
    ):
        """Initialize movement manager."""
        self.current_robot = current_robot

        # Single timing source for durations
        self._now = time.monotonic

        # Movement state
        self.state = MovementState()
        self.state.last_activity_time = self._now()
        neutral_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.state.last_primary_pose = (neutral_pose, (0.0, 0.0), 0.0)

        # Move queue (primary moves)
        self.move_queue: deque[Move] = deque()

        # Configuration
        self.idle_inactivity_delay = 0.3  # seconds
        self.target_frequency = CONTROL_LOOP_FREQUENCY_HZ
        self.target_period = 1.0 / self.target_frequency

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._is_listening = False
        self._last_commanded_pose: FullBodyPose = clone_full_body_pose(self.state.last_primary_pose)
        self._listening_antennas: Tuple[float, float] = self._last_commanded_pose[1]
        self._antenna_unfreeze_blend = 1.0
        self._antenna_blend_duration = 0.4  # seconds to blend back after listening
        self._last_listening_blend_time = self._now()
        self._idle_active = False  # true when idle move is running or queued
        self._listening_debounce_s = 0.15
        self._last_listening_toggle_time = self._now()
        self._last_set_target_err = 0.0
        self._set_target_err_interval = 1.0  # seconds between error logs
        self._set_target_err_suppressed = 0

        # Cross-thread signalling
        self._command_queue: "Queue[Tuple[str, Any]]" = Queue()

        self._shared_state_lock = threading.Lock()
        self._shared_last_activity_time = self.state.last_activity_time
        self._shared_is_listening = self._is_listening
        self._status_lock = threading.Lock()
        self._freq_stats = LoopFrequencyStats()
        self._freq_snapshot = LoopFrequencyStats()

    def queue_move(self, move: Move) -> None:
        """Queue a primary move to run after the currently executing one.

        Thread-safe: the move is enqueued via the worker command queue so the
        control loop remains the sole mutator of movement state.
        """
        self._command_queue.put(("queue_move", move))

    def clear_move_queue(self) -> None:
        """Stop the active move and discard any queued primary moves.

        Thread-safe: executed by the worker thread via the command queue.
        """
        self._command_queue.put(("clear_queue", None))

    def set_moving_state(self, duration: float) -> None:
        """Mark the robot as actively moving for the provided duration.

        Legacy hook used by goto helpers to keep inactivity and idle logic
        aware of manual motions. Thread-safe via the command queue.
        """
        self._command_queue.put(("set_moving_state", duration))

    def is_idle(self) -> bool:
        """Return True when the robot has been inactive longer than the idle delay."""
        with self._shared_state_lock:
            last_activity = self._shared_last_activity_time
            listening = self._shared_is_listening

        if listening:
            return False

        return self._now() - last_activity >= self.idle_inactivity_delay

    def set_listening(self, listening: bool) -> None:
        """Enable or disable listening mode without touching shared state directly.

        While listening:
        - Antenna positions are frozen at the last commanded values.
        - Blending is reset so that upon unfreezing the antennas return smoothly.
        - Idle behaviour is suppressed.

        Thread-safe: the change is posted to the worker command queue.
        """
        with self._shared_state_lock:
            if self._shared_is_listening == listening:
                return
        self._command_queue.put(("set_listening", listening))

    def _poll_signals(self, current_time: float) -> None:
        """Apply queued commands."""
        while True:
            try:
                command, payload = self._command_queue.get_nowait()
            except Empty:
                break
            self._handle_command(command, payload, current_time)

    def _handle_command(self, command: str, payload: Any, current_time: float) -> None:
        """Handle a single cross-thread command."""
        if command == "queue_move":
            if isinstance(payload, Move):
                self.move_queue.append(payload)
                self.state.update_activity()
                duration = getattr(payload, "duration", None)
                if duration is not None:
                    try:
                        duration_str = f"{float(duration):.2f}"
                    except (TypeError, ValueError):
                        duration_str = str(duration)
                else:
                    duration_str = "?"
                logger.debug(
                    "Queued move with duration %ss, queue size: %s",
                    duration_str,
                    len(self.move_queue),
                )
            else:
                logger.warning("Ignored queue_move command with invalid payload: %s", payload)
        elif command == "clear_queue":
            self.move_queue.clear()
            self.state.current_move = None
            self.state.move_start_time = None
            self._idle_active = False
            logger.info("Cleared move queue and stopped current move")
        elif command == "set_moving_state":
            try:
                duration = float(payload)
            except (TypeError, ValueError):
                logger.warning("Invalid moving state duration: %s", payload)
                return
            self.state.update_activity()
        elif command == "mark_activity":
            self.state.update_activity()
        elif command == "set_listening":
            desired_state = bool(payload)
            now = self._now()
            if now - self._last_listening_toggle_time < self._listening_debounce_s:
                return
            self._last_listening_toggle_time = now

            if self._is_listening == desired_state:
                return

            self._is_listening = desired_state
            self._last_listening_blend_time = now
            if desired_state:
                # Freeze: snapshot current commanded antennas and reset blend
                self._listening_antennas = (
                    float(self._last_commanded_pose[1][0]),
                    float(self._last_commanded_pose[1][1]),
                )
                self._antenna_unfreeze_blend = 0.0
            else:
                # Unfreeze: restart blending from frozen pose
                self._antenna_unfreeze_blend = 0.0
            self.state.update_activity()
        else:
            logger.warning("Unknown command received by MovementManager: %s", command)

    def _publish_shared_state(self) -> None:
        """Expose idle-related state for external threads."""
        with self._shared_state_lock:
            self._shared_last_activity_time = self.state.last_activity_time
            self._shared_is_listening = self._is_listening

    def _manage_move_queue(self, current_time: float) -> None:
        """Manage the primary move queue (sequential execution)."""
        if self.state.current_move is None or (
            self.state.move_start_time is not None
            and current_time - self.state.move_start_time >= self.state.current_move.duration
        ):
            self.state.current_move = None
            self.state.move_start_time = None

            if self.move_queue:
                self.state.current_move = self.move_queue.popleft()
                self.state.move_start_time = current_time
                # Any real move cancels idle mode flag
                self._idle_active = isinstance(self.state.current_move, IdleMove)
                logger.debug(f"Starting new move, duration: {self.state.current_move.duration}s")

    def _manage_idle(self, current_time: float) -> None:
        """Manage automatic idle behaviour when inactive."""
        if (
            self.state.current_move is None
            and not self.move_queue
            and not self._is_listening
            and not self._idle_active
        ):
            idle_for = current_time - self.state.last_activity_time
            if idle_for >= self.idle_inactivity_delay:
                try:
                    # These 2 functions return the latest available sensor data from the robot, but don't perform I/O synchronously.
                    # Therefore, we accept calling them inside the control loop.
                    _, current_antennas = self.current_robot.get_current_joint_positions()
                    current_head_pose = self.current_robot.get_current_head_pose()

                    self._idle_active = True
                    self.state.update_activity()

                    idle_move = IdleMove(
                        interpolation_start_pose=current_head_pose,
                        interpolation_start_antennas=current_antennas,
                        interpolation_duration=1.0,
                    )
                    self.move_queue.append(idle_move)
                    logger.debug("Started idle behaviour after %.1fs of inactivity", idle_for)
                except Exception as e:
                    self._idle_active = False
                    logger.error("Failed to start idle behaviour: %s", e)

        if isinstance(self.state.current_move, IdleMove) and self.move_queue:
            self.state.current_move = None
            self.state.move_start_time = None
            self._idle_active = False
            logger.debug("Stopping idle behaviour due to new move activity")

        if self.state.current_move is not None and not isinstance(self.state.current_move, IdleMove):
            self._idle_active = False

    def _get_primary_pose(self, current_time: float) -> FullBodyPose:
        """Get the primary full body pose from current move or neutral."""
        # When a primary move is playing, sample it and cache the resulting pose
        if self.state.current_move is not None and self.state.move_start_time is not None:
            move_time = current_time - self.state.move_start_time
            head, antennas, body_yaw = self.state.current_move.evaluate(move_time)

            if head is None:
                head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            if antennas is None:
                antennas = np.array([-0.1745, 0.1745])  # ~10° offset
            if body_yaw is None:
                body_yaw = 0.0

            antennas_tuple = (float(antennas[0]), float(antennas[1]))
            head_copy = head.copy()
            primary_full_body_pose = (
                head_copy,
                antennas_tuple,
                float(body_yaw),
            )

            self.state.last_primary_pose = clone_full_body_pose(primary_full_body_pose)
        # Otherwise reuse the last primary pose so we avoid jumps between moves
        elif self.state.last_primary_pose is not None:
            primary_full_body_pose = clone_full_body_pose(self.state.last_primary_pose)
        else:
            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            primary_full_body_pose = (neutral_head_pose, (0.0, 0.0), 0.0)
            self.state.last_primary_pose = clone_full_body_pose(primary_full_body_pose)

        return primary_full_body_pose

    def _update_primary_motion(self, current_time: float) -> None:
        """Advance queue state and idle behaviours for this tick."""
        self._manage_move_queue(current_time)
        self._manage_idle(current_time)

    def _calculate_blended_antennas(self, target_antennas: Tuple[float, float]) -> Tuple[float, float]:
        """Blend target antennas with listening freeze state and update blending."""
        now = self._now()
        listening = self._is_listening
        listening_antennas = self._listening_antennas
        blend = self._antenna_unfreeze_blend
        blend_duration = self._antenna_blend_duration
        last_update = self._last_listening_blend_time
        self._last_listening_blend_time = now

        if listening:
            antennas_cmd = listening_antennas
            new_blend = 0.0
        else:
            dt = max(0.0, now - last_update)
            if blend_duration <= 0:
                new_blend = 1.0
            else:
                new_blend = min(1.0, blend + dt / blend_duration)
            antennas_cmd = (
                listening_antennas[0] * (1.0 - new_blend) + target_antennas[0] * new_blend,
                listening_antennas[1] * (1.0 - new_blend) + target_antennas[1] * new_blend,
            )

        if listening:
            self._antenna_unfreeze_blend = 0.0
        else:
            self._antenna_unfreeze_blend = new_blend
            if new_blend >= 1.0:
                self._listening_antennas = (
                    float(target_antennas[0]),
                    float(target_antennas[1]),
                )

        return antennas_cmd

    def _issue_control_command(
        self, head: NDArray[np.float32], antennas: Tuple[float, float], body_yaw: float
    ) -> None:
        """Send the pose to the robot with throttled error logging."""
        try:
            self.current_robot.set_target(head=head, antennas=antennas, body_yaw=body_yaw)
        except Exception as e:
            now = self._now()
            if now - self._last_set_target_err >= self._set_target_err_interval:
                msg = f"Failed to set robot target: {e}"
                if self._set_target_err_suppressed:
                    msg += f" (suppressed {self._set_target_err_suppressed} repeats)"
                    self._set_target_err_suppressed = 0
                logger.error(msg)
                self._last_set_target_err = now
            else:
                self._set_target_err_suppressed += 1
        else:
            with self._status_lock:
                self._last_commanded_pose = clone_full_body_pose((head, antennas, body_yaw))

    def _update_frequency_stats(
        self,
        loop_start: float,
        prev_loop_start: float,
        stats: LoopFrequencyStats,
    ) -> LoopFrequencyStats:
        """Update frequency statistics based on the current loop start time."""
        period = loop_start - prev_loop_start
        if period > 0:
            stats.last_freq = 1.0 / period
            stats.count += 1
            delta = stats.last_freq - stats.mean
            stats.mean += delta / stats.count
            stats.m2 += delta * (stats.last_freq - stats.mean)
            stats.min_freq = min(stats.min_freq, stats.last_freq)
        return stats

    def _schedule_next_tick(self, loop_start: float, stats: LoopFrequencyStats) -> Tuple[float, LoopFrequencyStats]:
        """Compute sleep time to maintain target frequency and update potential freq."""
        computation_time = self._now() - loop_start
        stats.potential_freq = 1.0 / computation_time if computation_time > 0 else float("inf")
        sleep_time = max(0.0, self.target_period - computation_time)
        return sleep_time, stats

    def _record_frequency_snapshot(self, stats: LoopFrequencyStats) -> None:
        """Store a thread-safe snapshot of current frequency statistics."""
        with self._status_lock:
            self._freq_snapshot = LoopFrequencyStats(
                mean=stats.mean,
                m2=stats.m2,
                min_freq=stats.min_freq,
                count=stats.count,
                last_freq=stats.last_freq,
                potential_freq=stats.potential_freq,
            )

    def _maybe_log_frequency(self, loop_count: int, print_interval_loops: int, stats: LoopFrequencyStats) -> None:
        """Emit frequency telemetry when enough loops have elapsed."""
        if loop_count % print_interval_loops != 0 or stats.count == 0:
            return

        variance = stats.m2 / stats.count if stats.count > 0 else 0.0
        lowest = stats.min_freq if stats.min_freq != float("inf") else 0.0
        logger.debug(
            "Loop freq - avg: %.2fHz, variance: %.4f, min: %.2fHz, last: %.2fHz, potential: %.2fHz, target: %.1fHz",
            stats.mean,
            variance,
            lowest,
            stats.last_freq,
            stats.potential_freq,
            self.target_frequency,
        )
        stats.reset()

    def start(self) -> None:
        """Start the worker thread that drives the 100 Hz control loop."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Move worker already running; start() ignored")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        logger.debug("Move worker started")

    def stop(self, reset_to_neutral: bool = True) -> None:
        """Request the worker thread to stop and wait for it to exit.

        Optionally resets the robot to a neutral position after stopping.
        """
        if self._thread is None or not self._thread.is_alive():
            logger.debug("Move worker not running; stop() ignored")
            return

        if reset_to_neutral:
            logger.info("Stopping movement manager and resetting to neutral position...")
        else:
            logger.info("Stopping movement manager...")

        # Clear any queued moves and stop current move
        self.clear_move_queue()

        # Stop the worker thread first so it doesn't interfere
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        logger.debug("Move worker stopped")

        if not reset_to_neutral:
            return

        # Reset to neutral position using goto_target (same approach as wake_up)
        try:
            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            neutral_antennas = [-0.1745, 0.1745]  # ~10° offset to reduce shaking
            neutral_body_yaw = 0.0

            # Use goto_target directly on the robot
            self.current_robot.goto_target(
                head=neutral_head_pose,
                antennas=neutral_antennas,
                duration=2.0,
                body_yaw=neutral_body_yaw,
            )

            logger.info("Reset to neutral position completed")

        except Exception as e:
            logger.error(f"Failed to reset to neutral position: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Return a lightweight status snapshot for observability."""
        with self._status_lock:
            pose_snapshot = clone_full_body_pose(self._last_commanded_pose)
            freq_snapshot = LoopFrequencyStats(
                mean=self._freq_snapshot.mean,
                m2=self._freq_snapshot.m2,
                min_freq=self._freq_snapshot.min_freq,
                count=self._freq_snapshot.count,
                last_freq=self._freq_snapshot.last_freq,
                potential_freq=self._freq_snapshot.potential_freq,
            )

        head_matrix = pose_snapshot[0].tolist() if pose_snapshot else None
        antennas = pose_snapshot[1] if pose_snapshot else None
        body_yaw = pose_snapshot[2] if pose_snapshot else None

        return {
            "queue_size": len(self.move_queue),
            "is_listening": self._is_listening,
            "idle_active": self._idle_active,
            "last_commanded_pose": {
                "head": head_matrix,
                "antennas": antennas,
                "body_yaw": body_yaw,
            },
            "loop_frequency": {
                "last": freq_snapshot.last_freq,
                "mean": freq_snapshot.mean,
                "min": freq_snapshot.min_freq,
                "potential": freq_snapshot.potential_freq,
                "samples": freq_snapshot.count,
            },
        }

    def working_loop(self) -> None:
        """Run the primary-move control loop with a single set_target() call per tick."""
        logger.debug("Starting enhanced movement control loop (100Hz)")

        loop_count = 0
        prev_loop_start = self._now()
        print_interval_loops = max(1, int(self.target_frequency * 2))
        freq_stats = self._freq_stats

        while not self._stop_event.is_set():
            loop_start = self._now()
            loop_count += 1

            if loop_count > 1:
                freq_stats = self._update_frequency_stats(loop_start, prev_loop_start, freq_stats)
            prev_loop_start = loop_start

            # 1) Poll external commands
            self._poll_signals(loop_start)

            # 2) Manage the primary move queue (start new move, end finished move, idle)
            self._update_primary_motion(loop_start)

            # 3) Build the primary full-body pose for this tick
            head, antennas, body_yaw = self._get_primary_pose(loop_start)

            # 4) Apply listening antenna freeze or blend-back
            antennas_cmd = self._calculate_blended_antennas(antennas)

            # 5) Single set_target call - the only control point
            self._issue_control_command(head, antennas_cmd, body_yaw)

            # 6) Adaptive sleep to align to next tick, then publish shared state
            sleep_time, freq_stats = self._schedule_next_tick(loop_start, freq_stats)
            self._publish_shared_state()
            self._record_frequency_snapshot(freq_stats)

            # 7) Periodic telemetry on loop frequency
            self._maybe_log_frequency(loop_count, print_interval_loops, freq_stats)

            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.debug("Movement control loop stopped")
