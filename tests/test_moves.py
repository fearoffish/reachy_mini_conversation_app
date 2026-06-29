import time
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from reachy_mini_conversation_app.moves import (
    IDLE_HEAD_FALL_S,
    IDLE_HEAD_HOLD_S,
    IDLE_HEAD_RISE_S,
    IDLE_ANTENNA_FALL_S,
    IDLE_ANTENNA_RISE_S,
    IDLE_ANTENNA_AMP_RAD,
    IdleMove,
    MovementManager,
    _gesture_envelope,
)


def _make_idle_move(interpolation_duration: float = 1.0) -> IdleMove:
    """Build an IdleMove starting from a neutral pose."""
    return IdleMove(
        interpolation_start_pose=np.eye(4, dtype=np.float32),
        interpolation_start_antennas=(0.0, 0.0),
        interpolation_duration=interpolation_duration,
    )


def test_gesture_envelope_ramps_holds_and_returns() -> None:
    """The envelope ramps 0->1 over rise, holds at 1, then ramps back to 0."""
    rise, hold, fall = 1.0, 1.0, 1.0
    assert _gesture_envelope(0.0, rise, hold, fall) == 0.0
    assert _gesture_envelope(-1.0, rise, hold, fall) == 0.0
    assert _gesture_envelope(rise, rise, hold, fall) == pytest.approx(1.0)
    assert _gesture_envelope(rise + hold / 2, rise, hold, fall) == pytest.approx(1.0)
    assert _gesture_envelope(rise + hold + fall, rise, hold, fall) == 0.0
    # Monotonic, bounded ramp up
    assert 0.0 < _gesture_envelope(rise / 2, rise, hold, fall) < 1.0


def test_idle_move_holds_still_between_gestures() -> None:
    """Before the first scheduled gesture the robot is fully still at neutral.

    Head gestures are scheduled at least 30s out and antenna twitches at least 60s
    out, so a few seconds into the idle phase nothing moves and there is no z-bob.
    """
    move = _make_idle_move(interpolation_duration=1.0)
    head_pose, antennas, body_yaw = move.evaluate(1.0 + 5.0)

    assert np.allclose(head_pose[:3, :3], np.eye(3), atol=1e-9)  # no rotation
    assert np.allclose(head_pose[:3, 3], 0.0, atol=1e-9)  # no translation / breathing bob
    assert np.allclose(antennas, move.neutral_antennas, atol=1e-9)
    assert body_yaw == 0.0


def test_idle_move_head_gesture_moves_head_then_returns() -> None:
    """A scheduled head gesture rotates the head at its peak and returns to neutral."""
    move = _make_idle_move(interpolation_duration=1.0)
    # Force a deterministic gesture starting at idle_time=0 with a pure yaw offset.
    move._head_start = 0.0
    move._head_target = np.array([0.0, 0.0, 8.0])  # roll, pitch, yaw (deg)
    move._antenna_start = 1e9  # keep antennas out of the way

    peak_t = 1.0 + IDLE_HEAD_RISE_S + IDLE_HEAD_HOLD_S / 2
    head_pose, _, _ = move.evaluate(peak_t)
    assert not np.allclose(head_pose[:3, :3], np.eye(3), atol=1e-3)  # head rotated

    # Well past the gesture it rests at neutral again (and reschedules).
    after_t = 1.0 + IDLE_HEAD_RISE_S + IDLE_HEAD_HOLD_S + IDLE_HEAD_FALL_S + 0.5
    head_pose_after, _, _ = move.evaluate(after_t)
    assert np.allclose(head_pose_after[:3, :3], np.eye(3), atol=1e-9)
    assert move._head_start > after_t - 1.0  # next gesture scheduled into the future


def test_idle_move_antenna_twitch_offsets_antennas() -> None:
    """A scheduled antenna twitch peaks at the configured amplitude and returns."""
    move = _make_idle_move(interpolation_duration=1.0)
    move._head_start = 1e9  # keep head out of the way
    move._antenna_start = 0.0
    move._antenna_sign = 1.0

    peak_t = 1.0 + IDLE_ANTENNA_RISE_S  # start of (zero-length) hold == full amplitude
    _, antennas, _ = move.evaluate(peak_t)
    expected = move.neutral_antennas + IDLE_ANTENNA_AMP_RAD
    assert np.allclose(antennas, expected, atol=1e-6)

    after_t = 1.0 + IDLE_ANTENNA_RISE_S + IDLE_ANTENNA_FALL_S + 0.5
    _, antennas_after, _ = move.evaluate(after_t)
    assert np.allclose(antennas_after, move.neutral_antennas, atol=1e-9)


def test_stop_can_skip_neutral_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sleep shutdown should stop the movement loop without undoing the sleep pose."""
    robot = MagicMock()
    manager = MovementManager(robot)
    started = threading.Event()

    def fake_working_loop() -> None:
        started.set()
        while not manager._stop_event.is_set():
            time.sleep(0.001)

    monkeypatch.setattr(manager, "working_loop", fake_working_loop)

    manager.start()
    assert started.wait(timeout=1.0)

    manager.stop(reset_to_neutral=False)

    assert manager._thread is None
    robot.goto_target.assert_not_called()
