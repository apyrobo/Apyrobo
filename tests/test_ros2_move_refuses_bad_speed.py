"""ROS2Adapter.move refuses a speed of 0, below 0, NaN or infinite.

On the /cmd_vel fallback ``speed or default`` turned 0 into the default
speed, a negative speed was published as linear.x < 0 (the robot drove
backwards until the navigation timeout) and NaN was published as is. A
Python-API caller reaches the adapter through Robot.move with no policy in
between, so the adapter itself refuses before anything is sent.
"""
from __future__ import annotations

import math

import pytest

# Importing the mocked bridge module installs the ROS 2 sys.modules mocks.
from tests.test_ros2_bridge_mocked import _build_adapter, bridge_module

BAD_SPEEDS = [0, 0.0, -0.1, -1.0, float("nan"), float("inf"), float("-inf")]


def _adapter(has_nav2: bool):
    adapter, _ = _build_adapter(has_nav2=has_nav2)
    adapter._nav_timeout_sec = 0.05
    adapter._position = (0.0, 0.0)
    adapter._orientation = 0.0
    adapter._cmd_vel_pub.publish.reset_mock()
    return adapter


@pytest.mark.parametrize("speed", BAD_SPEEDS, ids=repr)
@pytest.mark.parametrize("has_nav2", [False, True], ids=["cmd_vel", "nav2"])
def test_bad_speed_is_refused_before_anything_is_sent(speed, has_nav2):
    adapter = _adapter(has_nav2)
    with pytest.raises(ValueError, match="speed must be a positive, finite number"):
        adapter.move(10.0, 0.0, speed=speed)
    adapter._cmd_vel_pub.publish.assert_not_called()
    adapter._nav2_client.send_goal_async.assert_not_called()
    assert adapter._state_machine == bridge_module.RobotState.IDLE


@pytest.mark.parametrize("speed", [True, "fast", [0.5]], ids=repr)
def test_non_number_speed_is_refused(speed):
    adapter = _adapter(False)
    with pytest.raises(ValueError, match="speed must be a number"):
        adapter.move(10.0, 0.0, speed=speed)
    adapter._cmd_vel_pub.publish.assert_not_called()


def test_cmd_vel_never_publishes_a_reverse_or_nan_speed():
    adapter = _adapter(False)
    for speed in (-1.0, float("nan")):
        with pytest.raises(ValueError):
            adapter.move(10.0, 0.0, speed=speed)
    for call in adapter._cmd_vel_pub.publish.call_args_list:
        x = call.args[0].linear.x
        if isinstance(x, float):
            assert math.isfinite(x) and x >= 0


@pytest.mark.parametrize("speed", [None, 0.2, 1])
def test_none_and_positive_speeds_still_drive(speed):
    adapter = _adapter(False)
    with pytest.raises(bridge_module.Ros2NavigationError):
        # far target + tiny timeout: it drives, then times out
        adapter.move(10.0, 0.0, speed=speed)
    sent = [c.args[0].linear.x for c in adapter._cmd_vel_pub.publish.call_args_list]
    expected = 0.5 if speed is None else float(speed)
    assert sent and sent[0] == pytest.approx(expected)


def test_check_move_speed_accepts_numpy_like_reals():
    np = pytest.importorskip("numpy")
    bridge_module.check_move_speed(np.float32(0.3))
    with pytest.raises(ValueError):
        bridge_module.check_move_speed(np.float64("nan"))
