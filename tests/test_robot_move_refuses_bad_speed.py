"""Robot.move refuses a speed of 0, below 0, NaN, infinite or not a number.

Only ros2:// checked the speed itself; gazebo:// and isaac_sim:// used
``speed or default``, so speed=0 drove at the default speed and a negative
speed went through. The check now runs in Robot.move, before any adapter
is called, so it holds for every scheme. None still means the default.
"""
from __future__ import annotations

import contextlib
from unittest.mock import MagicMock

import pytest

from apyrobo.core import adapters as adapters_module
from apyrobo.core.robot import Robot

# Register the adapters that live in their own modules too.
for _mod in ("isaac_adapter", "unitree_adapter", "vda5050_adapter"):
    with contextlib.suppress(ImportError):
        __import__(f"apyrobo.core.{_mod}")

BAD_SPEEDS = [0, 0.0, -0.1, -1, float("nan"), float("inf"), float("-inf")]
NOT_NUMBERS = ["0.5", True, [0.5], object()]


@pytest.mark.parametrize("speed", BAD_SPEEDS, ids=repr)
@pytest.mark.parametrize("uri", ["mock://bot", "gazebo://turtlebot4"])
def test_bad_speed_refused_and_robot_does_not_move(uri, speed):
    robot = Robot.discover(uri)
    before = robot.get_position()
    with pytest.raises(ValueError, match="positive, finite number"):
        robot.move(2.0, 0.0, speed=speed)
    assert robot.get_position() == before


@pytest.mark.parametrize("speed", NOT_NUMBERS, ids=repr)
def test_non_number_speed_refused(speed):
    robot = Robot.discover("mock://bot")
    with pytest.raises(ValueError, match="must be a number"):
        robot.move(2.0, 0.0, speed=speed)
    assert robot.get_position() == (0.0, 0.0)


@pytest.mark.parametrize("scheme", sorted(adapters_module._ADAPTER_REGISTRY))
@pytest.mark.parametrize("speed", [0, -0.5, float("nan")], ids=repr)
def test_every_registered_adapter_is_never_called(scheme, speed):
    cls = adapters_module._ADAPTER_REGISTRY[scheme]
    adapter = MagicMock(spec=cls)
    robot = Robot(robot_id="r", adapter=adapter)
    with pytest.raises(ValueError):
        robot.move(1.0, 1.0, speed=speed)
    adapter.move.assert_not_called()


@pytest.mark.parametrize("speed", [None, 0.5, 2])
def test_mock_still_moves_with_default_or_positive_speed(speed):
    robot = Robot.discover("mock://bot")
    robot.move(2.0, 3.0, speed=speed)
    assert robot.get_position() == pytest.approx((2.0, 3.0))


def test_ros2_bridge_still_exports_the_check():
    from apyrobo.core.adapters import check_move_speed

    ros2_bridge = pytest.importorskip("apyrobo.core.ros2_bridge")
    assert ros2_bridge.check_move_speed is check_move_speed
