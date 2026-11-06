"""A move sent without a speed stays within the policy's max_speed."""
from __future__ import annotations

from typing import Any

import pytest

from apyrobo.core.adapters import CapabilityAdapter, MockAdapter
from apyrobo.core.robot import Robot
from apyrobo.core.schemas import RobotCapability
from apyrobo.safety.enforcer import SafetyEnforcer, SafetyPolicy


class DefaultSpeedAdapter(CapabilityAdapter):
    """Drives at *default* when move() gets no speed, like ros2://'s cmd_vel path."""

    def __init__(self, robot_name: str, default: Any = None, **kwargs: Any) -> None:
        super().__init__(robot_name, **kwargs)
        self.default_speed = default
        self.sent: list[float | None] = []

    def get_capabilities(self) -> RobotCapability:
        return RobotCapability(robot_id=self.robot_name, name=self.robot_name)

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        self.sent.append(speed)

    def stop(self) -> None:
        pass


def enforcer_for(adapter: CapabilityAdapter, max_speed: float) -> SafetyEnforcer:
    policy = SafetyPolicy(name="t", max_speed=max_speed, move_timeout=60.0)
    return SafetyEnforcer(Robot("r", adapter), policy=policy)


@pytest.mark.parametrize("speed", [None, 0, 0.0], ids=["none", "zero", "zero-float"])
def test_a_default_above_the_limit_is_lowered_to_the_limit(speed):
    adapter = DefaultSpeedAdapter("r", default=0.5)
    enforcer = enforcer_for(adapter, max_speed=0.2)
    enforcer.move(1.0, 0.0, speed=speed)
    assert adapter.sent == [0.2]
    assert enforcer.last_speed == 0.2
    assert enforcer.interventions[-1]["type"] == "speed_clamped"
    assert enforcer.interventions[-1]["default"] == 0.5
    assert enforcer.interventions[-1]["enforced"] == 0.2


def test_a_default_within_the_limit_is_kept_not_raised():
    adapter = DefaultSpeedAdapter("r", default=0.5)
    enforcer = enforcer_for(adapter, max_speed=1.5)
    enforcer.move(1.0, 0.0)
    assert adapter.sent == [0.5]
    assert enforcer.interventions == []


@pytest.mark.parametrize("default", [None, "fast", True, float("nan"), -1.0, 0.0])
def test_an_unknown_default_gets_the_limit(default):
    adapter = DefaultSpeedAdapter("r", default=default)
    enforcer = enforcer_for(adapter, max_speed=0.3)
    enforcer.move(1.0, 0.0)
    assert adapter.sent == [0.3]


def test_an_asked_speed_is_untouched_by_the_default():
    adapter = DefaultSpeedAdapter("r", default=0.5)
    enforcer = enforcer_for(adapter, max_speed=1.0)
    enforcer.move(1.0, 0.0, speed=0.1)
    assert adapter.sent == [0.1]


def test_robot_reports_its_adapters_default():
    assert Robot("r", DefaultSpeedAdapter("r", default=0.4)).default_speed == 0.4
    assert Robot("r", MockAdapter("r")).default_speed is None


def test_ros2_cmd_vel_default_is_declared():
    ros2_bridge = pytest.importorskip("apyrobo.core.ros2_bridge")
    adapter_cls = getattr(ros2_bridge, "ROS2Adapter", None)
    if adapter_cls is None:
        pytest.skip("rclpy not installed")
    assert adapter_cls.default_speed == 0.5
