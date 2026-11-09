"""A rotate sent without a speed stays within the policy's max_angular_speed.

SafetyEnforcer.rotate clamped a speed it was given but passed ``speed=None``
through, so the robot turned at its own default rate: on unitree:// that is up
to 1.0 rad/s, and on mqtt:// and http:// the robot's firmware picks. A policy
of 0.3 rad/s did not hold for "rotate 90 degrees" sent without a speed, while
apyrobo mcp told the caller "default speed (policy limit 0.3 rad/s)".
The enforcer now hands the robot the policy limit, the same as move() does
for a move sent without a speed on a robot whose default it cannot read.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from apyrobo.core.adapters import CapabilityAdapter
from apyrobo.core.robot import Robot
from apyrobo.core.schemas import RobotCapability
from apyrobo.mcp_server import build_server
from apyrobo.safety.enforcer import SafetyEnforcer, SafetyPolicy


class FirmwareTurnAdapter(CapabilityAdapter):
    """Turns at its own *default* rate when rotate() gets no speed."""

    def __init__(self, robot_name: str, default: float = 3.0, **kwargs: Any) -> None:
        super().__init__(robot_name, **kwargs)
        self.default = default
        self.rates: list[float] = []

    def get_capabilities(self) -> RobotCapability:
        return RobotCapability(robot_id=self.robot_name, name=self.robot_name)

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        pass

    def rotate(self, angle_rad: float, speed: float | None = None) -> None:
        self.rates.append(self.default if speed is None else abs(speed))

    def stop(self) -> None:
        pass


def enforcer_for(adapter: CapabilityAdapter, limit: float) -> SafetyEnforcer:
    policy = SafetyPolicy(name="t", max_angular_speed=limit, move_timeout=60.0)
    return SafetyEnforcer(Robot("r", adapter), policy=policy)


def test_a_rotate_without_a_speed_turns_no_faster_than_the_limit():
    adapter = FirmwareTurnAdapter("r", default=3.0)
    enforcer = enforcer_for(adapter, limit=0.3)
    enforcer.rotate(1.57)
    assert adapter.rates == [0.3]
    assert enforcer.last_speed == 0.3


@pytest.mark.parametrize("angle", [1.57, -1.57], ids=["left", "right"])
def test_the_limit_is_sent_as_a_positive_rate_either_way(angle):
    adapter = FirmwareTurnAdapter("r")
    enforcer = enforcer_for(adapter, limit=0.3)
    enforcer.rotate(angle)
    assert adapter.rates == [0.3]


def test_a_given_speed_within_the_limit_is_sent_unchanged():
    adapter = FirmwareTurnAdapter("r")
    enforcer = enforcer_for(adapter, limit=0.5)
    enforcer.rotate(1.0, speed=0.2)
    assert adapter.rates == [0.2]
    assert enforcer.interventions == []


def test_a_given_speed_over_the_limit_is_still_clamped():
    adapter = FirmwareTurnAdapter("r")
    enforcer = enforcer_for(adapter, limit=0.5)
    enforcer.rotate(1.0, speed=2.0)
    assert adapter.rates == [0.5]
    assert enforcer.interventions[-1]["type"] == "angular_speed_clamped"


def test_mock_robot_receives_the_limit():
    robot = Robot.discover("mock://rotate_default")
    policy = SafetyPolicy(name="t", max_angular_speed=0.4, move_timeout=60.0)
    SafetyEnforcer(robot, policy=policy).rotate(0.5)
    assert robot._adapter.rotate_history[-1]["speed"] == 0.4


def test_apyrobo_mcp_rotate_without_a_speed_logs_the_limit(tmp_path):
    log = tmp_path / "audit.jsonl"
    server = build_server(audit_log=log)
    limit = server.toolbox.enforcer.policy.max_angular_speed
    request = {
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "rotate", "arguments": {"angle_rad": 1.0}},
    }
    reply = server.handle(request)
    assert not reply["result"].get("isError")
    last = json.loads(log.read_text().splitlines()[-1])
    assert last["enforced_speed"] == limit
