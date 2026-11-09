"""apyrobo skill compose (ComposeREPL) drives the robot through a SafetyEnforcer.

Before the fix, ``test navigate_to x=1 y=1 speed=9`` sent move(speed=9.0) to
the raw robot and reported completed, and no no-go zone was checked, against
docs/comparison.md ("Every command passes through the SafetyEnforcer before
reaching the robot").
"""
from __future__ import annotations

from typing import Any

import pytest

from apyrobo import Robot
from apyrobo.safety.enforcer import SafetyEnforcer, SafetyPolicy
from apyrobo.skills.compose import ComposeREPL
from apyrobo.skills.library import SkillLibrary

ZONE = {"x_min": 2.0, "x_max": 4.0, "y_min": 2.0, "y_max": 5.0}


def _spy_robot(uri: str = "mock://turtlebot4") -> tuple[Any, list[dict[str, Any]]]:
    robot = Robot.discover(uri)
    calls: list[dict[str, Any]] = []
    orig = robot.move

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return orig(*args, **kwargs)

    robot.move = spy  # type: ignore[method-assign]
    return robot, calls


def _repl(robot: Any, **kwargs: Any) -> ComposeREPL:
    return ComposeREPL(robot, SkillLibrary.from_decorated(), **kwargs)


def test_test_command_clamps_to_default_policy() -> None:
    robot, calls = _spy_robot()
    out = _repl(robot).run_command("test navigate_to x=1 y=1 speed=9")
    assert "completed" in out
    assert len(calls) == 1
    assert calls[0]["speed"] == pytest.approx(1.5)


def test_run_command_clamps_to_given_policy() -> None:
    robot, calls = _spy_robot()
    repl = _repl(robot, safety_policy=SafetyPolicy(name="slow", max_speed=0.3))
    repl.run_command("add navigate_to x=1 y=1 speed=1.0")
    out = repl.run_command("run")
    assert "COMPLETED" in out
    assert [c["speed"] for c in calls] == [pytest.approx(0.3)]


def test_in_limit_speed_is_unchanged() -> None:
    robot, calls = _spy_robot()
    repl = _repl(robot)
    repl.run_command("add navigate_to x=1 y=1 speed=0.4")
    repl.run_command("run")
    assert calls[0]["speed"] == pytest.approx(0.4)


def test_test_command_refuses_no_go_zone() -> None:
    robot, calls = _spy_robot()
    policy = SafetyPolicy(name="zoned", collision_zones=[ZONE])
    out = _repl(robot, safety_policy=policy).run_command("test navigate_to x=3 y=4 speed=0.5")
    assert "failed" in out
    assert calls == []


def test_run_command_refuses_no_go_zone() -> None:
    robot, calls = _spy_robot()
    policy = SafetyPolicy(name="zoned", collision_zones=[ZONE])
    repl = _repl(robot, safety_policy=policy)
    repl.run_command("add navigate_to x=3 y=4 speed=0.5")
    out = repl.run_command("run")
    assert "FAILED" in out
    assert calls == []


def test_policy_by_registered_name() -> None:
    robot, calls = _spy_robot()
    out = _repl(robot, safety_policy="strict").run_command("test navigate_to x=1 y=1 speed=9")
    assert "completed" in out
    strict = SafetyEnforcer(Robot.discover("mock://other"), policy="strict").policy.max_speed
    assert calls[0]["speed"] == pytest.approx(strict)
    assert calls[0]["speed"] < 1.5


def test_robot_is_wrapped() -> None:
    robot, _ = _spy_robot()
    repl = _repl(robot)
    assert isinstance(repl._robot, SafetyEnforcer)


def test_existing_enforcer_is_used_as_is() -> None:
    robot, calls = _spy_robot()
    enforcer = SafetyEnforcer(robot, policy=SafetyPolicy(name="own", max_speed=0.2))
    repl = _repl(enforcer)
    assert repl._robot is enforcer
    repl.run_command("test navigate_to x=1 y=1 speed=1")
    assert calls[0]["speed"] == pytest.approx(0.2)


def test_switched_robot_is_wrapped_with_same_policy() -> None:
    robot, _ = _spy_robot()
    policy = SafetyPolicy(name="slow", max_speed=0.3)
    repl = _repl(robot, safety_policy=policy)
    out = repl.run_command("robot mock://other")
    assert "Switched" in out
    assert isinstance(repl._robot, SafetyEnforcer)
    assert repl._robot.policy.max_speed == pytest.approx(0.3)


def test_switched_robot_default_policy_clamps() -> None:
    robot, _ = _spy_robot()
    repl = _repl(robot)
    repl.run_command("robot mock://other")
    assert isinstance(repl._robot, SafetyEnforcer)
    assert repl._robot.policy.max_speed == pytest.approx(1.5)
