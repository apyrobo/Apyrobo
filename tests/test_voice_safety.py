"""Voice commands run through the SafetyEnforcer.

docs/comparison.md and docs/architecture.md say every command passes through
the SafetyEnforcer before reaching the robot. voice_loop() (used by
``apyrobo voice``) and VoiceAgent used to hand the raw robot to the agent, so
a heard "go to (3, 4) at speed 5" was sent to the robot at 5 m/s and no
no-go zone was checked.
"""

from __future__ import annotations

import pytest

from apyrobo.core.robot import Robot
from apyrobo.safety.enforcer import SafetyEnforcer, SafetyPolicy
from apyrobo.skills.agent import Agent
from apyrobo.voice import MockVoiceAdapter, VoiceAgent, voice_loop


def _spy_robot() -> tuple[Robot, list[dict]]:
    robot = Robot.discover("mock://turtlebot4")
    moves: list[dict] = []
    real_move = robot.move

    def spy(*args, **kwargs):
        moves.append(dict(kwargs))
        return real_move(*args, **kwargs)

    robot.move = spy  # type: ignore[method-assign]
    return robot, moves


def _loop(robot, text, **kwargs):
    return voice_loop(
        agent=Agent(provider="rule"),
        robot=robot,
        adapter=MockVoiceAdapter([text, "stop"]),
        max_turns=3,
        **kwargs,
    )


def _voice_agent(robot, text, **kwargs) -> str:
    va = VoiceAgent(
        agent=Agent(provider="rule"),
        robot=robot,
        stt=MockVoiceAdapter(responses=[text]),
        **kwargs,
    )
    return va.run()


# --- voice_loop -----------------------------------------------------------

def test_voice_loop_clamps_speed_to_default_policy():
    robot, moves = _spy_robot()
    _loop(robot, "go to (3, 4) at speed 5")
    assert len(moves) == 1
    assert moves[0]["speed"] == pytest.approx(1.5)


def test_voice_loop_clamps_speed_to_given_policy():
    robot, moves = _spy_robot()
    policy = SafetyPolicy(name="slow", max_speed=0.2)
    _loop(robot, "go to (3, 4) at speed 5", safety_policy=policy)
    assert len(moves) == 1
    assert moves[0]["speed"] == pytest.approx(0.2)


def test_voice_loop_accepts_registered_policy_name():
    robot, moves = _spy_robot()
    _loop(robot, "go to (3, 4) at speed 5", safety_policy="strict")
    assert len(moves) == 1
    from apyrobo.safety.enforcer import POLICY_REGISTRY

    assert moves[0]["speed"] == pytest.approx(POLICY_REGISTRY["strict"].max_speed)


def test_voice_loop_refuses_target_in_no_go_zone():
    robot, moves = _spy_robot()
    policy = SafetyPolicy(
        name="zone",
        collision_zones=[{"x_min": 2, "x_max": 4, "y_min": 2, "y_max": 4}],
    )
    turns = _loop(robot, "go to (3, 3)", safety_policy=policy)
    assert moves == []
    assert len(turns) == 1
    assert turns[0]["result"].status.value != "completed"


def test_voice_loop_speed_within_limit_is_unchanged():
    robot, moves = _spy_robot()
    _loop(robot, "go to (3, 4) at speed 0.3")
    assert moves[0]["speed"] == pytest.approx(0.3)


def test_voice_loop_uses_given_enforcer_as_is():
    robot, moves = _spy_robot()
    enforcer = SafetyEnforcer(robot, policy=SafetyPolicy(name="e", max_speed=0.4))
    _loop(enforcer, "go to (3, 4) at speed 5")
    assert moves[0]["speed"] == pytest.approx(0.4)
    # The caller's enforcer saw the command (no second wrapper hid it).
    assert enforcer.moves_sent == 1


# --- VoiceAgent ------------------------------------------------------------

def test_voice_agent_clamps_speed_to_default_policy():
    robot, moves = _spy_robot()
    _voice_agent(robot, "go to (1, 1) at speed 9")
    assert len(moves) == 1
    assert moves[0]["speed"] == pytest.approx(1.5)


def test_voice_agent_clamps_speed_to_given_policy():
    robot, moves = _spy_robot()
    _voice_agent(
        robot, "go to (1, 1) at speed 9",
        safety_policy=SafetyPolicy(name="slow", max_speed=0.25),
    )
    assert moves[0]["speed"] == pytest.approx(0.25)


def test_voice_agent_refuses_target_in_no_go_zone():
    robot, moves = _spy_robot()
    policy = SafetyPolicy(
        name="zone",
        collision_zones=[{"x_min": 0.5, "x_max": 1.5, "y_min": 0.5, "y_max": 1.5}],
    )
    summary = _voice_agent(robot, "go to (1, 1)", safety_policy=policy)
    assert moves == []
    assert "completed" not in summary.lower().split(":")[0]


def test_voice_agent_uses_given_enforcer_as_is():
    robot, moves = _spy_robot()
    enforcer = SafetyEnforcer(robot, policy=SafetyPolicy(name="e", max_speed=0.4))
    _voice_agent(enforcer, "go to (1, 1) at speed 9")
    assert moves[0]["speed"] == pytest.approx(0.4)
    assert enforcer.moves_sent == 1
