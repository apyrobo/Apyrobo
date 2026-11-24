"""A state precondition nobody has set yet says how to set it.

A fresh SkillExecutor has no state, so a skill whose precondition is
``robot_idle=True`` fails. That stays (assuming the robot is idle would pass
a safety-relevant check silently), but the error now names the fix.
"""

from __future__ import annotations

from apyrobo.core.robot import Robot
from apyrobo.skills.executor import SkillExecutor
from apyrobo.skills.skill import Condition, Skill, SkillStatus


def _skill(value: object = True) -> Skill:
    return Skill(
        skill_id="needs_idle",
        name="Needs Idle",
        preconditions=[
            Condition(
                name="robot_idle",
                check_type="state",
                parameters={"key": "robot_idle", "value": value},
            )
        ],
    )


def _executor() -> SkillExecutor:
    return SkillExecutor(Robot.discover("mock://precondition_hint"))


def test_unset_state_names_the_fix() -> None:
    executor = _executor()
    ok, reason = executor.check_preconditions(_skill(), executor._robot)
    assert not ok
    assert reason.startswith(
        "State precondition 'robot_idle' not met: "
        "expected robot_idle=True, got None"
    )
    assert "run a skill that sets it first" in reason
    assert "executor.state.set('robot_idle', True)" in reason


def test_following_the_hint_passes() -> None:
    executor = _executor()
    executor.state.set("robot_idle", True)
    ok, reason = executor.check_preconditions(_skill(), executor._robot)
    assert ok, reason


def test_value_set_but_wrong_has_no_hint() -> None:
    executor = _executor()
    executor.state.set("robot_idle", False)
    ok, reason = executor.check_preconditions(_skill(), executor._robot)
    assert not ok
    assert reason == (
        "State precondition 'robot_idle' not met: "
        "expected robot_idle=True, got False"
    )


def test_hint_uses_the_expected_value() -> None:
    executor = _executor()
    ok, reason = executor.check_preconditions(_skill("docked"), executor._robot)
    assert not ok
    assert "executor.state.set('robot_idle', 'docked')" in reason


def test_executing_the_skill_still_fails_and_carries_the_hint() -> None:
    executor = _executor()
    events = []
    executor.on_event(lambda e: events.append(e))
    status = executor.execute_skill(_skill())
    assert status == SkillStatus.FAILED
    failed = [e for e in events if e.status == SkillStatus.FAILED]
    assert failed
    assert "executor.state.set('robot_idle', True)" in failed[-1].message


def test_skill_authoring_guide_quotes_the_real_error() -> None:
    from pathlib import Path

    guide = (
        Path(__file__).resolve().parent.parent / "docs" / "skill_authoring.md"
    ).read_text(encoding="utf-8")
    executor = _executor()
    _, reason = executor.check_preconditions(_skill(), executor._robot)
    assert f"`{reason}`" in guide
