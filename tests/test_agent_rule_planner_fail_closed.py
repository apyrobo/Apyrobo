"""Agent.plan / Agent.execute with the rule planner fail closed.

A task the rule planner cannot plan as asked (no readable target, or steps it
would drop) used to drive the robot to a made-up target and end "completed".
The Python API now makes the same checks ``apyrobo mcp`` and ``apyrobo plan``
make: no plan, ``last_refusal`` says why, and the robot does not move.
"""

from __future__ import annotations

import pytest

from apyrobo.core.robot import Robot
from apyrobo.skills.agent import Agent, RuleBasedProvider


@pytest.fixture
def robot() -> Robot:
    return Robot.discover("mock://fail_closed_bot")


def _status(result) -> str:
    return getattr(result.status, "value", str(result.status))


def _position(robot: Robot) -> tuple[float, float]:
    x, y = robot.get_position()
    return float(x), float(y)


@pytest.mark.parametrize(
    "task, words",
    [
        ("go to the kitchen", "No target"),
        ("deliver the package to the dock", "No target"),
        ("go to (1, 1) then pick up the box", "drop the pick-up"),
        ("zzz_no_match_at_all_xyzzy", "did not recognise this task"),
    ],
)
def test_plan_refuses_with_the_reason(robot, task, words):
    agent = Agent(provider="rule")
    graph = agent.plan(task, robot)
    assert len(graph) == 0
    assert agent.last_refusal is not None
    assert words in agent.last_refusal


@pytest.mark.parametrize(
    "task",
    [
        "go to the kitchen",
        "go to (1, 1) then pick up the box",
        "go to (1, 1) and come back",
        "patrol the perimeter",
    ],
)
def test_execute_reports_failed_and_the_robot_stays(robot, task):
    agent = Agent(provider="rule")
    before = _position(robot)
    result = agent.execute(task, robot)
    assert _status(result) == "failed"
    assert result.error == agent.last_refusal
    assert result.error and result.error.startswith(("No target", "Not planned"))
    assert _position(robot) == pytest.approx(before)


def test_refusal_names_the_python_api_not_mcp_tools(robot):
    agent = Agent(provider="rule")
    agent.plan("zzz_no_match_at_all_xyzzy", robot)
    assert 'Agent(provider="llm")' in agent.last_refusal
    assert "move_to" not in agent.last_refusal
    assert "--provider" not in agent.last_refusal


def test_a_readable_task_still_runs(robot):
    agent = Agent(provider="rule")
    result = agent.execute("go to (2, 1)", robot)
    assert _status(result) == "completed"
    assert agent.last_refusal is None
    assert _position(robot) == pytest.approx((2.0, 1.0))


def test_last_refusal_clears_on_the_next_plan(robot):
    agent = Agent(provider="rule")
    agent.plan("go to the kitchen", robot)
    assert agent.last_refusal is not None
    agent.plan("go to (2, 1)", robot)
    assert agent.last_refusal is None


def test_a_pattern_the_user_registered_skips_the_check(robot):
    agent = Agent(provider="rule", patterns=[(r"wave hello", "report_status")])
    graph = agent.plan("wave hello", robot)
    assert [s.skill_id.rsplit("_", 1)[0] for s in graph.get_execution_order()] == [
        "report_status"
    ]
    assert agent.last_refusal is None


def test_check_false_returns_the_plan_unchecked(robot):
    agent = Agent(provider="rule")
    graph = agent.plan("zzz_no_match_at_all_xyzzy", robot, check=False)
    ids = [s.skill_id.rsplit("_", 1)[0] for s in graph.get_execution_order()]
    assert ids == ["report_status"]
    assert agent.last_refusal is None


def test_the_check_reuses_the_plan_it_has(robot, monkeypatch):
    calls: list[str] = []
    original = RuleBasedProvider.plan

    def counted(self, task, *args, **kwargs):
        calls.append(task)
        return original(self, task, *args, **kwargs)

    monkeypatch.setattr(RuleBasedProvider, "plan", counted)
    agent = Agent(provider="rule")
    agent.plan("go to (1, 1) then pick up the box", robot)
    agent.plan("go to (2, 1)", robot)
    assert len(calls) == 2
