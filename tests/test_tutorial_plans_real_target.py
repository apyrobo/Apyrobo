"""The tasks apyrobo tutorial and apyrobo shell suggest are ones the CLI plans.

Both used to suggest 'navigate to the dock'. The rule planner has no map, so it
turned that into a drive to (0, 0), and apyrobo plan / execute / mcp refuse it
('No target'). The examples now give coordinates the planner reads exactly.
"""

from __future__ import annotations

import re
from pathlib import Path

import apyrobo.cli as cli_module
from apyrobo.cli import _TUTORIAL_STEPS, _rule_plan_check
from apyrobo.core.robot import Robot
from apyrobo.safety.enforcer import SafetyEnforcer
from apyrobo.skills.agent import Agent

_PLAN_CALL = re.compile(r"agent\.plan\('([^']+)', robot\)")


def _tutorial_task() -> str:
    step = next(s for s in _TUTORIAL_STEPS if s["title"].startswith("Step 3"))
    found = _PLAN_CALL.findall(step["code"])
    assert len(found) == 1
    return found[0]


def _shell_task() -> str:
    source = Path(cli_module.__file__).read_text(encoding="utf-8")
    start = source.index("def cmd_shell(")
    found = _PLAN_CALL.findall(source[start:source.index("_TUTORIAL_STEPS = [", start)])
    assert len(found) == 1
    return found[0]


def _check(task: str):
    robot = Robot.discover("mock://tutorial_target")
    agent = Agent(provider="rule")
    graph = agent.plan(task, robot)
    return _rule_plan_check(
        agent, robot, SafetyEnforcer(robot), "mock://tutorial_target", task, graph
    )


def test_tutorial_task_passes_the_cli_plan_check() -> None:
    graph, _, reason = _check(_tutorial_task())
    assert reason is None
    steps = graph.get_execution_order()
    assert [s.skill_id for s in steps] == ["navigate_to_0"]
    params = graph.get_parameters("navigate_to_0")
    assert (params["x"], params["y"]) == (2.0, 1.0)


def test_shell_hint_task_passes_the_cli_plan_check() -> None:
    _, _, reason = _check(_shell_task())
    assert reason is None


def test_named_place_is_still_refused() -> None:
    _, _, reason = _check("navigate to the dock")
    assert reason is not None and reason.startswith("No target")
