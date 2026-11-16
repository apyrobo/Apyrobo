"""Every rule-planner task shown in the docs plans the way ``apyrobo mcp`` accepts.

The rule planner makes up a target for a task that gives no coordinates
("deliver package to dock 3" drove to (5, 5)) and drops steps it cannot read,
then reports completed. ``apyrobo mcp``, ``plan`` and ``execute`` refuse such
tasks; an example a reader copies must not be one of them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from apyrobo import Robot
from apyrobo.mcp_server import build_server

ROOT = Path(__file__).resolve().parent.parent
# README.md is left out until its 'Use an AI Agent' task is changed (another
# pending change edits README.md).
DOCS = sorted([*(ROOT / "docs").rglob("*.md"), *(ROOT / "demos").rglob("*.md")])
BLOCK = re.compile(r"```(?:python|bash)\n(.*?)```", re.S)
TASK = re.compile(
    r"""(?:\bagent\.(?:execute|plan)|\bcoordinator\.execute_task)\(\s*(?:task=)?"""
    r"""(["'])(?P<task>[^"'\n]+)\1"""
)
RULE_AGENT = re.compile(r"""Agent\(\s*provider\s*=\s*["']rule["']""")


def _rule_tasks() -> list[tuple[str, str]]:
    found = []
    for path in DOCS:
        for block in BLOCK.findall(path.read_text(encoding="utf-8")):
            # Only blocks whose agent is the rule planner on its own: an LLM
            # plans open tasks, and a custom skill library adds tasks the
            # bare rule planner does not know.
            if not RULE_AGENT.search(block) or "library=" in block:
                continue
            for match in TASK.finditer(block.split('provider="llm"')[0]):
                found.append((str(path.relative_to(ROOT)), match.group("task")))
    return sorted(set(found))


def _plan(task: str) -> dict:
    server = build_server("mock://docs_check")
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "plan_task", "arguments": {"task": task}},
        }
    )
    return json.loads(response["result"]["content"][0]["text"])


def test_the_docs_show_rule_planner_tasks():
    tasks = {task for _, task in _rule_tasks()}
    assert "deliver from (1, 1) to (5, 1)" in tasks
    assert "go to (2, 3)" in tasks


@pytest.mark.parametrize("doc,task", _rule_tasks())
def test_a_documented_rule_task_is_accepted(doc, task):
    plan = _plan(task)
    assert plan["can_proceed"] is True, f"{doc}: {task!r}: {plan.get('reason')}"


@pytest.mark.parametrize(
    "task", ["deliver package to dock 3", "navigate to point A",
             "navigate to position 2.5 3.0"],
)
def test_the_old_doc_tasks_were_refused(task):
    assert _plan(task)["can_proceed"] is False


def test_the_fleet_example_sends_robots_to_the_named_places():
    from apyrobo import Agent, SwarmBus, SwarmCoordinator

    robots = [Robot.discover(f"mock://bot_{i}") for i in (1, 2, 3)]
    bus = SwarmBus()
    for robot in robots:
        bus.register(robot)
    coordinator = SwarmCoordinator(bus, robots)
    result = coordinator.execute_task("deliver from (1, 1) to (5, 1)", Agent(provider="rule"))
    assert result.status.value == "completed"
    positions = {robot.get_position() for robot in robots}
    assert {(1.0, 1.0), (5.0, 1.0)} <= positions
