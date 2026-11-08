"""The rule planner refuses a task it would misread, instead of guessing.

A negation ("don't go to (3, 3)") used to plan a drive to (3, 3); "(1e3, 2)"
drove to (3, 2); "(1, 2, 3)" dropped the 3; "(.5, 1)" and "(-.5, 1)" drove to
(5, 1). Each is now refused, in the
planner and through MCP, and the robot stays put.
"""

from __future__ import annotations

import json

import pytest

from apyrobo.mcp_server import build_server
from apyrobo.skills.agent import RuleBasedProvider

NEGATED = [
    "do not go to (1,1)",
    "don't go to (3,3)",
    "Don’t drive to (2, 2)",
    "dont move to (2, 2)",
    "never navigate to (2, 2)",
    "go to (2, 2), not (1, 1)",
]
UNREADABLE = [
    "go to (1e3, 2)",
    "go to (2, 1E-2)",
    "go to (1,2,3)",
    "go to 1, 2, 3",
    "go to (.5, 1)",
    "go to (-.5, 1)",
    "go to x=.5 y=1",
    "move to 1 .5",
    "go to (1_0, 1)",
    "pick up the box at (2_0, 1)",
]
READABLE = [
    "go to (3, 3)",
    "go to (-1.5, 2)",
    "deliver from (1, 1) to (5, 1)",
    "navigate to (0.5, 0.25) then stop",
    "report status",
    "go to the note at (1, 1)",
    "go to (5., 1)",
    "go to (1, 2) at speed 0.2.",
]


@pytest.mark.parametrize("task", NEGATED + UNREADABLE)
def test_planner_refuses_a_task_it_would_misread(task):
    provider = RuleBasedProvider()
    assert provider.cannot_read(task)
    assert provider.plan(task, [], []) == []


@pytest.mark.parametrize("task", READABLE)
def test_planner_still_plans_readable_tasks(task):
    provider = RuleBasedProvider()
    assert provider.cannot_read(task) is None
    assert provider.plan(task, [], [])


def test_negation_also_blocks_custom_patterns():
    provider = RuleBasedProvider(patterns=[(r"charge", "go_to_dock")])
    assert provider.plan("don't charge", [], []) == []
    assert provider.plan("charge", [], []) == [{"skill_id": "go_to_dock", "parameters": {}}]


def call(server, name, arguments=None):
    msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    }
    result = server.handle(msg)["result"]
    return result["isError"], result["content"][0]["text"]


def position(server):
    return json.loads(call(server, "robot_status")[1])["position"]


@pytest.mark.parametrize("task", NEGATED + UNREADABLE)
def test_execute_task_refuses_and_moves_nothing(task):
    server = build_server("mock://turtlebot4")
    before = position(server)
    is_error, text = call(server, "execute_task", {"task": task})
    assert is_error
    assert text.startswith("Not planned:")
    assert position(server) == before


@pytest.mark.parametrize("task", NEGATED + UNREADABLE)
def test_plan_task_says_it_cannot_proceed(task):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "plan_task", {"task": task})
    assert not is_error
    plan = json.loads(text)
    assert plan["steps"] == []
    assert plan["can_proceed"] is False
    assert plan["reason"].startswith("Not planned:")


def test_execute_task_still_drives_a_plain_target():
    server = build_server("mock://turtlebot4")
    is_error, _ = call(server, "execute_task", {"task": "go to (1, 2)"})
    assert not is_error
    assert position(server) == {"x": 1.0, "y": 2.0}


@pytest.mark.parametrize("task", ["go to (.5, 1)", "go to (-.5, 1)"])
def test_leading_point_is_refused_not_read_as_ten_times_further(task):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "execute_task", {"task": task})
    assert is_error
    assert "starts with a decimal point" in text
    assert position(server) == {"x": 0.0, "y": 0.0}
