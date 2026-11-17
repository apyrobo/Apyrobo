"""The rule planner refuses a list of numbered places instead of driving to two of them.

"deliver packages to docks 1, 2, and 3" read 1 and 2 as one coordinate pair,
and through apyrobo mcp the robot drove to (1, 2). A list of three or more
numbers ("1, 2, and 3", "1, 2 or 3") or a plural noun before a number pair
("docks 1, 2") is now refused, in the planner and through MCP, and the robot
stays put.
"""

from __future__ import annotations

import json

import pytest

from apyrobo.mcp_server import build_server
from apyrobo.skills.agent import RuleBasedProvider

LISTS = [
    "deliver packages to docks 1, 2, and 3",
    "deliver packages to docks 1, 2 and 3",
    "go to 1, 2 and 3",
    "go to 1, 2, or 3",
    "go to docks 1, 2",
    "visit rooms 3, 4",
    "check shelves 1.5, 2",
]
STILL_READ = [
    "go to (1, 2)",
    "go to 1, 2",
    "go to coordinates 1, 2",
    "go to (1, 2) and then (3, 4)",
    "go to (1, 2), and then (3, 4)",
    "deliver from (1, 1) to (5, 1)",
    "go to (1, 2) at 0.2 m/s",
]


@pytest.mark.parametrize("task", LISTS)
def test_planner_refuses_a_list_of_numbered_places(task):
    provider = RuleBasedProvider()
    reason = provider.cannot_read(task)
    assert reason is not None
    assert reason.startswith("Not planned:")
    assert "(x, y)" in reason
    assert provider.plan(task, [], []) == []


@pytest.mark.parametrize("task", STILL_READ)
def test_planner_still_reads_a_coordinate_pair(task):
    assert RuleBasedProvider.cannot_read(task) is None


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


def test_execute_task_does_not_drive_to_two_dock_numbers():
    server = build_server("mock://turtlebot4")
    before = position(server)
    is_error, text = call(
        server, "execute_task", {"task": "deliver packages to docks 1, 2, and 3"}
    )
    assert is_error
    assert text.startswith("Not planned:")
    assert "list of numbers" in text
    assert position(server) == before


def test_plan_task_says_it_cannot_proceed():
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "plan_task", {"task": "go to docks 1, 2"})
    assert not is_error
    plan = json.loads(text)
    assert plan["steps"] == []
    assert plan["can_proceed"] is False
    assert "list of numbers" in plan["reason"]
