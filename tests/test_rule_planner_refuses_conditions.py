"""The rule planner refuses a task that sets a condition it cannot check.

'go to (3, 3) if the door is open', 'unless someone is in the way', 'wait for
a person then go to (3, 3)' and the like planned the drive alone, so through
apyrobo mcp the robot drove whether or not the condition held. The planner
has no way to check a condition, so these tasks are now refused with a reason;
plain moves, and words that only look like conditions, still plan.
"""

from __future__ import annotations

import json

import pytest

from apyrobo.mcp_server import build_server
from apyrobo.skills.agent import RuleBasedProvider

CONDITIONAL = [
    "go to (3, 3) if the door is open",
    "if the path is clear, go to (3, 3)",
    "go to (3, 3) only if it is safe",
    "go to (3, 3) unless the door is closed",
    "go to (3, 3) unless someone is in the way",
    "go to (3, 3) when the battery is above 50%",
    "whenever the alarm rings go to (3, 3)",
    "go to (3, 3) until you see a wall",
    "go to (3, 3) once the area is empty",
    "go to (3, 3) after the human leaves",
    "go to (3, 3) before someone arrives",
    "wait for a person then go to (3, 3)",
    "go to (3, 3) as long as it is clear",
    "in case of fire go to (3, 3)",
    "go to (3, 3) provided the door is open",
    "go to (3, 3) every time the bell rings",
    "deliver from (1, 1) to (5, 1) if nobody is there",
    "Go To (3, 3) IF The Door Is Open",
]
KEPT = [
    "go to (3, 3)",
    "go to (1, 1) and stop",
    "go to (1, 1) then report status",
    "go to (2, 2) at 0.2 m/s",
    "go to (1, 1) and after that stop",
    "drive to the lift at (2, 3)",
    "go to the shelf at (2, 2)",
]


def _status(server):
    msg = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "robot_status", "arguments": {}},
    }
    return json.loads(server.handle(msg)["result"]["content"][0]["text"])


def _call(server, name, task):
    msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": {"task": task}},
    }
    return server.handle(msg)["result"]


@pytest.mark.parametrize("task", CONDITIONAL)
def test_cannot_read_names_the_condition(task):
    reason = RuleBasedProvider.cannot_read(task)
    assert reason is not None
    assert "condition" in reason


@pytest.mark.parametrize("task", CONDITIONAL)
def test_rule_planner_makes_no_plan(task):
    assert RuleBasedProvider().plan(task, [], []) == []


@pytest.mark.parametrize("task", KEPT)
def test_plain_moves_still_plan(task):
    assert RuleBasedProvider.cannot_read(task) is None
    assert RuleBasedProvider().plan(task, [], [])


@pytest.mark.parametrize("task", CONDITIONAL)
def test_mcp_plan_task_says_it_cannot_proceed(task):
    server = build_server("mock://turtlebot4")
    result = _call(server, "plan_task", task)
    body = json.loads(result["content"][0]["text"])
    assert body["can_proceed"] is False
    assert "condition" in json.dumps(body)


@pytest.mark.parametrize("task", CONDITIONAL)
def test_mcp_execute_task_refuses_and_robot_stays(task):
    server = build_server("mock://turtlebot4")
    result = _call(server, "execute_task", task)
    assert result["isError"] is True
    assert "condition" in result["content"][0]["text"]
    pos = _status(server)["position"]
    assert (pos["x"], pos["y"]) == (0.0, 0.0)
