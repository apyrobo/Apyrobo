"""The rule planner makes one motion per task, or refuses.

"go to (3, 3) and rotate 90 degrees" drove and dropped the turn;
"rotate 90 degrees then go to (1, 1)" drove only; "rotate 90 degrees twice"
turned once; "turn left 90 degrees then turn right 90 degrees" turned left
and stopped there. Each reported "completed". They are now refused through
MCP with a "Not planned:" reason, and the robot does not move.
"""

from __future__ import annotations

import json

import pytest

from apyrobo.mcp_server import build_server

MANY_MOTIONS = [
    "go to (3, 3) and rotate 90 degrees",
    "rotate 90 degrees then go to (1, 1)",
    "go to (1, 1) and turn left",
    "pick up the box at (2, 1) and turn right 45 degrees",
    "rotate 90 degrees twice",
    "rotate 45 degrees two times",
    "turn left 90 degrees then turn right 90 degrees",
    "rotate 90 degrees and 45 degrees",
    "rotate 30 degrees, then again",
]
ONE_MOTION = [
    ("rotate 90 degrees", "rotate"),
    ("turn right 45 deg", "rotate"),
    ("rotate 1.5 rad", "rotate"),
    ("go to (3, 3)", "navigate_to"),
    ("navigate to (2, 2)", "navigate_to"),
    ("deliver from (1, 1) to (5, 1)", "navigate_to"),
]


def call(server, name, arguments=None):
    msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    }
    result = server.handle(msg)["result"]
    return result["isError"], result["content"][0]["text"]


def status(server):
    return json.loads(call(server, "robot_status")[1])


@pytest.mark.parametrize("task", MANY_MOTIONS)
def test_execute_task_refuses_and_moves_nothing(task):
    server = build_server("mock://turtlebot4")
    before = status(server)
    is_error, text = call(server, "execute_task", {"task": task})
    assert is_error
    assert text.startswith("Not planned:")
    assert "one motion per call" in text
    assert status(server) == before


@pytest.mark.parametrize("task", MANY_MOTIONS)
def test_plan_task_says_it_cannot_proceed(task):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "plan_task", {"task": task})
    assert not is_error
    plan = json.loads(text)
    assert plan["steps"] == []
    assert plan["can_proceed"] is False
    assert "one motion per call" in plan["reason"]


@pytest.mark.parametrize(("task", "skill"), ONE_MOTION)
def test_one_motion_still_plans(task, skill):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "plan_task", {"task": task})
    assert not is_error
    plan = json.loads(text)
    assert plan["can_proceed"] is True
    assert plan["steps"][0]["skill_id"].startswith(skill)
