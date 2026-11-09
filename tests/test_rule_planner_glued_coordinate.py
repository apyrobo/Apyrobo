"""The rule planner refuses a coordinate with a digit or sign glued to it.

The coordinate reader takes only the number next to the comma, so
"go to (1 000, 2)" drove to (0, 2), "go to (3+1, 3)" to (1, 3),
"go to (3-1, 3)" to (-1, 3) and "go to (2, 3+1)" to (2, 3); each reported
"completed". These tasks are now refused through MCP with a "Not planned:"
reason, and the robot does not move.
"""

from __future__ import annotations

import json

import pytest

from apyrobo.mcp_server import build_server

GLUED = [
    "go to (1 000, 2)",
    "go to (3+1, 3)",
    "go to (3*2, 3)",
    "go to (3-1, 3)",
    "go to (3x2, 3)",
    "go to (2, 3+1)",
    "go to (2, 3 000)",
    "go to (2, 2*2)",
    "go to 1 2, 3",
    "deliver from (1, 1) to (2 0, 2)",
    "deliver from (1+1, 1) to (2, 2)",
]
STILL_READ = [
    ("go to (3 , 3)", (3.0, 3.0), None),
    ("go to (+3, 3)", (3.0, 3.0), None),
    ("go to (-3, -3)", (-3.0, -3.0), None),
    ("go to ( 3 , 3 ) then report", (3.0, 3.0), None),
    ("go to 2, 3", (2.0, 3.0), None),
    ("go to room 2, (3, 4)", (3.0, 4.0), None),
    ("go to (3, 3) at speed 0.2", (3.0, 3.0), 0.2),
    ("go to (3, 3) at 0.2 m/s", (3.0, 3.0), 0.2),
    ("go to (1.5, 2) at speed 0.2. then report", (1.5, 2.0), 0.2),
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


@pytest.mark.parametrize("task", GLUED)
def test_execute_task_refuses_and_moves_nothing(task):
    server = build_server("mock://turtlebot4")
    before = status(server)
    is_error, text = call(server, "execute_task", {"task": task})
    assert is_error
    assert text.startswith("Not planned:")
    assert "only in part" in text
    assert status(server) == before


@pytest.mark.parametrize("task", GLUED)
def test_plan_task_says_it_cannot_proceed(task):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "plan_task", {"task": task})
    assert not is_error
    plan = json.loads(text)
    assert plan["steps"] == []
    assert plan["can_proceed"] is False
    assert plan["reason"].startswith("Not planned:")


@pytest.mark.parametrize(("task", "target", "speed"), STILL_READ)
def test_plain_coordinates_still_plan(task, target, speed):
    server = build_server("mock://turtlebot4")
    _, text = call(server, "plan_task", {"task": task})
    plan = json.loads(text)
    assert plan["can_proceed"] is True
    params = plan["steps"][0]["parameters"]
    assert (params["x"], params["y"]) == target
    if speed is not None:
        assert params["speed"] == speed
