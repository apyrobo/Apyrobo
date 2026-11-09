"""The rule planner refuses a number it would read only in part.

"go to (0x10, 1)" drove to (10, 1), "go to (1.5.2, 1)" to (5.2, 1) and
"go to (2:30, 1)" to (30, 1), and "at speed +0.2" ran at the default speed;
each reported "completed". These tasks are now refused through MCP with a
"Not planned:" reason, and the robot does not move.
"""

from __future__ import annotations

import json

import pytest

from apyrobo.mcp_server import build_server

PARTLY_READ = [
    "go to (0x10, 1)",
    "go to (0b11, 1)",
    "go to (0o7, 1)",
    "go to (1.5.2, 1)",
    "go to (1..5, 2)",
    "go to (2:30, 1)",
    "go to (1, 2 : 5)",
    "deliver from (1, 1) to (0x5, 1)",
]
PLUS_SPEED = [
    "go to (3, 3) at speed +0.2",
    "go to (3, 3) at + 0.2 m/s",
    "go to (3, 3) speed: +0.2",
]
STILL_READ = [
    ("go to (+3, 3)", (3.0, 3.0), None),
    ("go to (3., 3)", (3.0, 3.0), None),
    ("go to x=3 y=4", (3.0, 4.0), None),
    ("go to (1.5, 2) at speed 0.2. then report", (1.5, 2.0), 0.2),
    ("go to (10, 0) speed:0.3", (10.0, 0.0), 0.3),
    ("go to (3, 3) at 0.2m/s", (3.0, 3.0), 0.2),
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


@pytest.mark.parametrize(
    ("task", "words"),
    [(t, "only in part") for t in PARTLY_READ] + [(t, "plus sign") for t in PLUS_SPEED],
)
def test_execute_task_refuses_and_moves_nothing(task, words):
    server = build_server("mock://turtlebot4")
    before = status(server)
    is_error, text = call(server, "execute_task", {"task": task})
    assert is_error
    assert text.startswith("Not planned:")
    assert words in text
    assert status(server) == before


@pytest.mark.parametrize("task", PARTLY_READ + PLUS_SPEED)
def test_plan_task_says_it_cannot_proceed(task):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "plan_task", {"task": task})
    assert not is_error
    plan = json.loads(text)
    assert plan["steps"] == []
    assert plan["can_proceed"] is False
    assert plan["reason"].startswith("Not planned:")


@pytest.mark.parametrize(("task", "target", "speed"), STILL_READ)
def test_numbers_it_reads_whole_still_plan(task, target, speed):
    server = build_server("mock://turtlebot4")
    _, text = call(server, "plan_task", {"task": task})
    plan = json.loads(text)
    assert plan["can_proceed"] is True
    params = plan["steps"][0]["parameters"]
    assert (params["x"], params["y"]) == target
    if speed is not None:
        assert params["speed"] == speed
