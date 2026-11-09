"""The rule planner refuses a speed unit or wording it does not read.

"go to (3, 3) at speed 0.2 m/min" drove at 0.2 m/s, and "go to (3, 3) at
most 0.2 m/s" ran at the default 0.5 m/s; each reported "completed". These
tasks are now refused through MCP with a "Not planned:" reason, and the
robot does not move.
"""

from __future__ import annotations

import json

import pytest

from apyrobo.mcp_server import build_server

OTHER_UNIT = [
    "go to (3, 3) at speed 0.2 m/min",
    "go to (3, 3) at speed 0.2 m/h",
    "go to (3, 3) at speed 0.2 in/s",
    "go to (3, 3) at speed 0.2 m/s²",
    "go to (3, 3) at speed 0.2 meter/s",
    "go to (3, 3) at 0.2 mps",
    "go to (3, 3) at 20 cm per second",
    "go to (3, 3) at 0.2 meters/second",
    "go to (3, 3) at speed 1 m per minute",
]
OTHER_WORDING = [
    "go to (3, 3) at most 0.2 m/s",
    "go to (3, 3) max 0.2 m/s",
    "go to (3, 3) under 0.2 m/s",
    "go to (3, 3), 0.2 m/s",
    "go to (3, 3) at a speed of about 0.2",
    "go to (3, 3) speed limit 0.2",
    "go to (3, 3) at speed up to 0.2",
    "go to (3, 3) speed <= 0.2",
]
STILL_READ = [
    ("go to (3, 3) at speed 0.2", 0.2),
    ("go to (3, 3) speed: 0.3", 0.3),
    ("go to (3, 3) speed=0.3", 0.3),
    ("go to (3, 3) at a speed of 0.2", 0.2),
    ("go to (3, 3) at a speed of 0.2 m/s", 0.2),
    ("go to (3, 3) at 0.1 m/s", 0.1),
    ("go to (3, 3) at 0.2m/s", 0.2),
    ("go to (3, 3) at speed 0.2 m / s", 0.2),
    ("go to (3, 3) at speed 0.2 m/s.", 0.2),
    ("deliver from (1, 1) to (5, 1) at 0.2 m/s", 0.2),
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
    [(t, "unit") for t in OTHER_UNIT] + [(t, "wording") for t in OTHER_WORDING],
)
def test_execute_task_refuses_and_moves_nothing(task, words):
    server = build_server("mock://turtlebot4")
    before = status(server)
    is_error, text = call(server, "execute_task", {"task": task})
    assert is_error
    assert text.startswith("Not planned:")
    assert words in text
    assert status(server) == before


@pytest.mark.parametrize("task", OTHER_UNIT + OTHER_WORDING)
def test_plan_task_says_it_cannot_proceed(task):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "plan_task", {"task": task})
    assert not is_error
    plan = json.loads(text)
    assert plan["steps"] == []
    assert plan["can_proceed"] is False
    assert plan["reason"].startswith("Not planned:")


@pytest.mark.parametrize(("task", "speed"), STILL_READ)
def test_speeds_it_reads_still_plan(task, speed):
    server = build_server("mock://turtlebot4")
    _, text = call(server, "plan_task", {"task": task})
    plan = json.loads(text)
    assert plan["can_proceed"] is True
    for step in plan["steps"]:
        if "x" in step["parameters"]:
            assert step["parameters"]["speed"] == speed
