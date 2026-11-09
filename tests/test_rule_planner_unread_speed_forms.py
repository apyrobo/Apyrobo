"""The rule planner refuses a speed or a digit it does not read.

"go to x=3, y=4 at 0.2" (a bare number after "at"), "go to (1,2) v=0.3"
and "go to (1, 2) quickly at 0.1" planned the default 0.5 m/s with
can_proceed true, "at speed ½" planned 0.5, and "go to (²3, 1)"
drove to (3, 1). Each would report completed. Such a task is now refused
through MCP with a "Not planned:" reason, and the robot does not move.
"""

from __future__ import annotations

import json

import pytest

from apyrobo.mcp_server import build_server
from apyrobo.skills.agent import RuleBasedProvider

UNREAD_SPEED = [
    "go to x=3, y=4 at 0.2",
    "go to (1, 2) quickly at 0.1",
    "go to (1, 2) at 0.3 please",
    "go to (1,2) v=0.3",
    "go to (1,2) v: 0.3",
    "go to (1,2) velocity 0.3",
    "go to (1,2) at a velocity of 0.3",
    "go to (1,2) spd=0.3",
    "go to (1,2) at 2 m/s then at 0.5",
]
ODD_DIGIT = [
    "go to (1, 2) at speed ½",
    "go to (1, 2) at ¼ m/s",
    "go to (²3, 1)",
    "go to (3², 1)",
    "go to (₃, 1)",
    "go to (1, ⅔)",
]
STILL_READ = [
    ("go to (1, 2) at 0.2 m/s", (1.0, 2.0)),
    ("go to (1, 2) at speed 0.2", (1.0, 2.0)),
    ("go to (3, 3) at speed 0.2.", (3.0, 3.0)),
    ("go to (1, 2)", (1.0, 2.0)),
    ("stop at 3, 4 after going to (1, 2)", (3.0, 4.0)),
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


def position(server):
    return json.loads(call(server, "robot_status")[1])["position"]


@pytest.mark.parametrize("task", UNREAD_SPEED + ODD_DIGIT)
def test_execute_task_refuses(task):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "execute_task", {"task": task})
    assert is_error
    assert text.startswith("Not planned:")
    assert position(server) == {"x": 0.0, "y": 0.0}


@pytest.mark.parametrize("task", UNREAD_SPEED + ODD_DIGIT)
def test_plan_task_says_it_cannot_proceed(task):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "plan_task", {"task": task})
    assert not is_error
    plan = json.loads(text)
    assert plan["can_proceed"] is False
    assert plan["steps"] == []


@pytest.mark.parametrize("task", UNREAD_SPEED)
def test_reason_names_the_speed(task):
    assert "speed the rule planner does not read" in RuleBasedProvider.cannot_read(task)


@pytest.mark.parametrize("task", ODD_DIGIT)
def test_reason_names_the_digit(task):
    assert "superscript" in RuleBasedProvider.cannot_read(task)


@pytest.mark.parametrize("task,target", STILL_READ)
def test_read_forms_still_drive(task, target):
    assert RuleBasedProvider.cannot_read(task) is None
    if task.startswith("stop"):
        return
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "execute_task", {"task": task})
    assert not is_error, text
    assert json.loads(text)["status"] == "completed"
    here = position(server)
    assert (here["x"], here["y"]) == target


def test_turn_at_an_angle_is_not_a_speed():
    assert RuleBasedProvider.cannot_read("rotate at 90 degrees") is None
