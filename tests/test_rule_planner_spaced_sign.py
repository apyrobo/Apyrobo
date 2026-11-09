"""The rule planner refuses a sign with a space before its number.

"go to (- 1, 2)" drove to (1, 2) and reported "completed": the coordinate
reader skips a minus that is not right next to its digit, so the target was
mirrored. The same held for "x = - 3", "rotate - 90 degrees" and
"at speed - 0.2". Such a task is now refused through MCP with a
"Not planned:" reason, and the robot does not move. A sign written next to
its number reads as before.
"""

from __future__ import annotations

import json

import pytest

from apyrobo.mcp_server import build_server
from apyrobo.skills.agent import RuleBasedProvider

SPACED = [
    "go to (- 1, 2)",
    "go to ( - 1, 2)",
    "go to (-  5, 0)",
    "go to (1, - 2)",
    "go to (- 1, - 2)",
    "go to x = - 3 y = 4",
    "go to x=- 3 y=4",
    "go to - 3 4",
    "go to (+ 1, 2)",
    "rotate - 90 degrees",
    "go to (1, 2) at speed - 0.2",
    "deliver from (- 1, 1) to (5, 1)",
]
SIGN_NEXT_TO_NUMBER = [
    ("go to (-1, 2)", (-1.0, 2.0)),
    ("go to (1,-2)", (1.0, -2.0)),
    ("go to (1 ,-2)", (1.0, -2.0)),
    ("go to ( -1 , 2 )", (-1.0, 2.0)),
    ("go to x=-3 y=4", (-3.0, 4.0)),
    ("go to x = -3 y = 4", (-3.0, 4.0)),
    ("go to (3, 3) - then report", (3.0, 3.0)),
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


@pytest.mark.parametrize("task", SPACED)
def test_execute_task_refuses_spaced_sign(task):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "execute_task", {"task": task})
    assert is_error
    assert text.startswith("Not planned:")
    assert "sign with a space" in text
    assert position(server) == {"x": 0.0, "y": 0.0}


@pytest.mark.parametrize("task", SPACED)
def test_plan_task_says_it_cannot_proceed(task):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "plan_task", {"task": task})
    assert not is_error
    plan = json.loads(text)
    assert plan["can_proceed"] is False
    assert plan["steps"] == []
    assert "sign with a space" in plan["reason"]


@pytest.mark.parametrize("task,target", SIGN_NEXT_TO_NUMBER)
def test_sign_next_to_number_still_drives(task, target):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "execute_task", {"task": task})
    assert not is_error, text
    assert json.loads(text)["status"] == "completed"
    here = position(server)
    assert (here["x"], here["y"]) == target


def test_negative_turn_next_to_number_still_turns():
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "execute_task", {"task": "rotate -90 degrees"})
    assert not is_error, text
    assert json.loads(text)["status"] == "completed"


def test_cannot_read_names_the_sign():
    assert "sign with a space" in RuleBasedProvider.cannot_read("go to (- 1, 2)")
    assert RuleBasedProvider.cannot_read("go to (-1, 2)") is None
