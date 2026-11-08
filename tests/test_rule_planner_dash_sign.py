"""The rule planner refuses a minus sign it does not read.

"go to (−5, 0)" (U+2212 MINUS SIGN) drove to (5, 0), and the same with an
en dash, em dash or full-width hyphen-minus, each reported "completed". The
coordinate reader only knows the ASCII hyphen-minus, so it dropped the sign.
These tasks are now refused through MCP with a "Not planned:" reason, and the
robot does not move.
"""

from __future__ import annotations

import json

import pytest

from apyrobo.mcp_server import build_server

DASH_SIGN = [
    "go to (−5, 0)",
    "go to (–5, 1)",
    "go to (—5, 1)",
    "go to (﹣5, 1)",
    "go to (－5, 1)",
    "go to (‐ 5, 1)",
    "go to (1, −2)",
    "rotate −90 degrees",
    "go to (5, 1) at speed −0.2",
    "deliver from (−1, 0) to (2, 0)",
]
PLAIN = [
    ("go to (-5, 0)", (-5.0, 0.0)),
    ("go to (1, -2)", (1.0, -2.0)),
    ("go to (3, 3) – carefully", (3.0, 3.0)),
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


@pytest.mark.parametrize("task", DASH_SIGN)
def test_execute_task_refuses_and_moves_nothing(task):
    server = build_server("mock://turtlebot4")
    before = status(server)
    is_error, text = call(server, "execute_task", {"task": task})
    assert is_error
    assert text.startswith("Not planned:")
    assert "hyphen-minus" in text
    assert status(server) == before


@pytest.mark.parametrize("task", DASH_SIGN)
def test_plan_task_says_it_cannot_proceed(task):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "plan_task", {"task": task})
    assert not is_error
    plan = json.loads(text)
    assert plan["steps"] == []
    assert plan["can_proceed"] is False
    assert "hyphen-minus" in plan["reason"]


@pytest.mark.parametrize(("task", "target"), PLAIN)
def test_plain_minus_and_a_dash_away_from_numbers_still_drive(task, target):
    server = build_server("mock://turtlebot4")
    is_error, _ = call(server, "execute_task", {"task": task})
    assert not is_error
    pos = status(server)["position"]
    assert (pos["x"], pos["y"]) == target
