"""The rule planner turns one way per task, or refuses.

"rotate 90 degrees left then right" turned left once, "turn right 90 degrees
and back left" turned left instead of right, and "rotate 90 degrees left or
right" picked left. Each reported "completed". A turn that names both
directions is now refused through MCP with a "Not planned:" reason, and the
robot does not move. "anti clockwise" (with a space) turned right; it now
turns left like "anticlockwise" and "anti-clockwise".
"""

from __future__ import annotations

import json
import math

import pytest

from apyrobo.mcp_server import build_server
from apyrobo.skills.agent import RuleBasedProvider

BOTH_DIRECTIONS = [
    "rotate 90 degrees left then right",
    "rotate 90 degrees right then left",
    "turn 90 degrees left and right",
    "turn left and right 90 degrees",
    "turn right 90 degrees and back left",
    "rotate 90 degrees left, 90 right",
    "rotate 90 degrees left or right",
    "turn 90 degrees clockwise then anticlockwise",
    "turn 45 degrees counter-clockwise, then clockwise",
    "turn right 45 degrees, rotating left",
]
ONE_DIRECTION = [
    ("rotate 90 degrees", 90.0),
    ("turn left 90 degrees", 90.0),
    ("turn right 45 deg", -45.0),
    ("rotate 90 degrees clockwise", -90.0),
    ("rotate 90 degrees counterclockwise", 90.0),
    ("rotate 90 degrees counter-clockwise", 90.0),
    ("rotate 90 degrees anti-clockwise", 90.0),
    ("rotate 90 degrees anticlockwise", 90.0),
    ("rotate 90 degrees anti clockwise", 90.0),
    ("rotate 90 degrees counter clockwise", 90.0),
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


@pytest.mark.parametrize("task", BOTH_DIRECTIONS)
def test_execute_task_refuses_and_moves_nothing(task):
    server = build_server("mock://turtlebot4")
    before = status(server)
    is_error, text = call(server, "execute_task", {"task": task})
    assert is_error
    assert text.startswith("Not planned:")
    assert "both directions" in text
    assert status(server) == before


@pytest.mark.parametrize("task", BOTH_DIRECTIONS)
def test_plan_task_says_it_cannot_proceed(task):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "plan_task", {"task": task})
    assert not is_error
    plan = json.loads(text)
    assert plan["steps"] == []
    assert plan["can_proceed"] is False
    assert "both directions" in plan["reason"]


@pytest.mark.parametrize(("task", "degrees"), ONE_DIRECTION)
def test_one_direction_plans_that_turn(task, degrees):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "plan_task", {"task": task})
    assert not is_error
    plan = json.loads(text)
    assert plan["can_proceed"] is True
    (step,) = plan["steps"]
    assert step["skill_id"].startswith("rotate")
    assert step["parameters"]["angle_rad"] == pytest.approx(math.radians(degrees))


@pytest.mark.parametrize(
    ("task", "found"),
    [
        ("rotate 90 degrees", set()),
        ("turn left 90 degrees", {"left"}),
        ("turn right 90 degrees", {"right"}),
        ("rotate 90 degrees anti clockwise", {"left"}),
        ("rotate 90 degrees counterclockwise", {"left"}),
        ("rotate 90 degrees clockwise", {"right"}),
        ("turn left then right 90 degrees", {"left", "right"}),
        ("turn anticlockwise then clockwise 90 degrees", {"left", "right"}),
    ],
)
def test_turn_directions(task, found):
    assert RuleBasedProvider.turn_directions(task) == found
