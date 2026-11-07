"""execute_task refuses a task that names more than one place to drive.

The rule planner keeps one coordinate per drive: 'go to (3, 3) then come back
to (0, 0)' planned only the drive to (0, 0) and reported completed, so a no-go
zone over (3, 3) was never refused in the reply. Such tasks are now refused,
asking for one target per call; a delivery from one point to another still
plans both.
"""

from __future__ import annotations

import json

import pytest

from apyrobo.mcp_server import build_server

ZONE = [{"x_min": 2.0, "y_min": 2.0, "x_max": 4.0, "y_max": 4.0}]

MANY_TARGETS = [
    "go to (3, 3) then come back to (0,0)",
    "go to (1,1) then (1,0)",
    "go to (1,1), pick up the box, then go to (3,3)",
    "go to x=1 y=1 then x=1 y=0",
    "go to 1 1 and then to 1 0",
    "deliver from (1, 1) to (5, 1) then (6, 6)",
]
ONE_TARGET = [
    "go to (1, 2)",
    "go to (1, 1) then (1, 1)",
    "go to (1, 2) at speed 0.1",
    "go to x=1 y=2",
    "pick up the box at (1, 1)",
    "deliver from (1, 1) to (5, 1)",
    "deliver the package to (5, 1)",
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


@pytest.mark.parametrize("task", MANY_TARGETS)
def test_execute_task_refuses_more_than_one_target_and_moves_nothing(task):
    server = build_server("mock://turtlebot4")
    before = position(server)
    is_error, text = call(server, "execute_task", {"task": task})
    assert is_error
    assert text.startswith("Not planned: this task names more than one place")
    assert "one target per call" in text
    assert position(server) == before


@pytest.mark.parametrize("task", MANY_TARGETS)
def test_plan_task_says_it_cannot_proceed_with_more_than_one_target(task):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "plan_task", {"task": task})
    assert not is_error
    plan = json.loads(text)
    assert plan["steps"] == []
    assert plan["can_proceed"] is False
    assert plan["reason"].startswith("Not planned: this task names more than one place")


@pytest.mark.parametrize("task", ONE_TARGET)
def test_one_target_and_deliveries_still_plan(task):
    server = build_server("mock://turtlebot4")
    _, text = call(server, "plan_task", {"task": task})
    plan = json.loads(text)
    assert plan["steps"]
    assert "reason" not in plan


def test_a_good_move_no_longer_hides_a_zone_in_the_same_task():
    # Before: planned only (0, 0), drove there and said completed.
    server = build_server("mock://turtlebot4", no_go_zones=ZONE)
    is_error, text = call(
        server, "execute_task", {"task": "go to (3, 3) then come back to (0,0)"}
    )
    assert is_error
    assert "one target per call" in text
    # Sent one at a time, the zone is refused and the other move runs.
    is_error, text = call(server, "execute_task", {"task": "go to (3, 3)"})
    assert is_error
    assert "collision zone" in text
    is_error, _ = call(server, "execute_task", {"task": "go to (1, 0)"})
    assert not is_error
    assert position(server) == {"x": 1.0, "y": 0.0}


def test_delivery_between_two_points_still_visits_both():
    server = build_server("mock://turtlebot4")
    is_error, text = call(
        server, "execute_task", {"task": "deliver from (1, 1) to (5, 1)"}
    )
    assert not is_error
    targets = [
        (step["parameters"]["x"], step["parameters"]["y"])
        for step in json.loads(text)["plan"]
        if step["skill_id"].startswith("navigate_to")
    ]
    assert targets == [(1.0, 1.0), (5.0, 1.0)]
    assert position(server) == {"x": 5.0, "y": 1.0}
