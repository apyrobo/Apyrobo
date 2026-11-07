"""plan_task says can_proceed false for a plan the safety policy refuses.

plan_task is how a model checks a task before running it. It used to return
can_proceed true and risk low for a target inside a no-go zone or outside every
allowed area, and then move_to and execute_task were refused. Each drive in the
plan is now checked against the zones without moving, with the same wording a
move gets.
"""

from __future__ import annotations

import json

import pytest

from apyrobo.mcp_server import build_server

ZONE = {"x_min": 2.0, "y_min": 2.0, "x_max": 4.0, "y_max": 4.0}
AREA = {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}


def call(server, name, arguments=None):
    msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    }
    result = server.handle(msg)["result"]
    return result["isError"], result["content"][0]["text"]


def plan(server, task):
    is_error, text = call(server, "plan_task", {"task": task})
    assert not is_error, text
    return json.loads(text)


def position(server):
    return json.loads(call(server, "robot_status")[1])["position"]


def refusal_of_move(server, x, y):
    is_error, text = call(server, "move_to", {"x": x, "y": y})
    assert is_error, text
    # "REFUSED by safety policy '<name>': <reason>. The robot did not move. ..."
    return text.split(": ", 1)[1].split(". The robot did not move.")[0]


def test_target_in_no_go_zone_cannot_proceed_with_the_move_reason():
    server = build_server("mock://turtlebot4", no_go_zones=[ZONE])
    before = position(server)
    result = plan(server, "go to (3,3)")
    assert result["can_proceed"] is False
    assert result["reason"].startswith("REFUSED by safety policy ")
    assert "at step 1: Position (3, 3) is inside collision zone" in result["reason"]
    assert result["steps"], "the plan is still shown"
    assert position(server) == before
    assert refusal_of_move(server, 3, 3) in result["reason"]


def test_target_outside_allowed_areas_cannot_proceed():
    server = build_server("mock://turtlebot4", allowed_areas=[AREA])
    result = plan(server, "go to (5,5)")
    assert result["can_proceed"] is False
    assert "Position (5, 5) is outside every allowed area" in result["reason"]
    assert refusal_of_move(server, 5, 5) in result["reason"]


def test_route_through_a_zone_cannot_proceed():
    server = build_server("mock://turtlebot4", no_go_zones=[ZONE])
    result = plan(server, "go to (6,6)")
    assert result["can_proceed"] is False
    assert "Path from (0, 0) to (6, 6) crosses collision zone" in result["reason"]
    assert refusal_of_move(server, 6, 6) in result["reason"]


def test_route_leaving_allowed_areas_cannot_proceed():
    areas = [AREA, {"x_min": 5.0, "y_min": 5.0, "x_max": 6.0, "y_max": 6.0}]
    server = build_server("mock://turtlebot4", allowed_areas=areas)
    result = plan(server, "go to (5.5,5.5)")
    assert result["can_proceed"] is False
    assert "Path from (0, 0) to (5.5, 5.5) leaves the allowed areas at" in result["reason"]
    assert refusal_of_move(server, 5.5, 5.5) in result["reason"]


def test_second_leg_of_a_delivery_is_checked_from_the_first_target():
    zone = {"x_min": 2.0, "y_min": 0.5, "x_max": 3.0, "y_max": 1.5}
    server = build_server("mock://turtlebot4", no_go_zones=[zone])
    result = plan(server, "deliver from (1, 1) to (5, 1)")
    assert result["can_proceed"] is False
    assert "Path from (1, 1) to (5, 1) crosses collision zone" in result["reason"]
    navigate = [
        i for i, s in enumerate(result["steps"], 1) if s["skill_id"].startswith("navigate_to")
    ]
    assert f"at step {navigate[1]}:" in result["reason"]


@pytest.mark.parametrize(
    "task", ["go to (1,1)", "go to (0.5, 0.5)", "deliver from (1, 0) to (0, 1)"]
)
def test_allowed_plans_still_proceed(task):
    server = build_server("mock://turtlebot4", no_go_zones=[ZONE])
    result = plan(server, task)
    assert result["can_proceed"] is True
    assert "reason" not in result


def test_no_zones_no_change():
    server = build_server("mock://turtlebot4")
    result = plan(server, "go to (30, 30)")
    assert result["can_proceed"] is True
    assert "reason" not in result


def test_plan_task_records_no_violation():
    server = build_server("mock://turtlebot4", no_go_zones=[ZONE])
    plan(server, "go to (3,3)")
    report = json.loads(call(server, "safety_report")[1])
    assert report["violations"] == []


def test_execute_task_is_still_refused_for_the_same_target():
    server = build_server("mock://turtlebot4", no_go_zones=[ZONE])
    is_error, text = call(server, "execute_task", {"task": "go to (3,3)"})
    assert is_error
    assert "Position (3, 3) is inside collision zone" in text


def test_route_not_checked_when_the_adapter_plans_its_own(monkeypatch):
    from apyrobo.core.robot import Robot

    server = build_server("mock://turtlebot4", no_go_zones=[ZONE])
    enforcer = server.toolbox.enforcer
    monkeypatch.setattr(Robot, "plans_own_path", property(lambda self: True))
    # The target is clear; only move() asks the robot's planner for the route.
    assert enforcer.move_problem(6.0, 6.0) is None
    # A target inside a zone is still refused.
    assert "inside collision zone" in enforcer.move_problem(3.0, 3.0)
