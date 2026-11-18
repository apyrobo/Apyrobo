"""plan_task never says risk_level low for a plan it says cannot proceed.

A refused plan (a target in a no-go zone, outside every allowed area, a route
through a zone, or a task the rule planner will not run) used to come back
with can_proceed false next to risk_level low, two answers a model reads
differently. It now says risk_level high, and a policy refusal carries the
same code, policy and violation execute_task's refusal does, so a model can
branch on the code before anything runs.
"""

from __future__ import annotations

import json

import pytest

from apyrobo.mcp_server import build_server

ZONE = {"x_min": 2.0, "y_min": 2.0, "x_max": 4.0, "y_max": 4.0}
AREA = {"x_min": -1.0, "y_min": -1.0, "x_max": 5.0, "y_max": 5.0}


def call(server, name, arguments=None):
    msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    }
    return server.handle(msg)["result"]


def plan(server, task):
    result = call(server, "plan_task", {"task": task})
    assert not result["isError"], result
    return json.loads(result["content"][0]["text"])


@pytest.fixture()
def server():
    server = build_server("mock://turtlebot4", no_go_zones=[ZONE], allowed_areas=[AREA])
    # structuredContent goes only to clients on MCP 2025-06-18 or later.
    server.handle(
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        }
    )
    return server


@pytest.mark.parametrize(
    "task,code",
    [
        ("go to (3, 3)", "collision_zone"),
        ("go to (5, 5)", "collision_zone_path"),  # the straight route crosses it
        ("go to (9, 0)", "outside_allowed_area"),
    ],
)
def test_policy_refusal_is_high_risk_with_the_execute_task_code(server, task, code):
    result = plan(server, task)
    assert result["can_proceed"] is False
    assert result["risk_level"] == "high"
    refusal = result["refusal"]
    assert refusal["status"] == "refused"
    assert refusal["code"] == code
    assert refusal["policy"] == "default+cli"
    assert refusal["step"] == 1
    assert refusal["steps_completed"] == 0

    run = call(server, "execute_task", {"task": task})
    assert run["isError"]
    assert run["structuredContent"] == refusal


def test_planner_refusal_is_high_risk(server):
    # A step the rule planner does not make: refused with a reason, no steps.
    result = plan(server, "go to (1, 1) then wait 5 seconds")
    assert result["can_proceed"] is False
    assert result["reason"]
    assert result["risk_level"] == "high"
    assert "refusal" not in result  # not a safety-policy refusal


def test_plan_that_can_proceed_is_unchanged(server):
    result = plan(server, "go to (1, 1)")
    assert result["can_proceed"] is True
    assert result["risk_level"] == "low"
    assert "reason" not in result
    assert "refusal" not in result
