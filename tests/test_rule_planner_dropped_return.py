"""The rule planner's checks refuse a drive that would drop the way back.

The rule planner drives to the places a task names, one per step, so 'go to
(1, 1) and come back' planned the drive out alone and reported completed with
the robot left at (1, 1). apyrobo mcp and apyrobo plan / execute now refuse
such tasks; a way back that names its place is still the many-targets refusal,
and a single drive still plans.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from apyrobo.mcp_server import build_server
from apyrobo.skills.agent import RuleBasedProvider

DROPPED = [
    "go to (1, 1) and come back",
    "go to (1,1) then return",
    "go to position 2, 3 and back",
    "drive to (2, 2) then go back",
    "go to (3, 1) then go home",
    "navigate to (1, 2) and head back home",
    "go to (4, 4) and return to the start",
    "deliver from (1, 1) to (5, 1) and come back",
    "go to (2, 0), round trip",
]
KEPT = [
    ("go to (2, 2)", ["navigate_to"]),
    ("pick up the box at (5, 3)", ["navigate_to", "pick_object"]),
    ("deliver from (1, 1) to (5, 1)",
     ["navigate_to", "pick_object", "navigate_to", "place_object"]),
    ("go to (1, 1) then report status", ["navigate_to"]),
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


@pytest.mark.parametrize("task", DROPPED)
def test_execute_task_refuses_a_dropped_way_back_and_moves_nothing(task):
    server = build_server("mock://turtlebot4")
    before = position(server)
    is_error, text = call(server, "execute_task", {"task": task})
    assert is_error
    assert text.startswith("Not planned: this task asks the robot to come back")
    assert "'go to (1, 1)', then 'go to (0, 0)'" in text
    assert position(server) == before


@pytest.mark.parametrize("task", DROPPED)
def test_plan_task_says_it_cannot_proceed(task):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "plan_task", {"task": task})
    plan = json.loads(text)
    assert not is_error
    assert plan["can_proceed"] is False
    assert plan["steps"] == []
    assert plan["reason"].startswith("Not planned: this task asks the robot to come back")


@pytest.mark.parametrize("task,skills", KEPT)
def test_other_tasks_still_proceed(task, skills):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "plan_task", {"task": task})
    plan = json.loads(text)
    assert not is_error
    assert plan["can_proceed"] is True, plan.get("reason")
    assert [s["skill_id"].rsplit("_", 1)[0] for s in plan["steps"]] == skills


def test_a_way_back_with_its_own_place_is_the_many_targets_refusal():
    server = build_server("mock://turtlebot4")
    plan = json.loads(
        call(server, "plan_task", {"task": "go to (1, 1) then come back to (0, 0)"})[1]
    )
    assert plan["can_proceed"] is False
    assert plan["reason"].startswith("Not planned: this task names more than one place")


@pytest.mark.parametrize(
    "task,expected",
    [
        ("go to (1, 1) and come back", True),
        ("go to (1, 1) then return to (0, 0)", False),
        ("return to base", False),
        ("return to (2, 2)", False),
        ("go to (1, 1)", False),
        ("go to (1, 1) and back", True),
    ],
)
def test_drops_return_leg(task, expected):
    assert RuleBasedProvider.drops_return_leg(task) is expected


@pytest.mark.parametrize("command", ["plan", "execute"])
def test_cli_refuses_and_exits_1(command):
    out = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", command,
         "go to (1, 1) and come back", "--robot", "mock://tb"],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 1
    assert "Not planned: this task asks the robot to come back" in out.stdout + out.stderr
