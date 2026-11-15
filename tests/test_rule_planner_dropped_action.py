"""The rule planner's checks refuse a drive that would drop a pick-up.

The rule planner plans the first kind of task it recognises, so 'go to (5, 3)
then pick up the object' planned the drive alone and reported completed with
nothing picked up. apyrobo mcp and apyrobo plan / execute now refuse such
tasks; a pick-up with its own target still plans the drive and the pick.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from apyrobo.mcp_server import build_server

DROPPED = [
    "go to (5, 3) then pick up the object",
    "go to position 5, 3 then pick up the object",
    "go to (5, 3) and grab the box",
    "drive to (1, 1) to get the cup",
    "navigate to (2, 2) and grasp the handle",
]
KEPT = [
    ("go to (2, 2)", ["navigate_to"]),
    ("pick up the object at (5, 3)", ["navigate_to", "pick_object"]),
    ("grab the box at (1, 1)", ["navigate_to", "pick_object"]),
    ("deliver from (1, 1) to (5, 1)",
     ["navigate_to", "pick_object", "navigate_to", "place_object"]),
    ("report status", ["report_status"]),
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
def test_execute_task_refuses_a_dropped_action_and_moves_nothing(task):
    server = build_server("mock://turtlebot4")
    before = position(server)
    is_error, text = call(server, "execute_task", {"task": task})
    assert is_error
    assert text.startswith("Not planned: this task asks for a drive and then a pick-up")
    assert position(server) == before


@pytest.mark.parametrize("task", DROPPED)
def test_plan_task_says_it_cannot_proceed(task):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "plan_task", {"task": task})
    plan = json.loads(text)
    assert not is_error
    assert plan["can_proceed"] is False
    assert plan["steps"] == []
    assert plan["reason"].startswith("Not planned: this task asks for a drive and then")


@pytest.mark.parametrize("task,skills", KEPT)
def test_other_tasks_still_proceed(task, skills):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "plan_task", {"task": task})
    plan = json.loads(text)
    assert not is_error
    assert plan["can_proceed"] is True, plan.get("reason")
    assert [s["skill_id"].rsplit("_", 1)[0] for s in plan["steps"]] == skills


def test_cli_plan_refuses_the_quickstart_task_it_used_to_drop():
    out = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", "plan",
         "go to position 5, 3 then pick up the object", "--robot", "mock://tb"],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 1
    assert "Not planned: this task asks for a drive and then a pick-up" in out.stdout + out.stderr


def test_quickstart_task_plans_the_pick_up():
    with open("docs/quickstart_5min.md", encoding="utf-8") as f:
        doc = f.read()
    assert 'agent.execute("pick up the object at (5, 3)", robot)' in doc
    server = build_server("mock://turtlebot4")
    plan = json.loads(call(server, "plan_task", {"task": "pick up the object at (5, 3)"})[1])
    assert plan["can_proceed"] is True
    assert {"x": 5.0, "y": 3.0}.items() <= plan["steps"][0]["parameters"].items()
