"""The rule planner's checks refuse a drive followed by a step it does not make.

The rule planner plans the first kind of task it recognises, so 'go to (1, 1)
then wait 5 seconds' and 'go to (1, 1) and take a photo' planned the drive
alone and reported completed, dropping the wait or the photo. apyrobo mcp and
apyrobo plan / execute now refuse such tasks; a stop, a status report, a speed
or a courtesy after the drive still plans.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from apyrobo.mcp_server import build_server
from apyrobo.skills.agent import RuleBasedProvider

DROPPED = [
    "go to (1, 1) then wait 5 seconds",
    "go to (1, 1) and take a photo",
    "go to (1,1) and then scan the room",
    "drive to (2, 2), take a picture",
    "navigate to (3, 1); beep twice",
    "go to (2, 2) at 0.2 m/s then wait",
    "move to 3 3 then look around",
    "go to (1, 1). Then wait there.",
    "deliver from (1, 1) to (5, 1) and then wait",
]
KEPT = [
    ("go to (2, 2)", ["navigate_to"]),
    ("go to (1, 1) and stop", ["navigate_to"]),
    ("go to (1, 1) then report status", ["navigate_to"]),
    ("go to (1, 1), please", ["navigate_to"]),
    ("go to (1.5, 2.5). Thanks.", ["navigate_to"]),
    ("go to (1, 1) at 0.2 m/s", ["navigate_to"]),
    ("move to 3 3 at speed 0.2", ["navigate_to"]),
    ("pick up the box at (5, 3)", ["navigate_to", "pick_object"]),
    ("deliver from (1, 1) to (5, 1) and place it on the shelf",
     ["navigate_to", "pick_object", "navigate_to", "place_object"]),
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
def test_execute_task_refuses_a_dropped_step_and_moves_nothing(task):
    server = build_server("mock://turtlebot4")
    before = position(server)
    is_error, text = call(server, "execute_task", {"task": task})
    assert is_error
    assert text.startswith("Not planned: this task asks for something after the drive")
    assert "makes drives, pick-ups, deliveries, one turn, stop and status reports" in text
    assert position(server) == before


@pytest.mark.parametrize("task", DROPPED)
def test_plan_task_says_it_cannot_proceed(task):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "plan_task", {"task": task})
    plan = json.loads(text)
    assert not is_error
    assert plan["can_proceed"] is False
    assert plan["steps"] == []
    assert plan["reason"].startswith(
        "Not planned: this task asks for something after the drive"
    )


@pytest.mark.parametrize("task,skills", KEPT)
def test_other_tasks_still_proceed(task, skills):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "plan_task", {"task": task})
    plan = json.loads(text)
    assert not is_error
    assert plan["can_proceed"] is True, plan.get("reason")
    assert [s["skill_id"].rsplit("_", 1)[0] for s in plan["steps"]] == skills


def test_a_way_back_keeps_its_own_reason():
    server = build_server("mock://turtlebot4")
    plan = json.loads(call(server, "plan_task", {"task": "go to (1, 1) and come back"})[1])
    assert plan["reason"].startswith("Not planned: this task asks the robot to come back")


def test_a_pick_up_keeps_its_own_reason():
    server = build_server("mock://turtlebot4")
    plan = json.loads(
        call(server, "plan_task", {"task": "go to (5, 3) then pick up the box"})[1]
    )
    assert plan["reason"].startswith("Not planned: this task asks for a drive and then a pick-up")


@pytest.mark.parametrize(
    "task,expected",
    [
        ("go to (1, 1) then wait 5 seconds", "wait"),
        ("go to (1, 1) and then take a photo", "take"),
        ("go to (1, 1) and stop", None),
        ("go to (1, 1) then report status", None),
        ("go to (1.5, 2.5)", None),
        ("wait 5 seconds", None),
        ("deliver from (1, 1) to (5, 1) and put it down", None),
        ("go to (1, 1) and put it down", "put"),
    ],
)
def test_dropped_action_after_drive(task, expected):
    assert RuleBasedProvider.dropped_action_after_drive(task) == expected


@pytest.mark.parametrize("command", ["plan", "execute"])
def test_cli_refuses_and_exits_1(command):
    out = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", command,
         "go to (1, 1) then wait 5 seconds", "--robot", "mock://tb"],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 1
    assert "Not planned: this task asks for something after the drive" in out.stdout + out.stderr
