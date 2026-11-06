"""execute_task refuses a task the rule planner did not recognise.

'dock', 'go forward 1 metre' and 'turn left' used to plan a single status
report and end "completed" with the robot where it was; 'rotate 90 degrees'
planned a turn of 0 rad. Unrecognised tasks are now refused with what the
planner understands, and a turn uses the angle the task states.
"""

from __future__ import annotations

import json
import math

import pytest

from apyrobo.core.adapters import CapabilityAdapter, MockAdapter
from apyrobo.mcp_server import build_server
from apyrobo.skills.agent import RuleBasedProvider

UNRECOGNISED = [
    "dock",
    "go forward 1 metre",
    "take the box to (1, 1) at 5 m/s",
    "dance",
    "charge",
]
NO_ANGLE = ["turn left", "rotate", "turn around"]
STILL_PLANNED = [
    "go to (1, 2)",
    "deliver the package to (5, 1)",
    "deliver from (1, 1) to (5, 1)",
    "pick up the box at (1, 1)",
    "report status",
    "where are you",
    "stop",
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


@pytest.mark.parametrize(
    ("task", "angle"),
    [
        ("rotate 90 degrees", math.pi / 2),
        ("turn 90°", math.pi / 2),
        ("rotate -45 deg", -math.pi / 4),
        ("turn right 45 degrees", -math.pi / 4),
        ("turn left 45 degrees", math.pi / 4),
        ("rotate 90 degrees clockwise", -math.pi / 2),
        ("rotate 90 degrees counterclockwise", math.pi / 2),
        ("rotate 1.5 rad", 1.5),
        ("turn 0.5 radians", 0.5),
    ],
)
def test_planner_reads_the_stated_angle(task, angle):
    assert RuleBasedProvider.extract_angle(task) == pytest.approx(angle)
    plan = RuleBasedProvider().plan(task, [], [])
    assert plan == [{"skill_id": "rotate", "parameters": {"angle_rad": pytest.approx(angle)}}]


@pytest.mark.parametrize("task", ["turn left", "rotate 90", "go to (1, 1)", "90 degrees"])
def test_planner_finds_no_angle_without_a_turn_and_a_unit(task):
    assert RuleBasedProvider.extract_angle(task) is None


@pytest.mark.parametrize("task", UNRECOGNISED)
def test_execute_task_refuses_an_unrecognised_task_and_moves_nothing(task):
    server = build_server("mock://turtlebot4")
    before = status(server)
    is_error, text = call(server, "execute_task", {"task": task})
    assert is_error
    assert text.startswith("Not planned: the rule planner did not recognise this task")
    assert "go to (x, y)" in text
    after = status(server)
    assert after["position"] == before["position"]
    assert after["heading_rad"] == before["heading_rad"]


@pytest.mark.parametrize("task", NO_ANGLE)
def test_execute_task_refuses_a_turn_with_no_angle(task):
    server = build_server("mock://turtlebot4")
    before = status(server)["heading_rad"]
    is_error, text = call(server, "execute_task", {"task": task})
    assert is_error
    assert text.startswith("Not planned: this task asks to turn but gives no angle")
    assert status(server)["heading_rad"] == before


@pytest.mark.parametrize("task", UNRECOGNISED + NO_ANGLE)
def test_plan_task_says_it_cannot_proceed(task):
    server = build_server("mock://turtlebot4")
    is_error, text = call(server, "plan_task", {"task": task})
    assert not is_error
    plan = json.loads(text)
    assert plan["steps"] == []
    assert plan["can_proceed"] is False
    assert plan["reason"].startswith("Not planned:")


@pytest.mark.parametrize("task", STILL_PLANNED)
def test_recognised_tasks_still_plan(task):
    server = build_server("mock://turtlebot4")
    _, text = call(server, "plan_task", {"task": task})
    plan = json.loads(text)
    assert plan["steps"]
    assert "reason" not in plan


def test_execute_task_turns_by_the_stated_angle():
    server = build_server("mock://turtlebot4")
    is_error, _ = call(server, "execute_task", {"task": "rotate 90 degrees"})
    assert not is_error
    assert status(server)["heading_rad"] == pytest.approx(math.pi / 2)


def test_deliver_to_one_coordinate_still_drives_there():
    server = build_server("mock://turtlebot4")
    is_error, _ = call(server, "execute_task", {"task": "deliver the package to (5, 1)"})
    assert not is_error
    assert status(server)["position"] == {"x": 5.0, "y": 1.0}


class NoRotateAdapter(MockAdapter):
    """A mock:// adapter that keeps the base no-op rotate, as ROS2Adapter does."""

    rotate = CapabilityAdapter.rotate


def test_execute_task_refuses_a_turn_the_robot_cannot_make():
    server = build_server("mock://turtlebot4")
    robot = server.toolbox.robot
    robot._adapter = NoRotateAdapter(robot_name=robot._adapter.robot_name)
    is_error, text = call(server, "execute_task", {"task": "rotate 90 degrees"})
    assert is_error
    assert "cannot turn in place" in text
