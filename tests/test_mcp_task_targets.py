"""plan_task and execute_task never invent where to drive.

The rule planner's templates hold placeholder targets. Under MCP a drive goes
only to a point the task named; anything else is refused and moves nothing.
"""

from __future__ import annotations

import json

import pytest

from apyrobo.mcp_server import _NO_TARGET, build_server
from apyrobo.skills.agent import Agent, AgentProvider, RuleBasedProvider


@pytest.fixture
def server():
    return build_server("mock://turtlebot4")


def agent_with(provider):
    agent = Agent(provider="rule")
    agent._provider = provider
    return agent


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


def drive_to_start(server):
    call(server, "move_to", {"x": 1, "y": 1})
    assert position(server) == {"x": 1.0, "y": 1.0}


NO_TARGET_TASKS = [
    "go to the kitchen",
    "pick up the box",
    "deliver the package",
    "deliver the package to dock 3",
    "go to room 3",
    "navigate somewhere nice",
    "move at speed 2",
]


@pytest.mark.parametrize("task", NO_TARGET_TASKS)
def test_execute_task_without_a_target_moves_nothing(server, task):
    drive_to_start(server)
    is_error, text = call(server, "execute_task", {"task": task})
    assert is_error
    assert text == _NO_TARGET
    assert position(server) == {"x": 1.0, "y": 1.0}


@pytest.mark.parametrize("task", NO_TARGET_TASKS)
def test_plan_task_without_a_target_gives_the_same_reason(server, task):
    is_error, text = call(server, "plan_task", {"task": task})
    assert not is_error
    plan = json.loads(text)
    assert plan["can_proceed"] is False
    assert plan["reason"] == _NO_TARGET
    assert plan["steps"] == []


@pytest.mark.parametrize(
    "task, target",
    [
        ("move to 3 3 at speed 2", (3.0, 3.0)),
        ("move to 1.5 -2", (1.5, -2.0)),
        ("go to x=2 y=1", (2.0, 1.0)),
        ("go to x=2, y=1", (2.0, 1.0)),
        ("go to (2, 1)", (2.0, 1.0)),
        ("go to (2, 1) at 0.3 m/s", (2.0, 1.0)),
        ("go to (0, 0)", (0.0, 0.0)),
        ("GO TO 2 1 AT SPEED 0.2", (2.0, 1.0)),
    ],
)
def test_a_named_target_is_driven_to(server, task, target):
    drive_to_start(server)
    is_error, text = call(server, "execute_task", {"task": task})
    assert not is_error, text
    assert (json.loads(text)["position"]["x"], json.loads(text)["position"]["y"]) == target


def test_a_speed_is_not_read_as_coordinates(server):
    plan = json.loads(call(server, "plan_task", {"task": "go to the dock at speed 2, 3"})[1])
    assert plan["can_proceed"] is False


def test_a_delivery_to_one_point_starts_where_the_robot_is(server):
    drive_to_start(server)
    plan = json.loads(call(server, "plan_task", {"task": "deliver the package to (5, 1)"})[1])
    assert plan["can_proceed"] is True
    drives = [s["parameters"] for s in plan["steps"] if s["skill_id"].startswith("navigate_to")]
    assert [(d["x"], d["y"]) for d in drives] == [(1.0, 1.0), (5.0, 1.0)]


def test_a_delivery_between_two_points_uses_both(server):
    plan = json.loads(
        call(server, "plan_task", {"task": "deliver the package from (1, 1) to (5, 1)"})[1]
    )
    drives = [s["parameters"] for s in plan["steps"] if s["skill_id"].startswith("navigate_to")]
    assert [(d["x"], d["y"]) for d in drives] == [(1.0, 1.0), (5.0, 1.0)]


def test_a_task_with_no_drive_still_runs(server):
    is_error, text = call(server, "execute_task", {"task": "report status"})
    assert not is_error, text
    assert json.loads(text)["status"] == "completed"


def test_a_failed_llm_falling_back_to_rules_does_not_invent_a_target():
    class Broken(AgentProvider):
        def plan(self, task, available_skills, capabilities):
            raise ConnectionError("no network")

    server = build_server("mock://turtlebot4", agent=agent_with(Broken()))
    drive_to_start(server)
    is_error, text = call(server, "execute_task", {"task": "move to 3 3"})
    assert is_error
    assert text == _NO_TARGET
    assert position(server) == {"x": 1.0, "y": 1.0}


def test_another_planner_may_choose_its_own_target():
    class Mapped(AgentProvider):
        def plan(self, task, available_skills, capabilities):
            return [{"skill_id": "navigate_to", "parameters": {"x": 2.0, "y": 1.0}}]

    server = build_server("mock://turtlebot4", agent=agent_with(Mapped()))
    is_error, text = call(server, "execute_task", {"task": "go to the kitchen"})
    assert not is_error, text
    assert json.loads(text)["position"] == {"x": 2.0, "y": 1.0}


def test_the_rule_planner_as_a_library_keeps_its_defaults():
    steps = RuleBasedProvider().plan("go to the kitchen", [], [])
    assert steps == [{"skill_id": "navigate_to", "parameters": {"x": 0.0, "y": 0.0}}]
