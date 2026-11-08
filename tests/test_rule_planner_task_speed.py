"""The rule planner uses the speed a task names.

'go to (1, 2) at speed 0.1' used to plan the default 0.5 m/s, so under
--max-speed 0.2 the robot drove at 0.2, twice what was asked, and the reply
named 0.5, a number nobody asked for.
"""

from __future__ import annotations

import json

import pytest

from apyrobo.mcp_server import build_server
from apyrobo.skills.agent import RuleBasedProvider


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


@pytest.mark.parametrize(
    ("task", "speed"),
    [
        ("go to (1, 2) at speed 0.1", 0.1),
        ("go to (1, 2) speed: 0.15", 0.15),
        ("go to (1, 2) at a speed of 0.05", 0.05),
        ("go to (1, 2) at 0.1 m/s", 0.1),
        ("move to 1 2 at speed 0.1", 0.1),
        ("go to x=1 y=2 at speed 0.1", 0.1),
        ("go to (1, 2) at speed 0.1.", 0.1),
        ("Go to (1, 2) at 0.1 m/s.", 0.1),
    ],
)
def test_execute_task_drives_at_the_speed_the_task_names(task, speed):
    server = build_server("mock://turtlebot4", max_speed=0.2)
    is_error, text = call(server, "execute_task", {"task": task})
    assert not is_error, text
    result = json.loads(text)
    assert result["status"] == "completed"
    assert result["plan"][0]["parameters"] == {"x": 1.0, "y": 2.0, "speed": speed}
    # At or under the limit: nothing clamped, nothing named that wasn't asked.
    assert result["safety_interventions"] == []
    assert result["position"] == {"x": 1.0, "y": 2.0}


def test_a_speed_above_the_limit_is_clamped_and_the_reply_names_what_was_asked():
    server = build_server("mock://turtlebot4", max_speed=0.2)
    _, text = call(server, "execute_task", {"task": "go to (1,1) at 5 m/s"})
    result = json.loads(text)
    assert result["status"] == "completed"
    assert result["safety_interventions"] == [
        {"type": "speed_clamped", "requested": 5.0, "enforced": 0.2, "max_allowed": 0.2}
    ]


def test_plan_task_shows_the_named_speed_and_says_it_will_be_clamped():
    server = build_server("mock://turtlebot4", max_speed=0.2)
    _, text = call(server, "plan_task", {"task": "go to (1,1) at 5 m/s"})
    plan = json.loads(text)
    assert plan["can_proceed"] is True
    assert plan["steps"][0]["parameters"]["speed"] == 5.0
    assert plan["speed_notes"] == [
        "Step 1 drives at 0.2 m/s (requested 5.0, clamped to the policy limit)."
    ]


def test_plan_task_has_no_speed_notes_when_the_speed_is_within_the_limit():
    server = build_server("mock://turtlebot4", max_speed=0.2)
    _, text = call(server, "plan_task", {"task": "go to (1, 2) at speed 0.1"})
    plan = json.loads(text)
    assert plan["steps"][0]["parameters"]["speed"] == 0.1
    assert "speed_notes" not in plan


def test_a_delivery_uses_the_speed_for_both_drives():
    server = build_server("mock://turtlebot4", max_speed=0.2)
    _, text = call(
        server, "plan_task", {"task": "deliver from (1, 1) to (2, 2) at 0.1 m/s"}
    )
    steps = json.loads(text)["steps"]
    drives = [s for s in steps if s["skill_id"].startswith("navigate_to")]
    assert [d["parameters"]["speed"] for d in drives] == [0.1, 0.1]


def test_a_speed_is_not_read_as_a_coordinate():
    server = build_server("mock://turtlebot4", max_speed=0.2)
    _, text = call(server, "execute_task", {"task": "move to 3 3 at speed 2"})
    result = json.loads(text)
    assert result["position"] == {"x": 3.0, "y": 3.0}
    assert result["plan"][0]["parameters"]["speed"] == 2.0


def test_no_speed_in_the_task_keeps_the_default():
    server = build_server("mock://turtlebot4", max_speed=0.2)
    _, text = call(server, "plan_task", {"task": "go to (1, 2)"})
    assert json.loads(text)["steps"][0]["parameters"]["speed"] == 0.5


@pytest.mark.parametrize(
    ("task", "reason"),
    [
        ("go to (1, 2) at speed 0.1 then at speed 0.3", "names more than one speed"),
        ("go to (1, 2) at speed -1", "names a negative speed"),
        ("go to (1, 2) at speed 1e3", "exponent form"),
        ("go to (1, 2) at speed 0", "names a speed of 0"),
        ("go to (1, 2) at 0 m/s", "names a speed of 0"),
        ("go to (1, 2) at speed .1", "starts with a decimal point"),
        ("go to (1, 2) at .1 m/s", "starts with a decimal point"),
        ("go to (1, 2) at speed 1/10", "a fraction"),
        ("go to (1, 2) at speed 0,1", "decimal comma"),
        ("go to (1, 2) at 0,1 m/s", "decimal comma"),
        ("go to (1, 2) at 0.1 meters per second", "reads m/s only"),
        ("go to (1, 2) at 0.1 metres per second", "reads m/s only"),
        ("go to (1, 2) at 1 km/h", "reads m/s only"),
        ("go to (1, 2) at 1 mph", "reads m/s only"),
        ("go to (1, 2) at 10 cm/s", "reads m/s only"),
        ("go to (1, 2) at 0.1 m/sec", "reads m/s only"),
        ("rotate 90 degrees at speed 0.3", "names a speed, but the plan has no drive"),
        ("go to (1, 2) at speed 0.1 or 0.2", "choice or range of speeds"),
        ("go to (1, 2) at speed 0.1-0.2", "choice or range of speeds"),
        ("go to (1, 2) at speed 0.1 m/s or 0.2", "choice or range of speeds"),
        ("go to (1, 2) at 0.1 or 0.2 m/s", "choice or range of speeds"),
        ("go to (1, 2) at 0.1 to 0.2 m/s", "choice or range of speeds"),
        ("go to (1, 2) at 0.1\u20130.2 m/s", "choice or range of speeds"),
        ("go to (1, 2) at speed 1_0", "underscore"),
        ("rotate 90 degrees at 0.1 rad/s", "turn speed"),
        ("turn right 45 degrees at 10 deg/s", "turn speed"),
        ("rotate 1.5 rad at 0.2 radians per second", "turn speed"),
        ("turn left 90 degrees at 30 degrees per second", "turn speed"),
        ("rotate 90 degrees at 5 rpm", "turn speed"),
    ],
)
def test_a_speed_the_planner_cannot_use_is_refused_and_moves_nothing(task, reason):
    server = build_server("mock://turtlebot4", max_speed=0.2)
    before = position(server)
    is_error, text = call(server, "execute_task", {"task": task})
    assert is_error
    assert text.startswith("Not planned:")
    assert reason in text
    assert position(server) == before
    _, planned = call(server, "plan_task", {"task": task})
    plan = json.loads(planned)
    assert plan["can_proceed"] is False
    assert plan["steps"] == []
    assert reason in plan["reason"]


def test_the_same_speed_twice_is_one_speed():
    assert RuleBasedProvider.cannot_read("go to (1, 2) at speed 0.1, speed 0.1") is None


def test_rule_provider_sets_the_speed_on_its_own():
    steps = RuleBasedProvider().plan("go to (1, 2) at 0.3 m/s", [], [])
    assert steps == [{"skill_id": "navigate_to", "parameters": {"x": 1.0, "y": 2.0, "speed": 0.3}}]


def test_confirm_question_shows_the_named_speed():
    server = build_server("mock://turtlebot4", max_speed=0.2, confirm_moves=True)
    question = server.toolbox.confirmation_prompt(
        "execute_task", {"task": "go to (1, 1) at speed 0.1"}
    )
    assert "1. Drive to x=1.0, y=1.0 at 0.1 m/s" in question


def test_a_turn_speed_in_a_task_does_not_turn_the_robot():
    # It used to turn at the default speed, dropping the 0.1 rad/s asked for.
    server = build_server("mock://turtlebot4")
    before = json.loads(call(server, "robot_status")[1])["heading_rad"]
    is_error, text = call(server, "execute_task", {"task": "rotate 90 degrees at 0.1 rad/s"})
    assert is_error
    assert "turn speed" in text
    assert json.loads(call(server, "robot_status")[1])["heading_rad"] == before


@pytest.mark.parametrize(
    "task",
    [
        "pick up the box at 1 -2",
        "go to (1, -2) at speed 0.2",
        "go to (1, 2) at 0.2 m/s to be safe",
        "go at speed 0.2 to 3 3",
        "rotate 90 degrees",
        "turn right 45 deg",
    ],
)
def test_one_speed_and_plain_turns_still_plan(task):
    assert RuleBasedProvider.cannot_read(task) is None
    assert RuleBasedProvider().plan(task, [], [])
