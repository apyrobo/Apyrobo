"""The rule planner reads a task's drive speed in km/h, mph, cm/s and more.

'go to (1, 2) at 30 cm/s' and 'go to (1, 2) at 1 km/h' used to be refused
("it reads m/s only"), though `apyrobo policy` reads rules in those units. A
speed in one of them is now converted to m/s, rounded down so the robot never
drives faster than asked. Other units and wordings still move nothing.
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


@pytest.mark.parametrize(
    ("task", "speed"),
    [
        ("go to (1, 2) at 30 cm/s", 0.3),
        ("go to (1, 2) at 30 cm/sec", 0.3),
        ("go to (1, 2) at speed 30 cm/s", 0.3),
        ("go to (1, 2) at a speed of 25 centimeters per second", 0.25),
        ("go to (1, 2) at 150 mm/s", 0.15),
        ("go to (1, 2) at 1 km/h", 0.277777),
        ("go to (1, 2) at 1 kmh", 0.277777),
        ("go to (1, 2) at 1 kph", 0.277777),
        ("go to (1, 2) at 1.8 km/hr", 0.5),
        ("go to (1, 2) at 2 kilometers per hour", 0.555555),
        ("go to (1, 2) at 1 mph", 0.44704),
        ("go to (1, 2) at 1 mile per hour", 0.44704),
        ("go to (1, 2) at 1 ft/s", 0.3048),
        ("go to (1, 2) at 1 foot per second", 0.3048),
        ("go to (1, 2) at 12 m/min", 0.2),
        ("go to (1, 2) at 6 meters per minute", 0.1),
        ("go to (1, 2) at 0.1 meters per second", 0.1),
        ("go to (1, 2) at 0.1 m/sec", 0.1),
        ("go to (1, 2) at 0.3 m/s", 0.3),
        ("go to (1, 2) at speed 0.3", 0.3),
    ],
)
def test_a_speed_in_a_known_unit_is_read_in_m_s(task, speed):
    assert RuleBasedProvider.cannot_read(task) is None
    assert RuleBasedProvider.extract_speed(task) == pytest.approx(speed, abs=1e-9)
    steps = RuleBasedProvider().plan(task, [], [])
    assert [step["skill_id"] for step in steps] == ["navigate_to"]
    assert steps[0]["parameters"]["x"] == 1.0
    assert steps[0]["parameters"]["y"] == 2.0
    assert steps[0]["parameters"]["speed"] == pytest.approx(speed, abs=1e-9)


def test_a_converted_speed_is_never_rounded_up():
    # 1 km/h is 0.2777...; 0.277778 would be faster than asked.
    assert RuleBasedProvider.extract_speed("go to (1, 2) at 1 km/h") == 0.277777


def test_the_same_speed_in_two_units_is_one_speed():
    assert RuleBasedProvider.cannot_read("go to (1, 2) at 30 cm/s, speed 0.3") is None


@pytest.mark.parametrize(
    ("task", "reason"),
    [
        ("go to (1, 2) at 30 cm/s then at 0.2 m/s", "names more than one speed"),
        ("go to (1, 2) at -10 cm/s", "names a negative speed"),
        ("go to (1, 2) at 0 km/h", "names a speed of 0"),
        ("go to (1, 2) at 0.0001 mm/s", "names a speed of 0"),
        ("go to (1, 2) at 2 knots", "a unit the rule planner does not read"),
        ("go to (1, 2) at speed 2 knots", "a unit the rule planner does not read"),
        ("go to (1, 2) at speed 1 kmh/s", "the rule planner does not read"),
        ("go to (1, 2) max 3 km/h", "a wording the rule planner does not read"),
        ("go to (1, 2) max 3 km/hr", "a wording the rule planner does not read"),
        ("go to (1, 2) 3 miles per hour", "a wording the rule planner does not read"),
        ("go to (1, 2) going 3 cm/sec", "a wording the rule planner does not read"),
        ("go to (1, 2) at 30 or 40 cm/s", "a wording the rule planner does not read"),
        ("rotate 90 degrees at 30 cm/s", "names a speed, but the plan has no drive"),
    ],
)
def test_a_speed_the_planner_cannot_use_still_moves_nothing(task, reason):
    server = build_server("mock://turtlebot4", max_speed=0.2)
    before = json.loads(call(server, "robot_status")[1])["position"]
    is_error, text = call(server, "execute_task", {"task": task})
    assert is_error
    assert text.startswith("Not planned:")
    assert reason in text
    assert json.loads(call(server, "robot_status")[1])["position"] == before


def test_mcp_plans_and_drives_at_the_converted_speed():
    server = build_server("mock://turtlebot4", max_speed=0.5)
    _, planned = call(server, "plan_task", {"task": "go to (1, 2) at 30 cm/s"})
    plan = json.loads(planned)
    assert plan["can_proceed"] is True
    assert plan["steps"][0]["parameters"]["speed"] == 0.3
    is_error, text = call(server, "execute_task", {"task": "go to (1, 2) at 1 km/h"})
    assert not is_error, text
    result = json.loads(text)
    assert result["position"] == {"x": 1.0, "y": 2.0}
    assert result["plan"][0]["parameters"]["speed"] == 0.277777


def test_mcp_clamps_a_converted_speed_to_the_limit():
    server = build_server("mock://turtlebot4", max_speed=0.2)
    _, planned = call(server, "plan_task", {"task": "go to (1, 2) at 2 km/h"})
    plan = json.loads(planned)
    assert plan["steps"][0]["parameters"]["speed"] == 0.555555
    assert plan.get("speed_notes")
