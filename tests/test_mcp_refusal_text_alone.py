"""Every MCP safety refusal reads on its own, without structuredContent.

Some clients drop structuredContent from an error result and hand the model
only the text. These tests read the text alone: it must name what tripped
(the zone or area, the route, the battery level) and what the agent can do
next. Today's sentences stay a prefix; anything new comes after them.
"""

from __future__ import annotations

import pytest

from apyrobo.mcp_server import build_server

DOCK = {"name": "dock", "x_min": 2.0, "y_min": 2.0, "x_max": 4.0, "y_max": 4.0}
FLOOR = {"name": "floor", "x_min": -1.0, "y_min": -1.0, "x_max": 6.0, "y_max": 6.0}
YARD = {"name": "yard", "x_min": 7.0, "y_min": 7.0, "x_max": 10.0, "y_max": 10.0}
CHOOSE = "Choose a different target."


def _server(version="2024-11-05", **kwargs):
    """A server on a protocol with no structuredContent: the text is all there is."""
    server = build_server(max_speed=0.5, **kwargs)
    server.handle({
        "jsonrpc": "2.0", "id": 0, "method": "initialize",
        "params": {"protocolVersion": version, "capabilities": {}},
    })
    return server


def _refusal_text(server, name, arguments):
    resp = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    })
    result = resp["result"]
    assert result["isError"] is True
    assert "structuredContent" not in result
    # execute_task puts the run's outcome under the sentence; read the sentence.
    return result["content"][0]["text"].split("\n", 1)[0]


def _plan_reason(server, task):
    import json

    resp = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "plan_task", "arguments": {"task": task}},
    })
    plan = json.loads(resp["result"]["content"][0]["text"])
    assert plan["can_proceed"] is False
    return plan["reason"]


@pytest.mark.parametrize("tool", ["move_to", "execute_task", "plan_task"])
def test_target_in_a_zone_names_the_zone_and_the_way_out(tool):
    server = _server(no_go_zones=[DOCK])
    if tool == "move_to":
        text = _refusal_text(server, tool, {"x": 3, "y": 3})
    elif tool == "execute_task":
        text = _refusal_text(server, tool, {"task": "go to (3, 3)"})
    else:
        text = _plan_reason(server, "go to (3, 3)")
    assert "(3, 3)" in text
    assert "collision zone 'dock' (x 2 to 4, y 2 to 4)" in text
    assert text.endswith(CHOOSE)


@pytest.mark.parametrize("tool", ["move_to", "execute_task"])
def test_target_outside_the_areas_lists_every_area(tool):
    server = _server(allowed_areas=[FLOOR])
    args = {"x": 9, "y": 9} if tool == "move_to" else {"task": "go to (9, 9)"}
    text = _refusal_text(server, tool, args)
    assert "(9, 9) is outside every allowed area: 'floor' (x -1 to 6, y -1 to 6)" in text
    assert text.endswith(CHOOSE)


def test_route_through_a_zone_names_the_zone_and_suggests_going_around():
    text = _refusal_text(_server(no_go_zones=[DOCK]), "move_to", {"x": 5, "y": 5})
    head, _, tail = text.partition(CHOOSE)
    # Today's wording is untouched and still comes first.
    assert head.endswith(
        "Path from (0, 0) to (5, 5) crosses collision zone 'dock' (x 2 to 4, y 2 to 4). "
        "The robot did not move. "
    )
    assert "straight line" in tail
    assert "shorter moves" in tail and "around it" in tail


def test_route_through_a_zone_in_a_plan_suggests_going_around():
    reason = _plan_reason(_server(no_go_zones=[DOCK]), "go to (5, 5)")
    assert "crosses collision zone 'dock'" in reason
    assert CHOOSE in reason
    assert "shorter moves" in reason.split(CHOOSE, 1)[1]


def test_route_leaving_the_areas_says_where_and_suggests_staying_inside():
    server = _server(allowed_areas=[FLOOR, YARD])
    text = _refusal_text(server, "move_to", {"x": 8, "y": 8})
    head, _, tail = text.partition(CHOOSE)
    assert "leaves the allowed areas at (6, 6)" in head
    assert "'floor' (x -1 to 6, y -1 to 6)" in head and "'yard' (x 7 to 10, y 7 to 10)" in head
    assert "shorter moves" in tail and "inside the allowed areas" in tail


def test_unknown_position_says_a_new_target_will_not_help():
    server = _server(no_go_zones=[DOCK])
    robot = server.toolbox.enforcer._robot

    def lost():
        raise RuntimeError("odometry lost")

    robot.get_position = lost
    text = _refusal_text(server, "move_to", {"x": 1, "y": 0})
    assert "Cannot check the route: position unavailable (odometry lost)" in text
    tail = text.split(CHOOSE, 1)[1]
    assert "will not help" in tail and "robot_status" in tail


class _Battery:
    def __init__(self, percentage, enough):
        self.percentage = percentage
        self._enough = enough

    def can_complete_trip(self, distance, position):
        return self._enough


def test_battery_below_the_floor_names_both_levels_and_says_charge():
    server = _server()
    enforcer = server.toolbox.enforcer
    enforcer.policy.min_battery_pct = 20.0
    enforcer._battery_monitor = _Battery(12.0, enough=True)
    text = _refusal_text(server, "move_to", {"x": 1, "y": 0})
    assert "Battery (12%) below minimum (20%)" in text
    tail = text.split(CHOOSE, 1)[1]
    assert "charge the robot" in tail


def test_battery_short_for_the_trip_suggests_a_nearer_target():
    server = _server()
    server.toolbox.enforcer._battery_monitor = _Battery(30.0, enough=False)
    text = _refusal_text(server, "move_to", {"x": 3, "y": 4})
    assert "Battery too low (30%) for 5.0m trip" in text
    tail = text.split(CHOOSE, 1)[1]
    assert "nearer target" in tail and "charge the robot" in tail


def test_new_protocols_get_the_same_text():
    old = _refusal_text(_server(no_go_zones=[DOCK]), "move_to", {"x": 5, "y": 5})
    resp = _server("2025-11-25", no_go_zones=[DOCK]).handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "move_to", "arguments": {"x": 5, "y": 5}},
    })
    assert resp["result"]["content"][0]["text"] == old
    assert resp["result"]["structuredContent"]["code"] == "collision_zone_path"
