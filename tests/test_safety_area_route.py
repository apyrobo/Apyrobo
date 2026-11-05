"""Allowed areas hold along the route: a move may not leave them on its way."""

from __future__ import annotations

import json

import pytest

from apyrobo.core.adapters import MockAdapter
from apyrobo.core.robot import Robot
from apyrobo.mcp_server import build_server
from apyrobo.safety.enforcer import (
    SafetyEnforcer,
    SafetyPolicy,
    SafetyViolation,
    route_leaves_areas,
)


def box(x_min, y_min, x_max, y_max):
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}


NEAR = box(0, 0, 1, 1)
FAR = box(5, 5, 6, 6)


def guard(*areas, **policy):
    return SafetyEnforcer(
        Robot.discover("mock://route_bot"),
        policy=SafetyPolicy(name="t", allowed_areas=list(areas), **policy),
    )


class Planner(MockAdapter):
    plans_own_path = True

    def __init__(self, name, route):
        super().__init__(name)
        self.route = route
        self.plans = 0

    def plan_path(self, x, y):
        self.plans += 1
        if isinstance(self.route, Exception):
            raise self.route
        return self.route


def planner_guard(route, *areas, **policy):
    adapter = Planner("p", route)
    enforcer = SafetyEnforcer(
        Robot(robot_id="p", adapter=adapter),
        policy=SafetyPolicy(name="p", allowed_areas=list(areas), **policy),
    )
    return enforcer, adapter


class TestRouteLeavesAreas:
    @pytest.mark.parametrize(
        "route, areas, left_at",
        [
            ([(0.5, 0.5), (0.9, 0.9)], [NEAR], None),
            ([(0.5, 0.5), (5.5, 5.5)], [NEAR, FAR], (1, 1)),
            ([(0.5, 0.5), (0.5, 3.0)], [NEAR], (0.5, 1)),
            # Touching: the shared edge is no gap, in either order.
            ([(1, 1), (3, 1)], [box(0, 0, 2, 2), box(2, 0, 4, 2)], None),
            ([(1, 1), (3, 1)], [box(2, 0, 4, 2), box(0, 0, 2, 2)], None),
            # Overlapping, then chained through a third.
            ([(1, 1), (9, 1)], [box(0, 0, 3, 2), box(2, 0, 6, 2), box(5, 0, 10, 2)], None),
            # Touching only at a corner still carries a line through that corner.
            ([(1, 1), (3, 3)], [box(0, 0, 2, 2), box(2, 2, 4, 4)], None),
            # A line that misses the corner leaves.
            ([(1, 1.5), (3, 2.5)], [box(0, 0, 2, 1.8), box(2, 2.2, 4, 4)], (1.6, 1.8)),
            # Starting outside, drive in: the way in is not checked.
            ([(-5, 0.5), (0.5, 0.5)], [NEAR], None),
            # In, then out again.
            ([(-5, 0.5), (3, 0.5)], [NEAR], (1, 0.5)),
            # Never inside at all: nothing to leave.
            ([(-5, -5), (-4, -4)], [NEAR], None),
            # Multi-point: the second stretch leaves, the third comes back.
            ([(0.5, 0.5), (0.9, 0.5), (0.9, 2), (0.5, 0.5)], [NEAR], (0.9, 1)),
            # A stretch wholly outside, after being inside, between two far points.
            ([(0.5, 1.0), (0.5, 1.0), (1.0, 1.0), (3, 1)], [NEAR], (1, 1)),
            # Along an edge counts as inside.
            ([(0, 0), (1, 0)], [NEAR], None),
        ],
    )
    def test_cases(self, route, areas, left_at):
        result = route_leaves_areas(route, areas)
        if left_at is None:
            assert result is None
        else:
            assert result == pytest.approx(left_at)

    def test_rounding_at_a_shared_edge_is_not_a_gap(self):
        left = box(0, 0, 0.1 + 0.2, 1)
        right = box(0.3, 0, 1, 1)
        assert route_leaves_areas([(0.1, 0.5), (0.9, 0.5)], [left, right]) is None

    def test_a_single_point_has_no_stretch_to_check(self):
        assert route_leaves_areas([(9, 9)], [NEAR]) is None


class TestStraightLine:
    def test_the_gap_between_two_areas_is_refused_and_the_robot_stays(self):
        enforcer = guard(NEAR, FAR)
        enforcer.move(0.5, 0.5)
        with pytest.raises(SafetyViolation, match=r"leaves the allowed areas at \(1, 1\)"):
            enforcer.move(5.5, 5.5)
        assert enforcer.get_position() == (0.5, 0.5)
        violation = enforcer.violations[-1]
        assert violation["type"] == "allowed_area_path"
        assert violation["start"] == (0.5, 0.5)
        assert violation["requested_position"] == (5.5, 5.5)

    def test_touching_areas_work(self):
        enforcer = guard(box(-1, -1, 2, 2), box(2, -1, 4, 2))
        enforcer.move(3.0, 1.0)
        assert enforcer.get_position() == (3.0, 1.0)

    def test_a_robot_outside_may_drive_in_and_then_is_held(self):
        enforcer = guard(FAR, box(20, 5, 21, 6))
        enforcer.move(5.5, 5.5)
        with pytest.raises(SafetyViolation, match="leaves the allowed areas"):
            enforcer.move(20.5, 5.5)

    def test_a_zone_is_named_first_when_the_route_does_both(self):
        zone = box(2, 2, 3, 3)
        enforcer = guard(NEAR, FAR, collision_zones=[zone])
        enforcer.move(0.5, 0.5)
        with pytest.raises(SafetyViolation, match="crosses collision zone"):
            enforcer.move(5.5, 5.5)

    def test_unset_areas_change_nothing(self):
        enforcer = guard()
        enforcer.move(50.0, 50.0)
        assert enforcer.get_position() == (50.0, 50.0)


class TestPlanner:
    def test_a_plan_that_stays_inside_runs_though_the_straight_line_would_not(self):
        u_shape = [box(0, 0, 1, 5), box(0, 4, 5, 5), box(4, 0, 5, 5)]
        enforcer, _ = planner_guard([(0.5, 0.5), (0.5, 4.5), (4.5, 4.5), (4.5, 0.5)], *u_shape)
        enforcer.move(4.5, 0.5)
        assert enforcer.get_position() == (4.5, 0.5)

    def test_a_plan_that_leaves_is_refused(self):
        enforcer, _ = planner_guard([(0, 0), (0, 3), (1, 1)], NEAR)
        with pytest.raises(SafetyViolation, match="leaves the allowed areas"):
            enforcer.move(1, 1)
        assert enforcer.get_position() == (0.0, 0.0)

    def test_the_plan_is_fetched_with_areas_and_no_zones(self):
        enforcer, adapter = planner_guard([(0, 0), (1, 1)], NEAR)
        enforcer.move(1, 1)
        assert adapter.plans == 1

    def test_no_plan_fails_closed_with_areas_and_no_zones(self):
        enforcer, _ = planner_guard(RuntimeError("down"), NEAR)
        with pytest.raises(SafetyViolation, match="no plan"):
            enforcer.move(1, 1)
        assert enforcer.get_position() == (0.0, 0.0)
        assert enforcer.violations[-1]["type"] == "path_unchecked"

    def test_zones_and_areas_share_one_plan(self):
        enforcer, adapter = planner_guard(
            [(0, 0), (1, 1)], NEAR, collision_zones=[box(8, 8, 9, 9)]
        )
        enforcer.move(1, 1)
        assert adapter.plans == 1


class TestServer:
    def test_the_boss_reproduction_is_refused_in_words(self):
        srv = build_server(allowed_areas=[NEAR, FAR])

        def call(name, **arguments):
            return srv.handle({
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            })["result"]

        assert not call("move_to", x=0.5, y=0.5).get("isError")
        result = call("move_to", x=5.5, y=5.5)
        assert result["isError"] is True
        assert "REFUSED" in result["content"][0]["text"]
        assert "leaves the allowed areas" in result["content"][0]["text"]
        position = json.loads(call("robot_status")["content"][0]["text"])["position"]
        assert (position["x"], position["y"]) == (0.5, 0.5)

    def test_confirmation_is_not_asked_for_a_route_that_will_be_refused(self):
        toolbox = build_server(allowed_areas=[box(-1, -1, 1, 1), FAR]).toolbox
        assert toolbox.confirmation_prompt("move_to", {"x": 5.5, "y": 5.5}) is None
        assert toolbox.confirmation_prompt("move_to", {"x": 0.5, "y": 0.5}) is not None
