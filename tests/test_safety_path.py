"""Moves that cross a no-go zone are refused, not only moves that end in one."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from apyrobo.core.adapters import MockAdapter
from apyrobo.core.robot import Robot
from apyrobo.mcp_selftest import free_targets, run_checks
from apyrobo.mcp_server import build_server
from apyrobo.safety.enforcer import (
    SafetyEnforcer,
    SafetyPolicy,
    SafetyViolation,
    segment_enters_zone,
)

ZONE = {"x_min": 2.0, "y_min": 2.0, "x_max": 4.0, "y_max": 4.0}


def enforcer(*zones: dict) -> SafetyEnforcer:
    robot = Robot.discover("mock://path")
    return SafetyEnforcer(
        robot, policy=SafetyPolicy(name="path", collision_zones=list(zones) or [ZONE])
    )


class TestSegment:
    @pytest.mark.parametrize(
        "start, end, expected",
        [
            ((0, 0), (6, 6), True),  # through the middle
            ((0, 3), (6, 3), True),  # horizontal
            ((3, 0), (3, 6), True),  # vertical
            ((0, 3), (3, 3), True),  # ends inside
            ((0, 0), (1, 1), False),  # stops short
            ((0, 0), (6, 0), False),  # passes below
            ((0, 5), (6, 5), False),  # passes above
            ((0, 2), (6, 2), True),  # runs along the edge: edges count
            ((1, 3), (3, 5), True),  # clips the corner (2, 4) and past it
            ((5, 5), (5, 5), False),  # no movement, outside
            ((5, 5), (6, 6), False),  # moving away
        ],
    )
    def test_cases(self, start, end, expected):
        assert segment_enters_zone(start, end, ZONE) is expected

    def test_grazing_one_corner_counts(self):
        assert segment_enters_zone((0, 6), (6, 0), {**ZONE, "x_max": 3.0, "y_max": 3.0})

    def test_start_inside_may_leave(self):
        assert not segment_enters_zone((3, 3), (6, 6), ZONE)

    def test_start_on_edge_may_leave_but_not_cross(self):
        assert not segment_enters_zone((2, 3), (0, 3), ZONE)
        assert segment_enters_zone((2, 3), (6, 3), ZONE)

    def test_open_ended_zone(self):
        assert segment_enters_zone((0, 0), (2, 0), {"x_min": 1.0})
        assert not segment_enters_zone((0, 0), (0.5, 9), {"x_min": 1.0})


class TestMove:
    def test_crossing_move_is_refused_and_robot_stays(self):
        guard = enforcer()
        with pytest.raises(SafetyViolation, match=r"from \(0, 0\) to \(6, 6\) crosses"):
            guard.move(6, 6)
        assert guard.get_position() == (0.0, 0.0)
        assert guard.violations[-1]["type"] == "collision_zone_path"

    def test_refusal_names_the_zone(self):
        guard = enforcer({**ZONE, "name": "loading dock"})
        with pytest.raises(SafetyViolation, match="loading dock"):
            guard.move(6, 6)

    def test_clear_move_runs(self):
        guard = enforcer()
        guard.move(6, 0)
        assert guard.get_position() == (6.0, 0.0)

    def test_robot_inside_a_zone_may_leave(self):
        guard = SafetyEnforcer(Robot.discover("mock://path"), policy=SafetyPolicy(name="free"))
        guard.move(3, 3)
        guard.policy.collision_zones.append(ZONE)
        guard.move(6, 6)
        assert guard.get_position() == (6.0, 6.0)

    def test_leaving_one_zone_does_not_excuse_crossing_another(self):
        guard = SafetyEnforcer(Robot.discover("mock://path"), policy=SafetyPolicy(name="free"))
        guard.move(3, 3)
        guard.policy.collision_zones.extend(
            [ZONE, {"x_min": 5.0, "y_min": 5.0, "x_max": 6.0, "y_max": 6.0}]
        )
        with pytest.raises(SafetyViolation, match="crosses"):
            guard.move(7, 7)

    def test_dynamic_zone_on_the_route_is_refused(self):
        world = SimpleNamespace(
            obstacles=[SimpleNamespace(x=3.0, y=3.0, radius=0.5, source="scan")],
            humans=[],
        )
        guard = SafetyEnforcer(Robot.discover("mock://path"), policy=SafetyPolicy(name="free"))
        guard.update_world_state(world)
        with pytest.raises(SafetyViolation, match="crosses"):
            guard.move(6, 6)

    def test_no_zones_never_reads_the_position(self):
        robot = Robot.discover("mock://path")
        robot.get_position = None  # type: ignore[method-assign]
        SafetyEnforcer(robot, policy=SafetyPolicy(name="free")).move(1, 1)

    @pytest.mark.parametrize("bad", [(math.nan, 0.0), (0.0, math.inf), "boom"])
    def test_unknown_position_fails_closed(self, bad):
        guard = enforcer()

        def position():
            if bad == "boom":
                raise RuntimeError("odom lost")
            return bad

        guard.robot.get_position = position  # type: ignore[method-assign]
        with pytest.raises(SafetyViolation, match="Cannot check the route"):
            guard.move(1, 0)


def test_a_runtime_zone_stays_with_its_robot():
    first = SafetyEnforcer(Robot.discover("mock://first"), policy="default")
    first.add_collision_zone(ZONE)
    second = SafetyEnforcer(Robot.discover("mock://second"), policy="default")
    assert first.policy.collision_zones == [ZONE]
    assert second.policy.collision_zones == []


class Planner(MockAdapter):
    """A planner adapter whose route is whatever the test sets."""

    plans_own_path = True

    def __init__(self, name, route=None):
        super().__init__(name)
        self.route = route
        self.plans = 0

    def plan_path(self, x, y):
        self.plans += 1
        if isinstance(self.route, Exception):
            raise self.route
        return self.route


AROUND = [(0, 0), (0, 5), (5, 5), (6, 6)]  # stays clear of ZONE (2..4)
THROUGH = [(0, 0), (1, 1), (3, 3), (6, 6)]


def planner_guard(route, zones=(ZONE,)):
    adapter = Planner("p", route)
    guard = SafetyEnforcer(
        Robot(robot_id="p", adapter=adapter),
        policy=SafetyPolicy(name="p", collision_zones=list(zones)),
    )
    return guard, adapter


class TestPlanner:
    def test_a_planned_route_around_the_zone_runs(self):
        guard, _ = planner_guard(AROUND)  # the straight line would cross
        guard.move(6, 6)
        assert guard.get_position() == (6.0, 6.0)

    def test_a_planned_route_through_the_zone_is_refused(self):
        guard, _ = planner_guard(THROUGH)
        with pytest.raises(SafetyViolation, match="crosses collision zone"):
            guard.move(6, 6)
        assert guard.get_position() == (0.0, 0.0)
        assert guard.violations[-1]["type"] == "collision_zone_path"

    def test_a_route_is_caught_between_two_far_apart_points(self):
        guard, _ = planner_guard([(0, 3), (6, 3)])  # no point lies in the zone
        with pytest.raises(SafetyViolation, match="crosses collision zone"):
            guard.move(6, 3)

    @pytest.mark.parametrize(
        "route, message",
        [
            (RuntimeError("Nav2 planner is not available"), "not available"),
            (NotImplementedError("Planner has no route planner"), "no route planner"),
            ([], "empty path"),
            (None, "no plan"),
            ([(0, 0), (float("nan"), 1)], "non-finite"),
        ],
    )
    def test_no_usable_plan_fails_closed(self, route, message):
        guard, _ = planner_guard(route)
        with pytest.raises(SafetyViolation, match=message):
            guard.move(6, 6)
        assert guard.get_position() == (0.0, 0.0)
        assert guard.violations[-1]["type"] == "path_unchecked"

    def test_no_zones_means_no_plan_is_requested(self):
        guard, adapter = planner_guard(RuntimeError("down"), zones=())
        guard.move(6, 6)
        assert adapter.plans == 0
        assert guard.get_position() == (6.0, 6.0)

    def test_a_robot_inside_a_zone_may_follow_a_route_out(self):
        guard, _ = planner_guard([(3, 3), (3, 5), (6, 6)])
        guard.move(6, 6)
        assert guard.get_position() == (6.0, 6.0)

    def test_a_route_out_of_a_zone_may_not_come_back_through_it(self):
        guard, _ = planner_guard([(3, 3), (3, 5), (3, 3.5), (3, 0)])
        with pytest.raises(SafetyViolation, match="crosses collision zone"):
            guard.move(3, 0)

    def test_planner_adapter_still_refuses_a_target_in_a_zone(self):
        guard, adapter = planner_guard(AROUND)
        with pytest.raises(SafetyViolation, match="inside collision zone"):
            guard.move(3, 3)
        assert adapter.plans == 0

    def test_default_adapters_drive_straight(self):
        assert Robot.discover("mock://path").plans_own_path is False

    def test_an_adapter_without_a_planner_says_so(self):
        with pytest.raises(NotImplementedError):
            Robot.discover("mock://path").plan_path(1, 1)


def call(server, name, **arguments):
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    return response["result"]


class TestMcp:
    def test_move_through_a_zone_is_refused_over_mcp(self):
        server = build_server(robot_uri="mock://path", no_go_zones=[ZONE])
        result = call(server, "move_to", x=6, y=6, speed=0.1)
        assert result["isError"] is True
        assert "REFUSED" in result["content"][0]["text"]
        assert "crosses" in result["content"][0]["text"]
        status = json.loads(call(server, "robot_status")["content"][0]["text"])
        assert status["position"]["x"] == 0

    def test_confirm_prompt_skips_a_move_that_will_be_refused(self):
        server = build_server(robot_uri="mock://path", no_go_zones=[ZONE])
        assert server.toolbox.confirmation_prompt("move_to", {"x": 6, "y": 6}) is None
        assert server.toolbox.confirmation_prompt("move_to", {"x": 6, "y": 0}) is not None


class TestSelftest:
    def test_crossing_is_reported_for_each_zone(self):
        other = {"x_min": -5.0, "y_min": -5.0, "x_max": -3.0, "y_max": -3.0}
        checks = run_checks(build_server(robot_uri="mock://path", no_go_zones=[ZONE, other]))
        assert all(c.ok for c in checks), checks
        assert sum(c.name.startswith("move across") for c in checks) == 2

    def test_a_wall_beside_the_start_leaves_the_normal_moves_working(self):
        wall = {"x_min": 0.5, "y_min": -9.0, "x_max": 0.8, "y_max": 9.0}
        first, second = free_targets([wall])
        assert first == (0.0, 1.0)
        checks = run_checks(build_server(robot_uri="mock://path", no_go_zones=[wall]))
        assert all(c.ok for c in checks), checks
