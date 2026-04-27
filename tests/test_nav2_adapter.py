"""
Tests for Nav2Config, NavigationGoal, NavigationResult, MockNav2Adapter.
"""

from __future__ import annotations

import math
import pytest

from apyrobo.nav2 import (
    Nav2Adapter,
    Nav2Config,
    NavigationGoal,
    NavigationResult,
    NavigationStatus,
    MockNav2Adapter,
)


class TestNav2Config:
    def test_defaults(self):
        cfg = Nav2Config()
        assert cfg.namespace == ""
        assert cfg.action_server == "navigate_to_pose"
        assert cfg.timeout_s == 60.0
        assert cfg.goal_tolerance_m == 0.1

    def test_custom_namespace(self):
        cfg = Nav2Config(namespace="/robot1", timeout_s=30.0)
        assert cfg.namespace == "/robot1"
        assert cfg.timeout_s == 30.0


class TestNavigationGoal:
    def test_defaults(self):
        goal = NavigationGoal(x=1.0, y=2.0)
        assert goal.z == 0.0
        assert goal.yaw == 0.0
        assert goal.frame_id == "map"

    def test_full_construction(self):
        goal = NavigationGoal(x=3.0, y=4.0, z=0.5, yaw=1.57, frame_id="odom")
        assert goal.x == 3.0
        assert goal.yaw == pytest.approx(1.57)
        assert goal.frame_id == "odom"

    def test_distance_to_same_point(self):
        g = NavigationGoal(x=1.0, y=1.0)
        assert g.distance_to(g) == 0.0

    def test_distance_to_along_x(self):
        a = NavigationGoal(x=0.0, y=0.0)
        b = NavigationGoal(x=3.0, y=4.0)
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_distance_to_3d(self):
        a = NavigationGoal(x=0.0, y=0.0, z=0.0)
        b = NavigationGoal(x=1.0, y=0.0, z=0.0)
        assert a.distance_to(b) == pytest.approx(1.0)

    def test_speed_limit_default_none(self):
        goal = NavigationGoal(x=0.0, y=0.0)
        assert goal.speed_limit is None


class TestNavigationResult:
    def test_success_property(self):
        r = NavigationResult(status=NavigationStatus.SUCCESS)
        assert r.success is True

    def test_failed_property(self):
        r = NavigationResult(status=NavigationStatus.FAILED, message="blocked")
        assert r.success is False

    def test_timed_out_not_success(self):
        r = NavigationResult(status=NavigationStatus.TIMED_OUT)
        assert r.success is False

    def test_cancelled_not_success(self):
        r = NavigationResult(status=NavigationStatus.CANCELLED)
        assert r.success is False

    def test_raw_defaults_empty(self):
        r = NavigationResult(status=NavigationStatus.SUCCESS)
        assert r.raw == {}


class TestMockNav2Adapter:
    def test_is_nav2_adapter(self):
        adapter = MockNav2Adapter()
        assert isinstance(adapter, Nav2Adapter)

    def test_navigate_to_success(self):
        adapter = MockNav2Adapter()
        goal = NavigationGoal(x=5.0, y=3.0)
        result = adapter.navigate_to(goal)
        assert result.success is True
        assert result.status == NavigationStatus.SUCCESS

    def test_navigate_to_records_goal(self):
        adapter = MockNav2Adapter()
        goal = NavigationGoal(x=1.0, y=2.0)
        adapter.navigate_to(goal)
        assert len(adapter.goals_received) == 1
        assert adapter.goals_received[0].x == 1.0

    def test_navigate_to_updates_current_pose(self):
        adapter = MockNav2Adapter()
        adapter.navigate_to(NavigationGoal(x=7.0, y=8.0))
        pose = adapter.get_current_pose()
        assert pose.x == pytest.approx(7.0)
        assert pose.y == pytest.approx(8.0)

    def test_navigate_to_computes_distance(self):
        adapter = MockNav2Adapter()
        adapter.navigate_to(NavigationGoal(x=3.0, y=4.0))
        result = adapter.navigate_to(NavigationGoal(x=6.0, y=8.0))
        assert result.distance_traveled_m == pytest.approx(5.0)

    def test_fail_next_triggers_failure(self):
        adapter = MockNav2Adapter()
        adapter.fail_next = True
        result = adapter.navigate_to(NavigationGoal(x=1.0, y=1.0))
        assert result.success is False
        assert result.status == NavigationStatus.FAILED

    def test_fail_next_resets_after_one_failure(self):
        adapter = MockNav2Adapter()
        adapter.fail_next = True
        adapter.navigate_to(NavigationGoal(x=1.0, y=0.0))
        result = adapter.navigate_to(NavigationGoal(x=2.0, y=0.0))
        assert result.success is True

    def test_is_navigating_false_initially(self):
        adapter = MockNav2Adapter()
        assert adapter.is_navigating() is False

    def test_cancel_returns_false_when_not_navigating(self):
        adapter = MockNav2Adapter()
        assert adapter.cancel() is False

    def test_navigate_through_all_waypoints(self):
        adapter = MockNav2Adapter()
        waypoints = [NavigationGoal(x=float(i), y=0.0) for i in range(3)]
        results = adapter.navigate_through(waypoints)
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_navigate_through_stops_on_failure(self):
        adapter = MockNav2Adapter()
        adapter.fail_next = True
        waypoints = [NavigationGoal(x=float(i), y=0.0) for i in range(3)]
        results = adapter.navigate_through(waypoints)
        assert len(results) == 1
        assert not results[0].success

    def test_reset_clears_state(self):
        adapter = MockNav2Adapter()
        adapter.navigate_to(NavigationGoal(x=1.0, y=1.0))
        adapter.reset()
        assert adapter.goals_received == []
        assert adapter.results_sent == []

    def test_initial_pose_at_origin(self):
        adapter = MockNav2Adapter()
        pose = adapter.get_current_pose()
        assert pose.x == 0.0
        assert pose.y == 0.0
