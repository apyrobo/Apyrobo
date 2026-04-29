"""Tests for new Nav2Adapter methods: get_position(), real-mode structure."""
from __future__ import annotations

import asyncio
import pytest

from apyrobo.nav2 import (
    MockNav2Adapter,
    Nav2Config,
    NavigationGoal,
)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestGetPosition:
    def test_get_position_returns_tuple(self):
        adapter = MockNav2Adapter()
        run(adapter.connect())
        pos = adapter.get_position()
        assert isinstance(pos, tuple)
        assert len(pos) == 2

    def test_get_position_starts_at_origin(self):
        adapter = MockNav2Adapter()
        run(adapter.connect())
        x, y = adapter.get_position()
        assert x == 0.0
        assert y == 0.0

    def test_get_position_updates_after_navigation(self):
        adapter = MockNav2Adapter()
        run(adapter.connect())
        goal = NavigationGoal(x=3.5, y=-1.2)
        run(adapter.navigate_to(goal))
        x, y = adapter.get_position()
        assert abs(x - 3.5) < 1e-6
        assert abs(y - (-1.2)) < 1e-6

    def test_get_position_and_get_current_pose_consistent(self):
        adapter = MockNav2Adapter()
        run(adapter.connect())
        goal = NavigationGoal(x=1.0, y=2.0, yaw=0.5)
        run(adapter.navigate_to(goal))
        x, y = adapter.get_position()
        pose = adapter.get_current_pose()
        assert x == pose["x"]
        assert y == pose["y"]


class TestSetInitialPose:
    def test_set_initial_pose_affects_get_position(self):
        adapter = MockNav2Adapter()
        run(adapter.connect())
        adapter.set_initial_pose(5.0, 3.0, 1.57)
        x, y = adapter.get_position()
        assert abs(x - 5.0) < 1e-6
        assert abs(y - 3.0) < 1e-6

    def test_set_initial_pose_yaw_stored(self):
        adapter = MockNav2Adapter()
        run(adapter.connect())
        adapter.set_initial_pose(0.0, 0.0, 1.57)
        pose = adapter.get_current_pose()
        assert abs(pose["yaw"] - 1.57) < 1e-6


class TestCancelNavigation:
    def test_cancel_clears_navigating_flag(self):
        adapter = MockNav2Adapter()
        run(adapter.connect())
        run(adapter.cancel_navigation())
        assert adapter.is_navigating() is False

    def test_cancel_with_no_goal_handle_is_safe(self):
        adapter = MockNav2Adapter()
        run(adapter.connect())
        adapter._goal_handle = None
        run(adapter.cancel_navigation())  # must not raise


class TestNav2Config:
    def test_odom_topic_default(self):
        cfg = Nav2Config()
        assert cfg.odom_topic == "/odom"

    def test_action_name_default(self):
        cfg = Nav2Config()
        assert cfg.action_name == "navigate_to_pose"

    def test_custom_action_name(self):
        cfg = Nav2Config(action_name="robot/navigate_to_pose")
        assert cfg.action_name == "robot/navigate_to_pose"
