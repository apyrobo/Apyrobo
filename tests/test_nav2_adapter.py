"""Tests for Nav2 navigation adapter."""
from __future__ import annotations

import asyncio
import pytest

from apyrobo.nav2 import (
    Nav2Config,
    NavigationGoal,
    NavigationResult,
    Nav2Adapter,
    MockNav2Adapter,
)


# ---------------------------------------------------------------------------
# MockNav2Adapter
# ---------------------------------------------------------------------------

class TestMockNav2Adapter:
    def test_connect_sets_connected(self):
        adapter = MockNav2Adapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        assert adapter._connected is True

    def test_disconnect_clears_connected(self):
        adapter = MockNav2Adapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        asyncio.get_event_loop().run_until_complete(adapter.disconnect())
        assert adapter._connected is False

    def test_navigate_to_returns_success(self):
        adapter = MockNav2Adapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        goal = NavigationGoal(x=1.0, y=2.0)
        result = asyncio.get_event_loop().run_until_complete(adapter.navigate_to(goal))
        assert isinstance(result, NavigationResult)
        assert result.success is True

    def test_navigate_to_updates_pose(self):
        adapter = MockNav2Adapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        goal = NavigationGoal(x=3.5, y=-1.0, yaw=1.57)
        asyncio.get_event_loop().run_until_complete(adapter.navigate_to(goal))
        pose = adapter.get_current_pose()
        assert pose["x"] == pytest.approx(3.5)
        assert pose["y"] == pytest.approx(-1.0)

    def test_cancel_navigation(self):
        adapter = MockNav2Adapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        asyncio.get_event_loop().run_until_complete(adapter.cancel_navigation())
        assert adapter.is_navigating() is False

    def test_get_current_pose_returns_dict(self):
        adapter = MockNav2Adapter()
        pose = adapter.get_current_pose()
        assert isinstance(pose, dict)
        assert "x" in pose
        assert "y" in pose
        assert "yaw" in pose

    def test_is_navigating_false_after_complete(self):
        adapter = MockNav2Adapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        goal = NavigationGoal(x=0.0, y=0.0)
        asyncio.get_event_loop().run_until_complete(adapter.navigate_to(goal))
        assert adapter.is_navigating() is False

    def test_navigate_not_connected_fails(self):
        adapter = MockNav2Adapter()
        goal = NavigationGoal(x=1.0, y=1.0)
        result = asyncio.get_event_loop().run_until_complete(adapter.navigate_to(goal))
        assert result.success is False

    def test_set_initial_pose(self):
        adapter = MockNav2Adapter()
        adapter.set_initial_pose(1.0, 2.0, 0.5)
        pose = adapter.get_current_pose()
        assert pose["x"] == pytest.approx(1.0)
        assert pose["y"] == pytest.approx(2.0)
        assert pose["yaw"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# NavigationGoal dataclass
# ---------------------------------------------------------------------------

class TestNavigationGoal:
    def test_required_fields(self):
        goal = NavigationGoal(x=1.0, y=2.0)
        assert goal.x == 1.0
        assert goal.y == 2.0

    def test_optional_defaults(self):
        goal = NavigationGoal(x=0.0, y=0.0)
        assert goal.z == 0.0
        assert goal.yaw == 0.0
        assert goal.frame_id == "map"

    def test_custom_frame(self):
        goal = NavigationGoal(x=1.0, y=1.0, frame_id="odom")
        assert goal.frame_id == "odom"


# ---------------------------------------------------------------------------
# Nav2Adapter stub mode (rclpy not available)
# ---------------------------------------------------------------------------

class TestNav2AdapterStubMode:
    def test_stub_mode_when_rclpy_unavailable(self):
        import apyrobo.nav2 as nav2_mod
        original = nav2_mod._RCLPY_AVAILABLE
        nav2_mod._RCLPY_AVAILABLE = False
        try:
            adapter = Nav2Adapter()
            assert adapter._stub_mode is True
        finally:
            nav2_mod._RCLPY_AVAILABLE = original

    def test_stub_connect_succeeds(self):
        adapter = MockNav2Adapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        assert adapter._connected is True

    def test_stub_navigate_returns_success(self):
        adapter = MockNav2Adapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        result = asyncio.get_event_loop().run_until_complete(
            adapter.navigate_to(NavigationGoal(x=0.5, y=0.5))
        )
        assert result.success is True
        assert result.elapsed_s >= 0.0
