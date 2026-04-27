"""Tests for MoveIt 2 motion planning adapter."""
from __future__ import annotations

import asyncio
import pytest

from apyrobo.moveit import (
    MoveItConfig,
    JointTarget,
    PoseTarget,
    MotionResult,
    MoveItAdapter,
    MockMoveItAdapter,
)


# ---------------------------------------------------------------------------
# MockMoveItAdapter
# ---------------------------------------------------------------------------

class TestMockMoveItAdapter:
    def test_connect_sets_connected(self):
        adapter = MockMoveItAdapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        assert adapter._connected is True

    def test_disconnect_clears_connected(self):
        adapter = MockMoveItAdapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        asyncio.get_event_loop().run_until_complete(adapter.disconnect())
        assert adapter._connected is False

    def test_move_to_joint_target_returns_success(self):
        adapter = MockMoveItAdapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        target = JointTarget(
            joint_names=["joint1", "joint2"],
            positions=[0.1, -0.2],
        )
        result = asyncio.get_event_loop().run_until_complete(
            adapter.move_to_joint_target(target)
        )
        assert isinstance(result, MotionResult)
        assert result.success is True

    def test_move_to_joint_target_updates_values(self):
        adapter = MockMoveItAdapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        target = JointTarget(
            joint_names=["j1", "j2", "j3"],
            positions=[1.0, 2.0, 3.0],
        )
        asyncio.get_event_loop().run_until_complete(adapter.move_to_joint_target(target))
        values = adapter.get_current_joint_values()
        assert values["j1"] == pytest.approx(1.0)
        assert values["j2"] == pytest.approx(2.0)

    def test_move_to_pose_target_returns_success(self):
        adapter = MockMoveItAdapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        target = PoseTarget(x=0.3, y=0.0, z=0.5)
        result = asyncio.get_event_loop().run_until_complete(
            adapter.move_to_pose_target(target)
        )
        assert result.success is True
        assert result.trajectory_length > 0

    def test_move_to_pose_target_updates_pose(self):
        adapter = MockMoveItAdapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        target = PoseTarget(x=0.4, y=0.1, z=0.6, yaw=0.5)
        asyncio.get_event_loop().run_until_complete(adapter.move_to_pose_target(target))
        pose = adapter.get_current_pose()
        assert pose["x"] == pytest.approx(0.4)
        assert pose["z"] == pytest.approx(0.6)

    def test_move_to_named_target_home(self):
        adapter = MockMoveItAdapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        result = asyncio.get_event_loop().run_until_complete(
            adapter.move_to_named_target("home")
        )
        assert result.success is True

    def test_move_to_named_target_ready(self):
        adapter = MockMoveItAdapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        result = asyncio.get_event_loop().run_until_complete(
            adapter.move_to_named_target("ready")
        )
        assert result.success is True

    def test_move_to_named_target_unknown_fails(self):
        adapter = MockMoveItAdapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        result = asyncio.get_event_loop().run_until_complete(
            adapter.move_to_named_target("nonexistent")
        )
        assert result.success is False
        assert "unknown" in result.message

    def test_stop_sets_not_moving(self):
        adapter = MockMoveItAdapter()
        adapter._moving = True
        adapter.stop()
        assert adapter.is_moving() is False

    def test_is_moving_false_after_motion(self):
        adapter = MockMoveItAdapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        target = JointTarget(joint_names=["j1"], positions=[0.0])
        asyncio.get_event_loop().run_until_complete(adapter.move_to_joint_target(target))
        assert adapter.is_moving() is False

    def test_not_connected_returns_failure(self):
        adapter = MockMoveItAdapter()
        target = JointTarget(joint_names=["j1"], positions=[0.0])
        result = asyncio.get_event_loop().run_until_complete(
            adapter.move_to_joint_target(target)
        )
        assert result.success is False


# ---------------------------------------------------------------------------
# JointTarget and PoseTarget dataclasses
# ---------------------------------------------------------------------------

class TestJointTarget:
    def test_creation(self):
        t = JointTarget(joint_names=["a", "b"], positions=[0.1, 0.2])
        assert t.joint_names == ["a", "b"]
        assert t.positions == [0.1, 0.2]

    def test_optional_velocities(self):
        t = JointTarget(joint_names=["a"], positions=[0.0])
        assert t.velocities is None

    def test_with_velocities(self):
        t = JointTarget(joint_names=["a"], positions=[0.0], velocities=[0.5])
        assert t.velocities == [0.5]


class TestPoseTarget:
    def test_required_fields(self):
        t = PoseTarget(x=1.0, y=2.0, z=3.0)
        assert t.x == 1.0
        assert t.y == 2.0
        assert t.z == 3.0

    def test_optional_defaults(self):
        t = PoseTarget(x=0.0, y=0.0, z=0.0)
        assert t.roll == 0.0
        assert t.pitch == 0.0
        assert t.yaw == 0.0
        assert t.frame_id == "base_link"


# ---------------------------------------------------------------------------
# MoveItAdapter stub mode
# ---------------------------------------------------------------------------

class TestMoveItAdapterStubMode:
    def test_stub_mode_when_rclpy_unavailable(self):
        import apyrobo.moveit as moveit_mod
        original = moveit_mod._RCLPY_AVAILABLE
        moveit_mod._RCLPY_AVAILABLE = False
        try:
            adapter = MoveItAdapter()
            assert adapter._stub_mode is True
        finally:
            moveit_mod._RCLPY_AVAILABLE = original

    def test_get_current_pose_returns_dict(self):
        adapter = MockMoveItAdapter()
        pose = adapter.get_current_pose()
        assert isinstance(pose, dict)
        assert "x" in pose

    def test_get_current_joint_values_returns_dict(self):
        adapter = MockMoveItAdapter()
        values = adapter.get_current_joint_values()
        assert isinstance(values, dict)
