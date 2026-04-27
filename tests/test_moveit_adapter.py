"""
Tests for MoveItConfig, JointTarget, PoseTarget, MotionResult, MockMoveItAdapter.
"""

from __future__ import annotations

import pytest

from apyrobo.moveit import (
    MoveItAdapter,
    MoveItConfig,
    JointTarget,
    PoseTarget,
    MotionResult,
    MotionStatus,
    MockMoveItAdapter,
)


class TestMoveItConfig:
    def test_defaults(self):
        cfg = MoveItConfig()
        assert cfg.group_name == "manipulator"
        assert cfg.planning_time == 5.0
        assert cfg.velocity_scaling == 0.5
        assert cfg.acceleration_scaling == 0.5

    def test_custom_group(self):
        cfg = MoveItConfig(group_name="arm", planning_time=10.0)
        assert cfg.group_name == "arm"
        assert cfg.planning_time == 10.0


class TestJointTarget:
    def test_basic_construction(self):
        target = JointTarget(
            joint_names=["joint1", "joint2"],
            positions=[0.0, 1.57],
        )
        assert len(target.joint_names) == 2
        assert target.positions[1] == pytest.approx(1.57)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            JointTarget(joint_names=["j1"], positions=[0.0, 1.0])

    def test_velocities_default_empty(self):
        target = JointTarget(joint_names=["j1"], positions=[0.5])
        assert target.velocities == []

    def test_tolerances_default_empty(self):
        target = JointTarget(joint_names=["j1"], positions=[0.5])
        assert target.tolerances == []


class TestPoseTarget:
    def test_defaults(self):
        target = PoseTarget(x=1.0, y=0.0, z=0.5)
        assert target.qw == 1.0
        assert target.qx == 0.0
        assert target.frame_id == "base_link"

    def test_full_construction(self):
        target = PoseTarget(
            x=0.3, y=0.1, z=0.7,
            qx=0.0, qy=0.0, qz=0.707, qw=0.707,
            frame_id="world",
            end_effector_link="tool0",
        )
        assert target.z == pytest.approx(0.7)
        assert target.frame_id == "world"


class TestMotionResult:
    def test_success_property(self):
        r = MotionResult(status=MotionStatus.SUCCESS)
        assert r.success is True

    def test_planning_failed_not_success(self):
        r = MotionResult(status=MotionStatus.PLANNING_FAILED)
        assert r.success is False

    def test_execution_failed_not_success(self):
        r = MotionResult(status=MotionStatus.EXECUTION_FAILED)
        assert r.success is False

    def test_total_time_s(self):
        r = MotionResult(
            status=MotionStatus.SUCCESS,
            planning_time_s=1.0,
            execution_time_s=2.0,
        )
        assert r.total_time_s == pytest.approx(3.0)

    def test_raw_defaults_empty(self):
        r = MotionResult(status=MotionStatus.SUCCESS)
        assert r.raw == {}


class TestMockMoveItAdapter:
    def test_is_moveit_adapter(self):
        adapter = MockMoveItAdapter()
        assert isinstance(adapter, MoveItAdapter)

    def test_move_to_joint_target_success(self):
        adapter = MockMoveItAdapter()
        target = JointTarget(
            joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
            positions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        )
        result = adapter.move_to_joint_target(target)
        assert result.success is True

    def test_joint_target_updates_joint_values(self):
        adapter = MockMoveItAdapter()
        positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        target = JointTarget(
            joint_names=[f"j{i}" for i in range(6)],
            positions=positions,
        )
        adapter.move_to_joint_target(target)
        assert adapter.get_current_joint_values() == pytest.approx(positions)

    def test_move_to_pose_target_success(self):
        adapter = MockMoveItAdapter()
        target = PoseTarget(x=0.5, y=0.0, z=0.8)
        result = adapter.move_to_pose_target(target)
        assert result.success is True

    def test_pose_target_updates_current_pose(self):
        adapter = MockMoveItAdapter()
        target = PoseTarget(x=0.5, y=0.0, z=0.8)
        adapter.move_to_pose_target(target)
        pose = adapter.get_current_pose()
        assert pose.x == pytest.approx(0.5)
        assert pose.z == pytest.approx(0.8)

    def test_move_to_named_target_home(self):
        adapter = MockMoveItAdapter()
        result = adapter.move_to_named_target("home")
        assert result.success is True
        assert adapter.get_current_joint_values() == pytest.approx([0.0] * 6)

    def test_move_to_named_target_unknown(self):
        adapter = MockMoveItAdapter()
        result = adapter.move_to_named_target("nonexistent_pose")
        assert result.status == MotionStatus.INVALID_TARGET

    def test_fail_next_on_joint_move(self):
        adapter = MockMoveItAdapter()
        adapter.fail_next = True
        target = JointTarget(joint_names=["j1"], positions=[1.0])
        result = adapter.move_to_joint_target(target)
        assert result.success is False
        assert result.status == MotionStatus.PLANNING_FAILED

    def test_fail_next_resets_after_one_failure(self):
        adapter = MockMoveItAdapter()
        adapter.fail_next = True
        target = JointTarget(joint_names=["j1"], positions=[1.0])
        adapter.move_to_joint_target(target)
        result = adapter.move_to_joint_target(target)
        assert result.success is True

    def test_stop_sets_stopped_flag(self):
        adapter = MockMoveItAdapter()
        adapter.stop()
        assert adapter._stopped is True

    def test_plan_joint_target_no_execution_time(self):
        adapter = MockMoveItAdapter()
        target = JointTarget(joint_names=["j1"], positions=[0.5])
        result = adapter.plan_joint_target(target)
        assert result.execution_time_s == 0.0

    def test_records_all_moves(self):
        adapter = MockMoveItAdapter()
        jt = JointTarget(joint_names=["j1"], positions=[0.0])
        pt = PoseTarget(x=0.1, y=0.0, z=0.5)
        adapter.move_to_joint_target(jt)
        adapter.move_to_pose_target(pt)
        adapter.move_to_named_target("home")
        assert len(adapter.joint_moves) == 1
        assert len(adapter.pose_moves) == 1
        assert len(adapter.named_moves) == 1

    def test_reset_clears_all_state(self):
        adapter = MockMoveItAdapter()
        adapter.move_to_named_target("home")
        adapter.stop()
        adapter.reset()
        assert adapter.joint_moves == []
        assert adapter.pose_moves == []
        assert adapter.named_moves == []
        assert adapter._stopped is False

    def test_initial_joint_values_are_zeros(self):
        adapter = MockMoveItAdapter()
        vals = adapter.get_current_joint_values()
        assert vals == [0.0] * 6
