"""Tests for new MoveItAdapter methods: home_arm, get_joint_states, plan/execute."""
from __future__ import annotations

import asyncio
import pytest

from apyrobo.moveit import (
    JointTarget,
    MockMoveItAdapter,
    MoveItConfig,
    MotionResult,
    PoseTarget,
)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestHomeArm:
    def test_home_arm_returns_success(self):
        adapter = MockMoveItAdapter()
        run(adapter.connect())
        result = run(adapter.home_arm())
        assert result.success is True

    def test_home_arm_updates_current_pose(self):
        adapter = MockMoveItAdapter()
        run(adapter.connect())
        run(adapter.home_arm())
        pose = adapter.get_current_pose()
        # Home pose: x=0.0, y=0.0, z=0.5
        assert abs(pose["x"]) < 1e-6
        assert abs(pose["z"] - 0.5) < 1e-6

    def test_home_arm_is_not_moving_after_completion(self):
        adapter = MockMoveItAdapter()
        run(adapter.connect())
        run(adapter.home_arm())
        assert adapter.is_moving() is False


class TestGetJointStates:
    def test_get_joint_states_returns_dict(self):
        adapter = MockMoveItAdapter()
        run(adapter.connect())
        states = adapter.get_joint_states()
        assert isinstance(states, dict)

    def test_get_joint_states_empty_before_motion(self):
        adapter = MockMoveItAdapter()
        run(adapter.connect())
        assert adapter.get_joint_states() == {}

    def test_get_joint_states_after_joint_motion(self):
        adapter = MockMoveItAdapter()
        run(adapter.connect())
        target = JointTarget(
            joint_names=["joint1", "joint2"],
            positions=[0.5, 1.0],
        )
        run(adapter.move_to_joint_target(target))
        states = adapter.get_joint_states()
        assert abs(states["joint1"] - 0.5) < 1e-6
        assert abs(states["joint2"] - 1.0) < 1e-6

    def test_get_joint_states_returns_copy(self):
        adapter = MockMoveItAdapter()
        run(adapter.connect())
        s1 = adapter.get_joint_states()
        s1["injected"] = 999
        s2 = adapter.get_joint_states()
        assert "injected" not in s2


class TestPlanExecuteSeparation:
    def test_plan_motion_succeeds(self):
        adapter = MockMoveItAdapter()
        run(adapter.connect())
        target = PoseTarget(x=0.3, y=0.0, z=0.6)
        result = run(adapter.plan_motion(target))
        assert result.success is True

    def test_execute_without_plan_fails(self):
        adapter = MockMoveItAdapter()
        run(adapter.connect())
        result = run(adapter.execute_motion())
        assert result.success is False
        assert "plan_motion" in result.message

    def test_plan_then_execute_updates_pose(self):
        adapter = MockMoveItAdapter()
        run(adapter.connect())
        target = PoseTarget(x=0.4, y=0.1, z=0.7)
        run(adapter.plan_motion(target))
        run(adapter.execute_motion())
        pose = adapter.get_current_pose()
        assert abs(pose["x"] - 0.4) < 1e-6

    def test_execute_clears_pending_trajectory(self):
        adapter = MockMoveItAdapter()
        run(adapter.connect())
        run(adapter.plan_motion(PoseTarget(x=0.1, y=0.0, z=0.5)))
        run(adapter.execute_motion())
        # Second execute should fail — plan was consumed
        result = run(adapter.execute_motion())
        assert result.success is False

    def test_plan_joint_target_then_execute(self):
        adapter = MockMoveItAdapter()
        run(adapter.connect())
        target = JointTarget(joint_names=["j1"], positions=[0.7])
        run(adapter.plan_motion(target))
        result = run(adapter.execute_motion())
        assert result.success is True
        assert adapter.get_joint_states().get("j1") is not None


class TestMoveItConfigDefaults:
    def test_joint_states_topic_default(self):
        cfg = MoveItConfig()
        assert cfg.joint_states_topic == "/joint_states"

    def test_planning_group_default(self):
        cfg = MoveItConfig()
        assert cfg.planning_group == "panda_arm"
