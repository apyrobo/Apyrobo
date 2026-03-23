"""
Tests for improved GazeboNativeAdapter (SIM-01 extended).
"""

from __future__ import annotations

import pytest

from apyrobo.sim.adapters import GazeboNativeAdapter, GazeboNotRunningError, JointState
from apyrobo.core.schemas import AdapterState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter():
    """Connected GazeboNativeAdapter with Gazebo available."""
    return GazeboNativeAdapter("turtlebot4", world="test_world")


@pytest.fixture
def unavailable_adapter():
    """Adapter simulating Gazebo not running."""
    return GazeboNativeAdapter("turtlebot4", gazebo_available=False)


# ---------------------------------------------------------------------------
# is_available property
# ---------------------------------------------------------------------------

class TestIsAvailable:
    def test_available_when_connected_and_gazebo_up(self, adapter):
        assert adapter.is_available is True

    def test_unavailable_when_gazebo_down(self, unavailable_adapter):
        assert unavailable_adapter.is_available is False

    def test_unavailable_when_disconnected(self, adapter):
        adapter._state = AdapterState.DISCONNECTED
        assert adapter.is_available is False


# ---------------------------------------------------------------------------
# spawn_entity
# ---------------------------------------------------------------------------

class TestSpawnEntity:
    def test_spawn_adds_entity(self, adapter):
        result = adapter.spawn_entity("box1", pose=(2.0, 3.0))
        assert result is True
        assert "box1" in adapter.list_entities()

    def test_spawn_adds_topics(self, adapter):
        adapter.spawn_entity("box1")
        topics = adapter.list_topics()
        assert "/box1/odom" in topics
        assert "/box1/scan" in topics

    def test_spawn_updates_robot_position(self, adapter):
        adapter.spawn_entity("turtlebot4", pose=(5.0, 7.0))
        assert adapter.get_position() == (5.0, 7.0)

    def test_spawn_raises_when_unavailable(self, unavailable_adapter):
        with pytest.raises(GazeboNotRunningError):
            unavailable_adapter.spawn_entity("box1")

    def test_spawn_multiple_entities(self, adapter):
        adapter.spawn_entity("box1")
        adapter.spawn_entity("box2")
        entities = adapter.list_entities()
        assert "box1" in entities
        assert "box2" in entities


# ---------------------------------------------------------------------------
# despawn_entity
# ---------------------------------------------------------------------------

class TestDespawnEntity:
    def test_despawn_removes_entity(self, adapter):
        adapter.spawn_entity("box1")
        result = adapter.despawn_entity("box1")
        assert result is True
        assert "box1" not in adapter.list_entities()

    def test_despawn_removes_topics(self, adapter):
        adapter.spawn_entity("box1")
        adapter.despawn_entity("box1")
        topics = adapter.list_topics()
        assert "/box1/odom" not in topics
        assert "/box1/scan" not in topics

    def test_despawn_nonexistent_returns_false(self, adapter):
        result = adapter.despawn_entity("ghost_entity")
        assert result is False

    def test_despawn_robot_itself_returns_false(self, adapter):
        result = adapter.despawn_entity("turtlebot4")
        assert result is False

    def test_despawn_raises_when_unavailable(self, unavailable_adapter):
        with pytest.raises(GazeboNotRunningError):
            unavailable_adapter.despawn_entity("box1")


# ---------------------------------------------------------------------------
# Joint states
# ---------------------------------------------------------------------------

class TestJointStates:
    def test_get_joint_states_empty_initially(self, adapter):
        states = adapter.get_joint_states()
        assert states == {}

    def test_set_joint_state(self, adapter):
        result = adapter.set_joint_state("left_wheel", position=1.57, velocity=0.5)
        assert result is True
        states = adapter.get_joint_states()
        assert "left_wheel" in states
        assert states["left_wheel"].position == pytest.approx(1.57)
        assert states["left_wheel"].velocity == pytest.approx(0.5)

    def test_set_joint_state_with_effort(self, adapter):
        adapter.set_joint_state("arm_joint", position=0.5, velocity=0.1, effort=2.0)
        assert adapter.get_joint_states()["arm_joint"].effort == pytest.approx(2.0)

    def test_set_joint_state_returns_joint_state_object(self, adapter):
        adapter.set_joint_state("j1", 0.0)
        js = adapter.get_joint_states()["j1"]
        assert isinstance(js, JointState)
        assert js.name == "j1"

    def test_multiple_joints(self, adapter):
        adapter.set_joint_state("j1", 1.0)
        adapter.set_joint_state("j2", 2.0)
        states = adapter.get_joint_states()
        assert len(states) == 2

    def test_get_joint_states_raises_when_unavailable(self, unavailable_adapter):
        with pytest.raises(GazeboNotRunningError):
            unavailable_adapter.get_joint_states()

    def test_set_joint_state_raises_when_unavailable(self, unavailable_adapter):
        with pytest.raises(GazeboNotRunningError):
            unavailable_adapter.set_joint_state("j1", 0.0)


# ---------------------------------------------------------------------------
# apply_force
# ---------------------------------------------------------------------------

class TestApplyForce:
    def test_apply_force_to_robot(self, adapter):
        result = adapter.apply_force("turtlebot4", fx=10.0, fy=0.0)
        assert result is True

    def test_apply_force_records_entry(self, adapter):
        adapter.apply_force("turtlebot4", fx=5.0, fy=2.0, fz=1.0, duration=0.5)
        forces = adapter._applied_forces
        assert len(forces) == 1
        assert forces[0]["entity"] == "turtlebot4"
        assert forces[0]["fx"] == 5.0
        assert forces[0]["duration"] == 0.5

    def test_apply_force_to_unknown_entity_raises(self, adapter):
        with pytest.raises(ValueError, match="not found"):
            adapter.apply_force("ghost", fx=1.0, fy=0.0)

    def test_apply_force_to_spawned_entity(self, adapter):
        adapter.spawn_entity("box1", pose=(1.0, 1.0))
        result = adapter.apply_force("box1", fx=0.0, fy=10.0)
        assert result is True

    def test_apply_force_raises_when_unavailable(self, unavailable_adapter):
        with pytest.raises(GazeboNotRunningError):
            unavailable_adapter.apply_force("turtlebot4", fx=1.0, fy=0.0)

    def test_multiple_forces(self, adapter):
        adapter.apply_force("turtlebot4", fx=1.0, fy=0.0)
        adapter.apply_force("turtlebot4", fx=0.0, fy=1.0)
        assert len(adapter._applied_forces) == 2


# ---------------------------------------------------------------------------
# reset_world
# ---------------------------------------------------------------------------

class TestResetWorld:
    def test_reset_clears_joint_states(self, adapter):
        adapter.set_joint_state("j1", 1.57)
        adapter.reset_world()
        assert adapter.get_joint_states() == {}

    def test_reset_clears_applied_forces(self, adapter):
        adapter.apply_force("turtlebot4", fx=10.0, fy=0.0)
        adapter.reset_world()
        assert adapter._applied_forces == []

    def test_reset_restores_robot_pose(self, adapter):
        adapter.spawn_entity("turtlebot4", pose=(0.0, 0.0))
        adapter.move(5.0, 5.0)
        adapter.reset_world()
        assert adapter.get_position() == (0.0, 0.0)

    def test_reset_restores_spawned_entities(self, adapter):
        adapter.spawn_entity("box1", pose=(3.0, 4.0))
        adapter.reset_world()
        # Entity still present (it was spawned before reset)
        assert "box1" in adapter.list_entities()
        assert adapter._entities["box1"] == (3.0, 4.0)

    def test_reset_resets_orientation(self, adapter):
        adapter.move(1.0, 1.0)
        adapter.reset_world()
        assert adapter.get_orientation() == 0.0

    def test_reset_returns_true(self, adapter):
        assert adapter.reset_world() is True

    def test_reset_raises_when_unavailable(self, unavailable_adapter):
        with pytest.raises(GazeboNotRunningError):
            unavailable_adapter.reset_world()


# ---------------------------------------------------------------------------
# smoke_test
# ---------------------------------------------------------------------------

class TestSmokeTest:
    def test_smoke_test_when_available(self, adapter):
        result = adapter.smoke_test()
        assert result["available"] is True
        assert result["spawned"] is True
        assert result["has_odom_topic"] is True
        assert "world" in result

    def test_smoke_test_when_unavailable(self, unavailable_adapter):
        result = unavailable_adapter.smoke_test()
        assert result["available"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# get_health
# ---------------------------------------------------------------------------

class TestGetHealth:
    def test_health_includes_expected_keys(self, adapter):
        health = adapter.get_health()
        assert health["adapter"] == "GazeboNativeAdapter"
        assert health["world"] == "test_world"
        assert "gazebo_available" in health
        assert "entities" in health
        assert "joints" in health

    def test_health_entities_count(self, adapter):
        adapter.spawn_entity("extra_bot")
        health = adapter.get_health()
        assert health["entities"] == 2  # robot + extra_bot


# ---------------------------------------------------------------------------
# Error handling — disconnected adapter
# ---------------------------------------------------------------------------

class TestDisconnectedAdapter:
    def test_spawn_raises_when_disconnected(self, adapter):
        adapter._state = AdapterState.DISCONNECTED
        with pytest.raises(GazeboNotRunningError, match="not connected"):
            adapter.spawn_entity("box1")

    def test_reset_raises_when_disconnected(self, adapter):
        adapter._state = AdapterState.DISCONNECTED
        with pytest.raises(GazeboNotRunningError, match="not connected"):
            adapter.reset_world()
