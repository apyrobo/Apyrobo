"""
Comprehensive tests for apyrobo/sim/adapters.py — targeting missing coverage lines.

Covers:
- GazeboNativeAdapter (init, spawn_entity with/without entity==robot_name,
  list_topics, smoke_test, get_capabilities, move with dx/dy=0 and dx/dy!=0,
  stop, get_position, get_orientation)
- MuJoCoAdapter (init, get_capabilities, move, stop, get_position)
- IsaacSimAdapter (init, get_capabilities, move, stop)
- DomainRandomizationConfig defaults
- DomainRandomizer.randomize with/without seed
- RealityGapCalibrator.calibrate (empty and non-empty metrics, avg_gap > 0.15, < 0.15)
- SimToRealTransferPipeline (train_in_sim, evaluate_on_real, report)
"""

from __future__ import annotations

import math

import pytest

from apyrobo.sim.adapters import (
    DomainRandomizationConfig,
    DomainRandomizer,
    GazeboNativeAdapter,
    IsaacSimAdapter,
    MuJoCoAdapter,
    RealityGapCalibrator,
    SimToRealTransferPipeline,
)
from apyrobo.core.schemas import AdapterState, CapabilityType


# ---------------------------------------------------------------------------
# GazeboNativeAdapter
# ---------------------------------------------------------------------------

class TestGazeboNativeAdapter:
    def test_init_default(self):
        adapter = GazeboNativeAdapter("tb4")
        assert adapter.robot_name == "tb4"
        assert adapter._position == (0.0, 0.0)
        assert adapter._orientation == 0.0
        assert adapter._moving is False
        assert adapter._state == AdapterState.CONNECTED

    def test_init_with_world_kwarg(self):
        adapter = GazeboNativeAdapter("tb4", world="hospital")
        assert adapter._world == "hospital"

    def test_init_default_world(self):
        adapter = GazeboNativeAdapter("tb4")
        assert adapter._world == "default"

    def test_init_topics_include_robot_topics(self):
        adapter = GazeboNativeAdapter("tb4")
        topics = adapter.list_topics()
        assert "/tb4/cmd_vel" in topics
        assert "/tb4/odom" in topics
        assert "/tb4/scan" in topics

    def test_spawn_entity_new_entity(self):
        adapter = GazeboNativeAdapter("tb4")
        result = adapter.spawn_entity("obstacle_1", pose=(1.0, 2.0))
        assert result is True
        assert "obstacle_1" in adapter._entities
        topics = adapter.list_topics()
        assert "/obstacle_1/odom" in topics
        assert "/obstacle_1/scan" in topics

    def test_spawn_entity_when_entity_is_robot_name_updates_position(self):
        adapter = GazeboNativeAdapter("tb4")
        adapter.spawn_entity("tb4", pose=(3.0, 4.0))
        assert adapter._position == (3.0, 4.0)

    def test_spawn_entity_other_entity_does_not_change_position(self):
        adapter = GazeboNativeAdapter("tb4")
        adapter.spawn_entity("obstacle_1", pose=(5.0, 6.0))
        assert adapter._position == (0.0, 0.0)

    def test_list_topics_sorted(self):
        adapter = GazeboNativeAdapter("tb4")
        topics = adapter.list_topics()
        assert topics == sorted(topics)

    def test_smoke_test_returns_dict(self):
        adapter = GazeboNativeAdapter("tb4")
        result = adapter.smoke_test()
        assert result["spawned"] is True
        assert result["has_odom_topic"] is True
        assert result["topic_count"] > 0

    def test_get_capabilities_returns_robot_capability(self):
        adapter = GazeboNativeAdapter("tb4")
        caps = adapter.get_capabilities()
        assert caps.robot_id == "tb4"
        cap_types = {c.capability_type for c in caps.capabilities}
        assert CapabilityType.NAVIGATE in cap_types
        assert CapabilityType.ROTATE in cap_types

    def test_get_capabilities_has_sensors(self):
        adapter = GazeboNativeAdapter("tb4")
        caps = adapter.get_capabilities()
        sensor_types = {s.sensor_type.value for s in caps.sensors}
        assert "lidar" in sensor_types
        assert "imu" in sensor_types

    def test_get_capabilities_max_speed(self):
        adapter = GazeboNativeAdapter("tb4")
        caps = adapter.get_capabilities()
        assert caps.max_speed == 1.0

    def test_get_capabilities_metadata(self):
        adapter = GazeboNativeAdapter("tb4", world="warehouse")
        caps = adapter.get_capabilities()
        assert caps.metadata["backend"] == "gazebo_native"
        assert caps.metadata["world"] == "warehouse"

    def test_move_with_nonzero_dx_dy_updates_orientation(self):
        adapter = GazeboNativeAdapter("tb4")
        adapter.move(x=1.0, y=1.0)
        assert adapter._position == (1.0, 1.0)
        # orientation should be atan2(1, 1) = pi/4
        assert abs(adapter._orientation - math.atan2(1.0, 1.0)) < 1e-6

    def test_move_dx_dy_zero_does_not_update_orientation(self):
        adapter = GazeboNativeAdapter("tb4")
        adapter._position = (2.0, 3.0)
        adapter._orientation = 1.0
        adapter.move(x=2.0, y=3.0)  # same position -> dx=dy=0
        # orientation should remain unchanged
        assert adapter._orientation == 1.0

    def test_move_updates_position(self):
        adapter = GazeboNativeAdapter("tb4")
        adapter.move(x=5.0, y=6.0)
        assert adapter._position == (5.0, 6.0)

    def test_move_not_moving_after_complete(self):
        adapter = GazeboNativeAdapter("tb4")
        adapter.move(x=1.0, y=1.0)
        assert adapter._moving is False

    def test_stop(self):
        adapter = GazeboNativeAdapter("tb4")
        adapter._moving = True
        adapter.stop()
        assert adapter._moving is False

    def test_get_position(self):
        adapter = GazeboNativeAdapter("tb4")
        adapter.move(x=3.0, y=4.0)
        pos = adapter.get_position()
        assert pos == (3.0, 4.0)

    def test_get_orientation(self):
        adapter = GazeboNativeAdapter("tb4")
        adapter.move(x=1.0, y=0.0)
        ori = adapter.get_orientation()
        assert abs(ori - math.atan2(0.0, 1.0)) < 1e-6

    def test_move_with_speed_arg(self):
        adapter = GazeboNativeAdapter("tb4")
        adapter.move(x=2.0, y=3.0, speed=0.5)
        assert adapter._position == (2.0, 3.0)


# ---------------------------------------------------------------------------
# MuJoCoAdapter
# ---------------------------------------------------------------------------

class TestMuJoCoAdapter:
    def test_init_default(self):
        adapter = MuJoCoAdapter("mujoco_bot")
        assert adapter.robot_name == "mujoco_bot"
        assert adapter._position == (0.0, 0.0)
        assert adapter._model == "point_mass"
        assert adapter._state == AdapterState.CONNECTED

    def test_init_with_model_kwarg(self):
        adapter = MuJoCoAdapter("bot", model="humanoid")
        assert adapter._model == "humanoid"

    def test_get_capabilities_returns_navigate(self):
        adapter = MuJoCoAdapter("bot")
        caps = adapter.get_capabilities()
        cap_types = {c.capability_type for c in caps.capabilities}
        assert CapabilityType.NAVIGATE in cap_types

    def test_get_capabilities_metadata(self):
        adapter = MuJoCoAdapter("bot")
        caps = adapter.get_capabilities()
        assert caps.metadata["backend"] == "mujoco"
        assert caps.metadata["model"] == "point_mass"

    def test_get_capabilities_max_speed(self):
        adapter = MuJoCoAdapter("bot")
        caps = adapter.get_capabilities()
        assert caps.max_speed == 2.0

    def test_move_updates_position(self):
        adapter = MuJoCoAdapter("bot")
        adapter.move(x=1.5, y=2.5)
        assert adapter._position == (1.5, 2.5)

    def test_move_with_speed(self):
        adapter = MuJoCoAdapter("bot")
        adapter.move(x=1.0, y=1.0, speed=1.5)
        assert adapter._position == (1.0, 1.0)

    def test_stop_does_not_raise(self):
        adapter = MuJoCoAdapter("bot")
        adapter.stop()

    def test_get_position(self):
        adapter = MuJoCoAdapter("bot")
        adapter.move(x=3.0, y=4.0)
        assert adapter.get_position() == (3.0, 4.0)


# ---------------------------------------------------------------------------
# IsaacSimAdapter
# ---------------------------------------------------------------------------

class TestIsaacSimAdapter:
    def test_init(self):
        adapter = IsaacSimAdapter("isaac_bot")
        assert adapter.robot_name == "isaac_bot"
        assert adapter._position == (0.0, 0.0)
        assert adapter._state == AdapterState.CONNECTED

    def test_get_capabilities_returns_navigate(self):
        adapter = IsaacSimAdapter("isaac_bot")
        caps = adapter.get_capabilities()
        cap_types = {c.capability_type for c in caps.capabilities}
        assert CapabilityType.NAVIGATE in cap_types

    def test_get_capabilities_metadata(self):
        adapter = IsaacSimAdapter("isaac_bot")
        caps = adapter.get_capabilities()
        assert caps.metadata["backend"] == "isaac_sim"
        assert caps.metadata["rendering"] == "rtx"

    def test_get_capabilities_max_speed(self):
        adapter = IsaacSimAdapter("isaac_bot")
        caps = adapter.get_capabilities()
        assert caps.max_speed == 1.5

    def test_move_updates_position(self):
        adapter = IsaacSimAdapter("isaac_bot")
        adapter.move(x=2.0, y=3.0)
        assert adapter._position == (2.0, 3.0)

    def test_move_with_speed(self):
        adapter = IsaacSimAdapter("isaac_bot")
        adapter.move(x=1.0, y=0.0, speed=1.0)
        assert adapter._position == (1.0, 0.0)

    def test_stop_does_not_raise(self):
        adapter = IsaacSimAdapter("isaac_bot")
        adapter.stop()


# ---------------------------------------------------------------------------
# DomainRandomizationConfig
# ---------------------------------------------------------------------------

class TestDomainRandomizationConfig:
    def test_defaults(self):
        cfg = DomainRandomizationConfig()
        assert cfg.lighting_range == (0.6, 1.4)
        assert cfg.texture_pool == ("matte", "metal", "plastic")
        assert cfg.position_jitter_m == 0.5

    def test_custom_values(self):
        cfg = DomainRandomizationConfig(
            lighting_range=(0.8, 1.2),
            texture_pool=("rough", "smooth"),
            position_jitter_m=0.1,
        )
        assert cfg.lighting_range == (0.8, 1.2)
        assert cfg.position_jitter_m == 0.1


# ---------------------------------------------------------------------------
# DomainRandomizer
# ---------------------------------------------------------------------------

class TestDomainRandomizer:
    def test_randomize_without_seed_varies(self):
        dr = DomainRandomizer()
        scene = {"name": "test_scene"}
        r1 = dr.randomize(scene)
        r2 = dr.randomize(scene)
        # With different seeds they likely differ (probabilistic, but usually)
        assert "lighting" in r1
        assert "texture" in r1
        assert "position_jitter_m" in r1

    def test_randomize_with_seed_deterministic(self):
        dr = DomainRandomizer()
        scene = {"name": "test"}
        r1 = dr.randomize(scene, seed=42)
        r2 = dr.randomize(scene, seed=42)
        assert r1["lighting"] == r2["lighting"]
        assert r1["texture"] == r2["texture"]

    def test_randomize_lighting_in_range(self):
        dr = DomainRandomizer()
        scene = {}
        for seed in range(10):
            r = dr.randomize(scene, seed=seed)
            assert 0.6 <= r["lighting"] <= 1.4

    def test_randomize_texture_in_pool(self):
        dr = DomainRandomizer()
        scene = {}
        for seed in range(10):
            r = dr.randomize(scene, seed=seed)
            assert r["texture"] in ("matte", "metal", "plastic")

    def test_randomize_position_jitter_set(self):
        dr = DomainRandomizer()
        scene = {}
        r = dr.randomize(scene, seed=1)
        assert r["position_jitter_m"] == 0.5

    def test_randomize_with_custom_config(self):
        cfg = DomainRandomizationConfig(
            lighting_range=(1.0, 1.0),
            texture_pool=("only_one",),
            position_jitter_m=0.0,
        )
        dr = DomainRandomizer(cfg)
        scene = {}
        r = dr.randomize(scene, seed=0)
        assert r["lighting"] == 1.0
        assert r["texture"] == "only_one"
        assert r["position_jitter_m"] == 0.0

    def test_randomize_preserves_original_scene_keys(self):
        dr = DomainRandomizer()
        scene = {"original_key": "original_value"}
        r = dr.randomize(scene, seed=0)
        assert r["original_key"] == "original_value"

    def test_randomize_does_not_mutate_original_scene(self):
        dr = DomainRandomizer()
        scene = {"key": "val"}
        dr.randomize(scene, seed=0)
        assert "lighting" not in scene


# ---------------------------------------------------------------------------
# RealityGapCalibrator
# ---------------------------------------------------------------------------

class TestRealityGapCalibrator:
    def test_calibrate_empty_metrics(self):
        cal = RealityGapCalibrator()
        result = cal.calibrate({}, {})
        assert result["discrepancies"] == {}
        assert result["avg_gap"] == 0.0
        assert result["recommendation"] == "within_tolerance"

    def test_calibrate_no_common_keys(self):
        cal = RealityGapCalibrator()
        result = cal.calibrate({"sim_only": 1.0}, {"real_only": 1.0})
        assert result["discrepancies"] == {}
        assert result["avg_gap"] == 0.0

    def test_calibrate_identical_metrics_zero_gap(self):
        cal = RealityGapCalibrator()
        result = cal.calibrate({"speed": 1.0}, {"speed": 1.0})
        assert result["discrepancies"]["speed"] == 0.0
        assert result["avg_gap"] == 0.0
        assert result["recommendation"] == "within_tolerance"

    def test_calibrate_large_gap_recommends_tune_sim(self):
        cal = RealityGapCalibrator()
        result = cal.calibrate({"speed": 2.0}, {"speed": 1.0})
        assert result["avg_gap"] > 0.15
        assert result["recommendation"] == "tune_sim"

    def test_calibrate_small_gap_within_tolerance(self):
        cal = RealityGapCalibrator()
        result = cal.calibrate({"speed": 1.01}, {"speed": 1.0})
        assert result["avg_gap"] < 0.15
        assert result["recommendation"] == "within_tolerance"

    def test_calibrate_multiple_metrics(self):
        cal = RealityGapCalibrator()
        result = cal.calibrate(
            {"speed": 1.0, "torque": 2.0},
            {"speed": 1.0, "torque": 1.0},
        )
        assert "speed" in result["discrepancies"]
        assert "torque" in result["discrepancies"]

    def test_calibrate_near_zero_real_value_uses_unit_denom(self):
        cal = RealityGapCalibrator()
        # real is near zero — denominator should clamp to 1.0
        result = cal.calibrate({"metric": 1.0}, {"metric": 0.0000001})
        # Should not divide by near-zero
        assert result["discrepancies"]["metric"] == pytest.approx(1.0, abs=0.01)

    def test_calibrate_boundary_avg_gap_exactly_015(self):
        # avg_gap == 0.15: recommendation is "within_tolerance" (not > 0.15)
        cal = RealityGapCalibrator()
        # Find values that produce exactly 0.15
        # |sim - real| / real = 0.15 → sim = 1.15, real = 1.0
        result = cal.calibrate({"x": 1.15}, {"x": 1.0})
        assert result["avg_gap"] == pytest.approx(0.15)
        assert result["recommendation"] == "within_tolerance"


# ---------------------------------------------------------------------------
# SimToRealTransferPipeline
# ---------------------------------------------------------------------------

class TestSimToRealTransferPipeline:
    def test_train_in_sim_returns_record(self):
        pipeline = SimToRealTransferPipeline()
        rec = pipeline.train_in_sim("ppo_v1", episodes=50)
        assert rec["stage"] == "sim_train"
        assert rec["policy"] == "ppo_v1"
        assert rec["episodes"] == 50
        assert "timestamp" in rec

    def test_train_in_sim_default_episodes(self):
        pipeline = SimToRealTransferPipeline()
        rec = pipeline.train_in_sim("sac")
        assert rec["episodes"] == 100

    def test_evaluate_on_real_returns_record(self):
        pipeline = SimToRealTransferPipeline()
        rec = pipeline.evaluate_on_real("ppo_v1", success_rate=0.85)
        assert rec["stage"] == "real_eval"
        assert rec["policy"] == "ppo_v1"
        assert rec["success_rate"] == 0.85

    def test_report_empty(self):
        pipeline = SimToRealTransferPipeline()
        rep = pipeline.report()
        assert rep["sim_runs"] == 0
        assert rep["real_runs"] == 0
        assert rep["history"] == []

    def test_report_after_runs(self):
        pipeline = SimToRealTransferPipeline()
        pipeline.train_in_sim("policy_a", episodes=100)
        pipeline.train_in_sim("policy_b", episodes=200)
        pipeline.evaluate_on_real("policy_a", success_rate=0.7)
        rep = pipeline.report()
        assert rep["sim_runs"] == 2
        assert rep["real_runs"] == 1
        assert len(rep["history"]) == 3

    def test_report_history_is_copy(self):
        pipeline = SimToRealTransferPipeline()
        pipeline.train_in_sim("p")
        rep1 = pipeline.report()
        rep2 = pipeline.report()
        assert rep1["history"] is not rep2["history"]
