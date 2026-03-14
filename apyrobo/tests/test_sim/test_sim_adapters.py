from apyrobo.core.robot import Robot
from apyrobo.sim.adapters import (
    DomainRandomizer,
    GazeboNativeAdapter,
    RealityGapCalibrator,
    SimToRealTransferPipeline,
)


def test_gazebo_native_adapter_smoke_spawn_and_topics():
    robot = Robot.discover("gazebo_native://tb4")
    adapter = robot._adapter
    assert isinstance(adapter, GazeboNativeAdapter)

    report = adapter.smoke_test()
    assert report["spawned"] is True
    assert report["has_odom_topic"] is True
    assert report["topic_count"] >= 3


def test_mujoco_and_isaac_adapters_are_registered():
    mujoco = Robot.discover("mujoco://m1")
    isaac = Robot.discover("isaac://i1")
    assert mujoco.capabilities().metadata["backend"] == "mujoco"
    assert isaac.capabilities().metadata["backend"] == "isaac_sim"


def test_domain_randomization_and_reality_gap_calibration():
    randomizer = DomainRandomizer()
    randomized = randomizer.randomize({"lighting": 1.0, "texture": "matte"}, seed=42)
    assert "lighting" in randomized
    assert "texture" in randomized

    calibrator = RealityGapCalibrator()
    report = calibrator.calibrate(
        sim_metrics={"tracking_error": 0.10, "stop_error": 0.12},
        real_metrics={"tracking_error": 0.20, "stop_error": 0.10},
    )
    assert "avg_gap" in report
    assert report["avg_gap"] >= 0.0


def test_sim_to_real_transfer_pipeline_records_metrics():
    pipeline = SimToRealTransferPipeline()
    pipeline.train_in_sim("policy_a", episodes=10)
    pipeline.evaluate_on_real("policy_a", success_rate=0.7)
    summary = pipeline.report()

    assert summary["sim_runs"] == 1
    assert summary["real_runs"] == 1
