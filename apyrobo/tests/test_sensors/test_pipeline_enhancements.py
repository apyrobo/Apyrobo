import time

from apyrobo.core.schemas import SensorType
from apyrobo.sensors.pipeline import Obstacle, SensorPipeline, SensorReading


def test_semantic_find_object_prefers_best_confidence():
    pipeline = SensorPipeline()
    pipeline.feed(SensorReading(
        sensor_id="cam0",
        sensor_type=SensorType.CAMERA,
        data=[
            {"id": "o1", "label": "red box", "x": 1.0, "y": 1.0, "confidence": 0.4},
            {"id": "o2", "label": "box", "x": 2.0, "y": 2.0, "confidence": 0.9},
        ],
    ))

    obj = pipeline.get_world_state().find_object("red")
    assert obj is not None
    assert obj.object_id == "o1"


def test_obstacle_staleness_expires_old_obstacles():
    pipeline = SensorPipeline()
    pipeline.set_obstacle_max_age(0.01)
    pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, [{"x": 1.0, "y": 1.0}]))
    # Age obstacle
    for obstacle in pipeline.get_world_state().obstacles:
        obstacle.timestamp = time.time() - 10

    world = pipeline.get_world_state()
    assert len(world.obstacles) == 0


def test_gps_force_torque_and_depth_processors_update_world_state():
    pipeline = SensorPipeline()
    pipeline.inject_mock_reading(SensorType.GPS, {"lat": 10.0, "lon": 20.0, "fix": True}, sensor_id="gps0")
    pipeline.inject_mock_reading(SensorType.FORCE_TORQUE, {"fx": 0.0, "fy": 0.0, "fz": 9.0}, sensor_id="ft0")
    pipeline.inject_mock_reading(
        SensorType.DEPTH,
        [{"x": 1.0, "y": 2.0, "z": 0.5, "confidence": 0.8}],
        sensor_id="depth0",
    )

    world = pipeline.get_world_state()
    assert world.gps_position == (10.0, 20.0)
    assert world.metadata["gps_fix"] is True
    assert world.metadata["contact_detected"] is True
    assert world.metadata["grasp_success"] is True
    assert len(world.obstacle_point_cloud) == 1


def test_scene_graph_and_mock_processor_override():
    pipeline = SensorPipeline()

    def fake_camera(reading):
        pipeline._world.detected_objects = []

    pipeline.register_processor(SensorType.CAMERA, fake_camera)
    pipeline.feed(SensorReading("cam0", SensorType.CAMERA, [{"label": "ignored"}]))

    # Re-register standard by creating fresh pipeline for scene graph check
    pipeline = SensorPipeline()
    pipeline.feed(SensorReading(
        "cam0",
        SensorType.CAMERA,
        [
            {"id": "a", "label": "box", "x": 0.0, "y": 0.0},
            {"id": "b", "label": "door", "x": 0.4, "y": 0.2},
        ],
    ))
    scene = pipeline.get_world_state().scene_graph
    assert any(edge["relation"] == "next_to" for edge in scene)


def test_sensor_fusion_merges_nearby_obstacles():
    pipeline = SensorPipeline()
    pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, [{"x": 1.0, "y": 1.0, "confidence": 0.6}]))
    pipeline.feed(SensorReading("lidar1", SensorType.LIDAR, [{"x": 1.1, "y": 1.05, "confidence": 0.7}]))

    world = pipeline.get_world_state()
    assert len(world.obstacles) == 1
