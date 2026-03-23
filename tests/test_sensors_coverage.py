"""
Comprehensive tests for apyrobo/sensors/pipeline.py.

Covers: SensorReading, Obstacle, DetectedObject, WorldState
(nearest_obstacle, obstacles_within, find_object, expire_stale_obstacles,
build_scene_graph, is_path_clear, to_dict), MockDetector, SensorPipeline
(feed, _process_lidar, _process_camera, _process_imu, _process_gps,
_process_force_torque, _process_depth, _fuse_obstacles, get_world_state,
set_obstacle_max_age, set_target_labels, set_detector, inject_mock_reading,
register_processor, latest_readings, reading_count, __repr__),
_load_detector.
"""
from __future__ import annotations

import math
import time
from unittest.mock import MagicMock, patch

import pytest

from apyrobo.core.schemas import SensorType
from apyrobo.sensors.pipeline import (
    SensorReading,
    Obstacle,
    DetectedObject,
    WorldState,
    SensorPipeline,
    ObjectDetector,
    MockDetector,
    _load_detector,
)


# ---------------------------------------------------------------------------
# SensorReading
# ---------------------------------------------------------------------------

class TestSensorReading:
    def test_basic_creation(self):
        r = SensorReading("lidar0", SensorType.LIDAR, {"data": [1, 2, 3]})
        assert r.sensor_id == "lidar0"
        assert r.sensor_type == SensorType.LIDAR
        assert r.data == {"data": [1, 2, 3]}
        assert r.timestamp > 0

    def test_custom_timestamp(self):
        ts = 1234567890.0
        r = SensorReading("cam0", SensorType.CAMERA, None, timestamp=ts)
        assert r.timestamp == ts

    def test_repr(self):
        r = SensorReading("lidar0", SensorType.LIDAR, None)
        rep = repr(r)
        assert "SensorReading" in rep
        assert "lidar0" in rep
        assert "lidar" in rep


# ---------------------------------------------------------------------------
# Obstacle
# ---------------------------------------------------------------------------

class TestObstacle:
    def test_basic_creation(self):
        obs = Obstacle(x=1.0, y=2.0, radius=0.3, confidence=0.9, source="lidar")
        assert obs.x == 1.0
        assert obs.y == 2.0
        assert obs.radius == 0.3

    def test_distance_to(self):
        obs = Obstacle(x=0.0, y=0.0)
        dist = obs.distance_to(3.0, 4.0)
        assert dist == pytest.approx(5.0)

    def test_distance_to_same_point(self):
        obs = Obstacle(x=1.0, y=1.0)
        dist = obs.distance_to(1.0, 1.0)
        assert dist == pytest.approx(0.0)

    def test_repr(self):
        obs = Obstacle(x=1.5, y=2.5, radius=0.3)
        rep = repr(obs)
        assert "Obstacle" in rep
        assert "1.5" in rep
        assert "2.5" in rep
        assert "r=0.3" in rep


# ---------------------------------------------------------------------------
# DetectedObject
# ---------------------------------------------------------------------------

class TestDetectedObject:
    def test_basic_creation(self):
        obj = DetectedObject("obj_0", "box", 1.0, 2.0, confidence=0.95, source="camera")
        assert obj.object_id == "obj_0"
        assert obj.label == "box"
        assert obj.x == 1.0
        assert obj.y == 2.0

    def test_repr(self):
        obj = DetectedObject("obj_0", "chair", 3.0, 4.0)
        rep = repr(obj)
        assert "chair" in rep
        assert "3.0" in rep
        assert "4.0" in rep


# ---------------------------------------------------------------------------
# WorldState
# ---------------------------------------------------------------------------

class TestWorldState:
    def setup_method(self):
        self.ws = WorldState()

    def test_nearest_obstacle_none(self):
        assert self.ws.nearest_obstacle() is None

    def test_nearest_obstacle_found(self):
        self.ws.obstacles = [
            Obstacle(1.0, 0.0),
            Obstacle(5.0, 0.0),
            Obstacle(2.0, 0.0),
        ]
        nearest = self.ws.nearest_obstacle(x=0.0, y=0.0)
        assert nearest.x == 1.0

    def test_nearest_obstacle_uses_robot_position(self):
        self.ws.robot_position = (4.0, 0.0)
        self.ws.obstacles = [Obstacle(1.0, 0.0), Obstacle(5.0, 0.0)]
        nearest = self.ws.nearest_obstacle()
        assert nearest.x == 5.0

    def test_obstacles_within(self):
        self.ws.obstacles = [
            Obstacle(1.0, 0.0),
            Obstacle(10.0, 0.0),
        ]
        within = self.ws.obstacles_within(radius=2.0, x=0.0, y=0.0)
        assert len(within) == 1
        assert within[0].x == 1.0

    def test_obstacles_within_uses_robot_position(self):
        self.ws.robot_position = (0.0, 0.0)
        self.ws.obstacles = [Obstacle(0.5, 0.0), Obstacle(5.0, 0.0)]
        within = self.ws.obstacles_within(radius=1.0)
        assert len(within) == 1

    def test_find_object_exact_match(self):
        self.ws.detected_objects = [
            DetectedObject("obj_0", "box", 1.0, 1.0, confidence=0.9),
            DetectedObject("obj_1", "chair", 2.0, 2.0),
        ]
        result = self.ws.find_object("box")
        assert result is not None
        assert result.label == "box"

    def test_find_object_prefers_highest_confidence(self):
        self.ws.detected_objects = [
            DetectedObject("obj_0", "box", 1.0, 1.0, confidence=0.5),
            DetectedObject("obj_1", "box", 2.0, 2.0, confidence=0.9),
        ]
        result = self.ws.find_object("box")
        assert result.confidence == 0.9

    def test_find_object_partial_match(self):
        self.ws.detected_objects = [
            DetectedObject("obj_0", "cardboard box", 1.0, 1.0),
        ]
        result = self.ws.find_object("box")
        assert result is not None
        assert "box" in result.label

    def test_find_object_token_match(self):
        self.ws.detected_objects = [
            DetectedObject("obj_0", "red wooden chair", 1.0, 1.0),
        ]
        result = self.ws.find_object("chair")
        assert result is not None

    def test_find_object_not_found(self):
        self.ws.detected_objects = [
            DetectedObject("obj_0", "box", 1.0, 1.0),
        ]
        result = self.ws.find_object("elephant")
        assert result is None

    def test_find_object_empty_label(self):
        result = self.ws.find_object("")
        assert result is None

    def test_find_object_whitespace_label(self):
        result = self.ws.find_object("   ")
        assert result is None

    def test_expire_stale_obstacles(self):
        old_obstacle = Obstacle(1.0, 0.0)
        old_obstacle.timestamp = time.time() - 100  # Very old
        fresh_obstacle = Obstacle(2.0, 0.0)
        self.ws.obstacles = [old_obstacle, fresh_obstacle]
        removed = self.ws.expire_stale_obstacles(max_age_s=10.0)
        assert removed == 1
        assert len(self.ws.obstacles) == 1

    def test_build_scene_graph(self):
        self.ws.detected_objects = [
            DetectedObject("obj_0", "box", 0.0, 0.0),
            DetectedObject("obj_1", "chair", 0.5, 0.0),  # near box
            DetectedObject("obj_2", "table", 10.0, 0.0),  # far
        ]
        edges = self.ws.build_scene_graph(near_threshold=1.0)
        assert len(edges) == 2  # box-chair and chair-box
        subjects = {e["subject"] for e in edges}
        assert "obj_0" in subjects
        assert "obj_1" in subjects

    def test_is_path_clear_no_obstacles(self):
        assert self.ws.is_path_clear(0.0, 0.0, 5.0, 0.0) is True

    def test_is_path_clear_blocked(self):
        self.ws.obstacles = [Obstacle(2.5, 0.0, radius=0.3)]
        assert self.ws.is_path_clear(0.0, 0.0, 5.0, 0.0, clearance=0.5) is False

    def test_is_path_clear_obstacle_off_path(self):
        self.ws.obstacles = [Obstacle(2.5, 5.0, radius=0.3)]  # Far from path
        assert self.ws.is_path_clear(0.0, 0.0, 5.0, 0.0, clearance=0.5) is True

    def test_to_dict(self):
        self.ws.robot_position = (1.0, 2.0)
        self.ws.gps_position = (51.5, -0.1)
        self.ws.obstacles = [Obstacle(3.0, 4.0)]
        self.ws.detected_objects = [DetectedObject("obj_0", "box", 5.0, 6.0)]
        d = self.ws.to_dict()
        assert d["robot_position"] == [1.0, 2.0]
        assert d["gps_position"] == [51.5, -0.1]
        assert len(d["obstacles"]) == 1
        assert len(d["detected_objects"]) == 1

    def test_to_dict_no_gps(self):
        d = self.ws.to_dict()
        assert d["gps_position"] is None

    def test_repr(self):
        self.ws.robot_position = (1.0, 2.0)
        self.ws.obstacles = [Obstacle(0, 0)]
        r = repr(self.ws)
        assert "WorldState" in r
        assert "obstacles=1" in r

    def test_point_to_segment_distance_zero_length(self):
        # When start == end, distance is just point distance to start
        dist = WorldState._point_to_segment_distance(0.0, 0.0, 3.0, 4.0, 3.0, 4.0)
        assert dist == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# MockDetector
# ---------------------------------------------------------------------------

class TestMockDetector:
    def test_detect_all(self):
        detector = MockDetector([
            {"label": "box", "x": 1.0, "y": 2.0, "confidence": 0.9},
            {"label": "chair", "x": 3.0, "y": 4.0},
        ])
        objects = detector.detect(None)
        assert len(objects) == 2
        assert objects[0].label == "box"

    def test_detect_with_target_labels(self):
        detector = MockDetector([
            {"label": "box", "x": 1.0, "y": 0.0},
            {"label": "chair", "x": 2.0, "y": 0.0},
        ])
        objects = detector.detect(None, target_labels=["box"])
        assert len(objects) == 1
        assert objects[0].label == "box"

    def test_detect_empty(self):
        detector = MockDetector([])
        objects = detector.detect(None)
        assert objects == []

    def test_detect_custom_id(self):
        detector = MockDetector([
            {"id": "my_box", "label": "box", "x": 0.0, "y": 0.0},
        ])
        objects = detector.detect(None)
        assert objects[0].object_id == "my_box"

    def test_detect_default_id(self):
        detector = MockDetector([
            {"label": "box", "x": 0.0, "y": 0.0},
        ])
        objects = detector.detect(None)
        assert objects[0].object_id == "mock_0"

    def test_set_detections(self):
        detector = MockDetector([])
        detector.set_detections([{"label": "new", "x": 0.0, "y": 0.0}])
        objects = detector.detect(None)
        assert len(objects) == 1
        assert objects[0].label == "new"


# ---------------------------------------------------------------------------
# _load_detector
# ---------------------------------------------------------------------------

class TestLoadDetector:
    def test_load_none(self):
        assert _load_detector("none") is None

    def test_load_mock(self):
        detector = _load_detector("mock")
        assert isinstance(detector, MockDetector)

    def test_load_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown detector"):
            _load_detector("unknown_backend")


# ---------------------------------------------------------------------------
# SensorPipeline — LIDAR
# ---------------------------------------------------------------------------

class TestSensorPipelineLidar:
    def setup_method(self):
        self.pipeline = SensorPipeline()

    def test_feed_lidar_xy_data(self):
        data = [{"x": 1.0, "y": 0.0}, {"x": 2.0, "y": 1.0}]
        self.pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, data))
        world = self.pipeline.get_world_state()
        assert len(world.obstacles) > 0

    def test_feed_lidar_polar_data(self):
        data = [{"angle": 0.0, "distance": 2.0}]
        self.pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, data))
        world = self.pipeline.get_world_state()
        assert len(world.obstacles) > 0

    def test_feed_lidar_not_list_ignored(self):
        self.pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, "invalid"))
        world = self.pipeline.get_world_state()
        assert len(world.obstacles) == 0

    def test_feed_lidar_with_custom_radius(self):
        data = [{"x": 1.0, "y": 0.0, "radius": 0.5}]
        self.pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, data))
        world = self.pipeline.get_world_state()
        assert world.obstacles[0].radius == 0.5


# ---------------------------------------------------------------------------
# SensorPipeline — Camera
# ---------------------------------------------------------------------------

class TestSensorPipelineCamera:
    def setup_method(self):
        self.pipeline = SensorPipeline()

    def test_feed_camera_detections(self):
        data = [
            {"id": "obj_0", "label": "box", "x": 1.0, "y": 2.0},
        ]
        self.pipeline.feed(SensorReading("cam0", SensorType.CAMERA, data))
        world = self.pipeline.get_world_state()
        assert len(world.detected_objects) == 1

    def test_feed_camera_center_xy(self):
        data = [
            {"label": "chair", "center_x": 3.0, "center_y": 4.0},
        ]
        self.pipeline.feed(SensorReading("cam0", SensorType.CAMERA, data))
        world = self.pipeline.get_world_state()
        assert world.detected_objects[0].x == 3.0
        assert world.detected_objects[0].y == 4.0

    def test_feed_camera_not_list_ignored(self):
        self.pipeline.feed(SensorReading("cam0", SensorType.CAMERA, "invalid"))
        world = self.pipeline.get_world_state()
        assert len(world.detected_objects) == 0

    def test_feed_camera_with_detector(self):
        mock_detector = MockDetector([
            {"label": "box", "x": 1.0, "y": 0.0, "confidence": 0.95},
        ])
        self.pipeline.set_detector(mock_detector)
        # Feed raw image data (any value — MockDetector ignores it)
        self.pipeline.feed(SensorReading("cam0", SensorType.CAMERA, b"fake_image"))
        world = self.pipeline.get_world_state()
        assert len(world.detected_objects) == 1
        assert world.detected_objects[0].label == "box"

    def test_feed_camera_with_detector_and_target_labels(self):
        mock_detector = MockDetector([
            {"label": "box", "x": 1.0, "y": 0.0},
            {"label": "chair", "x": 2.0, "y": 0.0},
        ])
        pipeline = SensorPipeline(target_labels=["box"])
        pipeline.set_detector(mock_detector)
        pipeline.feed(SensorReading("cam0", SensorType.CAMERA, b"img"))
        world = pipeline.get_world_state()
        assert len(world.detected_objects) == 1
        assert world.detected_objects[0].label == "box"


# ---------------------------------------------------------------------------
# SensorPipeline — IMU
# ---------------------------------------------------------------------------

class TestSensorPipelineIMU:
    def test_feed_imu_position_and_yaw(self):
        pipeline = SensorPipeline()
        data = {"x": 2.0, "y": 3.0, "yaw": 1.57}
        pipeline.feed(SensorReading("imu0", SensorType.IMU, data))
        world = pipeline.get_world_state()
        assert world.robot_position == (2.0, 3.0)
        assert world.robot_orientation == pytest.approx(1.57)

    def test_feed_imu_only_yaw(self):
        pipeline = SensorPipeline()
        data = {"yaw": 0.5}
        pipeline.feed(SensorReading("imu0", SensorType.IMU, data))
        world = pipeline.get_world_state()
        assert world.robot_orientation == pytest.approx(0.5)

    def test_feed_imu_not_dict_ignored(self):
        pipeline = SensorPipeline()
        pipeline.feed(SensorReading("imu0", SensorType.IMU, "invalid"))
        # Should not change position
        world = pipeline.get_world_state()
        assert world.robot_position == (0.0, 0.0)


# ---------------------------------------------------------------------------
# SensorPipeline — GPS
# ---------------------------------------------------------------------------

class TestSensorPipelineGPS:
    def test_feed_gps_xy(self):
        pipeline = SensorPipeline()
        data = {"x": 10.0, "y": 20.0}
        pipeline.feed(SensorReading("gps0", SensorType.GPS, data))
        world = pipeline.get_world_state()
        assert world.gps_position == (10.0, 20.0)

    def test_feed_gps_latlon(self):
        pipeline = SensorPipeline()
        data = {"lat": 51.5, "lon": -0.1}
        pipeline.feed(SensorReading("gps0", SensorType.GPS, data))
        world = pipeline.get_world_state()
        assert world.gps_position == pytest.approx((51.5, -0.1))

    def test_feed_gps_fix_metadata(self):
        pipeline = SensorPipeline()
        data = {"x": 1.0, "y": 2.0, "fix": False}
        pipeline.feed(SensorReading("gps0", SensorType.GPS, data))
        world = pipeline.get_world_state()
        assert world.metadata.get("gps_fix") is False

    def test_feed_gps_not_dict_ignored(self):
        pipeline = SensorPipeline()
        pipeline.feed(SensorReading("gps0", SensorType.GPS, "invalid"))
        world = pipeline.get_world_state()
        assert world.gps_position is None


# ---------------------------------------------------------------------------
# SensorPipeline — Force/Torque
# ---------------------------------------------------------------------------

class TestSensorPipelineForceTorque:
    def test_feed_force_torque(self):
        pipeline = SensorPipeline()
        data = {"fx": 1.0, "fy": 0.0, "fz": 0.0, "contact_threshold": 0.5}
        pipeline.feed(SensorReading("ft0", SensorType.FORCE_TORQUE, data))
        world = pipeline.get_world_state()
        ft = world.metadata.get("force_torque")
        assert ft is not None
        assert ft["fx"] == 1.0
        assert world.metadata["contact_detected"] is True

    def test_feed_force_torque_no_contact(self):
        pipeline = SensorPipeline()
        data = {"fx": 0.1, "fy": 0.0, "fz": 0.0}
        pipeline.feed(SensorReading("ft0", SensorType.FORCE_TORQUE, data))
        world = pipeline.get_world_state()
        assert world.metadata["contact_detected"] is False

    def test_feed_force_torque_grasp_success(self):
        pipeline = SensorPipeline()
        data = {"fx": 5.0, "fy": 5.0, "fz": 5.0, "grasp_threshold": 5.0}
        pipeline.feed(SensorReading("ft0", SensorType.FORCE_TORQUE, data))
        world = pipeline.get_world_state()
        assert world.metadata["grasp_success"] is True

    def test_feed_force_torque_not_dict_ignored(self):
        pipeline = SensorPipeline()
        pipeline.feed(SensorReading("ft0", SensorType.FORCE_TORQUE, "invalid"))
        world = pipeline.get_world_state()
        assert "force_torque" not in world.metadata


# ---------------------------------------------------------------------------
# SensorPipeline — Depth
# ---------------------------------------------------------------------------

class TestSensorPipelineDepth:
    def test_feed_depth_data(self):
        pipeline = SensorPipeline()
        data = [
            {"x": 1.0, "y": 0.0, "z": 0.5},
            {"x": 2.0, "y": 0.0, "z": 0.5},
        ]
        pipeline.feed(SensorReading("depth0", SensorType.DEPTH, data))
        world = pipeline.get_world_state()
        assert len(world.obstacle_point_cloud) == 2
        assert len(world.obstacles) > 0

    def test_feed_depth_not_list_ignored(self):
        pipeline = SensorPipeline()
        pipeline.feed(SensorReading("depth0", SensorType.DEPTH, "invalid"))
        world = pipeline.get_world_state()
        assert len(world.obstacle_point_cloud) == 0

    def test_feed_depth_non_dict_points_skipped(self):
        pipeline = SensorPipeline()
        data = [
            {"x": 1.0, "y": 0.0, "z": 0.5},
            "invalid_point",  # Should be skipped
        ]
        pipeline.feed(SensorReading("depth0", SensorType.DEPTH, data))
        world = pipeline.get_world_state()
        assert len(world.obstacle_point_cloud) == 1

    def test_feed_depth_custom_radius_confidence(self):
        pipeline = SensorPipeline()
        data = [{"x": 1.0, "y": 0.0, "z": 0.5, "radius": 0.2, "confidence": 0.8}]
        pipeline.feed(SensorReading("depth0", SensorType.DEPTH, data))
        world = pipeline.get_world_state()
        assert world.obstacles[0].radius == 0.2
        assert world.obstacles[0].confidence == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# SensorPipeline — _fuse_obstacles
# ---------------------------------------------------------------------------

class TestFuseObstacles:
    def test_nearby_obstacles_merged(self):
        pipeline = SensorPipeline()
        # Two obstacles very close together — should be merged
        data = [
            {"x": 0.0, "y": 0.0},
            {"x": 0.1, "y": 0.0},  # Within 0.35m merge distance
        ]
        pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, data))
        world = pipeline.get_world_state()
        assert len(world.obstacles) == 1

    def test_far_obstacles_not_merged(self):
        pipeline = SensorPipeline()
        data = [
            {"x": 0.0, "y": 0.0},
            {"x": 5.0, "y": 0.0},  # Far apart
        ]
        pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, data))
        world = pipeline.get_world_state()
        assert len(world.obstacles) == 2


# ---------------------------------------------------------------------------
# SensorPipeline — get_world_state with stale obstacle expiry
# ---------------------------------------------------------------------------

class TestGetWorldState:
    def test_set_obstacle_max_age(self):
        pipeline = SensorPipeline()
        pipeline.set_obstacle_max_age(max_age_s=0.001)

        # Add a stale obstacle
        data = [{"x": 1.0, "y": 0.0}]
        pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, data))
        time.sleep(0.01)  # Wait for it to become stale

        world = pipeline.get_world_state()
        assert len(world.obstacles) == 0

    def test_no_max_age_preserves_obstacles(self):
        pipeline = SensorPipeline()
        data = [{"x": 1.0, "y": 0.0}]
        pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, data))
        world = pipeline.get_world_state()
        assert len(world.obstacles) > 0


# ---------------------------------------------------------------------------
# SensorPipeline — misc
# ---------------------------------------------------------------------------

class TestSensorPipelineMisc:
    def test_unknown_sensor_type_no_crash(self):
        # Use a sensor type with no registered processor
        pipeline = SensorPipeline()
        # Create a reading with an unregistered sensor type
        reading = SensorReading("custom0", SensorType.LIDAR, None)
        # Remove the lidar processor to simulate unknown type
        pipeline._processors.pop(SensorType.LIDAR, None)
        pipeline.feed(reading)  # Should not raise

    def test_set_target_labels(self):
        pipeline = SensorPipeline()
        pipeline.set_target_labels(["box", "chair"])
        assert pipeline._target_labels == ["box", "chair"]

    def test_set_detector(self):
        pipeline = SensorPipeline()
        mock_detector = MockDetector([])
        pipeline.set_detector(mock_detector)
        assert pipeline.detector is mock_detector

    def test_set_detector_none(self):
        pipeline = SensorPipeline(detector_backend="mock")
        pipeline.set_detector(None)
        assert pipeline.detector is None

    def test_detector_property(self):
        pipeline = SensorPipeline()
        assert pipeline.detector is None

    def test_inject_mock_reading(self):
        pipeline = SensorPipeline()
        data = [{"x": 1.0, "y": 0.0}]
        pipeline.inject_mock_reading(SensorType.LIDAR, data, sensor_id="test_lidar")
        world = pipeline.get_world_state()
        assert len(world.obstacles) > 0

    def test_register_processor(self):
        pipeline = SensorPipeline()
        custom_calls = []

        def custom_processor(reading):
            custom_calls.append(reading)

        pipeline.register_processor(SensorType.LIDAR, custom_processor)
        data = [{"x": 1.0, "y": 0.0}]
        pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, data))
        assert len(custom_calls) == 1

    def test_latest_readings(self):
        pipeline = SensorPipeline()
        pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, []))
        pipeline.feed(SensorReading("cam0", SensorType.CAMERA, []))
        readings = pipeline.latest_readings
        assert "lidar0" in readings
        assert "cam0" in readings

    def test_reading_count(self):
        pipeline = SensorPipeline()
        assert pipeline.reading_count == 0
        pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, []))
        pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, []))
        assert pipeline.reading_count == 2

    def test_repr(self):
        pipeline = SensorPipeline()
        pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, []))
        r = repr(pipeline)
        assert "SensorPipeline" in r
        assert "sensors=" in r
        assert "readings=" in r

    def test_feed_updates_latest_reading(self):
        pipeline = SensorPipeline()
        pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, []))
        pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, [{"x": 5.0, "y": 0.0}]))
        # Latest should be the second reading
        latest = pipeline.latest_readings["lidar0"]
        assert latest.data == [{"x": 5.0, "y": 0.0}]

    def test_pipeline_init_with_detector_backend(self):
        pipeline = SensorPipeline(detector_backend="mock")
        assert isinstance(pipeline.detector, MockDetector)

    def test_pipeline_init_with_none_backend(self):
        pipeline = SensorPipeline(detector_backend="none")
        assert pipeline.detector is None

    def test_pipeline_init_with_target_labels(self):
        pipeline = SensorPipeline(detector_backend="none", target_labels=["box"])
        assert pipeline._target_labels == ["box"]
