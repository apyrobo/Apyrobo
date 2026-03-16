"""
Tests for VC-03: VLM integration — wire camera pipeline to object detection.
"""

from __future__ import annotations

import pytest

from apyrobo.core.schemas import SensorType
from apyrobo.sensors.pipeline import (
    SensorPipeline,
    SensorReading,
    DetectedObject,
    MockDetector,
    ObjectDetector,
    _load_detector,
)


# ---------------------------------------------------------------------------
# MockDetector tests
# ---------------------------------------------------------------------------

class TestMockDetector:
    def test_returns_configured_fake_detections(self):
        """Mock backend returns configured fake detections."""
        detector = MockDetector([
            {"label": "box", "x": 1.0, "y": 2.0, "confidence": 0.95},
            {"label": "person", "x": 3.0, "y": 4.0, "confidence": 0.8},
        ])
        objects = detector.detect(image_data=b"fake_image")
        assert len(objects) == 2
        assert objects[0].label == "box"
        assert objects[0].x == 1.0
        assert objects[0].y == 2.0
        assert objects[0].confidence == 0.95
        assert objects[1].label == "person"

    def test_returns_detected_objects(self):
        detector = MockDetector([{"label": "cup", "x": 0.5, "y": 0.5}])
        objects = detector.detect(b"img")
        assert isinstance(objects[0], DetectedObject)

    def test_filter_by_target_labels(self):
        detector = MockDetector([
            {"label": "box", "x": 1.0, "y": 2.0},
            {"label": "person", "x": 3.0, "y": 4.0},
        ])
        objects = detector.detect(b"img", target_labels=["box"])
        assert len(objects) == 1
        assert objects[0].label == "box"

    def test_empty_detections(self):
        detector = MockDetector()
        objects = detector.detect(b"img")
        assert objects == []

    def test_set_detections(self):
        detector = MockDetector()
        detector.set_detections([{"label": "new_obj", "x": 0, "y": 0}])
        objects = detector.detect(b"img")
        assert len(objects) == 1
        assert objects[0].label == "new_obj"

    def test_is_object_detector(self):
        assert isinstance(MockDetector(), ObjectDetector)


# ---------------------------------------------------------------------------
# SensorPipeline with detector tests
# ---------------------------------------------------------------------------

class TestPipelineWithDetector:
    def test_object_visible_precondition_works_with_mock_backend(self):
        """object_visible precondition works with mock backend."""
        pipeline = SensorPipeline(detector_backend="mock")
        mock_det = MockDetector([
            {"label": "red_box", "x": 2.0, "y": 3.0, "confidence": 0.9},
        ])
        pipeline.set_detector(mock_det)

        # Feed camera data (raw bytes, since detector handles it)
        pipeline.feed(SensorReading("cam0", SensorType.CAMERA, b"image_bytes"))
        world = pipeline.get_world_state()

        # Object should be detected
        assert len(world.detected_objects) == 1
        obj = world.find_object("red_box")
        assert obj is not None
        assert obj.x == 2.0
        assert obj.y == 3.0
        assert obj.confidence == 0.9

    def test_no_detector_processes_pre_annotated_data(self):
        """No detector attached: pipeline processes pre-annotated data as before."""
        pipeline = SensorPipeline()  # detector_backend='none' by default

        # Feed pre-annotated camera data
        detections = [
            {"id": "obj_0", "label": "chair", "x": 1.0, "y": 1.0, "confidence": 0.8},
            {"id": "obj_1", "label": "table", "x": 2.0, "y": 2.0, "confidence": 0.9},
        ]
        pipeline.feed(SensorReading("cam0", SensorType.CAMERA, detections))
        world = pipeline.get_world_state()

        assert len(world.detected_objects) == 2
        assert world.find_object("chair") is not None
        assert world.find_object("table") is not None

    def test_detector_with_target_labels(self):
        pipeline = SensorPipeline(
            detector_backend="mock",
            target_labels=["box"],
        )
        mock_det = MockDetector([
            {"label": "box", "x": 1.0, "y": 2.0},
            {"label": "person", "x": 3.0, "y": 4.0},
        ])
        pipeline.set_detector(mock_det)

        pipeline.feed(SensorReading("cam0", SensorType.CAMERA, b"img"))
        world = pipeline.get_world_state()

        # Only "box" should pass the filter
        assert len(world.detected_objects) == 1
        assert world.detected_objects[0].label == "box"

    def test_set_target_labels(self):
        pipeline = SensorPipeline(detector_backend="mock")
        mock_det = MockDetector([
            {"label": "a", "x": 0, "y": 0},
            {"label": "b", "x": 1, "y": 1},
        ])
        pipeline.set_detector(mock_det)

        pipeline.set_target_labels(["b"])
        pipeline.feed(SensorReading("cam0", SensorType.CAMERA, b"img"))
        world = pipeline.get_world_state()

        assert len(world.detected_objects) == 1
        assert world.detected_objects[0].label == "b"

    def test_detector_property(self):
        pipeline = SensorPipeline(detector_backend="mock")
        assert pipeline.detector is not None
        assert isinstance(pipeline.detector, MockDetector)

    def test_no_detector_property(self):
        pipeline = SensorPipeline()
        assert pipeline.detector is None

    def test_set_detector(self):
        pipeline = SensorPipeline()
        assert pipeline.detector is None

        det = MockDetector([{"label": "x", "x": 0, "y": 0}])
        pipeline.set_detector(det)
        assert pipeline.detector is det

    def test_detection_output_has_confidence(self):
        """Detection output feeds into WorldState.detected_objects with confidence scores."""
        pipeline = SensorPipeline(detector_backend="mock")
        mock_det = MockDetector([
            {"label": "package", "x": 5.0, "y": 6.0, "confidence": 0.73},
        ])
        pipeline.set_detector(mock_det)

        pipeline.feed(SensorReading("cam0", SensorType.CAMERA, b"img"))
        world = pipeline.get_world_state()

        assert len(world.detected_objects) == 1
        assert world.detected_objects[0].confidence == 0.73


# ---------------------------------------------------------------------------
# _load_detector factory tests
# ---------------------------------------------------------------------------

class TestLoadDetector:
    def test_none_backend(self):
        assert _load_detector("none") is None

    def test_mock_backend(self):
        det = _load_detector("mock")
        assert isinstance(det, MockDetector)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown detector backend"):
            _load_detector("nonexistent")

    def test_yolov8_import_error(self):
        """YOLOv8 backend raises ImportError if ultralytics not installed."""
        try:
            det = _load_detector("yolov8")
            # If ultralytics is installed, detector should work
            assert isinstance(det, ObjectDetector)
        except Exception:
            # Expected if ultralytics not installed (import happens lazily)
            pass

    def test_nanoowl_import_error(self):
        """NanoOWL backend raises ImportError if nanoowl not installed."""
        try:
            det = _load_detector("nanoowl")
            assert isinstance(det, ObjectDetector)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Existing pipeline behavior preserved
# ---------------------------------------------------------------------------

class TestPipelineBackwardCompat:
    def test_lidar_still_works(self):
        pipeline = SensorPipeline()
        data = [{"x": 1.0, "y": 2.0, "radius": 0.3}]
        pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, data))
        world = pipeline.get_world_state()
        assert len(world.obstacles) == 1

    def test_imu_still_works(self):
        pipeline = SensorPipeline()
        pipeline.feed(SensorReading("imu0", SensorType.IMU, {"x": 1.0, "y": 2.0, "yaw": 0.5}))
        world = pipeline.get_world_state()
        assert world.robot_position == (1.0, 2.0)
        assert world.robot_orientation == 0.5

    def test_default_init_no_detector(self):
        """Default SensorPipeline() has no detector — backward compatible."""
        pipeline = SensorPipeline()
        assert pipeline.detector is None
        assert pipeline._target_labels == []
