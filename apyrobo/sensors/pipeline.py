"""
Sensor Pipeline — ingests sensor data and builds a structured world state.

Subscribes to ROS 2 sensor topics (or mock data), normalises readings
into a unified WorldState object that AI agents can reason about.

The WorldState is the agent's view of the physical world — obstacles,
objects, robot pose, and environment metadata.
"""

from __future__ import annotations

import abc
import logging
import math
import time
from typing import Any

from apyrobo.core.schemas import SensorType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sensor readings
# ---------------------------------------------------------------------------

class SensorReading:
    """A single reading from a sensor at a point in time."""

    def __init__(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        data: Any,
        timestamp: float | None = None,
    ) -> None:
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.data = data
        self.timestamp = timestamp or time.time()

    def __repr__(self) -> str:
        return f"<SensorReading {self.sensor_id} ({self.sensor_type.value}) t={self.timestamp:.1f}>"


# ---------------------------------------------------------------------------
# Obstacles and objects
# ---------------------------------------------------------------------------

class Obstacle:
    """A detected obstacle in the environment."""

    def __init__(self, x: float, y: float, radius: float = 0.3,
                 confidence: float = 1.0, source: str = "lidar") -> None:
        self.x = x
        self.y = y
        self.radius = radius
        self.confidence = confidence
        self.source = source
        self.timestamp = time.time()

    def distance_to(self, x: float, y: float) -> float:
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def __repr__(self) -> str:
        return f"<Obstacle ({self.x:.1f}, {self.y:.1f}) r={self.radius:.1f}>"


class DetectedObject:
    """An object detected by vision or other sensors."""

    def __init__(self, object_id: str, label: str, x: float, y: float,
                 confidence: float = 1.0, source: str = "camera") -> None:
        self.object_id = object_id
        self.label = label
        self.x = x
        self.y = y
        self.confidence = confidence
        self.source = source
        self.timestamp = time.time()

    def __repr__(self) -> str:
        return f"<Object {self.label!r} at ({self.x:.1f}, {self.y:.1f})>"


# ---------------------------------------------------------------------------
# World State
# ---------------------------------------------------------------------------

class WorldState:
    """
    The agent's structured view of the physical world.

    Built by fusing data from multiple sensors into a single queryable
    object that AI agents use for planning decisions.
    """

    def __init__(self) -> None:
        self.robot_position: tuple[float, float] = (0.0, 0.0)
        self.gps_position: tuple[float, float] | None = None
        self.robot_orientation: float = 0.0  # radians
        self.obstacles: list[Obstacle] = []
        self.detected_objects: list[DetectedObject] = []
        self.obstacle_point_cloud: list[dict[str, float]] = []
        self.scene_graph: list[dict[str, Any]] = []
        self.timestamp: float = time.time()
        self.metadata: dict[str, Any] = {}

    def nearest_obstacle(self, x: float | None = None,
                         y: float | None = None) -> Obstacle | None:
        """Find the obstacle closest to a point (defaults to robot position)."""
        if not self.obstacles:
            return None
        px = x if x is not None else self.robot_position[0]
        py = y if y is not None else self.robot_position[1]
        return min(self.obstacles, key=lambda o: o.distance_to(px, py))

    def obstacles_within(self, radius: float, x: float | None = None,
                         y: float | None = None) -> list[Obstacle]:
        """Find all obstacles within a radius of a point."""
        px = x if x is not None else self.robot_position[0]
        py = y if y is not None else self.robot_position[1]
        return [o for o in self.obstacles if o.distance_to(px, py) <= radius]

    def find_object(self, label: str) -> DetectedObject | None:
        """Semantic object search using exact/substring/token matches."""
        label_l = label.strip().lower()
        if not label_l:
            return None

        # Prefer highest confidence exact match
        exact = [o for o in self.detected_objects if o.label.lower() == label_l]
        if exact:
            return max(exact, key=lambda o: o.confidence)

        tokens = set(label_l.split())
        partial: list[DetectedObject] = []
        for obj in self.detected_objects:
            obj_label = obj.label.lower()
            if label_l in obj_label or obj_label in label_l:
                partial.append(obj)
                continue
            if tokens and tokens.intersection(set(obj_label.split())):
                partial.append(obj)

        if partial:
            return max(partial, key=lambda o: o.confidence)
        return None

    def expire_stale_obstacles(self, max_age_s: float) -> int:
        """Remove obstacle detections older than *max_age_s* seconds."""
        now = time.time()
        before = len(self.obstacles)
        self.obstacles = [o for o in self.obstacles if (now - o.timestamp) <= max_age_s]
        return before - len(self.obstacles)

    def build_scene_graph(self, near_threshold: float = 1.0) -> list[dict[str, Any]]:
        """Build simple spatial relationships between detected objects."""
        edges: list[dict[str, Any]] = []
        for i, a in enumerate(self.detected_objects):
            for b in self.detected_objects[i + 1:]:
                d = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
                if d <= near_threshold:
                    edges.append({
                        "subject": a.object_id,
                        "relation": "next_to",
                        "object": b.object_id,
                        "distance": d,
                    })
                    edges.append({
                        "subject": b.object_id,
                        "relation": "next_to",
                        "object": a.object_id,
                        "distance": d,
                    })
        self.scene_graph = edges
        return edges

    def is_path_clear(self, from_x: float, from_y: float,
                      to_x: float, to_y: float, clearance: float = 0.5) -> bool:
        """
        Check if a straight-line path is clear of obstacles.

        Uses a simple point-to-line-segment distance check.
        """
        for obstacle in self.obstacles:
            dist = self._point_to_segment_distance(
                obstacle.x, obstacle.y, from_x, from_y, to_x, to_y
            )
            if dist < clearance + obstacle.radius:
                return False
        return True

    @staticmethod
    def _point_to_segment_distance(
        px: float, py: float,
        ax: float, ay: float,
        bx: float, by: float,
    ) -> float:
        """Distance from point (px, py) to line segment (ax,ay)-(bx,by)."""
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
        t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / length_sq))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a dict for agent consumption."""
        return {
            "robot_position": list(self.robot_position),
            "gps_position": list(self.gps_position) if self.gps_position else None,
            "robot_orientation": self.robot_orientation,
            "obstacle_point_cloud": list(self.obstacle_point_cloud),
            "scene_graph": list(self.scene_graph),
            "obstacles": [
                {"x": o.x, "y": o.y, "radius": o.radius, "confidence": o.confidence}
                for o in self.obstacles
            ],
            "detected_objects": [
                {"id": o.object_id, "label": o.label, "x": o.x, "y": o.y,
                 "confidence": o.confidence}
                for o in self.detected_objects
            ],
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"<WorldState pos=({self.robot_position[0]:.1f}, {self.robot_position[1]:.1f}) "
            f"obstacles={len(self.obstacles)} objects={len(self.detected_objects)}>"
        )


# ---------------------------------------------------------------------------
# Sensor Pipeline
# ---------------------------------------------------------------------------

class ObjectDetector(abc.ABC):
    """Abstract interface for object detection backends (VC-03)."""

    @abc.abstractmethod
    def detect(self, image_data: Any, target_labels: list[str] | None = None) -> list[DetectedObject]:
        """Run detection on image data, returning DetectedObject instances."""
        ...


class MockDetector(ObjectDetector):
    """
    VC-03: Mock detector that returns configurable fake detections for testing.

    Usage:
        detector = MockDetector([
            {"label": "box", "x": 1.0, "y": 2.0, "confidence": 0.95},
        ])
    """

    def __init__(self, detections: list[dict[str, Any]] | None = None) -> None:
        self._detections = detections or []

    def detect(self, image_data: Any, target_labels: list[str] | None = None) -> list[DetectedObject]:
        objects = []
        for i, det in enumerate(self._detections):
            label = det.get("label", "object")
            if target_labels and label not in target_labels:
                continue
            objects.append(DetectedObject(
                object_id=det.get("id", f"mock_{i}"),
                label=label,
                x=det.get("x", 0.0),
                y=det.get("y", 0.0),
                confidence=det.get("confidence", 1.0),
                source="mock_detector",
            ))
        return objects

    def set_detections(self, detections: list[dict[str, Any]]) -> None:
        """Update the fake detections returned by detect()."""
        self._detections = detections


class YOLOv8Detector(ObjectDetector):
    """
    VC-03: YOLOv8 backend using ultralytics.

    Requires: pip install ultralytics
    """

    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        self._model_name = model_name
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self._model_name)
        return self._model

    def detect(self, image_data: Any, target_labels: list[str] | None = None) -> list[DetectedObject]:
        model = self._load_model()
        results = model(image_data, verbose=False)
        objects = []
        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls[0])]
                if target_labels and label not in target_labels:
                    continue
                # Use center of bounding box as position
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                conf = float(box.conf[0])
                objects.append(DetectedObject(
                    object_id=f"yolo_{len(objects)}",
                    label=label,
                    x=cx,
                    y=cy,
                    confidence=conf,
                    source="yolov8",
                ))
        return objects


class NanoOWLDetector(ObjectDetector):
    """
    VC-03: NanoOWL open-vocabulary detection from text prompts.

    Requires: nanoowl package (NVIDIA Jetson optimized)
    """

    def __init__(self, model_name: str = "google/owlvit-base-patch32") -> None:
        self._model_name = model_name
        self._predictor: Any = None

    def _load_model(self) -> Any:
        if self._predictor is None:
            from nanoowl.owl_predictor import OwlPredictor
            self._predictor = OwlPredictor(self._model_name)
        return self._predictor

    def detect(self, image_data: Any, target_labels: list[str] | None = None) -> list[DetectedObject]:
        if not target_labels:
            return []
        predictor = self._load_model()
        output = predictor.predict(
            image=image_data,
            text=target_labels,
            threshold=0.1,
        )
        objects = []
        for i, (box, label_idx, score) in enumerate(
            zip(output.boxes, output.labels, output.scores)
        ):
            x1, y1, x2, y2 = box.tolist()
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            objects.append(DetectedObject(
                object_id=f"owl_{i}",
                label=target_labels[label_idx],
                x=cx,
                y=cy,
                confidence=float(score),
                source="nanoowl",
            ))
        return objects


def _load_detector(backend: str) -> ObjectDetector | None:
    """Factory function to create a detector by backend name."""
    if backend == "none":
        return None
    if backend == "mock":
        return MockDetector()
    if backend == "yolov8":
        return YOLOv8Detector()
    if backend == "nanoowl":
        return NanoOWLDetector()
    raise ValueError(f"Unknown detector backend: {backend!r}. "
                     f"Available: none, mock, yolov8, nanoowl")


class SensorPipeline:
    """
    Ingests raw sensor data and builds a WorldState.

    In production, subscribes to ROS 2 topics. In mock mode, accepts
    readings programmatically.

    VC-03: Supports pluggable object detection backends via detector_backend.

    Usage:
        pipeline = SensorPipeline()
        pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, scan_data))
        pipeline.feed(SensorReading("cam0", SensorType.CAMERA, detections))
        world = pipeline.get_world_state()

        # With object detection:
        pipeline = SensorPipeline(detector_backend="yolov8")
        pipeline = SensorPipeline(detector_backend="mock", target_labels=["box"])
    """

    def __init__(self, detector_backend: str = "none",
                 target_labels: list[str] | None = None) -> None:
        self._readings: dict[str, SensorReading] = {}  # latest per sensor
        self._all_readings: list[SensorReading] = []
        self._world: WorldState = WorldState()
        self._processors: dict[SensorType, Any] = {
            SensorType.LIDAR: self._process_lidar,
            SensorType.CAMERA: self._process_camera,
            SensorType.IMU: self._process_imu,
            SensorType.GPS: self._process_gps,
            SensorType.FORCE_TORQUE: self._process_force_torque,
            SensorType.DEPTH: self._process_depth,
        }
        self._obstacle_max_age_s: float | None = None
        self._detector: ObjectDetector | None = _load_detector(detector_backend)
        self._target_labels: list[str] = target_labels or []

    def set_obstacle_max_age(self, max_age_s: float | None) -> None:
        """Configure automatic stale obstacle expiration."""
        self._obstacle_max_age_s = max_age_s

    def set_target_labels(self, labels: list[str]) -> None:
        """VC-03: Set target labels for open-vocab detectors (e.g. NanoOWL)."""
        self._target_labels = labels

    @property
    def detector(self) -> ObjectDetector | None:
        """VC-03: The attached object detector, or None."""
        return self._detector

    def set_detector(self, detector: ObjectDetector | None) -> None:
        """VC-03: Attach or replace the object detection backend."""
        self._detector = detector

    def feed(self, reading: SensorReading) -> None:
        """Ingest a sensor reading and update the world state."""
        self._readings[reading.sensor_id] = reading
        self._all_readings.append(reading)

        processor = self._processors.get(reading.sensor_type)
        if processor:
            processor(reading)
        else:
            logger.debug("No processor for sensor type %s", reading.sensor_type)

    def get_world_state(self) -> WorldState:
        """Return the current fused world state."""
        if self._obstacle_max_age_s is not None:
            self._world.expire_stale_obstacles(self._obstacle_max_age_s)
        self._world.build_scene_graph()
        self._world.timestamp = time.time()
        return self._world

    # ------------------------------------------------------------------
    # Sensor-specific processors
    # ------------------------------------------------------------------

    def _process_lidar(self, reading: SensorReading) -> None:
        """
        Process lidar data into obstacles.

        Expected data format: list of {"x": float, "y": float, "distance": float}
        or list of {"angle": float, "distance": float} for polar data.
        """
        if not isinstance(reading.data, list):
            return

        obstacles = []
        for point in reading.data:
            if isinstance(point, dict):
                if "x" in point and "y" in point:
                    obstacles.append(Obstacle(
                        x=point["x"], y=point["y"],
                        radius=point.get("radius", 0.2),
                        source=reading.sensor_id,
                    ))
                elif "angle" in point and "distance" in point:
                    # Convert polar to cartesian
                    angle = point["angle"]
                    dist = point["distance"]
                    x = self._world.robot_position[0] + dist * math.cos(angle)
                    y = self._world.robot_position[1] + dist * math.sin(angle)
                    obstacles.append(Obstacle(
                        x=x, y=y, radius=0.2, source=reading.sensor_id,
                    ))

        self._world.obstacles = obstacles
        self._fuse_obstacles()
        logger.debug("Lidar: detected %d obstacles", len(obstacles))

    def _process_camera(self, reading: SensorReading) -> None:
        """
        Process camera data into detected objects.

        VC-03: When a detector backend is attached, runs detection on image data.
        Otherwise, accepts pre-annotated detections as before.

        Expected data format (no detector):
            list of {"id": str, "label": str, "x": float, "y": float}
        Expected data format (with detector):
            raw image bytes or numpy array
        """
        if self._detector:
            # VC-03: Run object detection on image data
            objects = self._detector.detect(reading.data, self._target_labels or None)
            self._world.detected_objects = objects
            self._fuse_obstacles()
            logger.debug("Camera (detector): detected %d objects", len(objects))
            return

        # Existing behavior: accept pre-annotated detections
        if not isinstance(reading.data, list):
            return

        objects = []
        for det in reading.data:
            if isinstance(det, dict) and "label" in det:
                objects.append(DetectedObject(
                    object_id=det.get("id", f"obj_{len(objects)}"),
                    label=det["label"],
                    x=det.get("x", det.get("center_x", 0.0)),
                    y=det.get("y", det.get("center_y", 0.0)),
                    confidence=det.get("confidence", 1.0),
                    source=reading.sensor_id,
                ))

        self._world.detected_objects = objects
        self._fuse_obstacles()
        logger.debug("Camera: detected %d objects", len(objects))

    def _process_imu(self, reading: SensorReading) -> None:
        """
        Process IMU data into robot pose.

        Expected data format: {"x": float, "y": float, "yaw": float}
        """
        if isinstance(reading.data, dict):
            if "x" in reading.data and "y" in reading.data:
                self._world.robot_position = (reading.data["x"], reading.data["y"])
            if "yaw" in reading.data:
                self._world.robot_orientation = reading.data["yaw"]

    def _process_gps(self, reading: SensorReading) -> None:
        """Process GPS data: {'lat': float, 'lon': float} or {'x':, 'y':} local."""
        if not isinstance(reading.data, dict):
            return
        if "x" in reading.data and "y" in reading.data:
            self._world.gps_position = (float(reading.data["x"]), float(reading.data["y"]))
        elif "lat" in reading.data and "lon" in reading.data:
            self._world.gps_position = (float(reading.data["lat"]), float(reading.data["lon"]))
        self._world.metadata["gps_fix"] = bool(reading.data.get("fix", True))

    def _process_force_torque(self, reading: SensorReading) -> None:
        """Process force-torque readings into contact/grasp indicators."""
        if not isinstance(reading.data, dict):
            return
        fx = float(reading.data.get("fx", 0.0))
        fy = float(reading.data.get("fy", 0.0))
        fz = float(reading.data.get("fz", 0.0))
        magnitude = math.sqrt(fx * fx + fy * fy + fz * fz)
        contact_threshold = float(reading.data.get("contact_threshold", 3.0))
        grasp_threshold = float(reading.data.get("grasp_threshold", 8.0))
        self._world.metadata["force_torque"] = {
            "fx": fx, "fy": fy, "fz": fz, "magnitude": magnitude,
        }
        self._world.metadata["contact_detected"] = magnitude >= contact_threshold
        self._world.metadata["grasp_success"] = magnitude >= grasp_threshold

    def _process_depth(self, reading: SensorReading) -> None:
        """Process depth image points into 3D point cloud and 2D obstacles."""
        if not isinstance(reading.data, list):
            return
        point_cloud: list[dict[str, float]] = []
        projected: list[Obstacle] = []
        for point in reading.data:
            if not isinstance(point, dict):
                continue
            x = float(point.get("x", 0.0))
            y = float(point.get("y", 0.0))
            z = float(point.get("z", 0.0))
            point_cloud.append({"x": x, "y": y, "z": z})
            projected.append(Obstacle(
                x=x,
                y=y,
                radius=float(point.get("radius", 0.15)),
                confidence=float(point.get("confidence", 0.7)),
                source=reading.sensor_id,
            ))
        self._world.obstacle_point_cloud = point_cloud
        # Keep depth-projected obstacles alongside existing observations.
        self._world.obstacles.extend(projected)
        self._fuse_obstacles()

    def _fuse_obstacles(self, merge_distance: float = 0.35) -> None:
        """SN-08: merge nearby obstacle observations with confidence weighting."""
        fused: list[Obstacle] = []
        for obstacle in self._world.obstacles:
            match = None
            for existing in fused:
                if existing.distance_to(obstacle.x, obstacle.y) <= merge_distance:
                    match = existing
                    break
            if match is None:
                fused.append(obstacle)
                continue

            total_conf = max(0.01, match.confidence + obstacle.confidence)
            match.x = ((match.x * match.confidence) + (obstacle.x * obstacle.confidence)) / total_conf
            match.y = ((match.y * match.confidence) + (obstacle.y * obstacle.confidence)) / total_conf
            match.radius = max(match.radius, obstacle.radius)
            match.confidence = min(1.0, total_conf / 2.0)
            match.timestamp = max(match.timestamp, obstacle.timestamp)
        self._world.obstacles = fused

    # SN-10 helpers
    def inject_mock_reading(self, sensor_type: SensorType, data: Any,
                            sensor_id: str = "mock_sensor") -> None:
        """Convenience API for tests to inject synthetic sensor readings."""
        self.feed(SensorReading(sensor_id=sensor_id, sensor_type=sensor_type, data=data))

    def register_processor(self, sensor_type: SensorType, processor: Any) -> None:
        """Register/override a sensor processor for a sensor type."""
        self._processors[sensor_type] = processor

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def latest_readings(self) -> dict[str, SensorReading]:
        return dict(self._readings)

    @property
    def reading_count(self) -> int:
        return len(self._all_readings)

    def __repr__(self) -> str:
        return (
            f"<SensorPipeline sensors={len(self._readings)} "
            f"readings={len(self._all_readings)}>"
        )
