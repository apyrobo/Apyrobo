"""
Sensor Pipeline — ingests sensor data and builds a structured world state.

Subscribes to ROS 2 sensor topics (or mock data), normalises readings
into a unified WorldState object that AI agents can reason about.

The WorldState is the agent's view of the physical world — obstacles,
objects, robot pose, and environment metadata.
"""

from __future__ import annotations

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
        self.robot_orientation: float = 0.0  # radians
        self.obstacles: list[Obstacle] = []
        self.detected_objects: list[DetectedObject] = []
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
        """Find the first detected object with a given label."""
        for obj in self.detected_objects:
            if obj.label.lower() == label.lower():
                return obj
        return None

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
            "robot_orientation": self.robot_orientation,
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

class SensorPipeline:
    """
    Ingests raw sensor data and builds a WorldState.

    In production, subscribes to ROS 2 topics. In mock mode, accepts
    readings programmatically.

    Usage:
        pipeline = SensorPipeline()
        pipeline.feed(SensorReading("lidar0", SensorType.LIDAR, scan_data))
        pipeline.feed(SensorReading("cam0", SensorType.CAMERA, detections))
        world = pipeline.get_world_state()
    """

    def __init__(self) -> None:
        self._readings: dict[str, SensorReading] = {}  # latest per sensor
        self._all_readings: list[SensorReading] = []
        self._world: WorldState = WorldState()
        self._processors: dict[SensorType, Any] = {
            SensorType.LIDAR: self._process_lidar,
            SensorType.CAMERA: self._process_camera,
            SensorType.IMU: self._process_imu,
        }

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
        logger.debug("Lidar: detected %d obstacles", len(obstacles))

    def _process_camera(self, reading: SensorReading) -> None:
        """
        Process camera data into detected objects.

        Expected data format: list of {"id": str, "label": str, "x": float, "y": float}
        """
        if not isinstance(reading.data, list):
            return

        objects = []
        for det in reading.data:
            if isinstance(det, dict) and "label" in det:
                objects.append(DetectedObject(
                    object_id=det.get("id", f"obj_{len(objects)}"),
                    label=det["label"],
                    x=det.get("x", 0.0),
                    y=det.get("y", 0.0),
                    confidence=det.get("confidence", 1.0),
                    source=reading.sensor_id,
                ))

        self._world.detected_objects = objects
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
