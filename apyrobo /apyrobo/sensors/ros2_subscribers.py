"""
ROS 2 Sensor Subscribers — bridges real ROS 2 sensor topics into APYROBO.

Subscribes to standard ROS 2 sensor message types:
    - sensor_msgs/LaserScan → lidar obstacle points
    - sensor_msgs/Image → camera detections (via pluggable detector)
    - sensor_msgs/Imu → robot pose/orientation

Feeds normalised data into the SensorPipeline, which builds the WorldState.

Usage:
    pipeline = SensorPipeline()
    ros_sensors = ROS2SensorBridge(node, pipeline, config={
        "scan": "/scan",
        "camera": "/oakd/rgb/preview/image_raw",
        "imu": "/imu",
    })
    # Pipeline now receives real sensor data via ROS 2 callbacks
    world = pipeline.get_world_state()

Requires rclpy + sensor_msgs (available inside Docker).
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from apyrobo.core.schemas import SensorType
from apyrobo.sensors.pipeline import SensorPipeline, SensorReading

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ROS 2 imports
# ---------------------------------------------------------------------------

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import LaserScan, Image, Imu
    from nav_msgs.msg import Odometry

    _HAS_ROS2 = True
except ImportError:
    _HAS_ROS2 = False
    logger.debug("rclpy not available — ROS 2 sensor bridge disabled")


def _sensor_qos() -> Any:
    """QoS for sensor topics — best-effort, volatile, small queue."""
    if not _HAS_ROS2:
        return None
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        depth=5,
    )


# ---------------------------------------------------------------------------
# LaserScan processor
# ---------------------------------------------------------------------------

class LaserScanProcessor:
    """
    Converts sensor_msgs/LaserScan into obstacle points.

    LaserScan is a 1D array of distances at evenly spaced angles.
    We convert each valid range to a (x, y) point in the robot's frame,
    then filter to only keep points within a useful range.
    """

    def __init__(self, max_range: float = 10.0, min_range: float = 0.1,
                 subsample: int = 4) -> None:
        self.max_range = max_range
        self.min_range = min_range
        self.subsample = subsample  # take every Nth point to reduce volume

    def process(self, msg: Any, robot_x: float = 0.0,
                robot_y: float = 0.0, robot_yaw: float = 0.0) -> list[dict[str, float]]:
        """
        Convert a LaserScan message to a list of obstacle points.

        Returns: [{"x": float, "y": float, "distance": float}, ...]
        Points are in the map frame (transformed by robot pose).
        """
        obstacles = []
        angle = msg.angle_min

        for i, distance in enumerate(msg.ranges):
            if i % self.subsample != 0:
                angle += msg.angle_increment
                continue

            if (math.isnan(distance) or math.isinf(distance)
                    or distance < self.min_range or distance > self.max_range):
                angle += msg.angle_increment
                continue

            # Polar to cartesian in robot frame
            local_x = distance * math.cos(angle)
            local_y = distance * math.sin(angle)

            # Transform to map frame
            cos_yaw = math.cos(robot_yaw)
            sin_yaw = math.sin(robot_yaw)
            map_x = robot_x + local_x * cos_yaw - local_y * sin_yaw
            map_y = robot_y + local_x * sin_yaw + local_y * cos_yaw

            obstacles.append({
                "x": map_x,
                "y": map_y,
                "distance": distance,
                "radius": 0.15,  # approximate obstacle size
            })

            angle += msg.angle_increment

        return obstacles


# ---------------------------------------------------------------------------
# IMU processor
# ---------------------------------------------------------------------------

class ImuProcessor:
    """
    Converts sensor_msgs/Imu into robot orientation data.

    Extracts yaw from the orientation quaternion and linear acceleration
    for basic motion detection.
    """

    @staticmethod
    def process(msg: Any) -> dict[str, float]:
        """Convert an Imu message to a pose dict."""
        q = msg.orientation
        # Quaternion to yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return {
            "yaw": yaw,
            "angular_velocity_z": msg.angular_velocity.z,
            "linear_accel_x": msg.linear_acceleration.x,
            "linear_accel_y": msg.linear_acceleration.y,
        }


# ---------------------------------------------------------------------------
# Camera processor (stub — real impl requires CV pipeline)
# ---------------------------------------------------------------------------

class CameraProcessor:
    """
    Processes sensor_msgs/Image into detected objects.

    This is a stub — real implementation would use:
    - OpenCV for basic detection
    - A VLM (e.g. via the agent's LLM provider) for object identification
    - NVIDIA Isaac Perceptor for GPU-accelerated perception

    For the MVP demo, we pass the raw image dimensions so the pipeline
    knows a camera is active, and rely on a separate detection node.
    """

    @staticmethod
    def process(msg: Any) -> list[dict[str, Any]]:
        """Convert an Image message to basic image metadata."""
        return [{
            "id": "camera_frame",
            "label": "__raw_image__",
            "width": msg.width,
            "height": msg.height,
            "encoding": msg.encoding,
            "confidence": 0.0,  # no detection, just metadata
        }]


# ---------------------------------------------------------------------------
# ROS 2 Sensor Bridge
# ---------------------------------------------------------------------------

if _HAS_ROS2:

    class ROS2SensorBridge:
        """
        Subscribes to ROS 2 sensor topics and feeds data into the SensorPipeline.

        Creates one subscriber per configured sensor. Each callback converts
        the ROS message type into APYROBO's normalised SensorReading format
        and feeds it to the pipeline.
        """

        DEFAULT_CONFIG = {
            "scan": "/scan",
            "camera": "/oakd/rgb/preview/image_raw",
            "imu": "/imu",
            "odom": "/odom",
        }

        def __init__(
            self,
            node: Node,
            pipeline: SensorPipeline,
            config: dict[str, str] | None = None,
            lidar_subsample: int = 4,
            lidar_max_range: float = 10.0,
        ) -> None:
            self._node = node
            self._pipeline = pipeline
            self._config = dict(self.DEFAULT_CONFIG)
            if config:
                self._config.update(config)

            # Processors
            self._lidar_proc = LaserScanProcessor(
                max_range=lidar_max_range, subsample=lidar_subsample,
            )
            self._imu_proc = ImuProcessor()
            self._camera_proc = CameraProcessor()

            # Robot pose (updated from odom for lidar frame transform)
            self._robot_x = 0.0
            self._robot_y = 0.0
            self._robot_yaw = 0.0

            # Subscriber count
            self._sub_count = 0
            qos = _sensor_qos()

            # --- Odometry (for frame transforms) ---
            if self._config.get("odom"):
                self._node.create_subscription(
                    Odometry, self._config["odom"],
                    self._odom_callback, qos,
                )
                self._sub_count += 1
                logger.info("Sensor bridge: subscribed to odom at %s", self._config["odom"])

            # --- LaserScan ---
            if self._config.get("scan"):
                self._node.create_subscription(
                    LaserScan, self._config["scan"],
                    self._scan_callback, qos,
                )
                self._sub_count += 1
                logger.info("Sensor bridge: subscribed to lidar at %s", self._config["scan"])

            # --- Camera ---
            if self._config.get("camera"):
                self._node.create_subscription(
                    Image, self._config["camera"],
                    self._camera_callback, qos,
                )
                self._sub_count += 1
                logger.info("Sensor bridge: subscribed to camera at %s", self._config["camera"])

            # --- IMU ---
            if self._config.get("imu"):
                self._node.create_subscription(
                    Imu, self._config["imu"],
                    self._imu_callback, qos,
                )
                self._sub_count += 1
                logger.info("Sensor bridge: subscribed to IMU at %s", self._config["imu"])

            logger.info("ROS2SensorBridge ready: %d subscribers", self._sub_count)

        # ------------------------------------------------------------------
        # Callbacks
        # ------------------------------------------------------------------

        def _odom_callback(self, msg: Odometry) -> None:
            """Track robot pose for lidar frame transforms."""
            pos = msg.pose.pose.position
            self._robot_x = pos.x
            self._robot_y = pos.y
            q = msg.pose.pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self._robot_yaw = math.atan2(siny, cosy)

            # Also feed IMU-like pose data to pipeline
            self._pipeline.feed(SensorReading(
                sensor_id="odom",
                sensor_type=SensorType.IMU,
                data={"x": self._robot_x, "y": self._robot_y, "yaw": self._robot_yaw},
            ))

        def _scan_callback(self, msg: LaserScan) -> None:
            """Convert LaserScan to obstacles and feed to pipeline."""
            obstacles = self._lidar_proc.process(
                msg,
                robot_x=self._robot_x,
                robot_y=self._robot_y,
                robot_yaw=self._robot_yaw,
            )
            self._pipeline.feed(SensorReading(
                sensor_id="lidar0",
                sensor_type=SensorType.LIDAR,
                data=obstacles,
            ))

        def _camera_callback(self, msg: Image) -> None:
            """Convert Image to detections and feed to pipeline."""
            detections = self._camera_proc.process(msg)
            self._pipeline.feed(SensorReading(
                sensor_id="camera0",
                sensor_type=SensorType.CAMERA,
                data=detections,
            ))

        def _imu_callback(self, msg: Imu) -> None:
            """Convert IMU data and feed to pipeline."""
            imu_data = self._imu_proc.process(msg)
            self._pipeline.feed(SensorReading(
                sensor_id="imu0",
                sensor_type=SensorType.IMU,
                data=imu_data,
            ))

        # ------------------------------------------------------------------
        # Queries
        # ------------------------------------------------------------------

        @property
        def subscriber_count(self) -> int:
            return self._sub_count

        def __repr__(self) -> str:
            return f"<ROS2SensorBridge subs={self._sub_count}>"

else:
    # Placeholder so imports don't break outside Docker
    class ROS2SensorBridge:  # type: ignore[no-redef]
        """Stub — rclpy not available."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("rclpy is not installed. Run inside Docker.")
