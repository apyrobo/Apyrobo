"""
Mocked tests for apyrobo/sensors/ros2_subscribers.py.

Patches all ROS2 modules at sys.modules level so the sensor bridge can be
imported and exercised without a real ROS2 installation.
"""

from __future__ import annotations

import math
import sys
from unittest.mock import MagicMock, patch
import pytest

# ---------------------------------------------------------------------------
# ROS2 sys.modules mocks — must happen before any ros2_subscribers import
# ---------------------------------------------------------------------------

_rclpy_mock = MagicMock()
_rclpy_node_mock = MagicMock()
_rclpy_qos_mock = MagicMock()
_sensor_msgs_mock = MagicMock()
_sensor_msgs_msg_mock = MagicMock()
_nav_msgs_mock = MagicMock()
_nav_msgs_msg_mock = MagicMock()
_geometry_msgs_mock = MagicMock()
_geometry_msgs_msg_mock = MagicMock()

_rclpy_mock.node = _rclpy_node_mock
_rclpy_mock.qos = _rclpy_qos_mock

ros2_mocks = {
    "rclpy": _rclpy_mock,
    "rclpy.node": _rclpy_node_mock,
    "rclpy.qos": _rclpy_qos_mock,
    "sensor_msgs": _sensor_msgs_mock,
    "sensor_msgs.msg": _sensor_msgs_msg_mock,
    "nav_msgs": _nav_msgs_mock,
    "nav_msgs.msg": _nav_msgs_msg_mock,
    "geometry_msgs": _geometry_msgs_mock,
    "geometry_msgs.msg": _geometry_msgs_msg_mock,
}
for mod, mock in ros2_mocks.items():
    sys.modules.setdefault(mod, mock)

# Force re-import so _HAS_ROS2 is True under our mocks
if "apyrobo.sensors.ros2_subscribers" in sys.modules:
    del sys.modules["apyrobo.sensors.ros2_subscribers"]

import apyrobo.sensors.ros2_subscribers as subs_module

assert subs_module._HAS_ROS2, (
    "ros2_subscribers._HAS_ROS2 should be True with mocked rclpy"
)

LaserScanProcessor = subs_module.LaserScanProcessor
ImuProcessor = subs_module.ImuProcessor
CameraProcessor = subs_module.CameraProcessor
ROS2SensorBridge = subs_module.ROS2SensorBridge
_sensor_qos = subs_module._sensor_qos

from apyrobo.sensors.pipeline import SensorPipeline, SensorReading
from apyrobo.core.schemas import SensorType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node():
    node = MagicMock()
    node.create_subscription.return_value = MagicMock()
    return node


def _make_pipeline():
    pipeline = MagicMock(spec=SensorPipeline)
    return pipeline


def _make_laser_msg(ranges, angle_min=0.0, angle_increment=0.1):
    msg = MagicMock()
    msg.ranges = ranges
    msg.angle_min = angle_min
    msg.angle_increment = angle_increment
    return msg


def _make_imu_msg(qw=1.0, qx=0.0, qy=0.0, qz=0.0,
                   ang_vel_z=0.1, lin_accel_x=0.0, lin_accel_y=0.0):
    msg = MagicMock()
    msg.orientation.w = qw
    msg.orientation.x = qx
    msg.orientation.y = qy
    msg.orientation.z = qz
    msg.angular_velocity.z = ang_vel_z
    msg.linear_acceleration.x = lin_accel_x
    msg.linear_acceleration.y = lin_accel_y
    return msg


def _make_odom_msg(x=1.0, y=2.0, qw=1.0, qz=0.0):
    msg = MagicMock()
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.orientation.w = qw
    msg.pose.pose.orientation.x = 0.0
    msg.pose.pose.orientation.y = 0.0
    msg.pose.pose.orientation.z = qz
    return msg


# ===========================================================================
# _sensor_qos
# ===========================================================================

class TestSensorQos:
    def test_returns_something(self):
        result = _sensor_qos()
        assert result is not None


# ===========================================================================
# LaserScanProcessor
# ===========================================================================

class TestLaserScanProcessor:
    def test_default_params(self):
        proc = LaserScanProcessor()
        assert proc.max_range == 10.0
        assert proc.min_range == 0.1
        assert proc.subsample == 4

    def test_custom_params(self):
        proc = LaserScanProcessor(max_range=5.0, min_range=0.2, subsample=2)
        assert proc.max_range == 5.0
        assert proc.min_range == 0.2
        assert proc.subsample == 2

    def test_empty_ranges(self):
        proc = LaserScanProcessor()
        msg = _make_laser_msg([])
        result = proc.process(msg)
        assert result == []

    def test_valid_range_produces_point(self):
        proc = LaserScanProcessor(subsample=1)
        msg = _make_laser_msg([1.0], angle_min=0.0, angle_increment=0.1)
        result = proc.process(msg)
        assert len(result) == 1
        pt = result[0]
        assert abs(pt["x"] - 1.0) < 1e-6  # angle=0, distance=1 → x=1, y=0
        assert abs(pt["y"]) < 1e-6
        assert pt["distance"] == 1.0
        assert "radius" in pt

    def test_nan_range_skipped(self):
        proc = LaserScanProcessor(subsample=1)
        msg = _make_laser_msg([float("nan")], angle_min=0.0, angle_increment=0.1)
        result = proc.process(msg)
        assert result == []

    def test_inf_range_skipped(self):
        proc = LaserScanProcessor(subsample=1)
        msg = _make_laser_msg([float("inf")], angle_min=0.0, angle_increment=0.1)
        result = proc.process(msg)
        assert result == []

    def test_too_close_range_skipped(self):
        proc = LaserScanProcessor(min_range=0.5, subsample=1)
        msg = _make_laser_msg([0.1], angle_min=0.0, angle_increment=0.1)
        result = proc.process(msg)
        assert result == []

    def test_too_far_range_skipped(self):
        proc = LaserScanProcessor(max_range=5.0, subsample=1)
        msg = _make_laser_msg([10.0], angle_min=0.0, angle_increment=0.1)
        result = proc.process(msg)
        assert result == []

    def test_subsample_filters_points(self):
        proc = LaserScanProcessor(subsample=2)
        # 4 valid readings, subsample=2 → indices 0 and 2 kept
        msg = _make_laser_msg([1.0, 1.0, 1.0, 1.0], angle_min=0.0, angle_increment=0.1)
        result = proc.process(msg)
        assert len(result) == 2

    def test_robot_pose_transform(self):
        proc = LaserScanProcessor(subsample=1)
        # Point at angle 0, distance 1 from robot at (1, 0) facing east
        msg = _make_laser_msg([1.0], angle_min=0.0, angle_increment=0.1)
        result = proc.process(msg, robot_x=1.0, robot_y=0.0, robot_yaw=0.0)
        assert len(result) == 1
        assert abs(result[0]["x"] - 2.0) < 1e-6

    def test_robot_yaw_transform(self):
        proc = LaserScanProcessor(subsample=1)
        # Point at angle 0, distance 1, robot facing north (pi/2)
        msg = _make_laser_msg([1.0], angle_min=0.0, angle_increment=0.1)
        result = proc.process(msg, robot_x=0.0, robot_y=0.0, robot_yaw=math.pi / 2)
        assert len(result) == 1
        # x should be ~0, y should be ~1
        assert abs(result[0]["x"]) < 1e-6
        assert abs(result[0]["y"] - 1.0) < 1e-6

    def test_multiple_valid_ranges(self):
        proc = LaserScanProcessor(subsample=1, max_range=10.0, min_range=0.1)
        ranges = [1.0, 2.0, 3.0, 0.05, float("nan")]  # last two invalid
        msg = _make_laser_msg(ranges, angle_min=0.0, angle_increment=0.1)
        result = proc.process(msg)
        assert len(result) == 3


# ===========================================================================
# ImuProcessor
# ===========================================================================

class TestImuProcessor:
    def test_identity_quaternion_gives_zero_yaw(self):
        msg = _make_imu_msg(qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        result = ImuProcessor.process(msg)
        assert abs(result["yaw"]) < 1e-9

    def test_90_degree_yaw(self):
        angle = math.pi / 2
        qw = math.cos(angle / 2)
        qz = math.sin(angle / 2)
        msg = _make_imu_msg(qw=qw, qx=0.0, qy=0.0, qz=qz)
        result = ImuProcessor.process(msg)
        assert abs(result["yaw"] - math.pi / 2) < 1e-6

    def test_returns_angular_velocity(self):
        msg = _make_imu_msg(ang_vel_z=0.5)
        result = ImuProcessor.process(msg)
        assert result["angular_velocity_z"] == pytest.approx(0.5)

    def test_returns_linear_acceleration(self):
        msg = _make_imu_msg(lin_accel_x=1.5, lin_accel_y=-0.3)
        result = ImuProcessor.process(msg)
        assert result["linear_accel_x"] == pytest.approx(1.5)
        assert result["linear_accel_y"] == pytest.approx(-0.3)

    def test_result_keys(self):
        msg = _make_imu_msg()
        result = ImuProcessor.process(msg)
        assert "yaw" in result
        assert "angular_velocity_z" in result
        assert "linear_accel_x" in result
        assert "linear_accel_y" in result


# ===========================================================================
# CameraProcessor
# ===========================================================================

class TestCameraProcessor:
    def test_init_auto_backend(self):
        # ultralytics not installed → backend is None
        proc = CameraProcessor(backend="auto")
        assert proc._backend is None

    def test_mock_detections_on_message(self):
        proc = CameraProcessor(backend="none")
        msg = MagicMock()
        msg.mock_detections = [{"id": "obj0", "label": "box", "confidence": 0.9}]
        result = proc.process(msg)
        assert result == [{"id": "obj0", "label": "box", "confidence": 0.9}]

    def test_fallback_metadata_when_no_mock_and_no_backend(self):
        proc = CameraProcessor(backend="none")
        msg = MagicMock(spec=[])  # no mock_detections attribute
        msg.width = 640
        msg.height = 480
        result = proc.process(msg)
        assert len(result) == 1
        assert result[0]["label"] == "__raw_image__"
        assert result[0]["width"] == 640
        assert result[0]["height"] == 480

    def test_fallback_includes_encoding(self):
        proc = CameraProcessor(backend="none")
        msg = MagicMock(spec=[])
        msg.width = 320
        msg.height = 240
        msg.encoding = "rgb8"
        result = proc.process(msg)
        assert result[0]["encoding"] == "rgb8"

    def test_fallback_missing_encoding(self):
        proc = CameraProcessor(backend="none")

        class MinimalMsg:
            width = 100
            height = 100

        result = proc.process(MinimalMsg())
        assert result[0]["encoding"] == "unknown"

    def test_mock_detections_list_empty(self):
        proc = CameraProcessor(backend="none")
        msg = MagicMock()
        msg.mock_detections = []
        result = proc.process(msg)
        assert result == []

    def test_yolov8_backend_unavailable_falls_back_to_none(self):
        # When ultralytics is not installed, backend init returns None
        proc = CameraProcessor(backend="yolov8")
        assert proc._backend is None


# ===========================================================================
# ROS2SensorBridge construction
# ===========================================================================

class TestROS2SensorBridgeInit:
    def test_default_config_creates_four_subscribers(self):
        node = _make_node()
        pipeline = _make_pipeline()
        bridge = ROS2SensorBridge(node, pipeline)
        # odom + scan + camera + imu = 4
        assert bridge.subscriber_count == 4
        assert node.create_subscription.call_count == 4

    def test_custom_config_overrides_topics(self):
        node = _make_node()
        pipeline = _make_pipeline()
        bridge = ROS2SensorBridge(
            node, pipeline,
            config={"scan": "/my_scan", "camera": "", "imu": "", "odom": ""},
        )
        # Only scan subscriber created (others empty)
        assert bridge.subscriber_count == 1

    def test_no_topics_no_subscribers(self):
        node = _make_node()
        pipeline = _make_pipeline()
        bridge = ROS2SensorBridge(
            node, pipeline,
            config={"scan": "", "camera": "", "imu": "", "odom": ""},
        )
        assert bridge.subscriber_count == 0

    def test_repr(self):
        node = _make_node()
        pipeline = _make_pipeline()
        bridge = ROS2SensorBridge(node, pipeline)
        assert "ROS2SensorBridge" in repr(bridge)
        assert "subs=4" in repr(bridge)

    def test_custom_lidar_params(self):
        node = _make_node()
        pipeline = _make_pipeline()
        bridge = ROS2SensorBridge(
            node, pipeline, lidar_subsample=2, lidar_max_range=8.0,
        )
        assert bridge._lidar_proc.subsample == 2
        assert bridge._lidar_proc.max_range == 8.0

    def test_initial_robot_pose_is_zero(self):
        node = _make_node()
        pipeline = _make_pipeline()
        bridge = ROS2SensorBridge(node, pipeline)
        assert bridge._robot_x == 0.0
        assert bridge._robot_y == 0.0
        assert bridge._robot_yaw == 0.0


# ===========================================================================
# ROS2SensorBridge callbacks
# ===========================================================================

class TestROS2SensorBridgeCallbacks:
    def _build_bridge(self):
        node = _make_node()
        pipeline = _make_pipeline()
        bridge = ROS2SensorBridge(node, pipeline)
        return bridge, pipeline

    def test_odom_callback_updates_pose(self):
        bridge, pipeline = self._build_bridge()
        msg = _make_odom_msg(x=3.0, y=4.0, qw=1.0, qz=0.0)
        bridge._odom_callback(msg)
        assert bridge._robot_x == 3.0
        assert bridge._robot_y == 4.0
        assert abs(bridge._robot_yaw) < 1e-9

    def test_odom_callback_feeds_pipeline(self):
        bridge, pipeline = self._build_bridge()
        msg = _make_odom_msg(x=1.0, y=2.0)
        bridge._odom_callback(msg)
        pipeline.feed.assert_called_once()
        reading = pipeline.feed.call_args[0][0]
        assert reading.sensor_id == "odom"
        assert reading.sensor_type == SensorType.IMU
        assert reading.data["x"] == 1.0
        assert reading.data["y"] == 2.0

    def test_odom_callback_90_degree_yaw(self):
        bridge, pipeline = self._build_bridge()
        angle = math.pi / 2
        qw = math.cos(angle / 2)
        qz = math.sin(angle / 2)
        msg = _make_odom_msg(x=0.0, y=0.0, qw=qw, qz=qz)
        bridge._odom_callback(msg)
        assert abs(bridge._robot_yaw - math.pi / 2) < 1e-6

    def test_scan_callback_feeds_pipeline(self):
        bridge, pipeline = self._build_bridge()
        ranges = [1.0, 2.0, 3.0, 4.0]  # subsample=4 → index 0 kept
        msg = _make_laser_msg(ranges, angle_min=0.0, angle_increment=0.1)
        bridge._scan_callback(msg)
        pipeline.feed.assert_called_once()
        reading = pipeline.feed.call_args[0][0]
        assert reading.sensor_id == "lidar0"
        assert reading.sensor_type == SensorType.LIDAR
        assert isinstance(reading.data, list)

    def test_scan_callback_uses_robot_pose(self):
        bridge, pipeline = self._build_bridge()
        bridge._robot_x = 5.0
        bridge._robot_y = 5.0
        bridge._robot_yaw = 0.0
        ranges = [1.0, 1.0, 1.0, 1.0]
        msg = _make_laser_msg(ranges, angle_min=0.0, angle_increment=0.1)
        bridge._scan_callback(msg)
        reading = pipeline.feed.call_args[0][0]
        # The point should be offset by robot_x=5
        if reading.data:
            assert reading.data[0]["x"] > 5.0

    def test_camera_callback_feeds_pipeline(self):
        bridge, pipeline = self._build_bridge()
        msg = MagicMock()
        msg.mock_detections = [{"id": "det0", "label": "person", "confidence": 0.8}]
        bridge._camera_callback(msg)
        pipeline.feed.assert_called_once()
        reading = pipeline.feed.call_args[0][0]
        assert reading.sensor_id == "camera0"
        assert reading.sensor_type == SensorType.CAMERA

    def test_camera_callback_fallback_metadata(self):
        bridge, pipeline = self._build_bridge()
        msg = MagicMock(spec=["width", "height"])
        msg.width = 640
        msg.height = 480
        bridge._camera_callback(msg)
        reading = pipeline.feed.call_args[0][0]
        assert reading.data[0]["label"] == "__raw_image__"

    def test_imu_callback_feeds_pipeline(self):
        bridge, pipeline = self._build_bridge()
        msg = _make_imu_msg(qw=1.0, qx=0.0, qy=0.0, qz=0.0, ang_vel_z=0.3)
        bridge._imu_callback(msg)
        pipeline.feed.assert_called_once()
        reading = pipeline.feed.call_args[0][0]
        assert reading.sensor_id == "imu0"
        assert reading.sensor_type == SensorType.IMU
        assert "yaw" in reading.data
        assert reading.data["angular_velocity_z"] == pytest.approx(0.3)

    def test_multiple_callbacks_accumulate_in_pipeline(self):
        bridge, pipeline = self._build_bridge()
        bridge._odom_callback(_make_odom_msg())
        bridge._scan_callback(
            _make_laser_msg([1.0, 1.0, 1.0, 1.0], angle_min=0.0, angle_increment=0.1)
        )
        bridge._imu_callback(_make_imu_msg())
        assert pipeline.feed.call_count == 3

    def test_odom_callback_updates_pipeline_yaw(self):
        bridge, pipeline = self._build_bridge()
        angle = math.pi / 4
        qw = math.cos(angle / 2)
        qz = math.sin(angle / 2)
        msg = _make_odom_msg(x=0.0, y=0.0, qw=qw, qz=qz)
        bridge._odom_callback(msg)
        reading = pipeline.feed.call_args[0][0]
        assert abs(reading.data["yaw"] - math.pi / 4) < 1e-6


# ===========================================================================
# Stub class (when _HAS_ROS2 is False)
# ===========================================================================

class TestROS2SensorBridgeStub:
    """The stub class raises RuntimeError when rclpy is not available."""

    def test_stub_is_not_tested_here(self):
        # The stub is only instantiated without ROS2. Under our mocks
        # _HAS_ROS2 is True, so we just verify the real class is loaded.
        assert ROS2SensorBridge is not None
        assert hasattr(ROS2SensorBridge, "_scan_callback")
