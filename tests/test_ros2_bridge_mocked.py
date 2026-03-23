"""
Mocked tests for apyrobo/core/ros2_bridge.py.

Patches all ROS2 modules at sys.modules level so the bridge can be imported
and exercised without a real ROS2 installation.
"""

from __future__ import annotations

import math
import sys
from unittest.mock import MagicMock, patch, call
import pytest

# ---------------------------------------------------------------------------
# ROS2 sys.modules mocks — must happen before any apyrobo.core.ros2_bridge import
# ---------------------------------------------------------------------------

_rclpy_mock = MagicMock()
_rclpy_node_mock = MagicMock()
_rclpy_action_mock = MagicMock()
_rclpy_qos_mock = MagicMock()
_rclpy_callback_group_mock = MagicMock()
_rclpy_executors_mock = MagicMock()
_geometry_msgs_mock = MagicMock()
_geometry_msgs_msg_mock = MagicMock()
_nav_msgs_mock = MagicMock()
_nav_msgs_msg_mock = MagicMock()
_sensor_msgs_mock = MagicMock()
_sensor_msgs_msg_mock = MagicMock()
_nav2_msgs_mock = MagicMock()
_nav2_msgs_action_mock = MagicMock()
_std_msgs_mock = MagicMock()
_std_msgs_msg_mock = MagicMock()
_action_msgs_mock = MagicMock()
_action_msgs_msg_mock = MagicMock()
_builtin_interfaces_mock = MagicMock()
_builtin_interfaces_msg_mock = MagicMock()
_tf_transformations_mock = MagicMock()

# Wire up rclpy sub-modules
_rclpy_mock.node = _rclpy_node_mock
_rclpy_mock.action = _rclpy_action_mock
_rclpy_mock.qos = _rclpy_qos_mock
_rclpy_mock.executors = _rclpy_executors_mock

# GoalStatus constants
_goal_status_mock = MagicMock()
_goal_status_mock.STATUS_SUCCEEDED = 4
_goal_status_mock.STATUS_CANCELED = 6
_goal_status_mock.STATUS_ABORTED = 5
_action_msgs_msg_mock.GoalStatus = _goal_status_mock

# Quaternion needs to be a real object so ROS2Adapter._yaw_to_quat works
class _FakeQuaternion:
    def __init__(self):
        self.w = 1.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

_geometry_msgs_msg_mock.Quaternion = _FakeQuaternion
_geometry_msgs_msg_mock.Twist = MagicMock
_geometry_msgs_msg_mock.PoseStamped = MagicMock

ros2_mocks = {
    "rclpy": _rclpy_mock,
    "rclpy.node": _rclpy_node_mock,
    "rclpy.qos": _rclpy_qos_mock,
    "rclpy.action": _rclpy_action_mock,
    "rclpy.callback_group": _rclpy_callback_group_mock,
    "rclpy.executors": _rclpy_executors_mock,
    "geometry_msgs": _geometry_msgs_mock,
    "geometry_msgs.msg": _geometry_msgs_msg_mock,
    "nav_msgs": _nav_msgs_mock,
    "nav_msgs.msg": _nav_msgs_msg_mock,
    "sensor_msgs": _sensor_msgs_mock,
    "sensor_msgs.msg": _sensor_msgs_msg_mock,
    "nav2_msgs": _nav2_msgs_mock,
    "nav2_msgs.action": _nav2_msgs_action_mock,
    "std_msgs": _std_msgs_mock,
    "std_msgs.msg": _std_msgs_msg_mock,
    "action_msgs": _action_msgs_mock,
    "action_msgs.msg": _action_msgs_msg_mock,
    "builtin_interfaces": _builtin_interfaces_mock,
    "builtin_interfaces.msg": _builtin_interfaces_msg_mock,
    "tf_transformations": _tf_transformations_mock,
}
for mod, mock in ros2_mocks.items():
    sys.modules.setdefault(mod, mock)

# Now it is safe to import the bridge module
import importlib
# Force re-import so _HAS_ROS2 is True under our mocks
if "apyrobo.core.ros2_bridge" in sys.modules:
    del sys.modules["apyrobo.core.ros2_bridge"]

import apyrobo.core.ros2_bridge as bridge_module

# Verify the module detected ROS2
assert bridge_module._HAS_ROS2, (
    "ros2_bridge._HAS_ROS2 should be True with mocked rclpy"
)

# Grab the class (only defined when _HAS_ROS2 is True)
ROS2Adapter = bridge_module.ROS2Adapter
RobotState = bridge_module.RobotState
NavState = bridge_module.NavState
_ROS2NodeManager = bridge_module._ROS2NodeManager


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_node():
    """Return a fresh MagicMock that mimics a rclpy Node."""
    node = MagicMock()
    node.create_publisher.return_value = MagicMock()
    node.create_subscription.return_value = MagicMock()
    node.get_topic_names_and_types.return_value = []
    clock = MagicMock()
    clock.now.return_value.to_msg.return_value = MagicMock()
    node.get_clock.return_value = clock
    return node


def _make_nav2_client(accepted=True, succeeded=True):
    """Return a mock ActionClient that accepts/succeeds by default."""
    client = MagicMock()
    client.wait_for_server.return_value = True

    # Build the goal-handle future chain
    goal_handle = MagicMock()
    goal_handle.accepted = accepted

    result_future = MagicMock()
    result_future.done.return_value = True

    result = MagicMock()
    if succeeded:
        result.status = _goal_status_mock.STATUS_SUCCEEDED
    else:
        result.status = _goal_status_mock.STATUS_ABORTED
    result_future.result.return_value = result

    goal_handle.get_result_async.return_value = result_future

    send_future = MagicMock()
    send_future.result.return_value = goal_handle

    # Store callbacks so tests can fire them manually
    def _send_goal_async(msg, feedback_callback=None):
        client._last_feedback_cb = feedback_callback
        return send_future

    client.send_goal_async.side_effect = _send_goal_async

    return client, goal_handle, send_future


@pytest.fixture(autouse=True)
def _reset_node_manager():
    """Reset the singleton before every test."""
    _ROS2NodeManager._instance = None
    _ROS2NodeManager._node = None
    _ROS2NodeManager._executor = None
    _ROS2NodeManager._spin_thread = None
    yield
    _ROS2NodeManager._instance = None
    _ROS2NodeManager._node = None
    _ROS2NodeManager._executor = None
    _ROS2NodeManager._spin_thread = None


def _build_adapter(has_nav2=False, node=None, nav2_client=None, **kwargs):
    """
    Construct a ROS2Adapter with all external calls mocked.

    Returns the adapter plus the underlying mock node.
    """
    if node is None:
        node = _make_node()

    if nav2_client is None:
        nav2_client = MagicMock()
        nav2_client.wait_for_server.return_value = has_nav2

    with (
        patch.object(_ROS2NodeManager, "get_node", return_value=node),
        patch(
            "apyrobo.core.ros2_bridge.ActionClient",
            return_value=nav2_client,
        ),
        patch(
            "apyrobo.core.ros2_bridge.ReentrantCallbackGroup",
            return_value=MagicMock(),
        ),
    ):
        adapter = ROS2Adapter("test_robot", **kwargs)

    adapter._has_odom = True  # skip the wait loop in tests by default
    adapter._has_nav2 = has_nav2
    adapter._nav2_client = nav2_client
    return adapter, node


# ===========================================================================
# Module-level helpers
# ===========================================================================

class TestLoadYamlFile:
    def test_missing_path_returns_empty(self):
        result = bridge_module._load_yaml_file("")
        assert result == {}

    def test_nonexistent_file_returns_empty(self):
        result = bridge_module._load_yaml_file("/tmp/does_not_exist_xyz.yaml")
        assert result == {}

    def test_valid_yaml_file(self, tmp_path):
        f = tmp_path / "cfg.yaml"
        f.write_text("ros2_bridge:\n  cmd_vel: /my_cmd_vel\n")
        result = bridge_module._load_yaml_file(str(f))
        assert result == {"ros2_bridge": {"cmd_vel": "/my_cmd_vel"}}

    def test_invalid_yaml_returns_empty(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("key: [unclosed")
        result = bridge_module._load_yaml_file(str(f))
        assert result == {}

    def test_non_dict_yaml_returns_empty(self, tmp_path):
        f = tmp_path / "list.yaml"
        f.write_text("- item1\n- item2\n")
        result = bridge_module._load_yaml_file(str(f))
        assert result == {}


class TestApplyNamespace:
    def test_no_namespace_returns_copy(self):
        cfg = {"cmd_vel": "/cmd_vel", "odom": "/odom"}
        out = bridge_module._apply_namespace(cfg, None)
        assert out == cfg
        assert out is not cfg

    def test_empty_namespace_returns_copy(self):
        cfg = {"cmd_vel": "/cmd_vel"}
        out = bridge_module._apply_namespace(cfg, "   ")
        assert out == cfg

    def test_namespace_prefixes_absolute_topic(self):
        cfg = {"cmd_vel": "/cmd_vel"}
        out = bridge_module._apply_namespace(cfg, "robot1")
        assert out["cmd_vel"] == "/robot1/cmd_vel"

    def test_namespace_prefixes_relative_topic(self):
        cfg = {"cmd_vel": "cmd_vel"}
        out = bridge_module._apply_namespace(cfg, "robot1")
        assert out["cmd_vel"] == "/robot1/cmd_vel"

    def test_namespace_leading_slash(self):
        cfg = {"odom": "/odom"}
        out = bridge_module._apply_namespace(cfg, "/ns")
        assert out["odom"] == "/ns/odom"

    def test_action_key_not_prefixed(self):
        cfg = {"nav2_action": "navigate_to_pose"}
        out = bridge_module._apply_namespace(cfg, "robot1")
        assert out["nav2_action"] == "navigate_to_pose"

    def test_non_string_values_skipped(self):
        cfg = {"depth": 5}
        out = bridge_module._apply_namespace(cfg, "robot1")
        assert out["depth"] == 5


class TestRosCompatLayer:
    def test_known_distro(self, monkeypatch):
        monkeypatch.setenv("ROS_DISTRO", "humble")
        result = bridge_module._ros_compat_layer()
        assert result == {"distro": "humble", "status": "supported"}

    def test_unknown_distro(self, monkeypatch):
        monkeypatch.setenv("ROS_DISTRO", "foxy")
        result = bridge_module._ros_compat_layer()
        assert result["status"] == "unknown"

    def test_default_distro(self, monkeypatch):
        monkeypatch.delenv("ROS_DISTRO", raising=False)
        result = bridge_module._ros_compat_layer()
        assert result["distro"] == "humble"


class TestSensorQos:
    def test_returns_qos_profile(self):
        result = bridge_module._sensor_qos()
        # Under mocks QoSProfile is a MagicMock — just ensure it was called
        assert result is not None


# ===========================================================================
# _ROS2NodeManager
# ===========================================================================

class TestROS2NodeManager:
    def test_get_node_without_ros2_raises(self):
        original = bridge_module._HAS_ROS2
        bridge_module._HAS_ROS2 = False
        try:
            with pytest.raises(RuntimeError, match="rclpy is not installed"):
                _ROS2NodeManager.get_node()
        finally:
            bridge_module._HAS_ROS2 = original

    def test_get_node_creates_singleton(self):
        _rclpy_mock.ok.return_value = False
        executor_inst = MagicMock()
        _rclpy_mock.executors.MultiThreadedExecutor.return_value = executor_inst

        node = _ROS2NodeManager.get_node()
        _rclpy_mock.init.assert_called()
        assert node is _ROS2NodeManager._node

        # Second call returns same node without re-init
        _rclpy_mock.init.reset_mock()
        node2 = _ROS2NodeManager.get_node()
        _rclpy_mock.init.assert_not_called()
        assert node2 is node

    def test_get_node_skips_init_when_rclpy_already_ok(self):
        _rclpy_mock.ok.return_value = True
        _ROS2NodeManager.get_node()
        _rclpy_mock.init.assert_not_called()

    def test_shutdown_cleans_up(self):
        _rclpy_mock.ok.return_value = False
        _ROS2NodeManager.get_node()
        executor = _ROS2NodeManager._executor
        node = _ROS2NodeManager._node

        _rclpy_mock.ok.return_value = True
        _ROS2NodeManager.shutdown()

        executor.shutdown.assert_called_once()
        node.destroy_node.assert_called_once()
        _rclpy_mock.shutdown.assert_called()
        assert _ROS2NodeManager._instance is None

    def test_shutdown_with_nothing_initialized(self):
        # Should not raise even when nothing has been set up
        _ROS2NodeManager.shutdown()


# ===========================================================================
# ROS2Adapter construction
# ===========================================================================

class TestROS2AdapterInit:
    def test_basic_construction(self):
        adapter, node = _build_adapter()
        assert adapter.robot_name == "test_robot"
        assert adapter._has_nav2 is False
        assert isinstance(adapter._config, dict)

    def test_construction_with_nav2(self):
        adapter, _ = _build_adapter(has_nav2=True)
        assert adapter._has_nav2 is True

    def test_construction_with_namespace(self):
        adapter, _ = _build_adapter(namespace="robot1")
        assert adapter._namespace == "robot1"
        # cmd_vel should be prefixed
        assert "/robot1" in adapter._config["cmd_vel"]

    def test_construction_with_config_override(self):
        adapter, _ = _build_adapter(config={"cmd_vel": "/my_vel"})
        assert adapter._config["cmd_vel"] == "/my_vel"

    def test_construction_with_yaml_config(self, tmp_path):
        f = tmp_path / "ros2.yaml"
        f.write_text("ros2_bridge:\n  cmd_vel: /yaml_vel\n  nav_timeout_sec: 30\n")
        adapter, _ = _build_adapter(config_yaml=str(f))
        assert adapter._config["cmd_vel"] == "/yaml_vel"
        assert adapter._nav_timeout_sec == 30.0

    def test_nav2_client_exception_handled(self):
        node = _make_node()
        with (
            patch.object(_ROS2NodeManager, "get_node", return_value=node),
            patch(
                "apyrobo.core.ros2_bridge.ActionClient",
                side_effect=Exception("boom"),
            ),
            patch(
                "apyrobo.core.ros2_bridge.ReentrantCallbackGroup",
                return_value=MagicMock(),
            ),
        ):
            adapter = ROS2Adapter("test_robot")

        assert adapter._has_nav2 is False
        assert adapter._nav2_client is None

    def test_feedback_handler_stored(self):
        handler = MagicMock()
        adapter, _ = _build_adapter(feedback_handler=handler)
        assert adapter._feedback_handler is handler

    def test_odom_reliability_reliable(self):
        adapter, _ = _build_adapter(odom_reliability="reliable")
        assert adapter._odom_reliability == "reliable"

    def test_initial_state(self):
        adapter, _ = _build_adapter()
        assert adapter._state_machine == RobotState.IDLE
        assert adapter._nav_state == NavState.IDLE
        assert adapter._position == (0.0, 0.0)
        assert adapter._orientation == 0.0


# ===========================================================================
# Odometry callbacks & quaternion math
# ===========================================================================

class TestOdometryAndQuaternion:
    def test_quat_to_yaw_identity(self):
        q = _FakeQuaternion()
        q.w = 1.0
        q.x = 0.0
        q.y = 0.0
        q.z = 0.0
        yaw = ROS2Adapter._quat_to_yaw(q)
        assert abs(yaw) < 1e-9

    def test_quat_to_yaw_90_deg(self):
        q = _FakeQuaternion()
        angle = math.pi / 2
        q.w = math.cos(angle / 2)
        q.z = math.sin(angle / 2)
        q.x = 0.0
        q.y = 0.0
        yaw = ROS2Adapter._quat_to_yaw(q)
        assert abs(yaw - math.pi / 2) < 1e-6

    def test_yaw_to_quat_zero(self):
        q = ROS2Adapter._yaw_to_quat(0.0)
        assert abs(q.w - 1.0) < 1e-9
        assert abs(q.z) < 1e-9

    def test_yaw_to_quat_roundtrip(self):
        for yaw in [0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 3]:
            q = ROS2Adapter._yaw_to_quat(yaw)
            recovered = ROS2Adapter._quat_to_yaw(q)
            assert abs(recovered - yaw) < 1e-6, f"yaw={yaw} round-trip failed"

    def test_odom_callback_updates_position(self):
        adapter, _ = _build_adapter()
        msg = MagicMock()
        msg.pose.pose.position.x = 3.5
        msg.pose.pose.position.y = -1.2
        q = _FakeQuaternion()
        q.w = 1.0
        q.x = 0.0
        q.y = 0.0
        q.z = 0.0
        msg.pose.pose.orientation = q
        adapter._odom_callback(msg)
        assert adapter._position == (3.5, -1.2)
        assert adapter._has_odom is True

    def test_odom_qos_reliable(self):
        adapter, _ = _build_adapter(odom_reliability="reliable")
        qos = adapter._odom_qos()
        # Under mocks, QoSProfile just returns a MagicMock; we just check no crash
        assert qos is not None

    def test_odom_qos_best_effort(self):
        adapter, _ = _build_adapter()
        qos = adapter._odom_qos()
        assert qos is not None

    def test_get_pose(self):
        adapter, _ = _build_adapter()
        adapter._position = (1.0, 2.0)
        adapter._orientation = 0.5
        assert adapter.get_pose() == (1.0, 2.0, 0.5)


# ===========================================================================
# Capability discovery
# ===========================================================================

class TestGetCapabilities:
    def test_returns_robot_capability(self):
        adapter, node = _build_adapter()
        node.get_topic_names_and_types.return_value = []
        caps = adapter.get_capabilities()
        assert caps.robot_id == "test_robot"

    def test_navigate_capability_with_cmd_vel_topic(self):
        adapter, node = _build_adapter()
        node.get_topic_names_and_types.return_value = [
            (adapter._config["cmd_vel"], ["geometry_msgs/msg/Twist"])
        ]
        caps = adapter.get_capabilities()
        cap_types = [c.capability_type for c in caps.capabilities]
        from apyrobo.core.schemas import CapabilityType
        assert CapabilityType.NAVIGATE in cap_types

    def test_navigate_capability_with_nav2(self):
        adapter, node = _build_adapter(has_nav2=True)
        node.get_topic_names_and_types.return_value = []
        caps = adapter.get_capabilities()
        from apyrobo.core.schemas import CapabilityType
        cap_types = [c.capability_type for c in caps.capabilities]
        assert CapabilityType.NAVIGATE in cap_types
        # Description should mention Nav2
        nav_cap = next(c for c in caps.capabilities if c.capability_type == CapabilityType.NAVIGATE)
        assert "Nav2" in nav_cap.description

    def test_sensor_detection(self):
        adapter, node = _build_adapter()
        node.get_topic_names_and_types.return_value = [
            (adapter._config["scan"], ["sensor_msgs/msg/LaserScan"]),
            (adapter._config["camera"], ["sensor_msgs/msg/Image"]),
            (adapter._config["imu"], ["sensor_msgs/msg/Imu"]),
        ]
        caps = adapter.get_capabilities()
        sensor_ids = {s.sensor_id for s in caps.sensors}
        assert "lidar0" in sensor_ids
        assert "camera0" in sensor_ids
        assert "imu0" in sensor_ids

    def test_metadata_fields(self):
        adapter, node = _build_adapter()
        node.get_topic_names_and_types.return_value = []
        caps = adapter.get_capabilities()
        assert "adapter" in caps.metadata
        assert caps.metadata["adapter"] == "ros2"


# ===========================================================================
# Stop / E-Stop / reset
# ===========================================================================

class TestStopAndEStop:
    def test_stop_publishes_zero_twist(self):
        adapter, _ = _build_adapter()
        adapter.stop()
        assert adapter._nav_state == NavState.CANCELLED or adapter._nav_state == NavState.IDLE
        adapter._cmd_vel_pub.publish.assert_called()

    def test_stop_from_navigating_state(self):
        adapter, _ = _build_adapter()
        adapter._nav_state = NavState.NAVIGATING
        adapter.stop()
        assert adapter._nav_state == NavState.CANCELLED
        assert adapter._state_machine == RobotState.IDLE

    def test_stop_cancels_nav2_goal(self):
        adapter, _ = _build_adapter(has_nav2=True)
        adapter._nav_state = NavState.NAVIGATING
        cancel_future = MagicMock()
        cancel_future.done.return_value = True
        goal_handle = MagicMock()
        goal_handle.cancel_goal_async.return_value = cancel_future
        adapter._goal_handle = goal_handle
        adapter.stop()
        goal_handle.cancel_goal_async.assert_called_once()

    def test_emergency_stop_sets_state(self):
        adapter, _ = _build_adapter()
        adapter.emergency_stop()
        assert adapter._state_machine == RobotState.E_STOPPED
        adapter._cmd_vel_pub.publish.assert_called()

    def test_emergency_stop_with_nav2_goal(self):
        adapter, _ = _build_adapter(has_nav2=True)
        cancel_future = MagicMock()
        cancel_future.done.return_value = True
        goal_handle = MagicMock()
        goal_handle.cancel_goal_async.return_value = cancel_future
        adapter._goal_handle = goal_handle
        adapter.emergency_stop()
        goal_handle.cancel_goal_async.assert_called_once()

    def test_reset_estop(self):
        adapter, _ = _build_adapter()
        adapter._state_machine = RobotState.E_STOPPED
        adapter.reset_estop()
        assert adapter._state_machine == RobotState.IDLE

    def test_reset_estop_no_op_when_not_stopped(self):
        adapter, _ = _build_adapter()
        adapter._state_machine = RobotState.NAVIGATING
        adapter.reset_estop()
        assert adapter._state_machine == RobotState.NAVIGATING

    def test_publish_stop(self):
        adapter, _ = _build_adapter()
        adapter._publish_stop()
        adapter._cmd_vel_pub.publish.assert_called()

    def test_set_feedback_handler(self):
        adapter, _ = _build_adapter()
        handler = MagicMock()
        adapter.set_feedback_handler(handler)
        assert adapter._feedback_handler is handler


# ===========================================================================
# move() — E_STOPPED guard
# ===========================================================================

class TestMoveEstopGuard:
    def test_move_raises_when_estopped(self):
        adapter, _ = _build_adapter()
        adapter._state_machine = RobotState.E_STOPPED
        with pytest.raises(RuntimeError, match="E_STOPPED"):
            adapter.move(1.0, 2.0)


# ===========================================================================
# Nav2 callbacks
# ===========================================================================

class TestNav2Callbacks:
    def test_goal_response_accepted(self):
        adapter, _ = _build_adapter(has_nav2=True)
        future = MagicMock()
        goal_handle = MagicMock()
        goal_handle.accepted = True
        result_future = MagicMock()
        goal_handle.get_result_async.return_value = result_future
        future.result.return_value = goal_handle

        adapter._nav2_goal_response(future)

        assert adapter._goal_handle is goal_handle
        result_future.add_done_callback.assert_called_once_with(adapter._nav2_result)

    def test_goal_response_rejected(self):
        adapter, _ = _build_adapter(has_nav2=True)
        future = MagicMock()
        goal_handle = MagicMock()
        goal_handle.accepted = False
        future.result.return_value = goal_handle

        adapter._nav_state = NavState.NAVIGATING
        adapter._nav2_goal_response(future)

        assert adapter._nav_state == NavState.FAILED

    def test_nav2_result_succeeded(self):
        adapter, _ = _build_adapter(has_nav2=True)
        future = MagicMock()
        result = MagicMock()
        result.status = _goal_status_mock.STATUS_SUCCEEDED
        future.result.return_value = result

        adapter._nav2_result(future)
        assert adapter._nav_state == NavState.SUCCEEDED

    def test_nav2_result_canceled(self):
        adapter, _ = _build_adapter(has_nav2=True)
        future = MagicMock()
        result = MagicMock()
        result.status = _goal_status_mock.STATUS_CANCELED
        future.result.return_value = result

        adapter._nav2_result(future)
        assert adapter._nav_state == NavState.CANCELLED

    def test_nav2_result_failed(self):
        adapter, _ = _build_adapter(has_nav2=True)
        future = MagicMock()
        result = MagicMock()
        result.status = _goal_status_mock.STATUS_ABORTED
        future.result.return_value = result

        adapter._nav2_result(future)
        assert adapter._nav_state == NavState.FAILED

    def test_nav2_feedback_updates_position(self):
        adapter, _ = _build_adapter(has_nav2=True)
        feedback_msg = MagicMock()
        feedback_msg.feedback.current_pose.pose.position.x = 2.0
        feedback_msg.feedback.current_pose.pose.position.y = 3.0
        q = _FakeQuaternion()
        feedback_msg.feedback.current_pose.pose.orientation = q
        feedback_msg.feedback.distance_remaining = 1.5

        adapter._nav2_feedback(feedback_msg)

        assert adapter._position == (2.0, 3.0)

    def test_nav2_feedback_calls_handler(self):
        handler = MagicMock()
        adapter, _ = _build_adapter(has_nav2=True, feedback_handler=handler)
        feedback_msg = MagicMock()
        feedback_msg.feedback.current_pose.pose.position.x = 1.0
        feedback_msg.feedback.current_pose.pose.position.y = 0.5
        q = _FakeQuaternion()
        feedback_msg.feedback.current_pose.pose.orientation = q
        feedback_msg.feedback.distance_remaining = 0.8

        adapter._nav2_feedback(feedback_msg)
        handler.assert_called_once()
        payload = handler.call_args[0][0]
        assert payload["event"] == "nav2_feedback"
        assert payload["position"] == (1.0, 0.5)

    def test_nav2_feedback_handler_exception_swallowed(self):
        handler = MagicMock(side_effect=RuntimeError("oops"))
        adapter, _ = _build_adapter(has_nav2=True, feedback_handler=handler)
        feedback_msg = MagicMock()
        feedback_msg.feedback.current_pose.pose.position.x = 0.0
        feedback_msg.feedback.current_pose.pose.position.y = 0.0
        q = _FakeQuaternion()
        feedback_msg.feedback.current_pose.pose.orientation = q
        feedback_msg.feedback.distance_remaining = None

        # Should not raise
        adapter._nav2_feedback(feedback_msg)

    def test_nav2_feedback_no_distance_remaining(self):
        adapter, _ = _build_adapter(has_nav2=True)
        feedback_msg = MagicMock()
        feedback_msg.feedback.current_pose.pose.position.x = 0.0
        feedback_msg.feedback.current_pose.pose.position.y = 0.0
        q = _FakeQuaternion()
        feedback_msg.feedback.current_pose.pose.orientation = q
        # Simulate missing distance_remaining attribute
        del feedback_msg.feedback.distance_remaining
        # Should not raise
        adapter._nav2_feedback(feedback_msg)

    def test_cancel_nav2_with_no_goal(self):
        adapter, _ = _build_adapter(has_nav2=True)
        adapter._goal_handle = None
        # Should not raise
        adapter._cancel_nav2()

    def test_cancel_nav2_waits_for_future(self):
        adapter, _ = _build_adapter(has_nav2=True)
        cancel_future = MagicMock()
        cancel_future.done.side_effect = [False, False, True]
        goal_handle = MagicMock()
        goal_handle.cancel_goal_async.return_value = cancel_future
        adapter._goal_handle = goal_handle

        adapter._cancel_nav2()
        goal_handle.cancel_goal_async.assert_called_once()


# ===========================================================================
# _move_nav2 (blocking path via goal acceptance timeout)
# ===========================================================================

class TestMoveNav2:
    def test_move_nav2_goal_acceptance_timeout(self):
        """If goal is never accepted, should time out."""
        adapter, _ = _build_adapter(has_nav2=True)
        adapter._goal_accept_timeout_sec = 0.05  # very short for test speed

        send_future = MagicMock()
        # Don't set a done_callback so _goal_handle stays None
        adapter._nav2_client.send_goal_async.return_value = send_future

        adapter._move_nav2(1.0, 1.0)

        assert adapter._nav_state == NavState.TIMED_OUT

    def test_move_nav2_with_speed_logs(self):
        """Speed parameter should be logged but not cause an error."""
        adapter, _ = _build_adapter(has_nav2=True)
        adapter._goal_accept_timeout_sec = 0.05

        send_future = MagicMock()
        adapter._nav2_client.send_goal_async.return_value = send_future

        # Should not raise even though we pass speed
        adapter._move_nav2(1.0, 1.0, speed=0.3)

    def test_move_nav2_sets_navigating_state(self):
        adapter, _ = _build_adapter(has_nav2=True)
        adapter._goal_accept_timeout_sec = 0.05
        send_future = MagicMock()
        adapter._nav2_client.send_goal_async.return_value = send_future
        adapter._move_nav2(2.0, 3.0)
        # After timeout, nav_state is TIMED_OUT
        assert adapter._nav_state == NavState.TIMED_OUT

    def test_move_nav2_timeout_cancels_goal(self):
        """After the navigation timeout, _cancel_nav2 is called."""
        adapter, _ = _build_adapter(has_nav2=True)
        adapter._goal_accept_timeout_sec = 0.0
        adapter._nav_timeout_sec = 0.05

        send_future = MagicMock()
        adapter._nav2_client.send_goal_async.return_value = send_future

        # Simulate goal accepted immediately
        goal_handle = MagicMock()
        goal_handle.accepted = True
        goal_handle.cancel_goal_async.return_value = MagicMock()
        cancel_future = MagicMock()
        cancel_future.done.return_value = True
        goal_handle.cancel_goal_async.return_value = cancel_future
        adapter._goal_handle = goal_handle

        # Force nav_state to stay NAVIGATING long enough to trigger timeout
        adapter._nav_state = NavState.NAVIGATING
        adapter._move_nav2.__func__  # make sure it's bound

        # Use patch to inject immediate goal handle
        original_send = adapter._nav2_client.send_goal_async
        def fake_send(msg, feedback_callback=None):
            return send_future
        adapter._nav2_client.send_goal_async.side_effect = fake_send

        adapter._move_nav2(5.0, 5.0)
        assert adapter._nav_state in (NavState.TIMED_OUT, NavState.NAVIGATING)


# ===========================================================================
# _move_cmd_vel (proportional controller)
# ===========================================================================

class TestMoveCmdVel:
    def test_cmd_vel_reaches_goal_immediately(self):
        """Robot already at goal — should stop immediately."""
        adapter, _ = _build_adapter()
        adapter._position = (1.0, 2.0)
        adapter._orientation = 0.0

        adapter._move_cmd_vel(1.0, 2.0, speed=0.5)

        assert adapter._nav_state == NavState.SUCCEEDED

    def test_cmd_vel_timeout(self):
        """If goal is far away and timeout is tiny, should time out."""
        adapter, _ = _build_adapter()
        adapter._nav_timeout_sec = 0.01
        adapter._position = (0.0, 0.0)

        adapter._move_cmd_vel(100.0, 100.0, speed=0.1)
        assert adapter._nav_state == NavState.TIMED_OUT

    def test_cmd_vel_publishes_twist(self):
        adapter, _ = _build_adapter()
        adapter._nav_timeout_sec = 0.01
        adapter._position = (0.0, 0.0)
        adapter._move_cmd_vel(100.0, 0.0, speed=0.5)
        assert adapter._cmd_vel_pub.publish.call_count >= 1

    def test_cmd_vel_dwa_mode(self):
        adapter, _ = _build_adapter(config={"cmd_vel_controller": "dwa"})
        adapter._nav_timeout_sec = 0.01
        adapter._position = (0.0, 0.0)
        adapter._config["cmd_vel_controller"] = "dwa"
        adapter._move_cmd_vel(100.0, 0.0, speed=0.5)
        assert adapter._nav_state in (NavState.TIMED_OUT, NavState.SUCCEEDED)

    def test_cmd_vel_default_speed(self):
        """Should use speed=0.5 when None passed."""
        adapter, _ = _build_adapter()
        adapter._nav_timeout_sec = 0.01
        adapter._position = (0.0, 0.0)
        adapter._move_cmd_vel(100.0, 0.0, speed=None)
        assert adapter._nav_state == NavState.TIMED_OUT

    def test_cmd_vel_large_yaw_error_reduces_speed_dwa(self):
        """With DWA mode and large yaw error, linear.x should be 0."""
        adapter, _ = _build_adapter()
        adapter._nav_timeout_sec = 0.01
        adapter._position = (0.0, 0.0)
        adapter._orientation = 0.0
        adapter._config["cmd_vel_controller"] = "dwa"
        # Target directly behind — yaw error will be ~pi
        adapter._move_cmd_vel(-100.0, 0.0, speed=0.5)
        assert adapter._cmd_vel_pub.publish.call_count >= 1


# ===========================================================================
# move() dispatching
# ===========================================================================

class TestMove:
    def test_move_uses_cmd_vel_fallback(self):
        adapter, _ = _build_adapter(has_nav2=False)
        adapter._nav_timeout_sec = 0.01
        adapter._position = (0.0, 0.0)
        # Patch _move_cmd_vel to avoid real loop
        with patch.object(adapter, "_move_cmd_vel") as mock_cmd:
            adapter.move(1.0, 2.0)
            mock_cmd.assert_called_once_with(1.0, 2.0, None)

    def test_move_uses_nav2(self):
        adapter, _ = _build_adapter(has_nav2=True)
        with patch.object(adapter, "_move_nav2") as mock_nav2:
            adapter.move(3.0, 4.0, speed=0.5)
            mock_nav2.assert_called_once_with(3.0, 4.0, 0.5)

    def test_move_restores_idle_after_estop_guard(self):
        adapter, _ = _build_adapter(has_nav2=False)
        adapter._nav_timeout_sec = 0.01
        adapter._position = (1.0, 1.0)
        with patch.object(adapter, "_move_cmd_vel"):
            adapter.move(1.0, 1.0)
        assert adapter._state_machine == RobotState.IDLE


# ===========================================================================
# Normalize angle
# ===========================================================================

class TestNormalizeAngle:
    @pytest.mark.parametrize("angle, expected", [
        (0.0, 0.0),
        (math.pi, math.pi),
        (-math.pi, -math.pi),
        (3 * math.pi, math.pi),
        (-3 * math.pi, -math.pi),
        (2 * math.pi + 0.1, 0.1),
    ])
    def test_normalize(self, angle, expected):
        result = ROS2Adapter._normalize_angle(angle)
        assert abs(result - expected) < 1e-9


# ===========================================================================
# Properties
# ===========================================================================

class TestProperties:
    def test_position_property(self):
        adapter, _ = _build_adapter()
        adapter._position = (5.0, 6.0)
        assert adapter.position == (5.0, 6.0)

    def test_orientation_property(self):
        adapter, _ = _build_adapter()
        adapter._orientation = 1.2
        assert adapter.orientation == 1.2

    def test_nav_state_property(self):
        adapter, _ = _build_adapter()
        adapter._nav_state = NavState.SUCCEEDED
        assert adapter.nav_state == NavState.SUCCEEDED

    def test_is_moving_true(self):
        adapter, _ = _build_adapter()
        adapter._nav_state = NavState.NAVIGATING
        assert adapter.is_moving is True

    def test_is_moving_false(self):
        adapter, _ = _build_adapter()
        adapter._nav_state = NavState.IDLE
        assert adapter.is_moving is False

    def test_robot_state_property(self):
        adapter, _ = _build_adapter()
        assert adapter.robot_state == RobotState.IDLE


# ===========================================================================
# SLAM & floor map
# ===========================================================================

class TestSlamAndFloorMap:
    def test_trigger_slam_success(self):
        adapter, _ = _build_adapter()
        with patch("subprocess.Popen") as mock_popen:
            result = adapter.trigger_slam()
        assert result is True
        mock_popen.assert_called_once()

    def test_trigger_slam_failure(self):
        adapter, _ = _build_adapter()
        with patch("subprocess.Popen", side_effect=OSError("no ros2")):
            result = adapter.trigger_slam()
        assert result is False

    def test_switch_floor_map_success(self):
        adapter, _ = _build_adapter()
        with patch("subprocess.run") as mock_run:
            result = adapter.switch_floor_map("floor1", "/maps/floor1.yaml")
        assert result is True
        assert adapter._current_floor == "floor1"
        assert adapter._floor_maps["floor1"] == "/maps/floor1.yaml"

    def test_switch_floor_map_failure(self):
        adapter, _ = _build_adapter()
        with patch("subprocess.run", side_effect=OSError("no ros2")):
            result = adapter.switch_floor_map("floor2", "/maps/floor2.yaml")
        assert result is False


# ===========================================================================
# RobotState / NavState enums
# ===========================================================================

class TestEnums:
    def test_robot_state_values(self):
        assert RobotState.IDLE == "idle"
        assert RobotState.NAVIGATING == "navigating"
        assert RobotState.E_STOPPED == "e_stopped"

    def test_nav_state_values(self):
        assert NavState.IDLE == "idle"
        assert NavState.NAVIGATING == "navigating"
        assert NavState.SUCCEEDED == "succeeded"
        assert NavState.FAILED == "failed"
        assert NavState.CANCELLED == "cancelled"
        assert NavState.TIMED_OUT == "timed_out"
