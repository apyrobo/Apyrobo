"""
ROS 2 Bridge Adapter — production-grade connection to real ROS 2 robots.

This adapter translates APYROBO's semantic commands into:
    - Nav2 NavigateToPose action goals (proper goal-based navigation)
    - cmd_vel Twist fallback for simple moves
    - Odometry subscription for continuous pose tracking
    - Topic introspection for auto capability discovery

Requires rclpy + nav2_msgs (available inside the Docker container).

Usage:
    robot = Robot.discover("ros2://turtlebot4")     # real robot OR live Gazebo sim
    # Note: gazebo:// is NOT this adapter — it is a physics-flavored mock
    # (see core/adapters.py). A live Gazebo sim is driven via ros2://.

Node lifecycle: all ros2:// adapters in a process share one rclpy node
("apyrobo_bridge") spun by a daemon thread, so it dies with the process.
Long-running hosts that want the DDS participant removed from the graph
promptly (rather than after lease timeout) should call
``_ROS2NodeManager.shutdown()`` when done with all robots. This is not
wired to ``atexit`` on purpose: teardown ordering against a still-spinning
executor at interpreter exit is unreliable.

Architecture:
    APYROBO Agent
        │
        ▼
    ROS2Adapter.move(x, y, speed)
        │
        ├──► Nav2 NavigateToPose action (preferred)
        │       └── goal tracking, feedback, timeout, cancel
        │
        └──► cmd_vel Twist (fallback if Nav2 unavailable)
                └── proportional controller loop
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import threading
import time
from enum import Enum
from typing import Any

from apyrobo.core.adapters import CapabilityAdapter, register_adapter
from apyrobo.core.health import ConnectionHealth
from apyrobo.core.schemas import (
    Capability,
    CapabilityType,
    RobotCapability,
    SensorInfo,
    SensorType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_yaml_file(path: str) -> dict[str, Any]:
    """Load YAML config if available; returns empty dict on failure."""
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Failed to load ROS2 config YAML %s: %s", path, e)
        return {}


def _apply_namespace(config: dict[str, Any], namespace: str | None) -> dict[str, Any]:
    """Prefix configured topic names with namespace (RT-01)."""
    if not namespace:
        return dict(config)
    ns = namespace.strip()
    if not ns:
        return dict(config)
    if not ns.startswith("/"):
        ns = f"/{ns}"
    ns = ns.rstrip("/")

    out = dict(config)
    for key, value in list(out.items()):
        if not isinstance(value, str):
            continue
        if key.endswith("_action"):
            continue
        if not value.startswith("/"):
            out[key] = f"{ns}/{value}"
        else:
            out[key] = f"{ns}{value}"
    return out


def _ros_compat_layer() -> dict[str, str]:
    """RT-10: expose active ROS distro compatibility metadata."""
    distro = os.getenv("ROS_DISTRO", "humble").lower()
    if distro in {"humble", "iron", "jazzy"}:
        return {"distro": distro, "status": "supported"}
    return {"distro": distro, "status": "unknown"}


# ---------------------------------------------------------------------------
# ROS 2 imports — fail gracefully outside Docker
# ---------------------------------------------------------------------------

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from geometry_msgs.msg import Twist, PoseStamped, Quaternion
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan, Image, Imu
    from nav2_msgs.action import ComputePathToPose, NavigateToPose
    from action_msgs.msg import GoalStatus
    import tf_transformations  # euler/quaternion conversion

    _HAS_ROS2 = True
    _ROS2_IMPORT_ERROR: str | None = None
except ImportError as _exc:
    import warnings
    _HAS_ROS2 = False
    _ROS2_IMPORT_ERROR = str(_exc)
    warnings.warn(
        f"ROS 2 import failed ({_exc}) — the ros2:// adapter will not be "
        "available. Robot.discover('ros2://...') will raise RuntimeError. "
        "To use ros2://, run APYROBO inside the Docker container: "
        "docker compose -f docker/docker-compose.yml exec apyrobo bash",
        stacklevel=1,
    )


# tf2_ros is separate: without it ros2:// still works, on /odom positions.
try:
    from rclpy.time import Time as _RosTime
    from tf2_ros import Buffer as _TfBuffer
    from tf2_ros import TransformListener as _TfListener
    _HAS_TF2 = True
except ImportError:
    _HAS_TF2 = False


# ---------------------------------------------------------------------------
# Navigation state
# ---------------------------------------------------------------------------

class RobotState(str, Enum):
    """RT-06: high-level robot execution state machine."""
    IDLE = "idle"
    NAVIGATING = "navigating"
    MANIPULATING = "manipulating"
    E_STOPPED = "e_stopped"


class NavState(str, Enum):
    """State of the current navigation goal."""
    IDLE = "idle"
    NAVIGATING = "navigating"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class Ros2NavigationError(RuntimeError):
    """Raised by ROS2Adapter.move() when the robot did not reach the goal.

    Covers a goal Nav2 rejected or aborted, a goal Nav2 never answered, a
    navigation timeout (Nav2 or the cmd_vel fallback) and a goal cancelled by
    stop() or emergency_stop() before it finished. ``nav_state`` holds the
    final NavState.
    """

    def __init__(self, message: str, nav_state: NavState) -> None:
        super().__init__(message)
        self.nav_state = nav_state


# ---------------------------------------------------------------------------
# ROS 2 Node Manager (singleton)
# ---------------------------------------------------------------------------

class _ROS2NodeManager:
    """
    Singleton that manages rclpy initialisation and a shared executor.

    All adapters share one rclpy node with a background spin thread.
    This avoids the common pitfall of multiple rclpy.init() calls.
    """

    _instance: _ROS2NodeManager | None = None
    _node: Any = None
    _executor: Any = None
    _spin_thread: threading.Thread | None = None
    _lock = threading.Lock()

    @classmethod
    def get_node(cls) -> Any:
        if not _HAS_ROS2:
            raise RuntimeError(
                "rclpy is not installed. Run inside the Docker container "
                "or install ROS 2 Humble. Use 'mock://' for testing."
            )
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                if not rclpy.ok():
                    rclpy.init()
                cls._node = rclpy.create_node("apyrobo_bridge")

                # Background spin so callbacks fire without blocking
                cls._executor = rclpy.executors.MultiThreadedExecutor()
                cls._executor.add_node(cls._node)
                cls._spin_thread = threading.Thread(
                    target=cls._executor.spin, daemon=True
                )
                cls._spin_thread.start()
                logger.info("ROS 2 bridge node initialised with background spin")
        return cls._node

    @classmethod
    def shutdown(cls) -> None:
        """Clean shutdown — call on program exit."""
        with cls._lock:
            if cls._executor is not None:
                cls._executor.shutdown()
            if cls._node is not None:
                cls._node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            cls._instance = None
            cls._node = None


# ---------------------------------------------------------------------------
# QoS profiles
# ---------------------------------------------------------------------------

def _sensor_qos() -> Any:
    """QoS for sensor topics (best-effort, volatile)."""
    if not _HAS_ROS2:
        return None
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        depth=5,
    )


# ---------------------------------------------------------------------------
# ROS 2 Adapter — only registered if rclpy is available
# ---------------------------------------------------------------------------

if _HAS_ROS2:

    @register_adapter("ros2")
    class ROS2Adapter(CapabilityAdapter):
        """
        Production adapter for ROS 2 robots (real or Gazebo simulated).

        Navigation strategy:
            1. Prefer Nav2 NavigateToPose action if available
            2. Fall back to cmd_vel proportional controller

        Pose tracking:
            - Subscribes to /odom with best-effort QoS
            - Extracts yaw from quaternion for heading

        Capability discovery:
            - Introspects live ROS 2 topics
            - Maps found topics to APYROBO capability types
        """

        # Default topic/action mappings (TurtleBot4-compatible)
        DEFAULT_CONFIG = {
            "cmd_vel": "/cmd_vel",
            "odom": "/odom",
            "scan": "/scan",
            "camera": "/oakd/rgb/preview/image_raw",
            "depth": "/oakd/stereo/image_raw",
            "imu": "/imu",
            "nav2_action": "navigate_to_pose",
            "nav2_plan_action": "compute_path_to_pose",
        }

        # Navigation tuning
        NAV_TIMEOUT_SEC = 120.0
        NAV2_SERVER_WAIT_SEC = 5.0
        ODOM_WAIT_SEC = 5.0
        MAP_TF_WAIT_SEC = 2.0
        # A map transform older than this, against the odometry message that
        # asked for it, means localisation stopped; /odom is then the honest
        # answer. Compared with the message stamp so sim time works.
        MAP_TF_MAX_AGE_SEC = 2.0
        GOAL_ACCEPT_TIMEOUT_SEC = 10.0
        # Resend an unanswered goal after this long. The first goal from a
        # freshly started node can be lost: Nav2 logs "Failed to send goal
        # response (timeout)" when it has not yet discovered our response
        # reader, and the client never hears back.
        GOAL_RESEND_SEC = 2.0
        PLAN_TIMEOUT_SEC = 10.0
        GOAL_TOLERANCE_M = 0.25
        CMD_VEL_HZ = 10.0
        PROPORTIONAL_GAIN_LINEAR = 0.8
        PROPORTIONAL_GAIN_ANGULAR = 2.0

        def __init__(self, robot_name: str, **kwargs: Any) -> None:
            super().__init__(robot_name, **kwargs)
            self._node = _ROS2NodeManager.get_node()
            self._cb_group = ReentrantCallbackGroup()

            # Merge user config over defaults + optional YAML + namespace (RT-01)
            yaml_path = kwargs.get("config_yaml")
            yaml_data = _load_yaml_file(yaml_path)
            yaml_cfg = yaml_data.get("ros2_bridge", yaml_data) if isinstance(yaml_data, dict) else {}

            self._config = dict(self.DEFAULT_CONFIG)
            if isinstance(yaml_cfg, dict):
                self._config.update(yaml_cfg.get("topics", {}))
                for key in (
                    "nav2_action", "nav2_plan_action",
                    "cmd_vel", "odom", "scan", "camera", "depth", "imu",
                ):
                    if key in yaml_cfg:
                        self._config[key] = yaml_cfg[key]
            self._config.update(kwargs.get("config", {}))

            self._namespace = kwargs.get("namespace") or (yaml_cfg.get("namespace") if isinstance(yaml_cfg, dict) else None)
            self._config = _apply_namespace(self._config, self._namespace)

            # Configurable timeouts/QoS (RT-02/RT-03)
            self._nav_timeout_sec = float(kwargs.get("nav_timeout_sec", yaml_cfg.get("nav_timeout_sec", self.NAV_TIMEOUT_SEC) if isinstance(yaml_cfg, dict) else self.NAV_TIMEOUT_SEC))
            self._nav2_server_wait_sec = float(kwargs.get("nav2_server_wait_sec", yaml_cfg.get("nav2_server_wait_sec", self.NAV2_SERVER_WAIT_SEC) if isinstance(yaml_cfg, dict) else self.NAV2_SERVER_WAIT_SEC))
            self._odom_wait_sec = float(kwargs.get("odom_wait_sec", yaml_cfg.get("odom_wait_sec", self.ODOM_WAIT_SEC) if isinstance(yaml_cfg, dict) else self.ODOM_WAIT_SEC))
            plan_timeout_default = (
                yaml_cfg.get("plan_timeout_sec", self.PLAN_TIMEOUT_SEC)
                if isinstance(yaml_cfg, dict) else self.PLAN_TIMEOUT_SEC
            )
            self._plan_timeout_sec = float(kwargs.get("plan_timeout_sec", plan_timeout_default))
            self._goal_accept_timeout_sec = float(kwargs.get("goal_accept_timeout_sec", yaml_cfg.get("goal_accept_timeout_sec", self.GOAL_ACCEPT_TIMEOUT_SEC) if isinstance(yaml_cfg, dict) else self.GOAL_ACCEPT_TIMEOUT_SEC))
            resend_default = (
                yaml_cfg.get("goal_resend_sec", self.GOAL_RESEND_SEC)
                if isinstance(yaml_cfg, dict) else self.GOAL_RESEND_SEC
            )
            self._goal_resend_sec = float(kwargs.get("goal_resend_sec", resend_default))
            self._odom_reliability = str(kwargs.get("odom_reliability", yaml_cfg.get("odom_reliability", "best_effort") if isinstance(yaml_cfg, dict) else "best_effort")).lower()

            # Feedback hook for RT-08
            self._feedback_handler: Any = kwargs.get("feedback_handler")
            self._floor_maps: dict[str, str] = {}
            self._current_floor: str | None = None
            self._compat = _ros_compat_layer()

            # Health monitoring (created/started in connect())
            self._health_monitor: ConnectionHealth | None = kwargs.get("health_monitor")
            self._auto_health: bool = bool(kwargs.get("auto_health", True))

            # --- Publishers ---
            self._cmd_vel_pub = self._node.create_publisher(
                Twist, self._config["cmd_vel"], 10,
            )

            # --- State ---
            self._position = (0.0, 0.0)
            self._orientation = 0.0  # yaw in radians
            self._nav_state = NavState.IDLE
            self._state_machine = RobotState.IDLE
            self._goal_handle: Any = None
            self._has_odom = False
            self._has_nav2 = False
            self._position_frame = "odom"
            self._map_frame_logged: bool | None = None
            self._tf_buffer: Any = None
            self._tf_listener: Any = None
            self._map_tf_wait_sec = float(kwargs.get("map_tf_wait_sec", self.MAP_TF_WAIT_SEC))
            if _HAS_TF2:
                try:
                    self._tf_buffer = _TfBuffer()
                    self._tf_listener = _TfListener(self._tf_buffer, self._node)
                except Exception as e:
                    self._tf_buffer = None
                    logger.warning("Could not listen to TF, positions stay on /odom: %s", e)

            # --- Odometry subscriber ---
            self._odom_sub = self._node.create_subscription(
                Odometry,
                self._config["odom"],
                self._odom_callback,
                self._odom_qos(),
                callback_group=self._cb_group,
            )

            # --- Nav2 action client ---
            try:
                self._nav2_client = ActionClient(
                    self._node,
                    NavigateToPose,
                    self._config["nav2_action"],
                    callback_group=self._cb_group,
                )
                # Wait briefly for Nav2 to appear
                self._has_nav2 = self._nav2_client.wait_for_server(timeout_sec=self._nav2_server_wait_sec)
                if self._has_nav2:
                    logger.info("Nav2 action server found")
                else:
                    logger.warning("Nav2 action server not found — using cmd_vel fallback")
            except Exception as e:
                logger.warning("Could not create Nav2 action client: %s", e)
                self._nav2_client = None
                self._has_nav2 = False

            # --- Nav2 planner client, used to check a route before driving it ---
            self._plan_client: Any = None
            if self._has_nav2:
                try:
                    self._plan_client = ActionClient(
                        self._node,
                        ComputePathToPose,
                        self._config["nav2_plan_action"],
                        callback_group=self._cb_group,
                    )
                except Exception as e:
                    logger.warning("Could not create Nav2 planner client: %s", e)

            # Wait for first odometry message
            t0 = time.time()
            while not self._has_odom and (time.time() - t0) < self._odom_wait_sec:
                time.sleep(0.1)
            if self._has_odom:
                logger.info("Odometry online — position: (%.2f, %.2f)", *self._position)
            else:
                logger.warning("No odometry received within %.1fs — position may be stale", self._odom_wait_sec)

            # Goals and zones are in map, so give localisation a moment to
            # answer before the first position is read.
            if self._has_nav2 and self._has_odom and self._tf_buffer is not None:
                t0 = time.time()
                while self._position_frame != "map" and (time.time() - t0) < self._map_tf_wait_sec:
                    time.sleep(0.1)

            logger.info(
                "ROS2Adapter ready for %s [nav2=%s, odom=%s, pos=(%.1f, %.1f)]",
                robot_name, self._has_nav2, self._has_odom, *self._position,
            )

        # ==================================================================
        # Odometry
        # ==================================================================

        def _odom_callback(self, msg: Odometry) -> None:
            """Update the pose: in map when TF can give it, else from odometry."""
            map_pose = self._lookup_map_pose(msg)
            if map_pose is not None:
                self._position, self._orientation = map_pose
                self._position_frame = "map"
            else:
                pos = msg.pose.pose.position
                self._position = (pos.x, pos.y)
                self._orientation = self._quat_to_yaw(msg.pose.pose.orientation)
                self._position_frame = "odom"
            self._log_position_frame()
            self._has_odom = True
            if self._health_monitor is not None:
                self._health_monitor.record_odom()

        def _lookup_map_pose(self, msg: Odometry) -> tuple[tuple[float, float], float] | None:
            """The map-to-base transform, or None if there is no fresh one."""
            if self._tf_buffer is None:
                return None
            base_frame = msg.child_frame_id
            if not isinstance(base_frame, str) or not base_frame:
                base_frame = "base_link"
            try:
                tf = self._tf_buffer.lookup_transform("map", base_frame, _RosTime())
                tf_time = self._stamp_sec(tf.header.stamp)
                odom_time = self._stamp_sec(msg.header.stamp)
                # A stamp of 0 is a static transform, which never goes stale.
                if tf_time and odom_time - tf_time > self.MAP_TF_MAX_AGE_SEC:
                    return None
                t = tf.transform.translation
                return (float(t.x), float(t.y)), self._quat_to_yaw(tf.transform.rotation)
            except Exception:
                # tf2's lookup errors: no map frame yet, or never will be.
                return None

        @staticmethod
        def _stamp_sec(stamp: Any) -> float:
            return float(stamp.sec) + float(stamp.nanosec) * 1e-9

        def _log_position_frame(self) -> None:
            """One line each time the source of the position changes."""
            in_map = self._position_frame == "map"
            if in_map == self._map_frame_logged:
                return
            if in_map:
                logger.info("Position now comes from the map-to-base transform")
            else:
                logger.warning(
                    "No map-to-base transform: position comes from /odom, which "
                    "can differ from the map frame that goals and zones use"
                )
            self._map_frame_logged = in_map

        @staticmethod
        def _quat_to_yaw(q: Quaternion) -> float:
            """Extract yaw from quaternion (avoids tf_transformations dependency)."""
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            return math.atan2(siny_cosp, cosy_cosp)

        @staticmethod
        def _yaw_to_quat(yaw: float) -> Quaternion:
            """Convert yaw angle to quaternion."""
            q = Quaternion()
            q.w = math.cos(yaw / 2.0)
            q.x = 0.0
            q.y = 0.0
            q.z = math.sin(yaw / 2.0)
            return q

        def _odom_qos(self) -> Any:
            """RT-03: configurable odometry reliability to avoid silent QoS mismatch."""
            reliability = (
                ReliabilityPolicy.RELIABLE
                if self._odom_reliability == "reliable"
                else ReliabilityPolicy.BEST_EFFORT
            )
            return QoSProfile(
                reliability=reliability,
                durability=DurabilityPolicy.VOLATILE,
                depth=5,
            )

        def get_pose(self) -> tuple[float, float, float]:
            """RT-04: return current pose as (x, y, yaw)."""
            return (self._position[0], self._position[1], self._orientation)

        # ==================================================================
        # Capability discovery
        # ==================================================================

        def get_capabilities(self) -> RobotCapability:
            """
            Introspect live ROS 2 topics and build the capability profile.

            This is called once and cached by Robot.capabilities().
            """
            topic_list = self._node.get_topic_names_and_types()
            topic_map = {name: types for name, types in topic_list}
            topic_names = set(topic_map.keys())

            capabilities = []
            sensors = []

            # --- Navigation ---
            if self._has_nav2 or self._config["cmd_vel"] in topic_names:
                capabilities.append(Capability(
                    capability_type=CapabilityType.NAVIGATE,
                    name="navigate_to",
                    description=(
                        "Navigate to 2D position via Nav2"
                        if self._has_nav2 else
                        "Navigate to 2D position via cmd_vel (proportional)"
                    ),
                    parameters={
                        "x": "float — target X in map frame",
                        "y": "float — target Y in map frame",
                        "speed": "float (optional) — max speed in m/s",
                    },
                ))

            # --- Sensors ---
            sensor_checks = [
                (self._config["scan"], SensorType.LIDAR, "lidar0", 10.0,
                 ["sensor_msgs/msg/LaserScan"]),
                (self._config["camera"], SensorType.CAMERA, "camera0", 30.0,
                 ["sensor_msgs/msg/Image"]),
                (self._config.get("depth", ""), SensorType.DEPTH, "depth0", 15.0,
                 ["sensor_msgs/msg/Image"]),
                (self._config["imu"], SensorType.IMU, "imu0", 100.0,
                 ["sensor_msgs/msg/Imu"]),
            ]
            for topic, stype, sid, hz, expected_types in sensor_checks:
                if topic in topic_names:
                    sensors.append(SensorInfo(
                        sensor_id=sid,
                        sensor_type=stype,
                        topic=topic,
                        hz=hz,
                    ))

            # --- Service-based capabilities (future) ---
            # Check for MoveIt action servers → CapabilityType.PICK, PLACE, MANIPULATE

            return RobotCapability(
                robot_id=self.robot_name,
                name=f"ROS2-{self.robot_name}",
                capabilities=capabilities,
                sensors=sensors,
                max_speed=1.0,
                metadata={
                    "adapter": "ros2",
                    "nav2_available": self._has_nav2,
                    "topics_found": len(topic_names),
                    "odom_online": self._has_odom,
                    "position_frame": self._position_frame,
                    "namespace": self._namespace,
                    "odom_reliability": self._odom_reliability,
                    "ros_compat": self._compat,
                },
            )

        # ==================================================================
        # Navigation — Nav2 (preferred)
        # ==================================================================

        @property
        def plans_own_path(self) -> bool:
            """Nav2 picks the route; the cmd_vel fallback drives straight."""
            return self._has_nav2

        def plan_path(self, x: float, y: float) -> list[tuple[float, float]]:
            """
            Ask Nav2 for its route to (x, y) without moving, as (x, y) map points.

            Raises Ros2NavigationError when no route comes back: planner not
            running, goal rejected, planning failed or timed out. Nav2 may
            still drive a different path than this one (controller, replans).
            """
            client = getattr(self, "_plan_client", None)
            if client is None or not client.wait_for_server(
                timeout_sec=self._nav2_server_wait_sec
            ):
                raise Ros2NavigationError("Nav2 planner is not available", NavState.FAILED)

            goal_msg = ComputePathToPose.Goal()
            goal_msg.goal = PoseStamped()
            goal_msg.goal.header.frame_id = "map"
            goal_msg.goal.header.stamp = self._node.get_clock().now().to_msg()
            goal_msg.goal.pose.position.x = x
            goal_msg.goal.pose.position.y = y
            goal_msg.goal.pose.orientation = self._yaw_to_quat(0.0)
            goal_msg.use_start = False  # plan from the robot's current pose

            deadline = time.time() + self._plan_timeout_sec
            handle = self._send_plan_goal(client, goal_msg, deadline)
            if not handle.accepted:
                raise Ros2NavigationError("Nav2 planner rejected the goal", NavState.FAILED)
            response = self._wait_for_plan(handle.get_result_async(), deadline)
            if response.status != GoalStatus.STATUS_SUCCEEDED:
                raise Ros2NavigationError(
                    f"Nav2 found no route to ({x:.2f}, {y:.2f}) (status={response.status})",
                    NavState.FAILED,
                )
            return [
                (p.pose.position.x, p.pose.position.y)
                for p in response.result.path.poses
            ]

        def _send_plan_goal(self, client: Any, goal_msg: Any, deadline: float) -> Any:
            """
            Send the plan goal, resending it while no answer comes back.

            The first goal from a fresh process can lose its answer (see
            GOAL_RESEND_SEC). A plan moves nothing, so sending it again is
            safe; whichever send is answered first gives the handle.
            """
            resend_sec = getattr(self, "_goal_resend_sec", self.GOAL_RESEND_SEC)
            sent: list[Any] = []
            while True:
                sent.append(client.send_goal_async(goal_msg))
                resend_at = time.time() + resend_sec if resend_sec > 0 else deadline
                give_up_at = min(resend_at, deadline)
                while time.time() <= give_up_at:
                    for future in sent:
                        if future.done():
                            return future.result()
                    time.sleep(0.02)
                if give_up_at >= deadline:
                    raise Ros2NavigationError(
                        "Nav2 planner did not answer in time", NavState.TIMED_OUT
                    )

        def _wait_for_plan(self, future: Any, deadline: float) -> Any:
            """Block on a planner future; the node's executor completes it."""
            while not future.done():
                if time.time() > deadline:
                    raise Ros2NavigationError(
                        "Nav2 planner did not answer in time", NavState.TIMED_OUT
                    )
                time.sleep(0.02)
            return future.result()

        def move(self, x: float, y: float, speed: float | None = None) -> None:
            """
            Navigate to (x, y) using Nav2 if available, else cmd_vel.

            This is a blocking call: it returns once the goal is reached and
            raises Ros2NavigationError when it is not (goal rejected, aborted
            or never answered by Nav2, navigation timed out, or cancelled by
            stop()/emergency_stop() before it finished). For the
            executor/agent, the blocking behavior is intentional: skills
            execute sequentially.
            """
            if self._state_machine == RobotState.E_STOPPED:
                raise RuntimeError("Robot is E_STOPPED; call reset_estop() before moving")
            self._state_machine = RobotState.NAVIGATING
            if self._has_nav2:
                self._move_nav2(x, y, speed)
            else:
                self._move_cmd_vel(x, y, speed)
            if self._state_machine != RobotState.E_STOPPED:
                self._state_machine = RobotState.IDLE
            self._raise_if_goal_not_reached(x, y)

        _GOAL_NOT_REACHED = {
            NavState.FAILED: "was rejected or aborted by Nav2",
            NavState.TIMED_OUT: "timed out (no answer from Nav2, or navigation took too long)",
            NavState.CANCELLED: "was cancelled by stop() before the robot reached it",
        }

        def _raise_if_goal_not_reached(self, x: float, y: float) -> None:
            """Turn a failed, timed-out or cancelled goal into an exception."""
            reason = self._GOAL_NOT_REACHED.get(self._nav_state)
            if reason is None:
                return
            pos = self._position
            raise Ros2NavigationError(
                f"move to ({x:.2f}, {y:.2f}) {reason}; "
                f"robot is at ({pos[0]:.2f}, {pos[1]:.2f})",
                self._nav_state,
            )

        def _move_nav2(self, x: float, y: float, speed: float | None = None) -> None:
            """Send a NavigateToPose goal to Nav2."""
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = PoseStamped()
            goal_msg.pose.header.frame_id = "map"
            goal_msg.pose.header.stamp = self._node.get_clock().now().to_msg()
            goal_msg.pose.pose.position.x = x
            goal_msg.pose.pose.position.y = y
            goal_msg.pose.pose.position.z = 0.0

            # Point toward the goal
            target_yaw = math.atan2(
                y - self._position[1], x - self._position[0]
            )
            goal_msg.pose.pose.orientation = self._yaw_to_quat(target_yaw)

            # TODO: Nav2 doesn't have a direct speed parameter.
            # To control speed, you'd configure Nav2's controller server
            # max_vel parameter dynamically via a ROS 2 parameter service.
            # For now we log the requested speed for future implementation.
            if speed is not None:
                logger.info("Requested speed %.2f m/s (Nav2 speed control pending)", speed)

            self._nav_state = NavState.NAVIGATING
            logger.info(
                "Nav2: sending goal (%.2f, %.2f) yaw=%.1f°",
                x, y, math.degrees(target_yaw),
            )

            # Send the goal and block until it is accepted or rejected. A goal
            # with no answer after goal_resend_sec is sent again (see
            # GOAL_RESEND_SEC); answers to earlier sends are then stale.
            self._goal_handle = None
            resend_sec = getattr(self, "_goal_resend_sec", self.GOAL_RESEND_SEC)
            t0 = time.time()
            self._send_nav2_goal(goal_msg)
            last_send = t0
            while self._nav_state == NavState.NAVIGATING and self._goal_handle is None:
                time.sleep(0.05)
                now = time.time()
                if now - t0 > self._goal_accept_timeout_sec:
                    logger.error("Nav2: goal acceptance timed out")
                    self._nav_state = NavState.TIMED_OUT
                    return
                if resend_sec > 0 and now - last_send >= resend_sec:
                    logger.warning(
                        "Nav2: no answer to the goal after %.1fs, sending it again",
                        now - last_send,
                    )
                    self._send_nav2_goal(goal_msg)
                    last_send = now

            # Now block until navigation completes (with configurable timeout)
            t0 = time.time()
            while self._nav_state == NavState.NAVIGATING:
                time.sleep(0.1)
                if time.time() - t0 > self._nav_timeout_sec:
                    logger.error("Nav2: navigation timed out after %.0fs", self._nav_timeout_sec)
                    self._cancel_nav2()
                    self._nav_state = NavState.TIMED_OUT
                    return

            logger.info("Nav2: navigation finished — state=%s", self._nav_state.value)

        def _send_nav2_goal(self, goal_msg: Any) -> None:
            """Send one NavigateToPose goal; only the latest send's answer counts."""
            self._goal_send_seq = getattr(self, "_goal_send_seq", 0) + 1
            seq = self._goal_send_seq
            send_future = self._nav2_client.send_goal_async(
                goal_msg,
                feedback_callback=self._nav2_feedback,
            )
            send_future.add_done_callback(
                lambda f, seq=seq: self._nav2_goal_response(f, seq)
            )

        def _nav2_goal_response(self, future: Any, seq: int | None = None) -> None:
            """Callback when Nav2 accepts or rejects the goal."""
            if seq is not None and seq != getattr(self, "_goal_send_seq", seq):
                # Answer to a send that was superseded by a resend of the same
                # goal: cancel it if accepted so only the latest one runs.
                stale = future.result()
                if stale is not None and getattr(stale, "accepted", False):
                    logger.info("Nav2: cancelling a superseded copy of the goal")
                    stale.cancel_goal_async()
                return
            self._goal_handle = future.result()
            if not self._goal_handle.accepted:
                logger.error("Nav2: goal REJECTED")
                self._nav_state = NavState.FAILED
                return

            logger.info("Nav2: goal accepted — tracking")
            result_future = self._goal_handle.get_result_async()
            result_future.add_done_callback(self._nav2_result)

        def _nav2_feedback(self, feedback_msg: Any) -> None:
            """Callback for Nav2 progress feedback."""
            fb = feedback_msg.feedback
            pos = fb.current_pose.pose.position
            self._position = (pos.x, pos.y)
            self._orientation = self._quat_to_yaw(fb.current_pose.pose.orientation)
            # Nav2 reports in its global frame, normally map.
            frame = fb.current_pose.header.frame_id
            if isinstance(frame, str) and frame:
                self._position_frame = frame
            # distance_remaining is available in nav2_msgs Feedback
            dist = getattr(fb, "distance_remaining", None)
            if dist is not None:
                logger.debug("Nav2 feedback: pos=(%.2f, %.2f) remaining=%.2fm",
                             pos.x, pos.y, dist)
            if self._feedback_handler is not None:
                try:
                    self._feedback_handler({
                        "event": "nav2_feedback",
                        "position": (pos.x, pos.y),
                        "orientation": self._orientation,
                        "distance_remaining": float(dist) if dist is not None else None,
                        "state": self._nav_state.value,
                    })
                except Exception as e:
                    logger.warning("Nav2 feedback handler error: %s", e)

        def _nav2_result(self, future: Any) -> None:
            """Callback when Nav2 finishes the goal."""
            result = future.result()
            status = result.status

            if status == GoalStatus.STATUS_SUCCEEDED:
                self._nav_state = NavState.SUCCEEDED
                logger.info("Nav2: goal SUCCEEDED")
            elif status == GoalStatus.STATUS_CANCELED:
                self._nav_state = NavState.CANCELLED
                logger.info("Nav2: goal CANCELLED")
            else:
                self._nav_state = NavState.FAILED
                logger.error("Nav2: goal FAILED (status=%d)", status)

        def _cancel_nav2(self) -> None:
            """Cancel the current Nav2 goal."""
            if self._goal_handle is not None:
                logger.info("Nav2: cancelling current goal")
                cancel_future = self._goal_handle.cancel_goal_async()
                # Wait briefly for cancel confirmation
                t0 = time.time()
                while not cancel_future.done() and time.time() - t0 < 5.0:
                    time.sleep(0.05)

        # ==================================================================
        # Navigation — cmd_vel fallback (proportional controller)
        # ==================================================================

        def _move_cmd_vel(self, x: float, y: float, speed: float | None = None) -> None:
            """
            Proportional controller via cmd_vel — used when Nav2 is unavailable.

            Runs a control loop at CMD_VEL_HZ until within GOAL_TOLERANCE_M
            or timeout.
            """
            speed = speed or 0.5
            self._nav_state = NavState.NAVIGATING
            dt = 1.0 / self.CMD_VEL_HZ
            t0 = time.time()

            controller_mode = self._config.get("cmd_vel_controller", "pure_pursuit")
            logger.info("cmd_vel: driving to (%.2f, %.2f) speed=%.2f mode=%s", x, y, speed, controller_mode)

            while time.time() - t0 < self._nav_timeout_sec:
                dx = x - self._position[0]
                dy = y - self._position[1]
                distance = math.sqrt(dx * dx + dy * dy)

                if distance < self.GOAL_TOLERANCE_M:
                    self._publish_stop()
                    self._nav_state = NavState.SUCCEEDED
                    logger.info("cmd_vel: reached goal (dist=%.2fm)", distance)
                    return

                target_yaw = math.atan2(dy, dx)
                yaw_error = self._normalize_angle(target_yaw - self._orientation)

                twist = Twist()
                if controller_mode == "dwa":
                    # Lightweight DWA-like fallback: favor heading alignment before translation.
                    if abs(yaw_error) > 0.7:
                        twist.linear.x = 0.0
                    else:
                        twist.linear.x = min(speed, distance * self.PROPORTIONAL_GAIN_LINEAR)
                    twist.angular.z = max(-1.5, min(1.5, yaw_error * self.PROPORTIONAL_GAIN_ANGULAR))
                else:
                    # Pure-pursuit style proportional control on heading+distance.
                    lookahead = max(0.3, min(1.5, distance))
                    curvature = (2.0 * math.sin(yaw_error)) / lookahead
                    twist.linear.x = min(speed, distance * self.PROPORTIONAL_GAIN_LINEAR)
                    if abs(yaw_error) > 0.6:
                        twist.linear.x *= 0.25
                    twist.angular.z = max(-1.5, min(1.5, twist.linear.x * curvature * self.PROPORTIONAL_GAIN_ANGULAR))

                self._cmd_vel_pub.publish(twist)
                time.sleep(dt)

            # Timeout
            self._publish_stop()
            self._nav_state = NavState.TIMED_OUT
            logger.error("cmd_vel: timed out after %.0fs", self._nav_timeout_sec)

        # ==================================================================
        # Stop
        # ==================================================================

        def stop(self) -> None:
            """Immediately halt all motion — cancels Nav2 goal if active."""
            if self._nav_state == NavState.NAVIGATING:
                if self._has_nav2 and self._goal_handle is not None:
                    self._cancel_nav2()
                self._nav_state = NavState.CANCELLED

            self._publish_stop()
            if self._state_machine != RobotState.E_STOPPED:
                self._state_machine = RobotState.IDLE
            logger.info("ROS2Adapter: STOPPED")

        def _publish_stop(self) -> None:
            """Publish zero velocity on cmd_vel."""
            self._cmd_vel_pub.publish(Twist())

        def emergency_stop(self) -> None:
            """RT-06: enter E_STOPPED state and publish zero velocity."""
            self._state_machine = RobotState.E_STOPPED
            self._publish_stop()
            if self._has_nav2 and self._goal_handle is not None:
                self._cancel_nav2()

        def reset_estop(self) -> None:
            """Leave E_STOPPED and return to IDLE."""
            if self._state_machine == RobotState.E_STOPPED:
                self._state_machine = RobotState.IDLE

        # ==================================================================
        # Lifecycle — health monitor wiring
        # ==================================================================

        def connect(self) -> None:
            """Establish connection and start the connection health monitor."""
            super().connect()
            if self._health_monitor is None and self._auto_health:
                self._health_monitor = ConnectionHealth(self)
            if self._health_monitor is not None:
                self._health_monitor.start()

        def disconnect(self) -> None:
            """Disconnect and stop health monitoring."""
            if self._health_monitor is not None:
                self._health_monitor.stop()
            super().disconnect()

        @property
        def health(self) -> ConnectionHealth | None:
            """The ConnectionHealth instance, or None if connect() has not been called."""
            return self._health_monitor

        def set_feedback_handler(self, handler: Any) -> None:
            """RT-08: register callback for Nav2 progress payloads."""
            self._feedback_handler = handler

        def trigger_slam(self, mode: str = "online_async") -> bool:
            """RT-05: trigger slam_toolbox launch via Python API wrapper."""
            cmd = ["ros2", "launch", "slam_toolbox", "online_async_launch.py", f"slam_params_file:={mode}"]
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logger.info("Triggered slam_toolbox launch: %s", " ".join(cmd))
                return True
            except Exception as e:
                logger.warning("Failed to trigger slam_toolbox: %s", e)
                return False

        def switch_floor_map(self, floor_id: str, map_yaml: str) -> bool:
            """RT-09: load/switch map yaml for a floor via map_server service CLI."""
            cmd = ["ros2", "service", "call", "/map_server/load_map", "nav2_msgs/srv/LoadMap", f"{{map_url: '{map_yaml}'}}"]
            try:
                subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._floor_maps[floor_id] = map_yaml
                self._current_floor = floor_id
                return True
            except Exception as e:
                logger.warning("Failed to switch floor map: %s", e)
                return False

        # ==================================================================
        # Helpers
        # ==================================================================

        @staticmethod
        def _normalize_angle(angle: float) -> float:
            """Normalize angle to [-pi, pi]."""
            while angle > math.pi:
                angle -= 2 * math.pi
            while angle < -math.pi:
                angle += 2 * math.pi
            return angle

        @property
        def position(self) -> tuple[float, float]:
            return self._position

        @property
        def orientation(self) -> float:
            return self._orientation

        @property
        def position_frame(self) -> str:
            """The frame position is in: "map", or "odom" as the fallback."""
            return self._position_frame

        # Robot.get_position()/get_orientation() and the SafetyEnforcer read
        # these; without them the base class answers (0, 0) and 0 rad, so the
        # odometry watchdog saw a robot that never left the origin.
        def get_position(self) -> tuple[float, float]:
            return self._position

        def get_orientation(self) -> float:
            return self._orientation

        @property
        def nav_state(self) -> NavState:
            return self._nav_state

        @property
        def is_moving(self) -> bool:
            return self._nav_state == NavState.NAVIGATING

        @property
        def robot_state(self) -> RobotState:
            return self._state_machine
