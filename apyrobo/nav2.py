"""ROS 2 Nav2 navigation adapter for apyrobo.

When rclpy is available this adapter uses the real NavigateToPose action
client and subscribes to /odom for live pose tracking.  Without rclpy it
operates in stub mode: navigate_to() resolves immediately with success and
the pose is updated in memory.

The interface matches what tests/integration/fake_turtlebot4.py serves:
  - action: navigate_to_pose  (nav2_msgs/action/NavigateToPose)
  - odom:   /odom             (nav_msgs/msg/Odometry, BEST_EFFORT QoS)
"""
from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import rclpy  # type: ignore
    _RCLPY_AVAILABLE = True
except ImportError:
    rclpy = None  # type: ignore
    _RCLPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Nav2Config:
    ros_namespace: str = "/"
    goal_tolerance: float = 0.1
    planner: str = "NavfnPlanner"
    controller: str = "DWBLocalPlanner"
    timeout_s: float = 30.0
    odom_topic: str = "/odom"
    action_name: str = "navigate_to_pose"


@dataclass
class NavigationGoal:
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0
    frame_id: str = "map"


@dataclass
class NavigationResult:
    success: bool
    final_pose: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    message: str = ""


# ---------------------------------------------------------------------------
# Nav2Adapter
# ---------------------------------------------------------------------------

class Nav2Adapter:
    """
    Nav2 navigation adapter.

    When rclpy is installed:
      - ``connect()`` initialises rclpy, creates a node, and subscribes to
        *odom_topic* for continuous pose tracking.
      - ``navigate_to()`` sends a NavigateToPose goal and blocks until the
        action server returns success or failure (or times out).
      - ``cancel_navigation()`` cancels the active goal.
      - ``get_position()`` returns the latest (x, y, yaw) from /odom.

    Without rclpy the adapter falls back to stub mode and resolves all goals
    immediately so the rest of the stack can be developed and tested offline.
    """

    def __init__(self, config: Nav2Config | None = None) -> None:
        self.config = config or Nav2Config()
        self._connected = False
        self._navigating = False
        self._current_pose: dict[str, float] = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self._node: Any = None
        self._executor: Any = None
        self._spin_thread: threading.Thread | None = None
        self._odom_sub: Any = None
        self._goal_handle: Any = None
        self._stub_mode = not _RCLPY_AVAILABLE

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        if self._stub_mode:
            logger.warning("rclpy not available — Nav2Adapter running in stub mode")
            self._connected = True
            return
        try:
            if not rclpy.ok():
                rclpy.init()
            self._node = rclpy.create_node("apyrobo_nav2")

            # Subscribe to /odom for live pose tracking
            from nav_msgs.msg import Odometry  # type: ignore
            from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy  # type: ignore

            odom_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                depth=5,
            )
            self._odom_sub = self._node.create_subscription(
                Odometry,
                self.config.odom_topic,
                self._odom_callback,
                odom_qos,
            )

            # Background executor so callbacks fire without blocking navigate_to()
            self._executor = rclpy.executors.MultiThreadedExecutor()
            self._executor.add_node(self._node)
            self._spin_thread = threading.Thread(
                target=self._executor.spin, daemon=True
            )
            self._spin_thread.start()

            self._connected = True
            logger.info("Nav2Adapter connected (namespace=%s)", self.config.ros_namespace)
        except Exception as exc:
            logger.warning("Nav2 connect failed (%s) — stub mode", exc)
            self._stub_mode = True
            self._connected = True

    async def disconnect(self) -> None:
        self._navigating = False
        if self._executor is not None:
            try:
                self._executor.shutdown(timeout_sec=2)
            except Exception:
                pass
            self._executor = None
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        if _RCLPY_AVAILABLE and not self._stub_mode:
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass
        self._connected = False

    # ------------------------------------------------------------------
    # Odometry callback (runs in background spin thread)
    # ------------------------------------------------------------------

    def _odom_callback(self, msg: Any) -> None:
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self._current_pose = {"x": pos.x, "y": pos.y, "yaw": yaw}

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    async def navigate_to(self, goal: NavigationGoal) -> NavigationResult:
        if not self._connected:
            return NavigationResult(success=False, message="not connected")

        self._navigating = True
        start = time.monotonic()
        try:
            if self._stub_mode:
                await asyncio.sleep(0.05)
                self._current_pose = {"x": goal.x, "y": goal.y, "yaw": goal.yaw}
                return NavigationResult(
                    success=True,
                    final_pose=dict(self._current_pose),
                    elapsed_s=time.monotonic() - start,
                    message="stub navigation complete",
                )

            # Real implementation: delegate to blocking helper in a thread
            # so we don't block the asyncio event loop.
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._navigate_sync, goal)
            if result.success:
                self._current_pose = result.final_pose
            return result
        finally:
            self._navigating = False

    def _navigate_sync(self, goal: NavigationGoal) -> NavigationResult:
        """Blocking navigation via rclpy NavigateToPose action client."""
        start = time.monotonic()
        try:
            from rclpy.action import ActionClient  # type: ignore
            from nav2_msgs.action import NavigateToPose  # type: ignore
            from geometry_msgs.msg import PoseStamped  # type: ignore
        except ImportError as exc:
            return NavigationResult(success=False, message=f"missing ROS packages: {exc}")

        client = ActionClient(self._node, NavigateToPose, self.config.action_name)
        if not client.wait_for_server(timeout_sec=min(10.0, self.config.timeout_s)):
            return NavigationResult(
                success=False, message="Nav2 action server not available"
            )

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = goal.frame_id
        goal_msg.pose.header.stamp = self._node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(goal.x)
        goal_msg.pose.pose.position.y = float(goal.y)
        goal_msg.pose.pose.position.z = float(goal.z)
        goal_msg.pose.pose.orientation.w = math.cos(goal.yaw / 2.0)
        goal_msg.pose.pose.orientation.z = math.sin(goal.yaw / 2.0)

        done_event = threading.Event()
        result_holder: dict[str, Any] = {}

        def _goal_response(future: Any) -> None:
            handle = future.result()
            if not handle.accepted:
                result_holder["nav"] = NavigationResult(
                    success=False, message="goal rejected by Nav2"
                )
                done_event.set()
                return
            self._goal_handle = handle
            handle.get_result_async().add_done_callback(_result_callback)

        def _result_callback(future: Any) -> None:
            from action_msgs.msg import GoalStatus  # type: ignore
            res = future.result()
            success = res.status == GoalStatus.STATUS_SUCCEEDED
            result_holder["nav"] = NavigationResult(
                success=success,
                final_pose={"x": goal.x, "y": goal.y, "yaw": goal.yaw},
                elapsed_s=time.monotonic() - start,
                message="succeeded" if success else f"failed (status={res.status})",
            )
            done_event.set()

        client.send_goal_async(goal_msg).add_done_callback(_goal_response)

        if not done_event.wait(timeout=self.config.timeout_s):
            return NavigationResult(
                success=False,
                elapsed_s=time.monotonic() - start,
                message="navigation timed out",
            )

        return result_holder.get(
            "nav", NavigationResult(success=False, message="unknown error")
        )

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def cancel_navigation(self) -> None:
        self._navigating = False
        if self._goal_handle is not None:
            try:
                cancel_future = self._goal_handle.cancel_goal_async()
                deadline = time.monotonic() + 5.0
                while not cancel_future.done() and time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
            except Exception as exc:
                logger.warning("Nav2Adapter: cancel failed: %s", exc)
            self._goal_handle = None
        logger.info("Nav2Adapter: navigation cancelled")

    # ------------------------------------------------------------------
    # Pose queries
    # ------------------------------------------------------------------

    def get_current_pose(self) -> dict[str, float]:
        """Return latest pose as ``{"x": float, "y": float, "yaw": float}``."""
        return dict(self._current_pose)

    def get_position(self) -> tuple[float, float]:
        """Return ``(x, y)`` from the latest odometry message."""
        return (self._current_pose["x"], self._current_pose["y"])

    def is_navigating(self) -> bool:
        return self._navigating

    def set_initial_pose(self, x: float, y: float, yaw: float = 0.0) -> None:
        self._current_pose = {"x": x, "y": y, "yaw": yaw}
        logger.info(
            "Nav2Adapter: initial pose set to (%.2f, %.2f, yaw=%.2f)", x, y, yaw
        )


# ---------------------------------------------------------------------------
# MockNav2Adapter — stub for tests
# ---------------------------------------------------------------------------

class MockNav2Adapter(Nav2Adapter):
    """
    Stub Nav2 adapter for tests and offline development.

    ``navigate_to()`` updates ``_current_pose`` immediately and returns
    success without touching ROS or the network.
    """

    def __init__(self, config: Nav2Config | None = None) -> None:
        super().__init__(config)
        self._stub_mode = True

    async def connect(self) -> None:
        self._connected = True
        logger.debug("MockNav2Adapter connected")

    async def disconnect(self) -> None:
        self._navigating = False
        self._connected = False
        logger.debug("MockNav2Adapter disconnected")
