"""ROS 2 Nav2 navigation adapter for apyrobo."""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    import rclpy  # type: ignore
    from rclpy.action import ActionClient as RclpyActionClient  # type: ignore
    _RCLPY_AVAILABLE = True
except ImportError:
    rclpy = None  # type: ignore
    RclpyActionClient = None  # type: ignore
    _RCLPY_AVAILABLE = False


@dataclass
class Nav2Config:
    ros_namespace: str = "/"
    goal_tolerance: float = 0.1
    planner: str = "NavfnPlanner"
    controller: str = "DWBLocalPlanner"
    timeout_s: float = 30.0


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


class Nav2Adapter:
    """Nav2 navigation adapter.  Falls back to stub when rclpy unavailable."""

    def __init__(self, config: Nav2Config | None = None) -> None:
        self.config = config or Nav2Config()
        self._connected = False
        self._navigating = False
        self._current_pose: dict = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self._node: Any = None
        self._stub_mode = not _RCLPY_AVAILABLE
        self._goal_handle: Any = None
        self._odom_sub: Any = None

    async def connect(self) -> None:
        if self._stub_mode:
            logger.warning("rclpy not available — Nav2Adapter running in stub mode")
            self._connected = True
            return
        try:
            rclpy.init()
            self._node = rclpy.create_node("apyrobo_nav2")
            self._setup_odom_subscription()
            self._connected = True
            logger.info("Nav2Adapter connected (namespace=%s)", self.config.ros_namespace)
        except Exception as exc:
            logger.warning("Nav2 connect failed (%s) — stub mode", exc)
            self._stub_mode = True
            self._connected = True

    def _setup_odom_subscription(self) -> None:
        """Subscribe to /odom to track current robot pose."""
        try:
            from nav_msgs.msg import Odometry  # type: ignore
            self._odom_sub = self._node.create_subscription(
                Odometry,
                "/odom",
                self._odom_callback,
                10,
            )
        except ImportError:
            logger.warning("nav_msgs not available — odom tracking disabled")

    def _odom_callback(self, msg: Any) -> None:
        """Update current pose from /odom topic."""
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self._current_pose = {"x": pos.x, "y": pos.y, "yaw": yaw}

    async def disconnect(self) -> None:
        self._navigating = False
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        if _RCLPY_AVAILABLE and not self._stub_mode:
            try:
                rclpy.shutdown()
            except Exception:
                pass
        self._connected = False

    async def navigate_to(self, goal: NavigationGoal) -> NavigationResult:
        if not self._connected:
            return NavigationResult(success=False, message="not connected")
        self._navigating = True
        start = time.monotonic()
        try:
            if self._stub_mode:
                await asyncio.sleep(0.05)
                self._current_pose = {"x": goal.x, "y": goal.y, "yaw": goal.yaw}
                elapsed = time.monotonic() - start
                return NavigationResult(
                    success=True,
                    final_pose=dict(self._current_pose),
                    elapsed_s=elapsed,
                    message="stub navigation complete",
                )
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._blocking_navigate, goal)
            return result
        finally:
            self._navigating = False

    def _blocking_navigate(self, goal: NavigationGoal) -> NavigationResult:
        """Blocking Nav2 NavigateToPose call — runs in thread executor."""
        from nav2_msgs.action import NavigateToPose  # type: ignore
        from geometry_msgs.msg import PoseStamped, Quaternion  # type: ignore

        start = time.monotonic()
        action_client = RclpyActionClient(self._node, NavigateToPose, "navigate_to_pose")
        if not action_client.wait_for_server(timeout_sec=self.config.timeout_s):
            action_client.destroy()
            return NavigationResult(
                success=False,
                message="navigate_to_pose action server not available",
            )

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = goal.frame_id
        goal_msg.pose.header.stamp = self._node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = goal.x
        goal_msg.pose.pose.position.y = goal.y
        goal_msg.pose.pose.position.z = goal.z
        qz = math.sin(goal.yaw / 2.0)
        qw = math.cos(goal.yaw / 2.0)
        goal_msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)

        send_future = action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self._node, send_future)
        goal_handle = send_future.result()

        if not goal_handle.accepted:
            action_client.destroy()
            return NavigationResult(success=False, message="goal rejected by Nav2")

        self._goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self._node, result_future)

        elapsed = time.monotonic() - start
        action_client.destroy()
        self._goal_handle = None
        return NavigationResult(
            success=True,
            final_pose=dict(self._current_pose),
            elapsed_s=elapsed,
            message="navigation complete",
        )

    async def cancel_navigation(self) -> None:
        self._navigating = False
        if self._goal_handle is not None and not self._stub_mode:
            try:
                loop = asyncio.get_event_loop()
                goal_handle = self._goal_handle

                def _cancel() -> None:
                    future = goal_handle.cancel_goal_async()
                    rclpy.spin_until_future_complete(self._node, future)

                await loop.run_in_executor(None, _cancel)
            except Exception as exc:
                logger.warning("Nav2Adapter: cancel failed: %s", exc)
            self._goal_handle = None
        logger.info("Nav2Adapter: navigation cancelled")

    def get_current_pose(self) -> dict:
        return dict(self._current_pose)

    def is_navigating(self) -> bool:
        return self._navigating

    def set_initial_pose(self, x: float, y: float, yaw: float = 0.0) -> None:
        self._current_pose = {"x": x, "y": y, "yaw": yaw}
        logger.info("Nav2Adapter: initial pose set to (%.2f, %.2f, %.2f)", x, y, yaw)


class MockNav2Adapter(Nav2Adapter):
    """Stub Nav2 adapter for tests — simulates navigation with asyncio.sleep."""

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
