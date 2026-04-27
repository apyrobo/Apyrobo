"""
Nav2 adapter — ROS 2 Navigation Stack integration for autonomous navigation.

Provides a backend-agnostic interface to the Nav2 action server with:
    - Nav2Adapter: ABC defining the navigation API
    - RosNav2Adapter: live rclpy-backed adapter (optional dependency)
    - MockNav2Adapter: in-process mock for unit tests
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Nav2Config:
    """Configuration for the Nav2 adapter."""
    namespace: str = ""
    action_server: str = "navigate_to_pose"
    costmap_topic: str = "global_costmap/costmap"
    timeout_s: float = 60.0
    goal_tolerance_m: float = 0.1
    angle_tolerance_rad: float = 0.1


@dataclass
class NavigationGoal:
    """A navigation pose goal in 2D space."""
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0                    # radians
    frame_id: str = "map"
    speed_limit: float | None = None    # m/s; None = use planner default

    def distance_to(self, other: NavigationGoal) -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)


class NavigationStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass
class NavigationResult:
    """Result returned after a navigation attempt."""
    status: NavigationStatus
    goal: NavigationGoal | None = None
    message: str = ""
    time_taken_s: float = 0.0
    distance_traveled_m: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == NavigationStatus.SUCCESS


# ---------------------------------------------------------------------------
# Abstract adapter
# ---------------------------------------------------------------------------

class Nav2Adapter:
    """
    Abstract interface to the ROS 2 Nav2 navigation stack.

    Concrete implementations provide the transport (rclpy, mock, etc.).
    All methods are synchronous from the caller's perspective; async
    Nav2 actions are awaited internally.
    """

    def navigate_to(
        self,
        goal: NavigationGoal,
        timeout_s: float | None = None,
    ) -> NavigationResult:
        """Send a navigation goal and block until the robot reaches it or fails."""
        raise NotImplementedError

    def cancel(self) -> bool:
        """Cancel the current navigation goal. Returns True if a goal was active."""
        raise NotImplementedError

    def get_current_pose(self) -> NavigationGoal | None:
        """Return the robot's current pose, or None if unavailable."""
        raise NotImplementedError

    def is_navigating(self) -> bool:
        """Return True if a navigation goal is currently active."""
        raise NotImplementedError

    def navigate_through(
        self,
        waypoints: list[NavigationGoal],
        timeout_s: float | None = None,
    ) -> list[NavigationResult]:
        """Navigate through a sequence of waypoints, returning a result per waypoint."""
        results = []
        for wp in waypoints:
            result = self.navigate_to(wp, timeout_s=timeout_s)
            results.append(result)
            if not result.success:
                break
        return results


# ---------------------------------------------------------------------------
# RosNav2Adapter — live rclpy implementation (optional dependency)
# ---------------------------------------------------------------------------

class RosNav2Adapter(Nav2Adapter):
    """
    Nav2 adapter backed by rclpy action client.

    Requires: ros2, nav2_msgs; rclpy must be available on PYTHONPATH.
    Call init_ros() before use if rclpy has not been initialised by the caller.
    """

    def __init__(self, config: Nav2Config | None = None) -> None:
        self.config = config or Nav2Config()
        self._node: Any = None
        self._client: Any = None
        self._active_goal: Any = None

    def init_ros(self, node_name: str = "apyrobo_nav2") -> None:
        try:
            import rclpy
            from rclpy.action import ActionClient
            from nav2_msgs.action import NavigateToPose

            if not rclpy.ok():
                rclpy.init()
            self._node = rclpy.create_node(node_name)
            self._client = ActionClient(
                self._node,
                NavigateToPose,
                self.config.action_server,
            )
        except ImportError as exc:
            raise RuntimeError(
                "rclpy or nav2_msgs not available. "
                "Install ROS 2 and source the workspace."
            ) from exc

    def navigate_to(
        self,
        goal: NavigationGoal,
        timeout_s: float | None = None,
    ) -> NavigationResult:
        if self._node is None:
            raise RuntimeError("Call init_ros() before navigating.")

        from nav2_msgs.action import NavigateToPose
        from geometry_msgs.msg import PoseStamped, Quaternion
        import rclpy

        deadline = time.time() + (timeout_s or self.config.timeout_s)

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = goal.frame_id
        pose_msg.pose.position.x = goal.x
        pose_msg.pose.position.y = goal.y
        pose_msg.pose.position.z = goal.z
        # Convert yaw to quaternion (z-axis rotation)
        pose_msg.pose.orientation = Quaternion(
            x=0.0, y=0.0,
            z=math.sin(goal.yaw / 2),
            w=math.cos(goal.yaw / 2),
        )

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_msg

        t0 = time.time()
        self._client.wait_for_server(timeout_sec=5.0)
        future = self._client.send_goal_async(goal_msg)

        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if future.done():
                break
        else:
            return NavigationResult(
                status=NavigationStatus.TIMED_OUT,
                goal=goal,
                time_taken_s=time.time() - t0,
            )

        handle = future.result()
        if not handle.accepted:
            return NavigationResult(
                status=NavigationStatus.FAILED,
                goal=goal,
                message="Goal rejected by Nav2",
                time_taken_s=time.time() - t0,
            )

        result_future = handle.get_result_async()
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if result_future.done():
                break

        elapsed = time.time() - t0
        if not result_future.done():
            return NavigationResult(
                status=NavigationStatus.TIMED_OUT,
                goal=goal,
                time_taken_s=elapsed,
            )

        return NavigationResult(
            status=NavigationStatus.SUCCESS,
            goal=goal,
            time_taken_s=elapsed,
        )

    def cancel(self) -> bool:
        if self._active_goal is None:
            return False
        self._active_goal.cancel_goal_async()
        self._active_goal = None
        return True

    def get_current_pose(self) -> NavigationGoal | None:
        return None  # requires /tf listener; left as exercise for real deployment

    def is_navigating(self) -> bool:
        return self._active_goal is not None


# ---------------------------------------------------------------------------
# MockNav2Adapter — in-process mock for unit and integration tests
# ---------------------------------------------------------------------------

class MockNav2Adapter(Nav2Adapter):
    """
    Deterministic Nav2 adapter for testing — no ROS runtime required.

    Configure fail_next to simulate failures, or override results per goal.
    """

    def __init__(self, config: Nav2Config | None = None) -> None:
        self.config = config or Nav2Config()
        self.goals_received: list[NavigationGoal] = []
        self.results_sent: list[NavigationResult] = []
        self._current_pose: NavigationGoal = NavigationGoal(x=0.0, y=0.0)
        self._navigating: bool = False
        self.fail_next: bool = False
        self.simulated_duration_s: float = 0.0

    def navigate_to(
        self,
        goal: NavigationGoal,
        timeout_s: float | None = None,
    ) -> NavigationResult:
        self.goals_received.append(goal)
        self._navigating = True

        if self.fail_next:
            self.fail_next = False
            self._navigating = False
            result = NavigationResult(
                status=NavigationStatus.FAILED,
                goal=goal,
                message="simulated failure",
                time_taken_s=self.simulated_duration_s,
            )
            self.results_sent.append(result)
            return result

        # Compute simulated travel distance
        dist = self._current_pose.distance_to(goal)
        self._current_pose = NavigationGoal(x=goal.x, y=goal.y, z=goal.z, yaw=goal.yaw)
        self._navigating = False
        result = NavigationResult(
            status=NavigationStatus.SUCCESS,
            goal=goal,
            time_taken_s=self.simulated_duration_s,
            distance_traveled_m=dist,
        )
        self.results_sent.append(result)
        return result

    def cancel(self) -> bool:
        was_navigating = self._navigating
        self._navigating = False
        return was_navigating

    def get_current_pose(self) -> NavigationGoal:
        return NavigationGoal(
            x=self._current_pose.x,
            y=self._current_pose.y,
            z=self._current_pose.z,
            yaw=self._current_pose.yaw,
        )

    def is_navigating(self) -> bool:
        return self._navigating

    def reset(self) -> None:
        self.goals_received.clear()
        self.results_sent.clear()
        self._current_pose = NavigationGoal(x=0.0, y=0.0)
        self._navigating = False
        self.fail_next = False
