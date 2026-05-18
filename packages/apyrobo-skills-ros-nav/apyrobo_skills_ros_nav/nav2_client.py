"""Nav2Client — manages the rclpy node lifecycle for Nav2 action clients.

This module requires a sourced ROS 2 workspace with Nav2 installed.  It is
intentionally isolated from the skill wrappers so that the rest of the package
can be imported (and tested) without ROS 2 present.

Typical lifecycle::

    client = Nav2Client()
    success = client.navigate_to_pose(x=2.0, y=1.5, yaw=0.0)
    client.destroy()

Or as a context manager::

    with Nav2Client() as client:
        client.navigate_to_pose(x=2.0, y=1.5, yaw=0.0)
"""
from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ROS 2 / Nav2 message type imports are guarded — the module still loads
# without ROS 2 installed; the client methods raise ImportError on first use.
# ---------------------------------------------------------------------------

_ROS2_MISSING_MSG = (
    "ROS 2 rclpy is required. Source your ROS 2 workspace first.\n"
    "  e.g.: source /opt/ros/humble/setup.bash && source install/setup.bash"
)


def _require_rclpy() -> Any:
    """Return the rclpy module or raise a clear ImportError."""
    try:
        import rclpy  # type: ignore[import]
        return rclpy
    except ImportError as exc:
        raise ImportError(_ROS2_MISSING_MSG) from exc


class Nav2Client:
    """Thin wrapper that manages one rclpy node and a set of Nav2 action clients.

    The node spins in a background daemon thread so callers can use
    synchronous-style calls without blocking the main thread's event loop.

    Args:
        node_name: Name of the rclpy node to create (default
            ``"apyrobo_nav2_client"``).
    """

    def __init__(self, node_name: str = "apyrobo_nav2_client") -> None:
        rclpy = _require_rclpy()

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node(node_name)
        self._executor = rclpy.executors.MultiThreadedExecutor()
        self._executor.add_node(self._node)

        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            daemon=True,
            name=f"{node_name}-spin",
        )
        self._spin_thread.start()
        logger.debug("Nav2Client: node '%s' spinning in background thread", node_name)

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Nav2Client":
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def navigate_to_pose(
        self,
        x: float,
        y: float,
        yaw: float = 0.0,
        frame_id: str = "map",
        timeout: float = 30.0,
    ) -> bool:
        """Send a NavigateToPose action goal and block until it completes.

        Args:
            x:        Target X position in *frame_id* (metres).
            y:        Target Y position in *frame_id* (metres).
            yaw:      Target heading in radians.
            frame_id: Coordinate frame for the goal pose (default ``"map"``).
            timeout:  Maximum seconds to wait for the goal to complete (default 30).

        Returns:
            ``True`` if navigation succeeded, ``False`` on failure, cancel, or
            timeout.
        """
        try:
            from nav2_msgs.action import NavigateToPose  # type: ignore[import]
            from geometry_msgs.msg import PoseStamped, Quaternion  # type: ignore[import]
            from rclpy.action import ActionClient  # type: ignore[import]
            import rclpy  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(_ROS2_MISSING_MSG) from exc

        action_client = ActionClient(
            self._node,
            NavigateToPose,
            "navigate_to_pose",
        )

        if not self._wait_for_action_server(action_client, timeout=5.0):
            logger.warning(
                "navigate_to_pose: action server not available after 5 s — aborting"
            )
            return False

        goal = NavigateToPose.Goal()
        goal.pose = self._build_pose_stamped(x, y, yaw, frame_id)

        send_future = action_client.send_goal_async(goal)
        goal_handle = self._await_future(send_future, timeout)
        if goal_handle is None or not goal_handle.accepted:
            logger.warning("navigate_to_pose: goal rejected by server")
            return False

        result_future = goal_handle.get_result_async()
        result = self._await_future(result_future, timeout)
        if result is None:
            logger.warning("navigate_to_pose: timed out waiting for result")
            goal_handle.cancel_goal_async()
            return False

        # NavigateToPose result has no explicit success field; lack of
        # cancellation/abort implies success.
        from action_msgs.msg import GoalStatus  # type: ignore[import]
        success = result.status == GoalStatus.STATUS_SUCCEEDED
        if not success:
            logger.warning(
                "navigate_to_pose: goal finished with status %d", result.status
            )
        return success

    def follow_path(
        self,
        waypoints: list[tuple[float, float]] | None = None,
        timeout: float = 120.0,
    ) -> bool:
        """Follow a sequence of (x, y) waypoints using the Nav2 FollowWaypoints action.

        Args:
            waypoints: List of ``(x, y)`` tuples in the ``map`` frame.
                       If ``None`` or empty the method returns ``True`` immediately.
            timeout:   Maximum seconds to wait for the entire path to complete.

        Returns:
            ``True`` if all waypoints were reached, ``False`` otherwise.
        """
        try:
            from nav2_msgs.action import FollowWaypoints  # type: ignore[import]
            from geometry_msgs.msg import PoseStamped  # type: ignore[import]
            from rclpy.action import ActionClient  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(_ROS2_MISSING_MSG) from exc

        if not waypoints:
            logger.debug("follow_path: no waypoints supplied — returning immediately")
            return True

        action_client = ActionClient(
            self._node,
            FollowWaypoints,
            "follow_waypoints",
        )

        if not self._wait_for_action_server(action_client, timeout=5.0):
            logger.warning(
                "follow_path: action server not available after 5 s — aborting"
            )
            return False

        goal = FollowWaypoints.Goal()
        goal.poses = [
            self._build_pose_stamped(x, y, 0.0, "map") for x, y in waypoints
        ]

        send_future = action_client.send_goal_async(goal)
        goal_handle = self._await_future(send_future, timeout)
        if goal_handle is None or not goal_handle.accepted:
            logger.warning("follow_path: goal rejected by server")
            return False

        result_future = goal_handle.get_result_async()
        result = self._await_future(result_future, timeout)
        if result is None:
            logger.warning("follow_path: timed out waiting for result")
            goal_handle.cancel_goal_async()
            return False

        from action_msgs.msg import GoalStatus  # type: ignore[import]
        success = result.status == GoalStatus.STATUS_SUCCEEDED
        if not success:
            missed = getattr(result.result, "missed_waypoints", [])
            logger.warning(
                "follow_path: finished with status %d; missed waypoints: %s",
                result.status, missed,
            )
        return success

    def clear_costmaps(self) -> bool:
        """Call the Nav2 clear_costmap_global_srv and clear_costmap_local_srv services.

        Returns:
            ``True`` if both service calls succeeded, ``False`` if either failed
            or a service is unavailable.
        """
        try:
            from nav2_msgs.srv import ClearEntireCostmap  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(_ROS2_MISSING_MSG) from exc

        results = []
        for service_name in (
            "/global_costmap/clear_entirely_global_costmap",
            "/local_costmap/clear_entirely_local_costmap",
        ):
            ok = self._call_empty_service(ClearEntireCostmap, service_name, timeout=5.0)
            results.append(ok)
            if not ok:
                logger.warning("clear_costmaps: failed to clear '%s'", service_name)

        return all(results)

    def recover(self) -> bool:
        """Trigger Nav2 recovery behaviours via the Backup action.

        Sends a short backup manoeuvre as the canonical recovery entry-point.
        Operators may wish to replace this with a ClearEntireCostmap call or
        the BackUp/Spin recoveries, depending on their nav2_params.

        Returns:
            ``True`` if recovery completed, ``False`` on failure or timeout.
        """
        try:
            from nav2_msgs.action import BackUp  # type: ignore[import]
            from geometry_msgs.msg import Point  # type: ignore[import]
            from rclpy.action import ActionClient  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(_ROS2_MISSING_MSG) from exc

        action_client = ActionClient(self._node, BackUp, "backup")

        if not self._wait_for_action_server(action_client, timeout=5.0):
            logger.warning("recover: backup action server not available after 5 s")
            return False

        goal = BackUp.Goal()
        goal.target = Point(x=0.15, y=0.0, z=0.0)  # back up 15 cm
        goal.speed = 0.025                           # m/s — conservative

        send_future = action_client.send_goal_async(goal)
        goal_handle = self._await_future(send_future, timeout=10.0)
        if goal_handle is None or not goal_handle.accepted:
            logger.warning("recover: backup goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        result = self._await_future(result_future, timeout=10.0)
        if result is None:
            logger.warning("recover: timed out waiting for backup result")
            goal_handle.cancel_goal_async()
            return False

        from action_msgs.msg import GoalStatus  # type: ignore[import]
        success = result.status == GoalStatus.STATUS_SUCCEEDED
        if not success:
            logger.warning("recover: backup finished with status %d", result.status)
        return success

    def destroy(self) -> None:
        """Shut down the action client node and stop the background spin thread.

        Safe to call multiple times.
        """
        try:
            self._executor.shutdown(timeout_sec=2.0)
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception:
            pass
        logger.debug("Nav2Client: node destroyed")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _wait_for_action_server(self, client: Any, timeout: float = 5.0) -> bool:
        """Return True if the action server becomes available within *timeout* s."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if client.server_is_ready():
                return True
            time.sleep(0.1)
        return False

    def _await_future(self, future: Any, timeout: float) -> Any:
        """Spin until *future* completes or *timeout* elapses; return result or None."""
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)
        try:
            return future.result()
        except Exception as exc:
            logger.error("Nav2Client: future raised: %s", exc)
            return None

    @staticmethod
    def _build_pose_stamped(
        x: float, y: float, yaw: float, frame_id: str
    ) -> Any:
        """Build a geometry_msgs/PoseStamped from x, y, yaw."""
        from geometry_msgs.msg import PoseStamped, Quaternion  # type: ignore[import]
        from builtin_interfaces.msg import Time  # type: ignore[import]
        import rclpy.time  # type: ignore[import]

        pose = PoseStamped()
        pose.header.frame_id = frame_id
        # Use zero timestamp so Nav2 uses the latest available TF
        pose.header.stamp = Time(sec=0, nanosec=0)

        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0

        # Convert yaw to quaternion (rotation about Z axis)
        half = yaw / 2.0
        pose.pose.orientation = Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(half),
            w=math.cos(half),
        )
        return pose

    def _call_empty_service(
        self, srv_type: Any, service_name: str, timeout: float = 5.0
    ) -> bool:
        """Create a service client, call it once, return success bool."""
        client = self._node.create_client(srv_type, service_name)
        deadline = time.monotonic() + timeout
        while not client.service_is_ready():
            if time.monotonic() >= deadline:
                logger.warning(
                    "_call_empty_service: '%s' not available after %.1f s",
                    service_name, timeout,
                )
                self._node.destroy_client(client)
                return False
            time.sleep(0.1)

        future = client.call_async(srv_type.Request())
        result = self._await_future(future, timeout)
        self._node.destroy_client(client)
        return result is not None
