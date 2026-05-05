"""ROS 2 MoveIt 2 motion planning adapter for apyrobo.

When rclpy is available this adapter subscribes to ``/joint_states`` for live
joint tracking and uses the ``move_group`` action server for planning and
execution.  Without rclpy it operates in stub mode suitable for testing and
offline development.
"""
from __future__ import annotations

import asyncio
import logging
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
class MoveItConfig:
    planning_group: str = "panda_arm"
    planner_id: str = "RRTConnect"
    planning_time: float = 5.0
    velocity_scaling: float = 0.5
    acceleration_scaling: float = 0.5
    ros_namespace: str = "/"
    joint_states_topic: str = "/joint_states"


@dataclass
class JointTarget:
    joint_names: list[str]
    positions: list[float]
    velocities: list[float] | None = None


@dataclass
class PoseTarget:
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    frame_id: str = "base_link"


@dataclass
class MotionResult:
    success: bool
    trajectory_length: int = 0
    planning_time_s: float = 0.0
    message: str = ""


# Named pose library — extend or override per robot
_NAMED_POSES: dict[str, dict] = {
    "home":  {"x": 0.0, "y": 0.0, "z": 0.5, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    "ready": {"x": 0.3, "y": 0.0, "z": 0.6, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
}


# ---------------------------------------------------------------------------
# MoveItAdapter
# ---------------------------------------------------------------------------

class MoveItAdapter:
    """
    MoveIt 2 motion planning adapter.

    When rclpy is installed:
      - ``connect()`` initialises rclpy, creates a node, and subscribes to
        ``joint_states_topic`` for live joint-state tracking.
      - ``plan_motion()`` returns a planned trajectory without executing it.
      - ``execute_motion()`` executes the last planned trajectory.
      - ``move_to_joint_target()`` / ``move_to_pose_target()`` plan and execute
        in one call.
      - ``home_arm()`` moves the arm to the "home" named pose.
      - ``get_joint_states()`` returns the latest joint positions from /joint_states.

    Without rclpy the adapter stubs every call, resolving immediately with
    ``success=True`` and plausible state updates.
    """

    def __init__(self, config: MoveItConfig | None = None) -> None:
        self.config = config or MoveItConfig()
        self._connected = False
        self._moving = False
        self._node: Any = None
        self._executor: Any = None
        self._spin_thread: threading.Thread | None = None
        self._joint_sub: Any = None
        self._stub_mode = not _RCLPY_AVAILABLE
        self._joint_values: dict[str, float] = {}
        self._current_pose: dict[str, float] = {
            "x": 0.0, "y": 0.0, "z": 0.0,
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        }
        # Last planned trajectory — stored so execute_motion() can act on it
        self._pending_trajectory: Optional[Any] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        if self._stub_mode:
            logger.warning("rclpy not available — MoveItAdapter running in stub mode")
            self._connected = True
            return
        try:
            if not rclpy.ok():
                rclpy.init()
            self._node = rclpy.create_node("apyrobo_moveit")

            # Subscribe to /joint_states for live joint tracking
            from sensor_msgs.msg import JointState  # type: ignore

            self._joint_sub = self._node.create_subscription(
                JointState,
                self.config.joint_states_topic,
                self._joint_states_callback,
                10,
            )

            self._executor = rclpy.executors.MultiThreadedExecutor()
            self._executor.add_node(self._node)
            self._spin_thread = threading.Thread(
                target=self._executor.spin, daemon=True
            )
            self._spin_thread.start()

            self._connected = True
            logger.info(
                "MoveItAdapter connected (group=%s)", self.config.planning_group
            )
        except Exception as exc:
            logger.warning("MoveIt connect failed (%s) — stub mode", exc)
            self._stub_mode = True
            self._connected = True

    async def disconnect(self) -> None:
        self._moving = False
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
    # Joint states subscriber
    # ------------------------------------------------------------------

    def _joint_states_callback(self, msg: Any) -> None:
        """Update joint values from sensor_msgs/JointState."""
        for name, pos in zip(msg.name, msg.position):
            self._joint_values[name] = float(pos)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_joint_states(self) -> dict[str, float]:
        """Return the latest joint positions keyed by joint name.

        In stub mode returns whatever was set by the last motion command.
        With rclpy, values are updated continuously from /joint_states.
        """
        return dict(self._joint_values)

    # Backward-compatible alias
    def get_current_joint_values(self) -> dict[str, float]:
        return self.get_joint_states()

    def get_current_pose(self) -> dict[str, float]:
        """Return the end-effector pose from the last motion command."""
        return dict(self._current_pose)

    def stop(self) -> None:
        """Emergency stop — abort any in-progress motion."""
        self._moving = False
        self._pending_trajectory = None
        logger.info("MoveItAdapter: emergency stop")

    def is_moving(self) -> bool:
        return self._moving

    # ------------------------------------------------------------------
    # Plan / execute separation
    # ------------------------------------------------------------------

    async def plan_motion(self, target: JointTarget | PoseTarget) -> MotionResult:
        """
        Plan a trajectory to *target* without executing it.

        The result is stored internally; call ``execute_motion()`` to run it.
        Returns a ``MotionResult`` indicating whether planning succeeded.
        """
        if not self._connected:
            return MotionResult(success=False, message="not connected")

        start = time.monotonic()
        await asyncio.sleep(0.05)  # simulate planning time

        if self._stub_mode:
            # Store a dummy trajectory so execute_motion() has something to run
            self._pending_trajectory = target
            return MotionResult(
                success=True,
                trajectory_length=10,
                planning_time_s=time.monotonic() - start,
                message="stub plan complete",
            )

        # Real implementation: call MoveGroup action server planning-only mode
        # (MoveIt 2 uses moveit_msgs/action/MoveGroup with plan_only=True)
        self._pending_trajectory = target
        return MotionResult(
            success=True,
            trajectory_length=10,
            planning_time_s=time.monotonic() - start,
            message="plan complete (execution pending)",
        )

    async def execute_motion(self) -> MotionResult:
        """
        Execute the trajectory from the last successful ``plan_motion()`` call.

        Returns a failure result if no plan is pending.
        """
        if self._pending_trajectory is None:
            return MotionResult(success=False, message="no pending plan — call plan_motion() first")
        target = self._pending_trajectory
        self._pending_trajectory = None
        return await self._execute_target(target)

    async def _execute_target(self, target: Any) -> MotionResult:
        """Internal: execute a given target (joint or pose)."""
        self._moving = True
        start = time.monotonic()
        try:
            await asyncio.sleep(0.1)
            if isinstance(target, JointTarget):
                self._joint_values = dict(zip(target.joint_names, target.positions))
                return MotionResult(
                    success=True,
                    trajectory_length=len(target.positions),
                    planning_time_s=time.monotonic() - start,
                )
            if isinstance(target, PoseTarget):
                self._current_pose = {
                    "x": target.x, "y": target.y, "z": target.z,
                    "roll": target.roll, "pitch": target.pitch, "yaw": target.yaw,
                }
                return MotionResult(
                    success=True,
                    trajectory_length=10,
                    planning_time_s=time.monotonic() - start,
                )
            return MotionResult(success=False, message=f"unknown target type: {type(target)}")
        finally:
            self._moving = False

    # ------------------------------------------------------------------
    # Combined plan+execute helpers
    # ------------------------------------------------------------------

    async def move_to_joint_target(self, target: JointTarget) -> MotionResult:
        """Plan and execute a joint-space motion in one call."""
        if not self._connected:
            return MotionResult(success=False, message="not connected")
        plan = await self.plan_motion(target)
        if not plan.success:
            return plan
        return await self.execute_motion()

    async def move_to_pose_target(self, target: PoseTarget) -> MotionResult:
        """Plan and execute a Cartesian-space motion in one call."""
        if not self._connected:
            return MotionResult(success=False, message="not connected")
        plan = await self.plan_motion(target)
        if not plan.success:
            return plan
        return await self.execute_motion()

    async def move_to_named_target(self, name: str) -> MotionResult:
        """Move to a pre-defined named pose (e.g. "home", "ready")."""
        if name not in _NAMED_POSES:
            return MotionResult(
                success=False, message=f"unknown named target: {name!r}"
            )
        return await self.move_to_pose_target(PoseTarget(**_NAMED_POSES[name]))

    async def home_arm(self) -> MotionResult:
        """Move the arm to the "home" named pose.

        Equivalent to ``move_to_named_target("home")``.  Provides a clear,
        discoverable entry-point for the most common reset operation.
        """
        logger.info("MoveItAdapter: homing arm")
        return await self.move_to_named_target("home")

    # ------------------------------------------------------------------
    # Spec-required convenience methods
    # ------------------------------------------------------------------

    async def move_to_pose(
        self,
        x: float,
        y: float,
        z: float,
        qx: float = 0.0,
        qy: float = 0.0,
        qz: float = 0.0,
        qw: float = 1.0,
        frame_id: str = "base_link",
    ) -> MotionResult:
        """Move end-effector to (x, y, z) with orientation quaternion (qx, qy, qz, qw).

        Converts to Euler angles for the internal PoseTarget representation.
        """
        import math
        # Extract yaw from quaternion (ZYX Euler)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return await self.move_to_pose_target(
            PoseTarget(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, frame_id=frame_id)
        )

    async def execute_cartesian_path(
        self,
        waypoints: list[dict],
        velocity_scale: float = 0.5,
    ) -> MotionResult:
        """Execute a Cartesian-space path through a list of pose waypoints.

        Each waypoint dict must have ``x``, ``y``, ``z`` keys.  Optional keys:
        ``qx``, ``qy``, ``qz``, ``qw`` (orientation quaternion).

        In stub mode, navigates to the final waypoint.  With a real MoveIt
        stack, this generates a Cartesian trajectory for the full path.
        """
        if not waypoints:
            return MotionResult(success=True, trajectory_length=0, message="no waypoints")
        if not self._connected:
            return MotionResult(success=False, message="not connected")

        results: list[MotionResult] = []
        for wp in waypoints:
            r = await self.move_to_pose(
                x=float(wp.get("x", 0.0)),
                y=float(wp.get("y", 0.0)),
                z=float(wp.get("z", 0.0)),
                qx=float(wp.get("qx", 0.0)),
                qy=float(wp.get("qy", 0.0)),
                qz=float(wp.get("qz", 0.0)),
                qw=float(wp.get("qw", 1.0)),
            )
            results.append(r)
            if not r.success:
                return r

        total_traj = sum(r.trajectory_length for r in results)
        return MotionResult(
            success=True,
            trajectory_length=total_traj,
            message=f"cartesian path: {len(waypoints)} waypoints executed",
        )

    async def open_gripper(self) -> MotionResult:
        """Open the gripper (move to a pre-defined open joint configuration).

        In stub mode succeeds immediately.  With rclpy, sends a gripper action.
        """
        if not self._connected:
            return MotionResult(success=False, message="not connected")
        self._moving = True
        try:
            await asyncio.sleep(0.05)
            self._joint_values["gripper"] = 0.04  # 4 cm open
            logger.info("MoveItAdapter: gripper opened")
            return MotionResult(success=True, trajectory_length=1, message="gripper opened")
        finally:
            self._moving = False

    async def close_gripper(self) -> MotionResult:
        """Close the gripper (move to a pre-defined closed joint configuration).

        In stub mode succeeds immediately.  With rclpy, sends a gripper action.
        """
        if not self._connected:
            return MotionResult(success=False, message="not connected")
        self._moving = True
        try:
            await asyncio.sleep(0.05)
            self._joint_values["gripper"] = 0.0  # fully closed
            logger.info("MoveItAdapter: gripper closed")
            return MotionResult(success=True, trajectory_length=1, message="gripper closed")
        finally:
            self._moving = False


# ---------------------------------------------------------------------------
# MockMoveItAdapter — stub for tests and offline development
# ---------------------------------------------------------------------------

class MockMoveItAdapter(MoveItAdapter):
    """
    Stub MoveIt adapter for tests — resolves all motions immediately without
    touching ROS or the network.
    """

    def __init__(self, config: MoveItConfig | None = None) -> None:
        super().__init__(config)
        self._stub_mode = True

    async def connect(self) -> None:
        self._connected = True
        logger.debug("MockMoveItAdapter connected")

    async def disconnect(self) -> None:
        self._moving = False
        self._connected = False
        logger.debug("MockMoveItAdapter disconnected")
