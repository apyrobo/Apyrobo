"""ROS 2 MoveIt 2 motion planning adapter for apyrobo."""
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
class MoveItConfig:
    planning_group: str = "panda_arm"
    planner_id: str = "RRTConnect"
    planning_time: float = 5.0
    velocity_scaling: float = 0.5
    acceleration_scaling: float = 0.5
    ros_namespace: str = "/"


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


_NAMED_POSES: dict[str, dict] = {
    "home": {"x": 0.0, "y": 0.0, "z": 0.5, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    "ready": {"x": 0.3, "y": 0.0, "z": 0.6, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
}


class MoveItAdapter:
    """MoveIt 2 motion planning adapter.  Falls back to stub when rclpy unavailable."""

    def __init__(self, config: MoveItConfig | None = None) -> None:
        self.config = config or MoveItConfig()
        self._connected = False
        self._moving = False
        self._node: Any = None
        self._stub_mode = not _RCLPY_AVAILABLE
        self._joint_values: dict = {}
        self._current_pose: dict = {"x": 0.0, "y": 0.0, "z": 0.0,
                                     "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        self._joint_state_sub: Any = None

    async def connect(self) -> None:
        if self._stub_mode:
            logger.warning("rclpy not available — MoveItAdapter running in stub mode")
            self._connected = True
            return
        try:
            rclpy.init()
            self._node = rclpy.create_node("apyrobo_moveit")
            self._setup_joint_state_subscription()
            self._connected = True
            logger.info("MoveItAdapter connected (group=%s)", self.config.planning_group)
        except Exception as exc:
            logger.warning("MoveIt connect failed (%s) — stub mode", exc)
            self._stub_mode = True
            self._connected = True

    def _setup_joint_state_subscription(self) -> None:
        """Subscribe to /joint_states to track current arm configuration."""
        try:
            from sensor_msgs.msg import JointState  # type: ignore
            self._joint_state_sub = self._node.create_subscription(
                JointState,
                "/joint_states",
                self._joint_state_callback,
                10,
            )
        except ImportError:
            logger.warning("sensor_msgs not available — joint state tracking disabled")

    def _joint_state_callback(self, msg: Any) -> None:
        """Update joint values from /joint_states topic."""
        self._joint_values = dict(zip(msg.name, msg.position))

    async def disconnect(self) -> None:
        self._moving = False
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

    async def move_to_joint_target(self, target: JointTarget) -> MotionResult:
        if not self._connected:
            return MotionResult(success=False, message="not connected")
        self._moving = True
        start = time.monotonic()
        try:
            if self._stub_mode:
                await asyncio.sleep(0.1)
                self._joint_values = dict(zip(target.joint_names, target.positions))
                return MotionResult(
                    success=True,
                    trajectory_length=len(target.positions),
                    planning_time_s=time.monotonic() - start,
                )
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._blocking_move_joints, target
            )
            return result
        finally:
            self._moving = False

    def _blocking_move_joints(self, target: JointTarget) -> MotionResult:
        """Blocking MoveGroup joint-space motion — runs in thread executor."""
        from moveit_msgs.action import MoveGroup  # type: ignore
        from moveit_msgs.msg import (  # type: ignore
            MotionPlanRequest,
            Constraints,
            JointConstraint,
        )

        start = time.monotonic()
        action_client = RclpyActionClient(self._node, MoveGroup, "move_group")
        if not action_client.wait_for_server(timeout_sec=self.config.planning_time):
            action_client.destroy()
            return MotionResult(success=False, message="move_group action server not available")

        request = MotionPlanRequest()
        request.group_name = self.config.planning_group
        request.planner_id = self.config.planner_id
        request.allowed_planning_time = self.config.planning_time
        request.max_velocity_scaling_factor = self.config.velocity_scaling
        request.max_acceleration_scaling_factor = self.config.acceleration_scaling
        request.num_planning_attempts = 5

        constraints = Constraints()
        for name, pos in zip(target.joint_names, target.positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        request.goal_constraints.append(constraints)

        goal_msg = MoveGroup.Goal()
        goal_msg.request = request

        send_future = action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self._node, send_future)
        goal_handle = send_future.result()

        if not goal_handle.accepted:
            action_client.destroy()
            return MotionResult(success=False, message="goal rejected by MoveGroup")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self._node, result_future)
        response = result_future.result().result.response

        planning_time = time.monotonic() - start
        action_client.destroy()

        if response.error_code.val != 1:  # moveit_msgs/MoveItErrorCodes SUCCESS = 1
            return MotionResult(
                success=False,
                planning_time_s=planning_time,
                message=f"MoveGroup error code {response.error_code.val}",
            )

        self._joint_values = dict(zip(target.joint_names, target.positions))
        traj_len = len(response.trajectory.joint_trajectory.points)
        return MotionResult(
            success=True,
            trajectory_length=traj_len,
            planning_time_s=planning_time,
        )

    async def move_to_pose_target(self, target: PoseTarget) -> MotionResult:
        if not self._connected:
            return MotionResult(success=False, message="not connected")
        self._moving = True
        start = time.monotonic()
        try:
            if self._stub_mode:
                await asyncio.sleep(0.1)
                self._current_pose = {
                    "x": target.x, "y": target.y, "z": target.z,
                    "roll": target.roll, "pitch": target.pitch, "yaw": target.yaw,
                }
                return MotionResult(
                    success=True,
                    trajectory_length=10,
                    planning_time_s=time.monotonic() - start,
                )
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._blocking_move_pose, target
            )
            return result
        finally:
            self._moving = False

    def _blocking_move_pose(self, target: PoseTarget) -> MotionResult:
        """Blocking MoveGroup Cartesian motion — runs in thread executor."""
        from moveit_msgs.action import MoveGroup  # type: ignore
        from moveit_msgs.msg import (  # type: ignore
            MotionPlanRequest,
            Constraints,
            PositionConstraint,
            OrientationConstraint,
            BoundingVolume,
        )
        from geometry_msgs.msg import Pose, Quaternion  # type: ignore
        from shape_msgs.msg import SolidPrimitive  # type: ignore

        start = time.monotonic()
        action_client = RclpyActionClient(self._node, MoveGroup, "move_group")
        if not action_client.wait_for_server(timeout_sec=self.config.planning_time):
            action_client.destroy()
            return MotionResult(success=False, message="move_group action server not available")

        request = MotionPlanRequest()
        request.group_name = self.config.planning_group
        request.planner_id = self.config.planner_id
        request.allowed_planning_time = self.config.planning_time
        request.max_velocity_scaling_factor = self.config.velocity_scaling
        request.max_acceleration_scaling_factor = self.config.acceleration_scaling
        request.num_planning_attempts = 5

        # Build position constraint
        pc = PositionConstraint()
        pc.header.frame_id = target.frame_id
        pc.link_name = "tool0"  # end-effector link
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.01]
        bv = BoundingVolume()
        bv.primitives.append(sphere)
        bv.primitive_poses.append(Pose())
        pc.constraint_region = bv
        pc.target_point_offset.x = target.x
        pc.target_point_offset.y = target.y
        pc.target_point_offset.z = target.z
        pc.weight = 1.0

        # Convert RPY to quaternion
        cr, cp, cy = math.cos(target.roll / 2), math.cos(target.pitch / 2), math.cos(target.yaw / 2)
        sr, sp, sy = math.sin(target.roll / 2), math.sin(target.pitch / 2), math.sin(target.yaw / 2)
        oc = OrientationConstraint()
        oc.header.frame_id = target.frame_id
        oc.link_name = "tool0"
        oc.orientation = Quaternion(
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
            w=cr * cp * cy + sr * sp * sy,
        )
        oc.absolute_x_axis_tolerance = 0.01
        oc.absolute_y_axis_tolerance = 0.01
        oc.absolute_z_axis_tolerance = 0.01
        oc.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pc)
        constraints.orientation_constraints.append(oc)
        request.goal_constraints.append(constraints)

        goal_msg = MoveGroup.Goal()
        goal_msg.request = request

        send_future = action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self._node, send_future)
        goal_handle = send_future.result()

        if not goal_handle.accepted:
            action_client.destroy()
            return MotionResult(success=False, message="goal rejected by MoveGroup")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self._node, result_future)
        response = result_future.result().result.response

        planning_time = time.monotonic() - start
        action_client.destroy()

        if response.error_code.val != 1:
            return MotionResult(
                success=False,
                planning_time_s=planning_time,
                message=f"MoveGroup error code {response.error_code.val}",
            )

        self._current_pose = {
            "x": target.x, "y": target.y, "z": target.z,
            "roll": target.roll, "pitch": target.pitch, "yaw": target.yaw,
        }
        traj_len = len(response.trajectory.joint_trajectory.points)
        return MotionResult(
            success=True,
            trajectory_length=traj_len,
            planning_time_s=planning_time,
        )

    async def move_to_named_target(self, name: str) -> MotionResult:
        if name not in _NAMED_POSES:
            return MotionResult(
                success=False, message=f"unknown named target: {name!r}"
            )
        pose = _NAMED_POSES[name]
        target = PoseTarget(**pose)
        return await self.move_to_pose_target(target)

    def get_current_joint_values(self) -> dict:
        return dict(self._joint_values)

    def get_current_pose(self) -> dict:
        return dict(self._current_pose)

    def stop(self) -> None:
        self._moving = False
        logger.info("MoveItAdapter: emergency stop")

    def is_moving(self) -> bool:
        return self._moving


class MockMoveItAdapter(MoveItAdapter):
    """Stub MoveIt adapter for tests — simulates motion with asyncio.sleep(0.1)."""

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
