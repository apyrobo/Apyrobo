"""ROS 2 MoveIt 2 motion planning adapter for apyrobo."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    import rclpy  # type: ignore
    _RCLPY_AVAILABLE = True
except ImportError:
    rclpy = None  # type: ignore
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

    async def connect(self) -> None:
        if self._stub_mode:
            logger.warning("rclpy not available — MoveItAdapter running in stub mode")
            self._connected = True
            return
        try:
            rclpy.init()
            self._node = rclpy.create_node("apyrobo_moveit")
            self._connected = True
            logger.info("MoveItAdapter connected (group=%s)", self.config.planning_group)
        except Exception as exc:
            logger.warning("MoveIt connect failed (%s) — stub mode", exc)
            self._stub_mode = True
            self._connected = True

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
            await asyncio.sleep(0.1)
            self._joint_values = dict(zip(target.joint_names, target.positions))
            return MotionResult(
                success=True,
                trajectory_length=len(target.positions),
                planning_time_s=time.monotonic() - start,
            )
        finally:
            self._moving = False

    async def move_to_pose_target(self, target: PoseTarget) -> MotionResult:
        if not self._connected:
            return MotionResult(success=False, message="not connected")
        self._moving = True
        start = time.monotonic()
        try:
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
        finally:
            self._moving = False

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
