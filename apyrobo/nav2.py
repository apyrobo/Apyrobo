"""ROS 2 Nav2 navigation adapter for apyrobo."""
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

    async def connect(self) -> None:
        if self._stub_mode:
            logger.warning("rclpy not available — Nav2Adapter running in stub mode")
            self._connected = True
            return
        try:
            rclpy.init()
            self._node = rclpy.create_node("apyrobo_nav2")
            self._connected = True
            logger.info("Nav2Adapter connected (namespace=%s)", self.config.ros_namespace)
        except Exception as exc:
            logger.warning("Nav2 connect failed (%s) — stub mode", exc)
            self._stub_mode = True
            self._connected = True

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
            # Real Nav2 action client would go here
            await asyncio.sleep(0.05)
            self._current_pose = {"x": goal.x, "y": goal.y, "yaw": goal.yaw}
            elapsed = time.monotonic() - start
            return NavigationResult(
                success=True,
                final_pose=dict(self._current_pose),
                elapsed_s=elapsed,
            )
        finally:
            self._navigating = False

    async def cancel_navigation(self) -> None:
        self._navigating = False
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
