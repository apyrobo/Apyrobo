"""
Robot — the primary interface for discovering and commanding robots.

This is the class that application code and AI agents use.  It hides
all ROS 2 details behind a clean, semantic API.

Usage (once a CapabilityAdapter is registered):

    robot = Robot.discover("gazebo://turtlebot4")
    caps  = robot.capabilities()
    robot.move(x=2.0, y=3.0, speed=0.5)
    robot.stop()
"""

from __future__ import annotations

import logging
from typing import Any

from apyrobo.core.adapters import CapabilityAdapter, get_adapter
from apyrobo.core.schemas import RobotCapability

logger = logging.getLogger(__name__)


class Robot:
    """High-level robot handle returned by discovery."""

    def __init__(self, robot_id: str, adapter: CapabilityAdapter) -> None:
        self._robot_id = robot_id
        self._adapter = adapter
        self._capability: RobotCapability | None = None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @classmethod
    def discover(cls, uri: str, **kwargs: Any) -> "Robot":
        """
        Discover a robot from a URI.

        Supported URI schemes (extensible via adapters):
            gazebo://turtlebot4   — simulated TurtleBot4 in Gazebo
            ros2://               — any robot on the local ROS 2 network
            mock://test           — in-memory mock for testing

        Returns a Robot instance ready for .capabilities() and commands.
        """
        scheme, _, robot_name = uri.partition("://")
        if not robot_name:
            raise ValueError(f"Invalid robot URI: {uri!r}  (expected scheme://name)")

        adapter = get_adapter(scheme, robot_name, **kwargs)
        robot = cls(robot_id=robot_name, adapter=adapter)
        logger.info("Discovered robot %s via %s adapter", robot_name, scheme)
        return robot

    # ------------------------------------------------------------------
    # Capability query
    # ------------------------------------------------------------------

    def capabilities(self, refresh: bool = False) -> RobotCapability:
        """
        Return the semantic capability profile for this robot.

        Caches the result after the first call unless *refresh* is True.
        """
        if self._capability is None or refresh:
            self._capability = self._adapter.get_capabilities()
        return self._capability

    # ------------------------------------------------------------------
    # Commands — thin wrappers that delegate to the adapter
    # ------------------------------------------------------------------

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        """Command the robot to move to (x, y) at optional *speed* m/s."""
        self._adapter.move(x=x, y=y, speed=speed)

    def stop(self) -> None:
        """Immediately halt all motion — always works regardless of task state."""
        self._adapter.stop()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    @property
    def robot_id(self) -> str:
        return self._robot_id

    def __repr__(self) -> str:
        return f"<Robot id={self._robot_id!r} adapter={type(self._adapter).__name__}>"
