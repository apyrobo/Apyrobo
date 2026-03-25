"""
Fleet Manager — load-balanced task assignment across a robot fleet.

Classes:
    RobotInfo    — registration record for one robot
    FleetManager — manages registration, heartbeats, and task assignment
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RobotInfo:
    """Registration record for a single robot."""

    robot_id: str
    capabilities: list[str]
    status: str = "idle"          # "idle" | "busy" | "offline"
    last_heartbeat: float = field(default_factory=time.time)
    current_task: str | None = None

    def is_available_for(self, required_caps: list[str]) -> bool:
        """True if robot is idle and has all required capabilities."""
        if self.status != "idle":
            return False
        return all(cap in self.capabilities for cap in required_caps)


class FleetManager:
    """
    Centralized fleet manager with load-balancing.

    Usage:
        fm = FleetManager()
        fm.register(RobotInfo("tb4_1", capabilities=["move", "gripper"]))
        fm.heartbeat("tb4_1")
        robot_id = fm.assign_task({"skill": "pick_object", "required": ["gripper"]})
    """

    def __init__(self) -> None:
        self._robots: dict[str, RobotInfo] = {}

    # ------------------------------------------------------------------
    # Registration & heartbeat
    # ------------------------------------------------------------------

    def register(self, robot: RobotInfo) -> None:
        """Register (or re-register) a robot."""
        self._robots[robot.robot_id] = robot
        logger.info("Fleet: registered robot %s (caps=%s)", robot.robot_id, robot.capabilities)

    def heartbeat(self, robot_id: str) -> None:
        """Update the last-seen timestamp for a robot."""
        robot = self._robots.get(robot_id)
        if robot is None:
            raise KeyError(f"Unknown robot: {robot_id!r}")
        robot.last_heartbeat = time.time()
        # Bring back online if it was marked offline
        if robot.status == "offline":
            robot.status = "idle"
            logger.info("Fleet: robot %s came back online", robot_id)

    # ------------------------------------------------------------------
    # Task assignment
    # ------------------------------------------------------------------

    def assign_task(self, task: dict[str, Any]) -> str | None:
        """
        Assign a task to the best available robot.

        Selection strategy:
        1. Filter to robots that are idle and have all required capabilities.
        2. Among those, pick the one with the *oldest* last_heartbeat (least
           recently used) — simple load-balancing without a separate load counter.

        Returns the robot_id of the assigned robot, or None if none available.
        """
        required_caps: list[str] = task.get("required", [])
        task_id: str = task.get("task_id", "")

        candidates = [
            r for r in self._robots.values()
            if r.is_available_for(required_caps)
        ]
        if not candidates:
            logger.debug(
                "Fleet: no robot available for task %r (required=%s)", task_id, required_caps
            )
            return None

        # Pick least-recently-active robot (simple load-balancing)
        chosen = min(candidates, key=lambda r: r.last_heartbeat)
        chosen.status = "busy"
        chosen.current_task = task_id
        logger.info("Fleet: assigned task %r to robot %s", task_id, chosen.robot_id)
        return chosen.robot_id

    def complete_task(self, robot_id: str) -> None:
        """Mark a robot as idle after its task finishes."""
        robot = self._robots.get(robot_id)
        if robot:
            robot.status = "idle"
            robot.current_task = None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return a summary of the fleet."""
        robots = [
            {
                "robot_id": r.robot_id,
                "status": r.status,
                "capabilities": r.capabilities,
                "current_task": r.current_task,
                "last_heartbeat": r.last_heartbeat,
            }
            for r in self._robots.values()
        ]
        idle = sum(1 for r in self._robots.values() if r.status == "idle")
        busy = sum(1 for r in self._robots.values() if r.status == "busy")
        offline = sum(1 for r in self._robots.values() if r.status == "offline")
        return {"total": len(robots), "idle": idle, "busy": busy, "offline": offline, "robots": robots}

    def offline_robots(self, timeout_sec: float = 30.0) -> list[str]:
        """Return robot_ids that have not sent a heartbeat within *timeout_sec*."""
        cutoff = time.time() - timeout_sec
        offline = []
        for robot in self._robots.values():
            if robot.last_heartbeat < cutoff:
                robot.status = "offline"
                offline.append(robot.robot_id)
        return offline

    def get_robot(self, robot_id: str) -> RobotInfo | None:
        return self._robots.get(robot_id)

    def __len__(self) -> int:
        return len(self._robots)
