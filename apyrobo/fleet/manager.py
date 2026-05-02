"""
Fleet Manager — load-balanced task assignment across a robot fleet.

Classes:
    RobotInfo    — registration record for one robot
    FleetManager — manages registration, heartbeats, task assignment,
                   and multi-robot task handoff on failure
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from apyrobo.core.robot import Robot
    from apyrobo.core.schemas import TaskResult
    from apyrobo.skills.library import SkillLibrary
    from apyrobo.skills.agent import Agent

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

    # ------------------------------------------------------------------
    # Multi-robot task handoff
    # ------------------------------------------------------------------

    def handoff_task(
        self,
        failed_robot_id: str,
        task_result: "TaskResult",
        exclude_robots: list[str] | None = None,
    ) -> str | None:
        """
        Find the best idle robot to take over from *failed_robot_id*.

        Marks the failed robot back to idle, selects a replacement using
        the same least-recently-active strategy as ``assign_task``, emits
        a ``task.handoff`` observability event, and returns the new robot's
        ID (or None if no candidate is available).

        Args:
            failed_robot_id: The robot whose task has failed.
            task_result: The TaskResult from the failed execution
                         (used for logging / observability metadata).
            exclude_robots: Additional robot IDs to skip (e.g. previously
                            failed handoff candidates).

        Returns:
            robot_id of the chosen replacement, or None.
        """
        # Release the failed robot
        failed = self._robots.get(failed_robot_id)
        if failed is not None:
            failed.status = "idle"
            failed.current_task = None

        excluded: set[str] = {failed_robot_id}
        if exclude_robots:
            excluded.update(exclude_robots)

        candidates = [
            r for r in self._robots.values()
            if r.status == "idle" and r.robot_id not in excluded
        ]
        if not candidates:
            logger.warning(
                "Fleet handoff: no available robot to take over from %s", failed_robot_id
            )
            return None

        chosen = min(candidates, key=lambda r: r.last_heartbeat)
        chosen.status = "busy"
        chosen.current_task = getattr(task_result, "task_name", "handoff_task")

        steps_done = getattr(task_result, "steps_completed", 0)
        steps_total = getattr(task_result, "steps_total", 0)
        error = getattr(task_result, "error", None)

        logger.info(
            "Fleet handoff: %s → %s  (steps %d/%d, error=%s)",
            failed_robot_id, chosen.robot_id, steps_done, steps_total, error,
        )

        try:
            from apyrobo.observability import emit_event
            emit_event(
                "task.handoff",
                from_robot=failed_robot_id,
                to_robot=chosen.robot_id,
                task_name=chosen.current_task,
                steps_completed=steps_done,
                steps_total=steps_total,
                error=error or "",
            )
        except Exception as exc:
            logger.debug("Could not emit task.handoff event: %s", exc)

        return chosen.robot_id

    def execute_with_handoff(
        self,
        task: str,
        library: "SkillLibrary",
        agent: "Agent",
        robots: dict[str, "Robot"],
        max_handoffs: int = 2,
    ) -> tuple["TaskResult", list[str]]:
        """
        Execute *task* against the fleet, automatically handing off to a
        new robot on failure up to *max_handoffs* times.

        The method:
        1. Picks the least-recently-active idle robot from ``robots``.
        2. Runs ``agent.execute(task, robot=robot_instance)``.
        3. On failure, calls ``handoff_task()`` to find a replacement and
           repeats from step 2.
        4. Returns the final ``TaskResult`` and the ordered list of
           robot IDs that were tried.

        Args:
            task:          Natural-language task string.
            library:       SkillLibrary to use for planning.
            agent:         Agent to use for execution.
            robots:        Mapping of robot_id → Robot instance.
                           Only robots registered with the FleetManager
                           AND present in this dict can be selected.
            max_handoffs:  Maximum number of handoff attempts (default 2).
                           Total attempts = max_handoffs + 1.

        Returns:
            (TaskResult, [robot_id, ...]) where the list contains every
            robot that was tried, in order.
        """
        from apyrobo.core.schemas import TaskResult, TaskStatus, RecoveryAction

        tried: list[str] = []
        exclude: list[str] = []

        # Pick an initial robot: idle, registered, and in robots dict
        candidates = [
            r for r in self._robots.values()
            if r.status == "idle" and r.robot_id in robots
        ]
        if not candidates:
            return (
                TaskResult(
                    task_name=task,
                    status=TaskStatus.FAILED,
                    error="No available robot in fleet",
                    recovery_actions_taken=[RecoveryAction.ABORT],
                ),
                tried,
            )

        current_id = min(candidates, key=lambda r: r.last_heartbeat).robot_id
        robot_info = self._robots[current_id]
        robot_info.status = "busy"
        robot_info.current_task = task

        attempts = 0
        result: TaskResult | None = None

        while attempts <= max_handoffs:
            tried.append(current_id)
            robot_instance = robots[current_id]

            logger.info(
                "Fleet execute_with_handoff: attempt %d/%d on robot %s",
                attempts + 1, max_handoffs + 1, current_id,
            )

            result = agent.execute(task, robot=robot_instance)

            if result.status.value == "completed":
                self.complete_task(current_id)
                return result, tried

            # Task failed — attempt handoff if budget allows
            if attempts < max_handoffs:
                exclude.append(current_id)
                next_id = self.handoff_task(
                    failed_robot_id=current_id,
                    task_result=result,
                    exclude_robots=exclude,
                )
                if next_id is None or next_id not in robots:
                    logger.warning(
                        "Fleet execute_with_handoff: no eligible robot for handoff after %s",
                        current_id,
                    )
                    break
                current_id = next_id
            else:
                # Budget exhausted — release final robot
                self.complete_task(current_id)

            attempts += 1

        # All attempts exhausted — return the last result
        assert result is not None
        return result, tried

    def __len__(self) -> int:
        return len(self._robots)
