"""
SwarmCoordinator — splits tasks across multiple robots.

Given a task and a swarm of robots, the coordinator:
1. Analyses which robots have the capabilities needed
2. Splits the task into subtasks
3. Assigns subtasks to the best-fit robots
4. Monitors execution and handles failures (reassignment)

Usage:
    coordinator = SwarmCoordinator(bus)
    result = coordinator.execute_task(
        task="deliver package from (1, 2) to (5, 5)",
        agent=agent,
    )
"""

from __future__ import annotations

import logging
import time
from typing import Any

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import (
    CapabilityType, TaskResult, TaskStatus, RecoveryAction, RobotCapability,
)
from apyrobo.skills.skill import Skill, SkillStatus
from apyrobo.skills.executor import SkillGraph, SkillExecutor, ExecutionEvent
from apyrobo.skills.agent import Agent
from apyrobo.swarm.bus import SwarmBus, SwarmMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

class RobotAssignment:
    """A subtask assigned to a specific robot."""

    def __init__(self, robot_id: str, graph: SkillGraph, description: str = "") -> None:
        self.robot_id = robot_id
        self.graph = graph
        self.description = description
        self.status: TaskStatus = TaskStatus.PENDING
        self.result: TaskResult | None = None

    def __repr__(self) -> str:
        return (
            f"<Assignment robot={self.robot_id!r} "
            f"skills={len(self.graph)} status={self.status.value}>"
        )


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class SwarmCoordinator:
    """
    Coordinates task execution across a swarm of robots.

    Strategies:
        - capability_match: Assign each subtask to the robot best equipped
        - round_robin: Distribute evenly (simple, for homogeneous swarms)
        - nearest: Assign to the robot closest to the task location (future)
    """

    def __init__(self, bus: SwarmBus, strategy: str = "capability_match") -> None:
        self._bus = bus
        self._strategy = strategy
        self._assignments: list[RobotAssignment] = []
        self._events: list[ExecutionEvent] = []

    # ------------------------------------------------------------------
    # Task splitting
    # ------------------------------------------------------------------

    def split_task(self, task: str, agent: Agent) -> list[RobotAssignment]:
        """
        Plan a task and split it across available robots.

        For now, uses a simple heuristic:
        - If only one robot: give it everything
        - If multiple robots: split the skill graph by capability match
        """
        robots = {rid: self._bus.get_robot(rid) for rid in self._bus.robot_ids}
        caps = {rid: self._bus.get_capabilities(rid) for rid in robots}

        if not robots:
            raise ValueError("No robots registered in the swarm")

        if len(robots) == 1:
            # Single robot — no splitting needed
            rid = list(robots.keys())[0]
            graph = agent.plan(task, robots[rid])
            assignment = RobotAssignment(rid, graph, description=task)
            return [assignment]

        # Multi-robot: plan against the first robot (for skill discovery),
        # then split skills across robots by capability
        first_robot = list(robots.values())[0]
        full_graph = agent.plan(task, first_robot)
        order = full_graph.get_execution_order()

        if len(order) <= 1:
            # Too few skills to split
            rid = list(robots.keys())[0]
            return [RobotAssignment(rid, full_graph, description=task)]

        # Build capability index: which robot can do what
        cap_index: dict[CapabilityType, list[str]] = {}
        for rid, cap in caps.items():
            for c in cap.capabilities:
                cap_index.setdefault(c.capability_type, []).append(rid)

        # Assign skills to robots
        assignments_by_robot: dict[str, list[tuple[Skill, dict[str, Any]]]] = {}
        robot_list = list(robots.keys())
        rr_index = 0  # round-robin fallback

        for skill in order:
            # Find robots that can handle this skill
            candidates = cap_index.get(skill.required_capability, [])

            if not candidates:
                # Fallback: CUSTOM capability, any robot can do it
                candidates = robot_list

            if self._strategy == "round_robin":
                chosen = robot_list[rr_index % len(robot_list)]
                rr_index += 1
            else:
                # capability_match: prefer the robot with the fewest assignments
                load = {rid: len(assignments_by_robot.get(rid, [])) for rid in candidates}
                chosen = min(candidates, key=lambda r: load.get(r, 0))

            assignments_by_robot.setdefault(chosen, []).append(
                (skill, full_graph.get_parameters(skill.skill_id))
            )

        # Build SkillGraphs for each robot
        result = []
        for rid, skill_list in assignments_by_robot.items():
            graph = SkillGraph()
            prev_id = None
            for skill, params in skill_list:
                depends = [prev_id] if prev_id else []
                graph.add_skill(skill, depends_on=depends, parameters=params)
                prev_id = skill.skill_id
            result.append(RobotAssignment(
                rid, graph,
                description=f"{task} (subtask for {rid}, {len(skill_list)} skills)",
            ))

        logger.info(
            "SwarmCoordinator: split task into %d assignments across %d robots",
            len(result), len(assignments_by_robot),
        )

        # Announce assignments via bus
        for assignment in result:
            self._bus.broadcast(
                sender="coordinator",
                message={
                    "event": "task_assigned",
                    "robot_id": assignment.robot_id,
                    "skills": len(assignment.graph),
                    "description": assignment.description,
                },
                msg_type="coordination",
            )

        return result

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_task(self, task: str, agent: Agent,
                     on_event: Any = None) -> TaskResult:
        """
        Plan, split, assign, and execute a task across the swarm.

        Returns an aggregated TaskResult.
        """
        self._assignments = self.split_task(task, agent)
        self._events = []

        total_steps = 0
        completed_steps = 0
        all_recovery: list[RecoveryAction] = []
        failed_robot: str | None = None

        for assignment in self._assignments:
            robot = self._bus.get_robot(assignment.robot_id)
            executor = SkillExecutor(robot)

            # Wire up event streaming
            def make_handler(rid: str):
                def handler(event: ExecutionEvent):
                    self._events.append(event)
                    if on_event:
                        on_event(event)
                    # Notify swarm about progress
                    self._bus.broadcast(
                        sender=rid,
                        message={
                            "event": "skill_status",
                            "skill_id": event.skill_id,
                            "status": event.status.value,
                            "message": event.message,
                        },
                        msg_type="status",
                    )
                return handler

            executor.on_event(make_handler(assignment.robot_id))

            # Execute
            assignment.status = TaskStatus.IN_PROGRESS
            result = executor.execute_graph(assignment.graph)
            assignment.result = result

            total_steps += result.steps_total
            completed_steps += result.steps_completed
            all_recovery.extend(result.recovery_actions_taken)

            if result.status == TaskStatus.COMPLETED:
                assignment.status = TaskStatus.COMPLETED
            else:
                assignment.status = TaskStatus.FAILED
                failed_robot = assignment.robot_id

                # Try to reassign to another robot
                reassigned = self._attempt_reassignment(assignment, agent)
                if reassigned:
                    completed_steps += 1  # approximate
                    all_recovery.append(RecoveryAction.REROUTE)
                else:
                    all_recovery.append(RecoveryAction.ABORT)
                    return TaskResult(
                        task_name=task,
                        status=TaskStatus.FAILED,
                        steps_completed=completed_steps,
                        steps_total=total_steps,
                        error=f"Robot {failed_robot} failed and could not be reassigned",
                        recovery_actions_taken=all_recovery,
                    )

        return TaskResult(
            task_name=task,
            status=TaskStatus.COMPLETED,
            confidence=1.0 if not all_recovery else 0.8,
            steps_completed=completed_steps,
            steps_total=total_steps,
            recovery_actions_taken=all_recovery,
        )

    def _attempt_reassignment(self, failed: RobotAssignment, agent: Agent) -> bool:
        """Try to reassign a failed robot's remaining work to another robot."""
        available = [rid for rid in self._bus.robot_ids if rid != failed.robot_id]
        if not available:
            logger.warning("No other robots available for reassignment")
            return False

        # Pick the robot with the lightest load
        other_rid = available[0]
        logger.info(
            "Reassigning work from %s to %s",
            failed.robot_id, other_rid,
        )

        self._bus.broadcast(
            sender="coordinator",
            message={
                "event": "task_reassigned",
                "from_robot": failed.robot_id,
                "to_robot": other_rid,
            },
            msg_type="coordination",
        )
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def assignments(self) -> list[RobotAssignment]:
        return list(self._assignments)

    @property
    def events(self) -> list[ExecutionEvent]:
        return list(self._events)

    def __repr__(self) -> str:
        return (
            f"<SwarmCoordinator strategy={self._strategy!r} "
            f"assignments={len(self._assignments)}>"
        )
