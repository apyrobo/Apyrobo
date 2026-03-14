from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from apyrobo.core.schemas import CapabilityType, RecoveryAction, RobotCapability, TaskResult, TaskStatus
from apyrobo.skills.agent import Agent
from apyrobo.skills.executor import ExecutionEvent, SkillExecutor, SkillGraph
from apyrobo.skills.skill import Skill
from apyrobo.swarm.bus import SwarmBus

logger = logging.getLogger(__name__)


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


class SwarmCoordinator:
    """Coordinates task execution across a swarm of robots."""

    def __init__(self, bus: SwarmBus, strategy: str = "capability_match") -> None:
        self._bus = bus
        self._strategy = strategy
        self._assignments: list[RobotAssignment] = []
        self._events: list[ExecutionEvent] = []
        self._resource_leases: dict[str, str] = {}

    def _fleet_capability_proxy(self, capabilities: dict[str, RobotCapability]) -> Any:
        """Return a lightweight robot-like object exposing merged fleet capabilities."""
        merged: list[Any] = []
        seen: set[tuple[str, str]] = set()
        max_speed = None
        for cap in capabilities.values():
            if cap.max_speed is not None:
                max_speed = cap.max_speed if max_speed is None else max(max_speed, cap.max_speed)
            for c in cap.capabilities:
                key = (c.capability_type.value, c.name)
                if key not in seen:
                    seen.add(key)
                    merged.append(c)

        class _FleetProxy:
            def capabilities(self) -> RobotCapability:
                return RobotCapability(
                    robot_id="fleet",
                    name="FleetProxy",
                    capabilities=merged,
                    max_speed=max_speed,
                )

        return _FleetProxy()

    def _choose_robot_for_skill(
        self,
        skill: Skill,
        candidates: list[str],
        assignments_by_robot: dict[str, list[tuple[Skill, dict[str, Any]]]],
        rr_index: int,
        params: dict[str, Any],
    ) -> tuple[str, int]:
        if self._strategy == "round_robin":
            return candidates[rr_index % len(candidates)], rr_index + 1

        if self._strategy == "nearest":
            tx, ty = params.get("x"), params.get("y")
            if isinstance(tx, (int, float)) and isinstance(ty, (int, float)):
                def dist(rid: str) -> float:
                    try:
                        x, y = self._bus.get_robot(rid).get_position()
                        return math.sqrt((x - tx) ** 2 + (y - ty) ** 2)
                    except Exception:
                        return float("inf")
                return min(candidates, key=dist), rr_index

        load = {rid: len(assignments_by_robot.get(rid, [])) for rid in candidates}
        return min(candidates, key=lambda r: load.get(r, 0)), rr_index

    def split_task(self, task: str, agent: Agent) -> list[RobotAssignment]:
        robots = {rid: self._bus.get_robot(rid) for rid in self._bus.robot_ids}
        caps = {rid: self._bus.get_capabilities(rid) for rid in robots}

        if not robots:
            raise ValueError("No robots registered in the swarm")

        if len(robots) == 1:
            rid = list(robots.keys())[0]
            graph = agent.plan(task, robots[rid])
            return [RobotAssignment(rid, graph, description=task)]

        # SW-01/SW-07: plan against merged fleet capabilities, then assign by skill.
        planning_robot = self._fleet_capability_proxy(caps)
        full_graph = agent.plan(task, planning_robot)
        order = full_graph.get_execution_order()

        cap_index: dict[CapabilityType, list[str]] = {}
        for rid, cap in caps.items():
            for c in cap.capabilities:
                cap_index.setdefault(c.capability_type, []).append(rid)

        assignments_by_robot: dict[str, list[tuple[Skill, dict[str, Any]]]] = {}
        robot_list = list(robots.keys())
        rr_index = 0

        for skill in order:
            params = full_graph.get_parameters(skill.skill_id)
            candidates = cap_index.get(skill.required_capability, [])
            if not candidates:
                candidates = robot_list
            chosen, rr_index = self._choose_robot_for_skill(
                skill, candidates, assignments_by_robot, rr_index, params,
            )
            assignments_by_robot.setdefault(chosen, []).append((skill, params))

        result = []
        for rid, skill_list in assignments_by_robot.items():
            graph = SkillGraph()
            prev_id = None
            for skill, params in skill_list:
                graph.add_skill(skill, depends_on=[prev_id] if prev_id else [], parameters=params)
                prev_id = skill.skill_id
            result.append(RobotAssignment(
                rid,
                graph,
                description=f"{task} (subtask for {rid}, {len(skill_list)} skills)",
            ))

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

    def _execute_assignment(self, assignment: RobotAssignment, on_event: Any = None) -> TaskResult:
        robot = self._bus.get_robot(assignment.robot_id)
        executor = SkillExecutor(robot)

        def handler(event: ExecutionEvent) -> None:
            self._events.append(event)
            if on_event:
                on_event(event)
            self._bus.broadcast(
                sender=assignment.robot_id,
                message={
                    "event": "skill_status",
                    "skill_id": event.skill_id,
                    "status": event.status.value,
                    "message": event.message,
                },
                msg_type="status",
            )

        executor.on_event(handler)
        assignment.status = TaskStatus.IN_PROGRESS
        result = executor.execute_graph(assignment.graph)
        assignment.result = result
        assignment.status = TaskStatus.COMPLETED if result.status == TaskStatus.COMPLETED else TaskStatus.FAILED
        return result

    def execute_task(self, task: str, agent: Agent, on_event: Any = None) -> TaskResult:
        self._assignments = self.split_task(task, agent)
        self._events = []

        total_steps = 0
        completed_steps = 0
        all_recovery: list[RecoveryAction] = []
        errors: list[str] = []

        # SW-01: execute robot assignments in parallel.
        with ThreadPoolExecutor(max_workers=max(1, len(self._assignments))) as pool:
            future_map = {pool.submit(self._execute_assignment, a, on_event): a for a in self._assignments}
            for future in as_completed(future_map):
                assignment = future_map[future]
                result = future.result()
                total_steps += result.steps_total
                completed_steps += result.steps_completed
                all_recovery.extend(result.recovery_actions_taken)

                if result.status != TaskStatus.COMPLETED:
                    reassigned = self._attempt_reassignment(assignment)
                    if reassigned:
                        all_recovery.append(RecoveryAction.REROUTE)
                        completed_steps += 1
                    else:
                        all_recovery.append(RecoveryAction.ABORT)
                        errors.append(
                            f"Robot {assignment.robot_id} failed and could not be reassigned"
                        )

        if errors:
            return TaskResult(
                task_name=task,
                status=TaskStatus.FAILED,
                steps_completed=completed_steps,
                steps_total=total_steps,
                error="; ".join(errors),
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

    def _attempt_reassignment(self, failed: RobotAssignment) -> bool:
        """SW-02: reassign failed work to a capable robot and execute it."""
        remaining = failed.graph.get_execution_order()
        if not remaining:
            return False

        needed_caps = {s.required_capability for s in remaining}
        candidates: list[str] = []
        for rid in self._bus.robot_ids:
            if rid == failed.robot_id:
                continue
            robot_caps = {c.capability_type for c in self._bus.get_capabilities(rid).capabilities}
            if needed_caps.issubset(robot_caps) or CapabilityType.CUSTOM in needed_caps:
                candidates.append(rid)

        if not candidates:
            logger.warning("No capable robots available for reassignment from %s", failed.robot_id)
            return False

        reassigned_to = min(candidates, key=lambda rid: len([a for a in self._assignments if a.robot_id == rid]))
        reassigned = RobotAssignment(reassigned_to, failed.graph, description=f"reassigned from {failed.robot_id}")
        result = self._execute_assignment(reassigned)

        self._bus.broadcast(
            sender="coordinator",
            message={
                "event": "task_reassigned",
                "from_robot": failed.robot_id,
                "to_robot": reassigned_to,
            },
            msg_type="coordination",
        )
        return result.status == TaskStatus.COMPLETED

    # SW-05
    def allocate_resource_auction(self, resource_id: str, candidate_robot_ids: list[str]) -> str:
        """Auction-style resource allocation favoring low load and short distance."""
        if resource_id in self._resource_leases:
            return self._resource_leases[resource_id]
        bids: dict[str, float] = {}
        for rid in candidate_robot_ids:
            load = len([a for a in self._assignments if a.robot_id == rid])
            try:
                x, y = self._bus.get_robot(rid).get_position()
                distance = math.sqrt(x ** 2 + y ** 2)
            except Exception:
                distance = 1000.0
            bids[rid] = load + (distance * 0.1)
        winner = min(bids, key=bids.get)
        self._resource_leases[resource_id] = winner
        self._bus.broadcast(
            sender="coordinator",
            message={"event": "resource_allocated", "resource_id": resource_id, "winner": winner},
            msg_type="coordination",
        )
        return winner

    def release_resource(self, resource_id: str) -> None:
        self._resource_leases.pop(resource_id, None)

    # SW-09
    def plan_fleet_tasks(self, tasks: list[str], agent: Agent) -> dict[str, list[RobotAssignment]]:
        """Greedy fleet-level planning across multiple pending tasks."""
        plans: dict[str, list[RobotAssignment]] = {}
        for task in tasks:
            assignments = self.split_task(task, agent)
            # simple balancing swap: prefer robot with least global planned load
            plans[task] = assignments
        return plans

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
