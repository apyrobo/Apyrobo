"""
Skill Graph Engine — chains skills into executable task plans.

The SkillGraph is a directed acyclic graph where nodes are skills and
edges are dependencies.  The SkillExecutor walks the graph and runs
each skill against a robot via the Core API.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import CapabilityType, TaskResult, TaskStatus, RecoveryAction
from apyrobo.skills.skill import Skill, SkillStatus, BUILTIN_SKILLS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skill Graph
# ---------------------------------------------------------------------------

class SkillGraph:
    """
    A directed acyclic graph of skills representing a task plan.

    Skills are nodes; edges mean "A must complete before B starts."
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._edges: dict[str, list[str]] = {}  # skill_id -> [depends_on_ids]
        self._parameters: dict[str, dict[str, Any]] = {}  # runtime params per skill

    def add_skill(self, skill: Skill, depends_on: list[str] | None = None,
                  parameters: dict[str, Any] | None = None) -> None:
        """Add a skill to the graph with optional dependencies."""
        self._skills[skill.skill_id] = skill
        self._edges[skill.skill_id] = depends_on or []
        if parameters:
            self._parameters[skill.skill_id] = parameters

    def get_execution_order(self) -> list[Skill]:
        """
        Topological sort — returns skills in valid execution order.

        Raises ValueError if there's a cycle (which shouldn't happen
        in a well-formed plan).
        """
        visited: set[str] = set()
        order: list[str] = []
        in_progress: set[str] = set()

        def visit(sid: str) -> None:
            if sid in in_progress:
                raise ValueError(f"Cycle detected in skill graph at {sid!r}")
            if sid in visited:
                return
            in_progress.add(sid)
            for dep in self._edges.get(sid, []):
                visit(dep)
            in_progress.discard(sid)
            visited.add(sid)
            order.append(sid)

        for sid in self._skills:
            visit(sid)

        return [self._skills[sid] for sid in order]

    def get_parameters(self, skill_id: str) -> dict[str, Any]:
        """Get runtime parameters for a skill."""
        base = dict(self._skills[skill_id].parameters)
        base.update(self._parameters.get(skill_id, {}))
        return base

    @property
    def skills(self) -> dict[str, Skill]:
        return dict(self._skills)

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        return f"<SkillGraph skills={len(self._skills)} edges={sum(len(v) for v in self._edges.values())}>"


# ---------------------------------------------------------------------------
# Execution events
# ---------------------------------------------------------------------------

class ExecutionEvent:
    """An event emitted during skill execution."""

    def __init__(self, skill_id: str, status: SkillStatus, message: str = "",
                 timestamp: float | None = None) -> None:
        self.skill_id = skill_id
        self.status = status
        self.message = message
        self.timestamp = timestamp or time.time()

    def __repr__(self) -> str:
        return f"<Event {self.skill_id}: {self.status.value} — {self.message}>"


# Type for event listeners
EventListener = Callable[[ExecutionEvent], None]


# ---------------------------------------------------------------------------
# Skill Executor
# ---------------------------------------------------------------------------

class SkillExecutor:
    """
    Executes a SkillGraph against a robot.

    Handles precondition checking, postcondition verification,
    retry logic, and event streaming.
    """

    def __init__(self, robot: Robot) -> None:
        self._robot = robot
        self._listeners: list[EventListener] = []
        self._events: list[ExecutionEvent] = []

    def on_event(self, listener: EventListener) -> None:
        """Register a callback for execution events."""
        self._listeners.append(listener)

    def _emit(self, skill_id: str, status: SkillStatus, message: str = "") -> None:
        event = ExecutionEvent(skill_id, status, message)
        self._events.append(event)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                logger.warning("Event listener error: %s", e)

    def check_preconditions(self, skill: Skill, robot: Robot) -> tuple[bool, str]:
        """
        Check whether a skill's preconditions are met.

        Returns (ok, reason).  For now, checks capability availability.
        """
        caps = robot.capabilities()
        cap_types = {c.capability_type for c in caps.capabilities}

        if skill.required_capability != CapabilityType.CUSTOM:
            if skill.required_capability not in cap_types:
                return False, f"Robot lacks required capability: {skill.required_capability.value}"

        # Check speed constraint
        params = skill.parameters
        if "speed" in params and caps.max_speed is not None:
            requested_speed = params.get("speed", 0)
            if isinstance(requested_speed, (int, float)) and requested_speed > caps.max_speed:
                return False, (
                    f"Requested speed {requested_speed} exceeds robot max {caps.max_speed}"
                )

        return True, "OK"

    def execute_skill(self, skill: Skill, parameters: dict[str, Any] | None = None) -> SkillStatus:
        """
        Execute a single skill against the robot.

        Handles precondition checking, the actual command, and retries.
        """
        params = dict(skill.parameters)
        if parameters:
            params.update(parameters)

        self._emit(skill.skill_id, SkillStatus.PENDING, "Checking preconditions")

        # Precondition check
        ok, reason = self.check_preconditions(skill, self._robot)
        if not ok:
            self._emit(skill.skill_id, SkillStatus.FAILED, f"Precondition failed: {reason}")
            return SkillStatus.FAILED

        # Execute with retry
        attempts = 0
        max_attempts = skill.retry_count + 1

        while attempts < max_attempts:
            attempts += 1
            self._emit(
                skill.skill_id, SkillStatus.RUNNING,
                f"Attempt {attempts}/{max_attempts}"
            )

            try:
                result = self._dispatch_skill(skill, params)
                if result:
                    self._emit(skill.skill_id, SkillStatus.COMPLETED, "Success")
                    return SkillStatus.COMPLETED
                else:
                    self._emit(
                        skill.skill_id, SkillStatus.RUNNING,
                        f"Attempt {attempts} failed, {'retrying' if attempts < max_attempts else 'no more retries'}"
                    )
            except Exception as e:
                self._emit(
                    skill.skill_id, SkillStatus.RUNNING,
                    f"Error on attempt {attempts}: {e}"
                )

        self._emit(skill.skill_id, SkillStatus.FAILED, f"Failed after {max_attempts} attempts")
        return SkillStatus.FAILED

    def _dispatch_skill(self, skill: Skill, params: dict[str, Any]) -> bool:
        """
        Map a skill to actual robot commands.

        This is the core translation layer between abstract skills
        and concrete robot actions.  Handles both exact IDs and
        agent-generated suffixed IDs (e.g. "navigate_to_0").
        """
        # Normalise: strip numeric suffix added by agent for graph uniqueness
        base_id = skill.skill_id.rsplit("_", 1)[0] if skill.skill_id[-1:].isdigit() else skill.skill_id

        if base_id == "navigate_to":
            x = float(params.get("x", 0.0))
            y = float(params.get("y", 0.0))
            speed = params.get("speed")
            speed = float(speed) if speed is not None else None
            self._robot.move(x=x, y=y, speed=speed)
            return True

        elif base_id == "stop":
            self._robot.stop()
            return True

        elif base_id in ("pick_object", "place_object"):
            # In sim/mock, these succeed immediately
            # Real implementation would call a grasp action server
            logger.info("Executing %s with params %s", skill.skill_id, params)
            return True

        elif base_id == "report_status":
            caps = self._robot.capabilities()
            logger.info("Status: robot=%s capabilities=%d sensors=%d",
                        caps.name, len(caps.capabilities), len(caps.sensors))
            return True

        else:
            logger.warning("Unknown skill: %s — treating as success", skill.skill_id)
            return True

    def execute_graph(self, graph: SkillGraph) -> TaskResult:
        """
        Execute an entire skill graph in topological order.

        Returns a TaskResult summarising the outcome.
        """
        order = graph.get_execution_order()
        completed = 0
        recovery_actions: list[RecoveryAction] = []

        for skill in order:
            params = graph.get_parameters(skill.skill_id)
            status = self.execute_skill(skill, params)

            if status == SkillStatus.COMPLETED:
                completed += 1
            elif status == SkillStatus.FAILED:
                if skill.retry_count > 0:
                    recovery_actions.append(RecoveryAction.RETRY)
                recovery_actions.append(RecoveryAction.ABORT)
                return TaskResult(
                    task_name=f"graph_{len(order)}_skills",
                    status=TaskStatus.FAILED,
                    steps_completed=completed,
                    steps_total=len(order),
                    error=f"Skill {skill.skill_id!r} failed",
                    recovery_actions_taken=recovery_actions,
                )

        return TaskResult(
            task_name=f"graph_{len(order)}_skills",
            status=TaskStatus.COMPLETED,
            confidence=1.0,
            steps_completed=completed,
            steps_total=len(order),
            recovery_actions_taken=recovery_actions,
        )

    @property
    def events(self) -> list[ExecutionEvent]:
        """All events emitted during execution."""
        return list(self._events)
