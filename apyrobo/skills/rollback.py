"""
Rollback execution — undo committed skill steps on plan failure.

When a multi-step plan fails partway through, completed state changes are
normally left in place. For example: navigate_to succeeds (robot has moved),
pick_object fails — the robot is now at the wrong position with no object.

RollbackExecutor wraps SkillExecutor.execute_graph() and undoes committed
steps in reverse order (LIFO) when a failure occurs.

Usage:
    registry = RollbackRegistry(robot)
    rollback_exec = RollbackExecutor(executor, registry)
    result = rollback_exec.execute_graph(graph)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Any

from apyrobo.skills.executor import SkillExecutor, SkillGraph, SkillStatus
from apyrobo.core.schemas import TaskResult, TaskStatus, RecoveryAction
from apyrobo.observability import emit_event

logger = logging.getLogger(__name__)


@dataclass
class RollbackAction:
    """An undo action for a completed skill step."""

    skill_id: str
    undo: Callable[[], None]
    description: str = ""


class RollbackRegistry:
    """
    Maps skill IDs to undo-action factories.

    Each factory receives the skill parameters and returns a Callable[[], None]
    that undoes that skill's effect.

    Built-in mappings:
    - navigate_to → robot.move(original_x, original_y) back to pre-skill position
    - pick_object → robot.gripper_open()
    - place_object → robot.gripper_close()
    - stop → no-op
    - Everything else → no-op by default

    The registry is extensible: callers can register custom undo factories.
    """

    def __init__(self, robot: Any) -> None:
        self._robot = robot
        self._factories: dict[str, Callable[[dict], Callable[[], None]]] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        robot = self._robot

        def _navigate_to_factory(params: dict) -> Callable[[], None]:
            # Capture the robot's position RIGHT NOW (before the skill runs)
            try:
                pre_x, pre_y = robot.get_position()
            except Exception:
                pre_x, pre_y = 0.0, 0.0

            def _undo() -> None:
                robot.move(pre_x, pre_y)

            return _undo

        def _pick_object_factory(params: dict) -> Callable[[], None]:
            def _undo() -> None:
                robot.gripper_open()

            return _undo

        def _place_object_factory(params: dict) -> Callable[[], None]:
            def _undo() -> None:
                robot.gripper_close()

            return _undo

        def _noop_factory(params: dict) -> Callable[[], None]:
            return lambda: None

        self._factories["navigate_to"] = _navigate_to_factory
        self._factories["pick_object"] = _pick_object_factory
        self._factories["place_object"] = _place_object_factory
        self._factories["stop"] = _noop_factory

    def register(
        self, skill_id: str, factory: Callable[[dict], Callable[[], None]]
    ) -> None:
        """Register a custom undo factory for a skill."""
        self._factories[skill_id] = factory

    def get_undo(
        self, skill_id: str, params: dict[str, Any]
    ) -> Callable[[], None]:
        """Return an undo callable for the given skill and params.

        The factory is called immediately (before the skill runs) so it can
        capture pre-execution state (e.g. current robot position). Returns a
        no-op if no factory is registered for the skill.
        """
        # Normalise suffixed skill IDs produced by the agent planner
        # (e.g. "navigate_to_0" → "navigate_to")
        base_id = skill_id
        if "_" in skill_id:
            # Strip trailing numeric suffix if present
            parts = skill_id.rsplit("_", 1)
            if parts[-1].isdigit():
                base_id = parts[0]

        factory = self._factories.get(skill_id) or self._factories.get(base_id)
        if factory is None:
            return lambda: None

        try:
            return factory(params)
        except Exception as e:
            logger.warning(
                "RollbackRegistry: factory for %r raised %s — using no-op",
                skill_id,
                e,
            )
            return lambda: None


class RollbackExecutor:
    """
    Wraps SkillExecutor to undo committed steps on plan failure.

    Tracks a rollback stack as skills complete. On failure, calls undo
    functions in reverse order (last-in, first-out).

    Safe: undo functions are called with best-effort; errors are logged
    but do not stop remaining undos.

    Usage:
        registry = RollbackRegistry(robot)
        rollback_exec = RollbackExecutor(executor, registry)
        result = rollback_exec.execute_graph(graph)
    """

    def __init__(
        self,
        executor: SkillExecutor,
        registry: RollbackRegistry | None = None,
    ) -> None:
        self._executor = executor
        self._registry = registry
        self._rollback_stack: list[RollbackAction] = []
        self._last_rollback: list[str] = []

    def execute_graph(self, graph: SkillGraph) -> TaskResult:
        """
        Execute graph with rollback on failure.

        On skill failure:
        1. Call undo functions in reverse order
        2. Emit "plan.rolled_back" event with list of rolled-back skill IDs
        3. Return TaskResult with status=FAILED and rolled_back=True in metadata
        """
        self._rollback_stack = []
        self._last_rollback = []

        order = graph.get_execution_order()
        completed = 0
        recovery_actions: list[RecoveryAction] = []

        for skill in order:
            params = graph.get_parameters(skill.skill_id)

            # Capture undo callable BEFORE executing the skill so it can
            # snapshot pre-execution state (e.g. robot position)
            if self._registry is not None:
                undo_fn = self._registry.get_undo(skill.skill_id, params)
            else:
                undo_fn = lambda: None  # noqa: E731

            status = self._executor.execute_skill(skill, params)

            if status == SkillStatus.COMPLETED:
                # Push onto rollback stack only after confirmed success
                action = RollbackAction(
                    skill_id=skill.skill_id,
                    undo=undo_fn,
                    description=f"Undo {skill.skill_id}",
                )
                self._rollback_stack.append(action)
                completed += 1
            elif status == SkillStatus.FAILED:
                if skill.retry_count > 0:
                    recovery_actions.append(RecoveryAction.RETRY)
                recovery_actions.append(RecoveryAction.ABORT)

                # Perform rollback in LIFO order
                rolled_back_ids = self._do_rollback(
                    reason=f"Skill {skill.skill_id!r} failed"
                )

                return TaskResult(
                    task_name=f"graph_{len(order)}_skills",
                    status=TaskStatus.FAILED,
                    steps_completed=completed,
                    steps_total=len(order),
                    error=f"Skill {skill.skill_id!r} failed",
                    recovery_actions_taken=recovery_actions,
                    metadata={
                        "rolled_back": True,
                        "rolled_back_skills": rolled_back_ids,
                    },
                )

        return TaskResult(
            task_name=f"graph_{len(order)}_skills",
            status=TaskStatus.COMPLETED,
            confidence=1.0,
            steps_completed=completed,
            steps_total=len(order),
            recovery_actions_taken=recovery_actions,
            metadata={"rolled_back": False},
        )

    def _do_rollback(self, reason: str) -> list[str]:
        """Call undo functions in LIFO order; log errors without stopping."""
        rolled_back_ids: list[str] = []

        for action in reversed(self._rollback_stack):
            rolled_back_ids.append(action.skill_id)
            try:
                action.undo()
                logger.info(
                    "Rollback: undid %r (%s)", action.skill_id, action.description
                )
            except Exception as e:
                logger.warning(
                    "Rollback: undo for %r raised %s — continuing",
                    action.skill_id,
                    e,
                )

        self._last_rollback = rolled_back_ids

        emit_event(
            "plan.rolled_back",
            rolled_back_skills=rolled_back_ids,
            reason=reason,
        )

        return rolled_back_ids

    @property
    def rollback_stack(self) -> list[RollbackAction]:
        """The current rollback stack (completed actions that can be undone)."""
        return list(self._rollback_stack)

    @property
    def last_rollback(self) -> list[str]:
        """Skill IDs that were rolled back in the last failed execution, in reverse order."""
        return list(self._last_rollback)
