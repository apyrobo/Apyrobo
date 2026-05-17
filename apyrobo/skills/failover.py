"""
Deterministic safe-state execution after skill failure.

When a skill fails or times out, the robot must be brought to a known-safe
state before the failure is propagated.  FailoverExecutor wraps SkillExecutor
and enforces this: every failure triggers a configured SafeStateAction before
returning the FAILED TaskResult.

Usage::

    policy = FailoverPolicy.default()
    failover = FailoverExecutor(executor, policy)
    result = failover.execute_graph(graph)
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from apyrobo.core.schemas import RecoveryAction, TaskResult, TaskStatus
from apyrobo.observability import emit_event
from apyrobo.skills.executor import SkillExecutor, SkillGraph
from apyrobo.skills.skill import SkillStatus

if TYPE_CHECKING:
    from apyrobo.core.robot import Robot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe-state action vocabulary
# ---------------------------------------------------------------------------

class SafeStateAction(enum.Enum):
    STOP = "stop"           # halt all motion — robot.stop()
    OPEN_GRIPPER = "open_gripper"   # robot.gripper_open()
    HOME = "home"           # home_fn() if provided, else robot.stop()
    DOCK = "dock"           # dock_fn() if provided, else robot.stop()
    NONE = "none"           # no-op (read-only skills like report_status)


# ---------------------------------------------------------------------------
# Failover policy
# ---------------------------------------------------------------------------

@dataclass
class FailoverPolicy:
    """Maps skill IDs to safe-state actions taken on failure or timeout."""

    default_action: SafeStateAction = SafeStateAction.STOP
    per_skill: dict[str, SafeStateAction] = field(default_factory=dict)

    def action_for(self, skill_id: str) -> SafeStateAction:
        """Return the safe-state action for a skill.

        Resolution order:
        1. Exact match in per_skill.
        2. Prefix match — any key in per_skill that is a prefix of skill_id
           (handles agent-generated suffixed IDs like ``pick_object_0``).
        3. default_action.
        """
        # Exact match
        if skill_id in self.per_skill:
            return self.per_skill[skill_id]

        # Prefix match (longest prefix wins for determinism)
        best_key: str | None = None
        for key in self.per_skill:
            if skill_id.startswith(key) and (best_key is None or len(key) > len(best_key)):
                best_key = key

        if best_key is not None:
            return self.per_skill[best_key]

        return self.default_action

    @classmethod
    def default(cls) -> "FailoverPolicy":
        """Sensible defaults: pick/place/grasp → OPEN_GRIPPER, read-only → NONE, rest → STOP."""
        return cls(
            default_action=SafeStateAction.STOP,
            per_skill={
                "pick_object": SafeStateAction.OPEN_GRIPPER,
                "place_object": SafeStateAction.OPEN_GRIPPER,
                "grasp": SafeStateAction.OPEN_GRIPPER,
                "report_status": SafeStateAction.NONE,
                "get_pose": SafeStateAction.NONE,
                "capture_image": SafeStateAction.NONE,
            },
        )


# ---------------------------------------------------------------------------
# Failover executor
# ---------------------------------------------------------------------------

class FailoverExecutor:
    """
    Wraps SkillExecutor to execute a safe-state action after any skill failure.

    On skill failure or timeout, calls the appropriate SafeStateAction before
    propagating the failure.  The robot is always left in a known-safe state.
    """

    def __init__(
        self,
        executor: SkillExecutor,
        policy: FailoverPolicy | None = None,
        home_fn: Callable[[], None] | None = None,
        dock_fn: Callable[[], None] | None = None,
    ) -> None:
        self._executor = executor
        self._policy = policy or FailoverPolicy.default()
        self._home_fn = home_fn
        self._dock_fn = dock_fn
        self._last_safe_state_action: SafeStateAction | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_safe_state(self, action: SafeStateAction, robot: Any) -> None:
        """Execute the safe-state action, swallowing errors (we're already in failure)."""
        try:
            if action == SafeStateAction.NONE:
                return
            elif action == SafeStateAction.STOP:
                robot.stop()
            elif action == SafeStateAction.OPEN_GRIPPER:
                robot.gripper_open()
            elif action == SafeStateAction.HOME:
                if self._home_fn is not None:
                    self._home_fn()
                else:
                    robot.stop()
            elif action == SafeStateAction.DOCK:
                if self._dock_fn is not None:
                    self._dock_fn()
                else:
                    robot.stop()
        except Exception as exc:
            logger.warning(
                "Safe-state action %s raised an error (suppressed): %s",
                action.value, exc,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_graph(self, graph: SkillGraph) -> TaskResult:
        """Execute graph; on first skill failure, run safe-state then return FAILED."""
        robot = self._executor._robot
        order = graph.get_execution_order()
        completed = 0

        for skill in order:
            params = graph.get_parameters(skill.skill_id)
            status = self._executor.execute_skill(skill, params)

            if status == SkillStatus.COMPLETED:
                completed += 1
            elif status == SkillStatus.FAILED:
                action = self._policy.action_for(skill.skill_id)
                self._last_safe_state_action = action

                # Execute safe-state action (errors suppressed)
                self._execute_safe_state(action, robot)

                # Emit observability event
                emit_event(
                    "skill.failover",
                    skill_id=skill.skill_id,
                    action=action.value,
                    reason="skill_failed",
                )

                return TaskResult(
                    task_name=f"graph_{len(order)}_skills",
                    status=TaskStatus.FAILED,
                    steps_completed=completed,
                    steps_total=len(order),
                    error=f"Skill {skill.skill_id!r} failed; safe-state: {action.value}",
                    recovery_actions_taken=[RecoveryAction.ABORT],
                )

        return TaskResult(
            task_name=f"graph_{len(order)}_skills",
            status=TaskStatus.COMPLETED,
            confidence=1.0,
            steps_completed=completed,
            steps_total=len(order),
            recovery_actions_taken=[],
        )

    @property
    def last_safe_state_action(self) -> SafeStateAction | None:
        """The last safe-state action that was triggered, or None."""
        return self._last_safe_state_action
