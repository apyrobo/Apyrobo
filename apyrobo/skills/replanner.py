"""LLM-based replanning when skill execution fails.

The Replanner is called by Agent.execute() on failure when replanning
is enabled. It asks the LLM provider to produce a revised plan given
context about what failed and what succeeded.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ReplanContext:
    """Context passed to the replanner on skill failure."""

    task: str
    failed_skill_id: str
    completed_steps: list[dict[str, Any]]
    replan_attempt: int
    available_skills: list[dict[str, Any]]
    capabilities: list[str]
    error: str = ""


class Replanner:
    """Asks an LLMProvider for a revised plan after a skill failure.

    The provider is called with an augmented task description that includes
    context about what failed and what succeeded, so the LLM can avoid
    repeating the same mistake.
    """

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    def replan(self, context: ReplanContext) -> list[dict[str, Any]]:
        """Return a new plan (list of skill step dicts) for the task."""
        completed_summary = (
            ", ".join(s["skill_id"] for s in context.completed_steps) or "none"
        )

        augmented_task = (
            f"{context.task}\n\n"
            f"[Replanning attempt {context.replan_attempt}]\n"
            f"Previous plan failed at skill '{context.failed_skill_id}': {context.error}\n"
            f"Completed steps: {completed_summary}\n"
            f"Please produce a revised plan that avoids the previous failure."
        )

        logger.info(
            "Replanning for task %r (attempt %d), failed at %r",
            context.task,
            context.replan_attempt,
            context.failed_skill_id,
        )

        return self._provider.plan(
            augmented_task, context.available_skills, context.capabilities
        )


class MockReplanner:
    """Deterministic replanner for tests — always returns a fixed plan."""

    def __init__(self, plan: list[dict[str, Any]]) -> None:
        self._plan = plan
        self.calls: list[ReplanContext] = []

    def replan(self, context: ReplanContext) -> list[dict[str, Any]]:
        self.calls.append(context)
        return list(self._plan)
