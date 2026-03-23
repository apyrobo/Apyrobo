"""
Execution Feedback Loop — records skill outcomes and adapts retry strategy.

Classes:
    ExecutionResult   — outcome of one skill execution
    FeedbackCollector — accumulates results, computes success rates
    AdaptiveExecutor  — wraps skill calls with automatic retry + recording
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Outcome of a single skill execution."""

    skill_name: str
    success: bool
    duration_ms: float
    error: str | None
    output: Any = None


class FeedbackCollector:
    """
    Accumulates ExecutionResults and exposes aggregate statistics.

    Usage:
        collector = FeedbackCollector()
        collector.record(ExecutionResult("navigate_to", True, 120.0, None))
        collector.success_rate("navigate_to")  # → 1.0
    """

    def __init__(self) -> None:
        self._results: dict[str, list[ExecutionResult]] = {}

    def record(self, result: ExecutionResult) -> None:
        self._results.setdefault(result.skill_name, []).append(result)
        status = "OK" if result.success else f"FAIL({result.error})"
        logger.debug(
            "Feedback: %s %s %.1f ms", result.skill_name, status, result.duration_ms
        )

    def success_rate(self, skill_name: str) -> float:
        """Return fraction of successful executions for *skill_name* (0.0–1.0)."""
        results = self._results.get(skill_name, [])
        if not results:
            return 1.0  # no data → assume healthy
        return sum(1 for r in results if r.success) / len(results)

    def degraded_skills(self, threshold: float = 0.8) -> list[str]:
        """Return skill names whose success rate is below *threshold*."""
        return [name for name in self._results if self.success_rate(name) < threshold]

    def summary(self) -> dict[str, Any]:
        """Return a dict with per-skill stats and overall degraded list."""
        per_skill = {}
        for name, results in self._results.items():
            durations = [r.duration_ms for r in results]
            per_skill[name] = {
                "total": len(results),
                "success": sum(1 for r in results if r.success),
                "success_rate": self.success_rate(name),
                "avg_duration_ms": sum(durations) / len(durations) if durations else 0.0,
            }
        return {
            "skills": per_skill,
            "degraded": self.degraded_skills(),
        }

    def clear(self) -> None:
        self._results.clear()


class AdaptiveExecutor:
    """
    Wraps skill calls with retry logic and feedback recording.

    Retry policy adapts based on observed success rate:
    - success_rate >= 0.9 → 1 attempt
    - success_rate >= 0.7 → 2 attempts
    - success_rate <  0.7 → max_retries attempts
    """

    def __init__(
        self,
        collector: FeedbackCollector | None = None,
        max_retries: int = 3,
        retry_delay_ms: float = 200.0,
    ) -> None:
        self.collector = collector or FeedbackCollector()
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms

    def _attempts_for(self, skill_name: str) -> int:
        # No prior data → use max_retries (safe default)
        if skill_name not in self.collector._results:
            return self.max_retries
        rate = self.collector.success_rate(skill_name)
        if rate >= 0.9:
            return 1
        if rate >= 0.7:
            return 2
        return self.max_retries

    def execute(
        self,
        skill_name: str,
        params: dict[str, Any],
        executor: Callable[..., Any],
    ) -> ExecutionResult:
        """
        Call *executor* with **params*, retrying on failure.

        Records every attempt in the FeedbackCollector and returns the
        result of the last attempt.
        """
        attempts = self._attempts_for(skill_name)
        last_result: ExecutionResult | None = None

        for attempt in range(1, attempts + 1):
            start = time.time()
            try:
                output = executor(**params)
                duration_ms = (time.time() - start) * 1000
                last_result = ExecutionResult(
                    skill_name=skill_name,
                    success=True,
                    duration_ms=duration_ms,
                    error=None,
                    output=output,
                )
                self.collector.record(last_result)
                return last_result
            except Exception as exc:
                duration_ms = (time.time() - start) * 1000
                last_result = ExecutionResult(
                    skill_name=skill_name,
                    success=False,
                    duration_ms=duration_ms,
                    error=str(exc),
                    output=None,
                )
                self.collector.record(last_result)
                if attempt < attempts:
                    logger.warning(
                        "Skill %r attempt %d/%d failed: %s — retrying",
                        skill_name, attempt, attempts, exc,
                    )
                    time.sleep(self.retry_delay_ms / 1000)

        assert last_result is not None
        return last_result
