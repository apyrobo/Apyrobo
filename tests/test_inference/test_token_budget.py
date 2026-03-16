"""
IN-02: Token budget tracking — integration tests for InferenceRouter.

Tests:
- BudgetExceeded raised when token limit exceeded
- Router routes to cheaper tier when premium tier is over budget
- budget.record() called with actual usage after successful call
- emit_event('budget_alert') fires at alert threshold
- get_budget_status() returns used/limit/percentage
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from apyrobo.inference.router import (
    BudgetExceeded,
    InferenceRouter,
    InferenceTier,
    TokenBudget,
    Urgency,
)
from apyrobo.skills.agent import AgentProvider, RuleBasedProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeProvider(AgentProvider):
    """A provider that returns a fixed plan."""

    def __init__(self, name: str = "fake", plan_result: list | None = None) -> None:
        self._name = name
        self._plan_result = plan_result or [{"skill_id": "navigate_to", "parameters": {"x": 1, "y": 2}}]

    def plan(self, task: str, available_skills: list[dict[str, Any]],
             capabilities: list[str]) -> list[dict[str, Any]]:
        return list(self._plan_result)


SAMPLE_SKILLS: list[dict[str, Any]] = [
    {"skill_id": "navigate_to", "description": "Navigate", "parameters": {"x": 0.0, "y": 0.0}},
]
CAPABILITIES = ["navigation"]


# ===========================================================================
# TokenBudget unit tests
# ===========================================================================

class TestTokenBudget:

    def test_check_raises_when_over_limit(self) -> None:
        """BudgetExceeded raised when token limit exceeded."""
        budget = TokenBudget(monthly_limit=1000)
        budget.record("cloud", input_tokens=900, output_tokens=0)
        # 900 used + 200 estimated > 1000 limit
        with pytest.raises(BudgetExceeded) as exc_info:
            budget.check(estimated=200)
        assert exc_info.value.used == 900
        assert exc_info.value.limit == 1000

    def test_check_passes_when_under_limit(self) -> None:
        """check() does not raise when within budget."""
        budget = TokenBudget(monthly_limit=1000)
        budget.record("cloud", input_tokens=100, output_tokens=0)
        budget.check(estimated=200)  # Should not raise

    def test_record_updates_usage(self) -> None:
        """budget.record() updates total_tokens correctly."""
        budget = TokenBudget(monthly_limit=10000)
        budget.record("cloud", input_tokens=100, output_tokens=50)
        assert budget.total_tokens == 150
        budget.record("edge", input_tokens=200, output_tokens=100)
        assert budget.total_tokens == 450

    def test_emit_event_budget_alert_at_threshold(self) -> None:
        """emit_event('budget_alert') fires when alert threshold reached."""
        budget = TokenBudget(monthly_limit=1000, alert_at_pct=90.0)

        with patch("apyrobo.inference.router.emit_event") as mock_emit:
            budget.record("cloud", input_tokens=900, output_tokens=5)
            mock_emit.assert_any_call(
                "budget_alert",
                pct=pytest.approx(90.5, abs=1),
                used=905,
                limit=1000,
            )

    def test_emit_event_budget_exceeded(self) -> None:
        """emit_event('budget_exceeded') fires at 100%."""
        budget = TokenBudget(monthly_limit=100)

        with patch("apyrobo.inference.router.emit_event") as mock_emit:
            budget.record("cloud", input_tokens=100, output_tokens=5)
            mock_emit.assert_any_call(
                "budget_exceeded",
                pct=100.0,
                used=105,
                limit=100,
            )

    def test_get_budget_status_dict(self) -> None:
        """get_budget_status() returns used/limit/percentage."""
        budget = TokenBudget(monthly_limit=10000)
        budget.record("cloud", input_tokens=500, output_tokens=500)
        status = budget.to_dict()
        assert status["total_tokens"] == 1000
        assert status["monthly_limit"] == 10000
        assert status["usage_pct"] == 10.0
        assert status["remaining"] == 9000
        assert status["is_over_budget"] is False


# ===========================================================================
# InferenceRouter budget integration tests
# ===========================================================================

class TestRouterBudgetIntegration:

    def test_router_skips_over_budget_tier(self) -> None:
        """Router routes to cheaper tier when premium tier is over budget."""
        premium_budget = TokenBudget(monthly_limit=100)
        # Exhaust the premium budget
        premium_budget.record("premium", input_tokens=100, output_tokens=0)

        cheap_provider = FakeProvider(name="cheap", plan_result=[
            {"skill_id": "navigate_to", "parameters": {"x": 5, "y": 5}},
        ])
        premium_provider = FakeProvider(name="premium", plan_result=[
            {"skill_id": "navigate_to", "parameters": {"x": 99, "y": 99}},
        ])

        router = InferenceRouter(enable_cache=False)
        router.add_tier("premium", premium_provider, priority=0, budget=premium_budget)
        router.add_tier("cheap", cheap_provider, priority=1)

        with patch("apyrobo.inference.router.emit_event"):
            plan = router.plan("go to 5,5", SAMPLE_SKILLS, CAPABILITIES)

        # Should get the cheap provider's plan since premium is over budget
        assert plan[0]["parameters"]["x"] == 5

    def test_router_uses_tier_when_under_budget(self) -> None:
        """Router uses the preferred tier when it's within budget."""
        budget = TokenBudget(monthly_limit=100000)
        premium_provider = FakeProvider(name="premium")

        router = InferenceRouter(enable_cache=False)
        router.add_tier("premium", premium_provider, priority=0, budget=budget)

        plan = router.plan("go to 1,2", SAMPLE_SKILLS, CAPABILITIES)
        assert len(plan) == 1
        assert plan[0]["skill_id"] == "navigate_to"

    def test_try_tier_records_on_per_tier_budget(self) -> None:
        """budget.record() called with actual usage after successful call."""
        tier_budget = TokenBudget(monthly_limit=100000)
        provider = FakeProvider()

        router = InferenceRouter(enable_cache=False)
        router.add_tier("cloud", provider, priority=0, budget=tier_budget)

        router.plan("go somewhere", SAMPLE_SKILLS, CAPABILITIES)

        # Per-tier budget should have been recorded
        assert tier_budget.total_tokens > 0

    def test_budget_exceeded_emits_event(self) -> None:
        """On BudgetExceeded: emit_event('budget_exceeded') is called."""
        budget = TokenBudget(monthly_limit=1)
        budget.record("tier", input_tokens=10, output_tokens=0)

        provider = FakeProvider()
        router = InferenceRouter(enable_cache=False)
        router.add_tier("expensive", provider, priority=0, budget=budget)
        router.add_tier("fallback", FakeProvider(), priority=1)

        with patch("apyrobo.inference.router.emit_event") as mock_emit:
            router.plan("go", SAMPLE_SKILLS, CAPABILITIES)
            mock_emit.assert_any_call("budget_exceeded", tier="expensive")

    def test_get_budget_status(self) -> None:
        """get_budget_status() returns used/limit/percentage."""
        budget = TokenBudget(monthly_limit=5000)
        tier_budget = TokenBudget(monthly_limit=2000)
        router = InferenceRouter(token_budget=budget, enable_cache=False)
        router.add_tier("cloud", FakeProvider(), priority=0, budget=tier_budget)

        router.plan("go to 1,2", SAMPLE_SKILLS, CAPABILITIES)

        status = router.get_budget_status()
        assert "total_tokens" in status
        assert "monthly_limit" in status
        assert "usage_pct" in status
        assert "tier_budgets" in status
        assert "cloud" in status["tier_budgets"]

    def test_global_budget_exceeded_falls_to_rule_based(self) -> None:
        """Global token budget exceeded → rule-based fallback."""
        global_budget = TokenBudget(monthly_limit=1)
        global_budget.record("any", input_tokens=10, output_tokens=0)

        router = InferenceRouter(token_budget=global_budget, enable_cache=False)
        router.add_tier("cloud", FakeProvider(), priority=0)

        plan = router.plan("go to position 1, 2", SAMPLE_SKILLS, CAPABILITIES)
        # Should get a result from rule-based fallback (may or may not match)
        assert isinstance(plan, list)
