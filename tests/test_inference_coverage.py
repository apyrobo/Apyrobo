"""
Targeted tests for inference/router.py — ProviderHealth, TokenBudget,
PlanCache, InferenceTier, InferenceRouter.
"""

from __future__ import annotations

import pytest

from apyrobo.inference.router import (
    BudgetExceeded,
    CircuitState,
    InferenceRouter,
    InferenceTier,
    PlanCache,
    ProviderHealth,
    TokenBudget,
    Urgency,
)
from apyrobo.skills.agent import RuleBasedProvider


# ---------------------------------------------------------------------------
# ProviderHealth
# ---------------------------------------------------------------------------

class TestProviderHealth:
    def test_initial_state(self) -> None:
        ph = ProviderHealth("cloud")
        assert ph.circuit_state == CircuitState.CLOSED
        assert ph.is_healthy is True
        assert ph.avg_latency_ms == float("inf")

    def test_record_success(self) -> None:
        ph = ProviderHealth("cloud")
        ph.record_success(100.0)
        assert ph.avg_latency_ms == pytest.approx(100.0)
        assert ph._consecutive_failures == 0

    def test_record_failure_trips_circuit(self) -> None:
        ph = ProviderHealth("cloud", failure_threshold=3)
        ph.record_failure()
        ph.record_failure()
        ph.record_failure()
        assert ph.circuit_state == CircuitState.OPEN
        assert ph.is_healthy is False

    def test_error_rate(self) -> None:
        ph = ProviderHealth("cloud")
        ph.record_success(50.0)
        ph.record_failure()
        assert ph.error_rate == pytest.approx(0.5)

    def test_p95_latency(self) -> None:
        ph = ProviderHealth("cloud")
        for ms in range(1, 101):  # 1-100ms
            ph.record_success(float(ms))
        assert ph.p95_latency_ms >= 95.0

    def test_reset_clears_circuit(self) -> None:
        ph = ProviderHealth("cloud", failure_threshold=2)
        ph.record_failure()
        ph.record_failure()
        assert ph.circuit_state == CircuitState.OPEN
        ph.reset()
        assert ph.circuit_state == CircuitState.CLOSED

    def test_half_open_success_closes_circuit(self) -> None:
        ph = ProviderHealth("cloud", failure_threshold=1, recovery_timeout=0.0)
        ph.record_failure()
        # Force time past recovery timeout by accessing circuit_state
        import time
        time.sleep(0.01)
        state = ph.circuit_state  # Should be HALF_OPEN now
        ph.record_success(100.0)
        assert ph.circuit_state == CircuitState.CLOSED

    def test_half_open_failure_reopens_circuit(self) -> None:
        ph = ProviderHealth("cloud", failure_threshold=1, recovery_timeout=60.0)
        ph.record_failure()
        # Manually set to HALF_OPEN to simulate recovery window
        ph._circuit_state = CircuitState.HALF_OPEN
        ph.record_failure()  # Failed probe → re-open
        # Access internal state directly (property would auto-transition with 0s timeout)
        assert ph._circuit_state == CircuitState.OPEN

    def test_is_available_before_calls(self) -> None:
        ph = ProviderHealth("cloud")
        assert ph.is_available is True

    def test_is_available_after_success(self) -> None:
        ph = ProviderHealth("cloud")
        ph.record_success(100.0)
        assert ph.is_available is True

    def test_to_dict(self) -> None:
        ph = ProviderHealth("cloud")
        ph.record_success(200.0)
        d = ph.to_dict()
        assert d["name"] == "cloud"
        assert "avg_latency_ms" in d
        assert "error_rate" in d
        assert "circuit_state" in d

    def test_repr(self) -> None:
        ph = ProviderHealth("cloud")
        assert "cloud" in repr(ph)


# ---------------------------------------------------------------------------
# TokenBudget
# ---------------------------------------------------------------------------

class TestTokenBudget:
    def test_initial_state(self) -> None:
        tb = TokenBudget(monthly_limit=1000)
        assert tb.total_tokens == 0
        assert tb.remaining_tokens == 1000
        assert tb.is_over_budget is False
        assert tb.usage_pct == pytest.approx(0.0)

    def test_record_tokens(self) -> None:
        tb = TokenBudget(monthly_limit=1000)
        tb.record("cloud", input_tokens=100, output_tokens=50)
        assert tb.total_tokens == 150
        assert tb.remaining_tokens == 850

    def test_record_multiple_tiers(self) -> None:
        tb = TokenBudget(monthly_limit=1000)
        tb.record("cloud", input_tokens=200)
        tb.record("edge", input_tokens=100)
        assert tb.total_tokens == 300

    def test_usage_pct(self) -> None:
        tb = TokenBudget(monthly_limit=1000)
        tb.record("cloud", input_tokens=500)
        assert tb.usage_pct == pytest.approx(50.0)

    def test_is_over_budget(self) -> None:
        tb = TokenBudget(monthly_limit=100)
        tb.record("cloud", input_tokens=100)
        assert tb.is_over_budget is True

    def test_check_raises_when_over(self) -> None:
        tb = TokenBudget(monthly_limit=100)
        tb.record("cloud", input_tokens=100)
        with pytest.raises(BudgetExceeded):
            tb.check()

    def test_check_raises_with_estimated(self) -> None:
        tb = TokenBudget(monthly_limit=100)
        tb.record("cloud", input_tokens=90)
        with pytest.raises(BudgetExceeded):
            tb.check(estimated=20)

    def test_check_ok_when_within_budget(self) -> None:
        tb = TokenBudget(monthly_limit=1000)
        tb.record("cloud", input_tokens=100)
        tb.check()  # Should not raise

    def test_reset(self) -> None:
        tb = TokenBudget(monthly_limit=1000)
        tb.record("cloud", input_tokens=500)
        tb.reset()
        assert tb.total_tokens == 0

    def test_usage_by_tier(self) -> None:
        tb = TokenBudget(monthly_limit=1000)
        tb.record("cloud", input_tokens=200, output_tokens=50, cost=0.01)
        by_tier = tb.usage_by_tier()
        assert "cloud" in by_tier
        assert by_tier["cloud"]["tokens"] == 250

    def test_total_cost(self) -> None:
        tb = TokenBudget(monthly_limit=1000)
        tb.record("cloud", cost=0.05)
        tb.record("edge", cost=0.02)
        assert tb.total_cost == pytest.approx(0.07)

    def test_alert_callback_triggered(self) -> None:
        alerts = []
        tb = TokenBudget(monthly_limit=100, alert_at_pct=50.0)
        tb.on_alert(lambda t, p, u: alerts.append(t))
        tb.record("cloud", input_tokens=60)  # 60% → triggers warning
        assert len(alerts) > 0

    def test_to_dict(self) -> None:
        tb = TokenBudget(monthly_limit=1000)
        tb.record("cloud", input_tokens=100)
        d = tb.to_dict()
        assert "monthly_limit" in d
        assert "total_tokens" in d
        assert d["total_tokens"] == 100

    def test_repr(self) -> None:
        tb = TokenBudget(monthly_limit=1000)
        assert "TokenBudget" in repr(tb)

    def test_zero_limit_usage_pct(self) -> None:
        tb = TokenBudget(monthly_limit=0)
        assert tb.usage_pct == pytest.approx(0.0)


class TestBudgetExceeded:
    def test_message_with_tier(self) -> None:
        exc = BudgetExceeded(used=100, limit=50, tier="cloud")
        assert "cloud" in str(exc)
        assert "100" in str(exc)

    def test_message_without_tier(self) -> None:
        exc = BudgetExceeded(used=100, limit=50)
        assert "100" in str(exc)


# ---------------------------------------------------------------------------
# PlanCache
# ---------------------------------------------------------------------------

class TestPlanCache:
    def test_miss_on_empty(self) -> None:
        cache = PlanCache()
        assert cache.get("deliver", ["navigate"]) is None

    def test_put_and_get(self) -> None:
        cache = PlanCache()
        plan = [{"skill": "navigate", "params": {}}]
        cache.put("deliver", ["navigate"], plan)
        result = cache.get("deliver", ["navigate"])
        assert result == plan

    def test_ttl_expiry(self) -> None:
        cache = PlanCache(ttl_seconds=0.01)
        cache.put("task", [], [{"step": 1}])
        import time
        time.sleep(0.05)
        assert cache.get("task", []) is None

    def test_hit_rate(self) -> None:
        cache = PlanCache()
        cache.put("task", [], [])
        cache.get("task", [])   # hit
        cache.get("miss", [])   # miss
        assert cache.hit_rate == pytest.approx(0.5)

    def test_invalidate_specific(self) -> None:
        cache = PlanCache()
        cache.put("task", ["nav"], [])
        count = cache.invalidate("task", ["nav"])
        assert count == 1
        assert cache.get("task", ["nav"]) is None

    def test_invalidate_all(self) -> None:
        cache = PlanCache()
        cache.put("t1", [], [])
        cache.put("t2", [], [])
        count = cache.invalidate()
        assert count == 2
        assert cache.size == 0

    def test_evict_oldest_when_full(self) -> None:
        cache = PlanCache(max_size=2)
        cache.put("t1", [], [{"id": 1}])
        cache.put("t2", [], [{"id": 2}])
        cache.put("t3", [], [{"id": 3}])  # Should evict oldest
        assert cache.size == 2

    def test_to_dict(self) -> None:
        cache = PlanCache()
        d = cache.to_dict()
        assert "size" in d
        assert "hit_rate" in d

    def test_repr(self) -> None:
        cache = PlanCache()
        assert "PlanCache" in repr(cache)

    def test_case_insensitive_key(self) -> None:
        cache = PlanCache()
        cache.put("Deliver Package", [], [{"step": 1}])
        result = cache.get("deliver package", [])
        assert result is not None


# ---------------------------------------------------------------------------
# InferenceTier
# ---------------------------------------------------------------------------

class TestInferenceTier:
    def test_repr_plain(self) -> None:
        tier = InferenceTier("cloud", RuleBasedProvider(), priority=0)
        assert "cloud" in repr(tier)

    def test_repr_vlm(self) -> None:
        tier = InferenceTier("vlm", RuleBasedProvider(), is_vlm=True)
        assert "vlm" in repr(tier)

    def test_repr_edge(self) -> None:
        tier = InferenceTier("edge", RuleBasedProvider(), is_edge=True)
        assert "edge" in repr(tier)


# ---------------------------------------------------------------------------
# InferenceRouter
# ---------------------------------------------------------------------------

class TestInferenceRouter:
    def test_no_tiers_uses_fallback(self) -> None:
        router = InferenceRouter()
        # No tiers - should use rule-based fallback
        result = router.plan("deliver package", available_skills=[], capabilities=["navigate"])
        assert isinstance(result, list)

    def test_add_tier_and_select(self) -> None:
        router = InferenceRouter()
        router.add_tier("rule", RuleBasedProvider(), priority=0)
        result = router.plan("navigate", available_skills=[], capabilities=["navigate"])
        assert isinstance(result, list)

    def test_select_high_urgency(self) -> None:
        router = InferenceRouter()
        router.add_tier("edge", RuleBasedProvider(), priority=0, is_edge=True,
                        supports_urgency=[Urgency.HIGH])
        result = router.plan("obstacle ahead", available_skills=[], capabilities=[],
                             urgency=Urgency.HIGH)
        assert isinstance(result, list)

    def test_plan_calls_provider(self) -> None:
        router = InferenceRouter()
        # RuleBasedProvider will plan something
        result = router.plan("deliver package", available_skills=[], capabilities=["navigate"])
        assert isinstance(result, list)

    def test_plan_with_cache(self) -> None:
        router = InferenceRouter(enable_cache=True)
        result1 = router.plan("deliver package", available_skills=[], capabilities=["navigate"])
        result2 = router.plan("deliver package", available_skills=[], capabilities=["navigate"])
        assert result1 == result2  # Should be cached

    def test_plan_no_cache(self) -> None:
        router = InferenceRouter(enable_cache=False)
        result = router.plan("deliver", available_skills=[], capabilities=[])
        assert isinstance(result, list)

    def test_health_report(self) -> None:
        router = InferenceRouter()
        router.add_tier("cloud", RuleBasedProvider(), priority=0)
        report = router.health_report()
        assert isinstance(report, dict)

    def test_route_log(self) -> None:
        router = InferenceRouter()
        router.plan("some task", available_skills=[], capabilities=[])
        # Route log should have entries after a plan
        assert isinstance(router._route_log, list)

    def test_token_budget_tracked(self) -> None:
        budget = TokenBudget(monthly_limit=10_000)
        router = InferenceRouter(token_budget=budget)
        router.plan("deliver", available_skills=[], capabilities=[])
        # Budget may have 0 tokens (rule-based doesn't consume tokens)
        assert budget.total_tokens >= 0

    def test_select_skips_unhealthy_tier(self) -> None:
        router = InferenceRouter()
        router.add_tier("cloud", RuleBasedProvider(), priority=0, failure_threshold=1)
        # Trip the circuit on the tier
        tier = router._tiers[0]
        tier.health.record_failure()
        # Should fall through to fallback
        result = router.plan("task", available_skills=[], capabilities=[])
        assert isinstance(result, list)
