"""
Inference Router — latency-aware routing between edge and cloud AI.

The router sits between the Agent and LLM providers. It manages multiple
inference tiers (cloud, edge, local, fallback) and dynamically routes
requests based on:

    - Measured latency per provider
    - Connectivity status (is cloud reachable?)
    - Task urgency (reactive decisions vs complex planning)
    - Provider health (error rate, consecutive failures)

Architecture:
    ┌─────────────────────────────────────────────────┐
    │  Agent.plan("deliver package")                  │
    │       │                                         │
    │       ▼                                         │
    │  InferenceRouter                                │
    │       │                                         │
    │       ├─── urgency=HIGH? ──► Edge/Local model   │
    │       │       (< 500ms, always available)       │
    │       │                                         │
    │       ├─── urgency=NORMAL? ──► Cloud preferred  │
    │       │       (better quality, ~1-3s latency)   │
    │       │       └── cloud down? ──► Edge fallback │
    │       │                                         │
    │       └─── urgency=LOW? ──► Cloud (batch OK)    │
    │               (best quality, latency tolerant)  │
    │                                                 │
    │  Safety enforcer + motor commands: ALWAYS LOCAL  │
    │  (never routed through LLM — hardcoded in ROS 2)│
    └─────────────────────────────────────────────────┘

Usage:
    router = InferenceRouter(config={
        "cloud": {"model": "claude-sonnet-4-20250514", "max_latency_ms": 5000},
        "edge":  {"model": "ollama/llama3", "max_latency_ms": 1000},
    })
    agent = Agent(provider="routed", router=router)

    # Router automatically picks the right tier
    result = agent.execute("deliver package", robot)

    # Force edge for time-critical reactive decisions
    result = agent.execute("obstacle ahead, reroute", robot, urgency="high")
"""

from __future__ import annotations

import logging
import time
import threading
from collections import deque
from enum import Enum
from typing import Any

from apyrobo.skills.agent import AgentProvider, RuleBasedProvider, LLMProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Urgency levels
# ---------------------------------------------------------------------------

class Urgency(str, Enum):
    """How time-critical is this inference request."""
    HIGH = "high"       # Reactive: obstacle avoidance, emergency reroute (<500ms)
    NORMAL = "normal"   # Standard planning: task decomposition (~1-5s OK)
    LOW = "low"         # Background: plan optimisation, learning (latency tolerant)


# ---------------------------------------------------------------------------
# Provider health tracking
# ---------------------------------------------------------------------------

class ProviderHealth:
    """Tracks latency, availability, and error rate for a single provider."""

    def __init__(self, name: str, max_history: int = 50) -> None:
        self.name = name
        self._latencies: deque[float] = deque(maxlen=max_history)
        self._errors: deque[bool] = deque(maxlen=max_history)
        self._consecutive_failures = 0
        self._last_success: float | None = None
        self._last_failure: float | None = None
        self._total_calls = 0
        self._total_errors = 0

    def record_success(self, latency_ms: float) -> None:
        """Record a successful inference call."""
        self._latencies.append(latency_ms)
        self._errors.append(False)
        self._consecutive_failures = 0
        self._last_success = time.time()
        self._total_calls += 1

    def record_failure(self, error: str = "") -> None:
        """Record a failed inference call."""
        self._errors.append(True)
        self._consecutive_failures += 1
        self._last_failure = time.time()
        self._total_calls += 1
        self._total_errors += 1
        logger.warning("Provider %s failed (%d consecutive): %s",
                       self.name, self._consecutive_failures, error)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency over recent calls."""
        if not self._latencies:
            return float("inf")
        return sum(self._latencies) / len(self._latencies)

    @property
    def p95_latency_ms(self) -> float:
        """95th percentile latency."""
        if not self._latencies:
            return float("inf")
        sorted_lat = sorted(self._latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def error_rate(self) -> float:
        """Recent error rate (0.0 to 1.0)."""
        if not self._errors:
            return 0.0
        return sum(1 for e in self._errors if e) / len(self._errors)

    @property
    def is_healthy(self) -> bool:
        """Is this provider currently usable."""
        # Unhealthy if 3+ consecutive failures
        if self._consecutive_failures >= 3:
            # But allow retry after 30 seconds
            if self._last_failure and time.time() - self._last_failure > 30:
                return True
            return False
        return True

    @property
    def is_available(self) -> bool:
        """Has this provider ever succeeded."""
        return self._last_success is not None or self._total_calls == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "error_rate": round(self.error_rate, 3),
            "consecutive_failures": self._consecutive_failures,
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "is_healthy": self.is_healthy,
        }

    def __repr__(self) -> str:
        return (
            f"<ProviderHealth {self.name}: "
            f"avg={self.avg_latency_ms:.0f}ms "
            f"err={self.error_rate:.0%} "
            f"{'OK' if self.is_healthy else 'DOWN'}>"
        )


# ---------------------------------------------------------------------------
# Inference Tier
# ---------------------------------------------------------------------------

class InferenceTier:
    """A configured inference provider with its constraints."""

    def __init__(
        self,
        name: str,
        provider: AgentProvider,
        max_latency_ms: float = 5000,
        priority: int = 0,
        supports_urgency: list[Urgency] | None = None,
    ) -> None:
        self.name = name
        self.provider = provider
        self.max_latency_ms = max_latency_ms
        self.priority = priority  # lower = preferred
        self.supports_urgency = supports_urgency or list(Urgency)
        self.health = ProviderHealth(name)

    def __repr__(self) -> str:
        return f"<Tier {self.name} pri={self.priority} {self.health}>"


# ---------------------------------------------------------------------------
# Inference Router
# ---------------------------------------------------------------------------

class InferenceRouter(AgentProvider):
    """
    Routes inference requests across multiple providers based on
    urgency, latency, and provider health.

    Implements the AgentProvider interface so it can be used directly
    as a provider in the Agent.

    Tiers are tried in priority order (lowest first). If a tier fails
    or exceeds its latency budget, the router falls through to the next.
    The rule-based provider is always the last-resort fallback.

    Usage:
        router = InferenceRouter()
        router.add_tier("cloud", LLMProvider(model="claude-sonnet-4-20250514"),
                        max_latency_ms=5000, priority=0,
                        supports_urgency=[Urgency.NORMAL, Urgency.LOW])
        router.add_tier("edge", LLMProvider(model="ollama/llama3:8b"),
                        max_latency_ms=1000, priority=1,
                        supports_urgency=[Urgency.HIGH, Urgency.NORMAL])

        # Router picks the best tier for each request
        plan = router.plan(task, skills, caps)

        # Or with explicit urgency
        plan = router.plan(task, skills, caps, urgency=Urgency.HIGH)
    """

    def __init__(self) -> None:
        self._tiers: list[InferenceTier] = []
        self._fallback = RuleBasedProvider()
        self._fallback_health = ProviderHealth("rule_fallback")
        self._lock = threading.Lock()
        self._route_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def add_tier(
        self,
        name: str,
        provider: AgentProvider,
        max_latency_ms: float = 5000,
        priority: int | None = None,
        supports_urgency: list[Urgency] | None = None,
    ) -> None:
        """Add an inference tier. Lower priority = preferred."""
        if priority is None:
            priority = len(self._tiers)
        tier = InferenceTier(
            name=name, provider=provider,
            max_latency_ms=max_latency_ms, priority=priority,
            supports_urgency=supports_urgency,
        )
        self._tiers.append(tier)
        self._tiers.sort(key=lambda t: t.priority)
        logger.info("Router: added tier %s (priority=%d, max_latency=%dms)",
                     name, priority, max_latency_ms)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> InferenceRouter:
        """
        Build a router from a config dict.

        Config format:
            {
                "cloud": {"model": "claude-sonnet-4-20250514", "max_latency_ms": 5000, "priority": 0},
                "edge": {"model": "ollama/llama3:8b", "max_latency_ms": 1000, "priority": 1},
            }
        """
        router = cls()
        for name, tier_config in config.items():
            model = tier_config.get("model")
            if model:
                try:
                    provider = LLMProvider(model=model)
                except Exception:
                    logger.warning("Could not create LLM provider for %s (%s), skipping",
                                   name, model)
                    continue
            else:
                provider = RuleBasedProvider()

            urgency_strs = tier_config.get("supports_urgency")
            urgency = [Urgency(u) for u in urgency_strs] if urgency_strs else None

            router.add_tier(
                name=name,
                provider=provider,
                max_latency_ms=tier_config.get("max_latency_ms", 5000),
                priority=tier_config.get("priority", len(router._tiers)),
                supports_urgency=urgency,
            )
        return router

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def plan(self, task: str, available_skills: list[dict[str, Any]],
             capabilities: list[str], urgency: Urgency | str = Urgency.NORMAL,
             **kwargs: Any) -> list[dict[str, Any]]:
        """
        Route a planning request to the best available tier.

        Tries tiers in priority order, filtering by urgency and health.
        Falls back to rule-based provider if all tiers fail.
        """
        if isinstance(urgency, str):
            urgency = Urgency(urgency)

        # Find eligible tiers
        eligible = [
            t for t in self._tiers
            if urgency in t.supports_urgency and t.health.is_healthy
        ]

        # Try each eligible tier
        for tier in eligible:
            result = self._try_tier(tier, task, available_skills, capabilities)
            if result is not None:
                return result

        # All tiers failed — use rule-based fallback
        logger.warning("Router: all tiers failed, using rule-based fallback")
        t0 = time.time()
        try:
            result = self._fallback.plan(task, available_skills, capabilities)
            latency = (time.time() - t0) * 1000
            self._fallback_health.record_success(latency)
            self._log_route("rule_fallback", latency, True, urgency)
            return result
        except Exception as e:
            self._fallback_health.record_failure(str(e))
            self._log_route("rule_fallback", 0, False, urgency)
            return []

    def _try_tier(
        self, tier: InferenceTier, task: str,
        available_skills: list[dict[str, Any]], capabilities: list[str],
    ) -> list[dict[str, Any]] | None:
        """Attempt a single tier. Returns None on failure."""
        t0 = time.time()
        try:
            result = tier.provider.plan(task, available_skills, capabilities)
            latency_ms = (time.time() - t0) * 1000

            # Check latency budget
            if latency_ms > tier.max_latency_ms:
                logger.warning(
                    "Router: tier %s responded in %.0fms (budget: %.0fms) — "
                    "succeeded but slow",
                    tier.name, latency_ms, tier.max_latency_ms,
                )
                # Still count as success but log the overage
                tier.health.record_success(latency_ms)
                self._log_route(tier.name, latency_ms, True, note="over_budget")
                return result

            tier.health.record_success(latency_ms)
            self._log_route(tier.name, latency_ms, True)
            logger.info("Router: tier %s responded in %.0fms", tier.name, latency_ms)
            return result

        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            tier.health.record_failure(str(e))
            self._log_route(tier.name, latency_ms, False, error=str(e))
            logger.warning("Router: tier %s failed in %.0fms: %s",
                           tier.name, latency_ms, e)
            return None

    # ------------------------------------------------------------------
    # Health monitoring
    # ------------------------------------------------------------------

    def health_report(self) -> dict[str, Any]:
        """Full health report for all tiers."""
        tiers = []
        for t in self._tiers:
            report = t.health.to_dict()
            report["max_latency_ms"] = t.max_latency_ms
            report["priority"] = t.priority
            report["supports_urgency"] = [u.value for u in t.supports_urgency]
            tiers.append(report)

        return {
            "tiers": tiers,
            "fallback": self._fallback_health.to_dict(),
            "total_routes": len(self._route_log),
            "tier_count": len(self._tiers),
        }

    def connectivity_check(self) -> dict[str, bool]:
        """Quick check: which tiers are currently reachable."""
        status = {}
        for tier in self._tiers:
            status[tier.name] = tier.health.is_healthy
        status["rule_fallback"] = True  # always available
        return status

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_route(self, tier_name: str, latency_ms: float, success: bool,
                   urgency: Urgency | None = None, error: str = "",
                   note: str = "") -> None:
        entry = {
            "timestamp": time.time(),
            "tier": tier_name,
            "latency_ms": round(latency_ms, 1),
            "success": success,
            "urgency": urgency.value if urgency else None,
            "error": error,
            "note": note,
        }
        with self._lock:
            self._route_log.append(entry)
            # Keep last 500 entries
            if len(self._route_log) > 500:
                self._route_log = self._route_log[-500:]

    @property
    def route_log(self) -> list[dict[str, Any]]:
        return list(self._route_log)

    @property
    def tier_names(self) -> list[str]:
        return [t.name for t in self._tiers]

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        healthy = sum(1 for t in self._tiers if t.health.is_healthy)
        return (
            f"<InferenceRouter tiers={len(self._tiers)} "
            f"healthy={healthy}/{len(self._tiers)} "
            f"routes={len(self._route_log)}>"
        )
