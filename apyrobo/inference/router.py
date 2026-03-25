"""
Inference Router — latency-aware routing between edge and cloud AI.

The router sits between the Agent and LLM providers. It manages multiple
inference tiers (cloud, edge, local, fallback) and dynamically routes
requests based on:

    - Measured latency per provider
    - Connectivity status (is cloud reachable?)
    - Task urgency (reactive decisions vs complex planning)
    - Provider health (error rate, consecutive failures)
    - Circuit-breaker state (IN-02)
    - Token budget tracking (IN-04)
    - Plan caching (IN-07)

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │  Agent.plan("deliver package")                      │
    │       │                                             │
    │       ▼                                             │
    │  InferenceRouter                                    │
    │       │                                             │
    │       ├─── urgency=HIGH? ──► Edge/Local model       │
    │       │       (< 500ms, always available)           │
    │       │                                             │
    │       ├─── urgency=NORMAL? ──► Cloud preferred      │
    │       │       (better quality, ~1-3s latency)       │
    │       │       └── cloud down? ──► Edge fallback     │
    │       │                                             │
    │       └─── urgency=LOW? ──► Cloud (batch OK)        │
    │               (best quality, latency tolerant)      │
    │                                                     │
    │  Safety enforcer + motor commands: ALWAYS LOCAL      │
    │  (never routed through LLM — hardcoded in ROS 2)   │
    └─────────────────────────────────────────────────────┘

Features:
    IN-01: Urgency forwarded through Agent.execute() via urgency= kwarg
    IN-02: Circuit-breaker with OPEN/HALF_OPEN/CLOSED states
    IN-03: Streaming plan support — yield skill steps as LLM streams
    IN-04: Token budget tracking per tier — alert on monthly spend
    IN-07: Plan caching — identical task + capabilities → reuse cached plan
    IN-08: VLM integration tier — vision-language model for spatial reasoning
    IN-10: Fine-tuned edge model tier support

Usage:
    router = InferenceRouter(config={
        "cloud": {"model": "claude-sonnet-4-20250514", "max_latency_ms": 5000},
        "edge":  {"model": "ollama/llama3", "max_latency_ms": 1000},
    })
    agent = Agent(provider="routed", router=router)

    # Router automatically picks the right tier
    result = agent.execute("deliver package", robot)

    # Force edge for time-critical reactive decisions (IN-01)
    result = agent.execute("obstacle ahead, reroute", robot, urgency="high")
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import threading
from collections import deque
from enum import Enum
from typing import Any, Generator

from apyrobo.skills.agent import AgentProvider, RuleBasedProvider, LLMProvider
from apyrobo.observability import emit_event

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
# Budget exceeded exception
# ---------------------------------------------------------------------------

class BudgetExceeded(Exception):
    """Raised when a tier's token budget has been exceeded."""

    def __init__(self, used: int, limit: int, tier: str = "") -> None:
        self.used = used
        self.limit = limit
        self.tier = tier
        msg = f"{used}/{limit} tokens used"
        if tier:
            msg = f"[{tier}] {msg}"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Circuit-breaker states (IN-02)
# ---------------------------------------------------------------------------

class CircuitState(str, Enum):
    """Circuit-breaker state for a provider tier."""
    CLOSED = "closed"       # Normal operation — requests flow through
    OPEN = "open"           # Tripped — all requests rejected immediately
    HALF_OPEN = "half_open" # Testing — one probe request allowed


# ---------------------------------------------------------------------------
# Provider health tracking (with circuit-breaker IN-02)
# ---------------------------------------------------------------------------

class ProviderHealth:
    """Tracks latency, availability, error rate, and circuit-breaker state."""

    def __init__(
        self, name: str, max_history: int = 50,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ) -> None:
        self.name = name
        self._latencies: deque[float] = deque(maxlen=max_history)
        self._errors: deque[bool] = deque(maxlen=max_history)
        self._consecutive_failures = 0
        self._last_success: float | None = None
        self._last_failure: float | None = None
        self._total_calls = 0
        self._total_errors = 0

        # IN-02: Circuit-breaker
        self._circuit_state = CircuitState.CLOSED
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._circuit_opened_at: float | None = None

    def record_success(self, latency_ms: float) -> None:
        """Record a successful inference call."""
        self._latencies.append(latency_ms)
        self._errors.append(False)
        self._consecutive_failures = 0
        self._last_success = time.time()
        self._total_calls += 1

        # IN-02: Success in HALF_OPEN → close circuit
        if self._circuit_state == CircuitState.HALF_OPEN:
            self._circuit_state = CircuitState.CLOSED
            logger.info("Circuit-breaker %s: HALF_OPEN → CLOSED (success)", self.name)

    def record_failure(self, error: str = "") -> None:
        """Record a failed inference call."""
        self._errors.append(True)
        self._consecutive_failures += 1
        self._last_failure = time.time()
        self._total_calls += 1
        self._total_errors += 1
        logger.warning("Provider %s failed (%d consecutive): %s",
                       self.name, self._consecutive_failures, error)

        # IN-02: Check if we should trip the circuit
        if self._circuit_state == CircuitState.HALF_OPEN:
            # Failed probe → re-open
            self._circuit_state = CircuitState.OPEN
            self._circuit_opened_at = time.time()
            logger.warning("Circuit-breaker %s: HALF_OPEN → OPEN (probe failed)", self.name)
        elif self._consecutive_failures >= self._failure_threshold:
            self._circuit_state = CircuitState.OPEN
            self._circuit_opened_at = time.time()
            logger.warning(
                "Circuit-breaker %s: CLOSED → OPEN (%d failures)",
                self.name, self._consecutive_failures,
            )

    @property
    def circuit_state(self) -> CircuitState:
        """Current circuit-breaker state (IN-02)."""
        if self._circuit_state == CircuitState.OPEN and self._circuit_opened_at:
            elapsed = time.time() - self._circuit_opened_at
            if elapsed >= self._recovery_timeout:
                self._circuit_state = CircuitState.HALF_OPEN
                logger.info(
                    "Circuit-breaker %s: OPEN → HALF_OPEN (%.0fs elapsed)",
                    self.name, elapsed,
                )
        return self._circuit_state

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
        """Is this provider currently usable (respects circuit-breaker)."""
        state = self.circuit_state
        if state == CircuitState.OPEN:
            return False
        # HALF_OPEN allows one probe request
        return True

    @property
    def is_available(self) -> bool:
        """Has this provider ever succeeded."""
        return self._last_success is not None or self._total_calls == 0

    def reset(self) -> None:
        """Manually reset the circuit-breaker to CLOSED."""
        self._circuit_state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._circuit_opened_at = None

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
            "circuit_state": self.circuit_state.value,
        }

    def __repr__(self) -> str:
        return (
            f"<ProviderHealth {self.name}: "
            f"avg={self.avg_latency_ms:.0f}ms "
            f"err={self.error_rate:.0%} "
            f"circuit={self.circuit_state.value}>"
        )


# ---------------------------------------------------------------------------
# Token Budget Tracker (IN-04)
# ---------------------------------------------------------------------------

class TokenBudget:
    """
    Tracks token usage per tier and alerts when spend exceeds thresholds.

    IN-04: Token budget tracking per tier — alert when monthly spend hits limit.
    """

    def __init__(self, monthly_limit: int = 1_000_000, alert_at_pct: float = 80.0) -> None:
        self.monthly_limit = monthly_limit
        self.alert_at_pct = alert_at_pct
        self._usage: dict[str, int] = {}  # tier_name → tokens used
        self._cost: dict[str, float] = {}  # tier_name → estimated cost
        self._reset_at: float = time.time()
        self._alerts_sent: set[str] = set()
        self._callbacks: list[Any] = []

    def check(self, estimated: int = 0) -> None:
        """Raise BudgetExceeded if budget is already exceeded or would be with estimated tokens."""
        if self.total_tokens >= self.monthly_limit or self.total_tokens + estimated > self.monthly_limit:
            raise BudgetExceeded(
                used=self.total_tokens, limit=self.monthly_limit,
            )

    def record(self, tier_name: str, input_tokens: int = 0, output_tokens: int = 0,
               cost: float = 0.0) -> None:
        """Record token usage for a tier."""
        total = input_tokens + output_tokens
        self._usage[tier_name] = self._usage.get(tier_name, 0) + total
        self._cost[tier_name] = self._cost.get(tier_name, 0.0) + cost

        # Check alert threshold
        total_usage = sum(self._usage.values())
        pct = (total_usage / self.monthly_limit * 100) if self.monthly_limit > 0 else 0
        if pct >= self.alert_at_pct and "budget_warning" not in self._alerts_sent:
            self._alerts_sent.add("budget_warning")
            logger.warning(
                "Token budget alert: %.0f%% used (%d/%d tokens)",
                pct, total_usage, self.monthly_limit,
            )
            emit_event("budget_alert", pct=pct, used=total_usage, limit=self.monthly_limit)
            for cb in self._callbacks:
                try:
                    cb("budget_warning", pct, total_usage)
                except Exception:
                    pass

        if total_usage >= self.monthly_limit and "budget_exceeded" not in self._alerts_sent:
            self._alerts_sent.add("budget_exceeded")
            logger.error(
                "Token budget EXCEEDED: %d/%d tokens",
                total_usage, self.monthly_limit,
            )
            emit_event("budget_exceeded", pct=100.0, used=total_usage, limit=self.monthly_limit)
            for cb in self._callbacks:
                try:
                    cb("budget_exceeded", 100.0, total_usage)
                except Exception:
                    pass

    def on_alert(self, callback: Any) -> None:
        """Register callback for budget alerts: callback(alert_type, pct, total)."""
        self._callbacks.append(callback)

    @property
    def total_tokens(self) -> int:
        return sum(self._usage.values())

    @property
    def total_cost(self) -> float:
        return sum(self._cost.values())

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.monthly_limit - self.total_tokens)

    @property
    def usage_pct(self) -> float:
        if self.monthly_limit <= 0:
            return 0.0
        return self.total_tokens / self.monthly_limit * 100

    @property
    def is_over_budget(self) -> bool:
        return self.total_tokens >= self.monthly_limit

    def usage_by_tier(self) -> dict[str, dict[str, Any]]:
        result = {}
        for tier, tokens in self._usage.items():
            result[tier] = {
                "tokens": tokens,
                "cost": round(self._cost.get(tier, 0.0), 4),
                "pct_of_budget": round(tokens / self.monthly_limit * 100, 1) if self.monthly_limit > 0 else 0,
            }
        return result

    def reset(self) -> None:
        """Reset usage (e.g. at month boundary)."""
        self._usage.clear()
        self._cost.clear()
        self._alerts_sent.clear()
        self._reset_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "monthly_limit": self.monthly_limit,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 4),
            "usage_pct": round(self.usage_pct, 1),
            "remaining": self.remaining_tokens,
            "is_over_budget": self.is_over_budget,
            "by_tier": self.usage_by_tier(),
        }

    def __repr__(self) -> str:
        return f"<TokenBudget {self.total_tokens}/{self.monthly_limit} ({self.usage_pct:.0f}%)>"


# ---------------------------------------------------------------------------
# Plan Cache (IN-07)
# ---------------------------------------------------------------------------

class PlanCache:
    """
    Caches plans keyed by (task, capabilities).

    IN-07: Identical task + capabilities → reuse cached plan.
    Entries expire after a configurable TTL.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: float = 3600.0) -> None:
        self._cache: dict[str, tuple[list[dict[str, Any]], float]] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(task: str, capabilities: list[str]) -> str:
        """Deterministic cache key from task and capabilities."""
        raw = json.dumps({"task": task.lower().strip(), "caps": sorted(capabilities)})
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, task: str, capabilities: list[str]) -> list[dict[str, Any]] | None:
        """Look up a cached plan. Returns None on miss or expiry."""
        key = self._key(task, capabilities)
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        plan, cached_at = entry
        if time.time() - cached_at > self._ttl:
            del self._cache[key]
            self._misses += 1
            return None
        self._hits += 1
        return plan

    def put(self, task: str, capabilities: list[str], plan: list[dict[str, Any]]) -> None:
        """Store a plan in the cache."""
        if len(self._cache) >= self._max_size:
            # Evict oldest entry
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        key = self._key(task, capabilities)
        self._cache[key] = (plan, time.time())

    def invalidate(self, task: str | None = None, capabilities: list[str] | None = None) -> int:
        """Invalidate cache entries. If no args, clear everything."""
        if task is None and capabilities is None:
            count = len(self._cache)
            self._cache.clear()
            return count
        if task is not None and capabilities is not None:
            key = self._key(task, capabilities)
            if key in self._cache:
                del self._cache[key]
                return 1
        return 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3),
        }

    def __repr__(self) -> str:
        return f"<PlanCache size={self.size}/{self._max_size} hit_rate={self.hit_rate:.0%}>"


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
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        is_vlm: bool = False,
        is_edge: bool = False,
        budget: TokenBudget | None = None,
    ) -> None:
        self.name = name
        self.provider = provider
        self.max_latency_ms = max_latency_ms
        self.priority = priority  # lower = preferred
        self.supports_urgency = supports_urgency or list(Urgency)
        self.health = ProviderHealth(
            name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
        self.is_vlm = is_vlm      # IN-08: vision-language model tier
        self.is_edge = is_edge    # IN-10: fine-tuned edge model
        self.budget = budget      # IN-02: per-tier token budget

    def __repr__(self) -> str:
        tags = []
        if self.is_vlm:
            tags.append("vlm")
        if self.is_edge:
            tags.append("edge")
        tag_str = f" [{','.join(tags)}]" if tags else ""
        return f"<Tier {self.name} pri={self.priority}{tag_str} {self.health}>"


# ---------------------------------------------------------------------------
# Inference Router
# ---------------------------------------------------------------------------

class InferenceRouter(AgentProvider):
    """
    Routes inference requests across multiple providers based on
    urgency, latency, provider health, and circuit-breaker state.

    Features:
        IN-01: Urgency forwarding via plan(..., urgency=)
        IN-02: Circuit-breaker (CLOSED/OPEN/HALF_OPEN) per tier
        IN-03: Streaming plan support via stream_plan()
        IN-04: Token budget tracking via TokenBudget
        IN-07: Plan caching via PlanCache
        IN-08: VLM tier selection via plan(..., use_vlm=True)
        IN-10: Edge model tier support
    """

    def __init__(
        self,
        token_budget: TokenBudget | None = None,
        plan_cache: PlanCache | None = None,
        enable_cache: bool = True,
    ) -> None:
        self._tiers: list[InferenceTier] = []
        self._fallback = RuleBasedProvider()
        self._fallback_health = ProviderHealth("rule_fallback")
        self._lock = threading.Lock()
        self._route_log: list[dict[str, Any]] = []

        # IN-04: Token budget
        self._token_budget = token_budget or TokenBudget()

        # IN-07: Plan cache
        self._plan_cache = plan_cache if plan_cache is not None else (PlanCache() if enable_cache else None)

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
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        is_vlm: bool = False,
        is_edge: bool = False,
        budget: TokenBudget | None = None,
    ) -> None:
        """Add an inference tier. Lower priority = preferred."""
        if priority is None:
            priority = len(self._tiers)
        tier = InferenceTier(
            name=name, provider=provider,
            max_latency_ms=max_latency_ms, priority=priority,
            supports_urgency=supports_urgency,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            is_vlm=is_vlm,
            is_edge=is_edge,
            budget=budget,
        )
        self._tiers.append(tier)
        self._tiers.sort(key=lambda t: t.priority)
        logger.info("Router: added tier %s (priority=%d, max_latency=%dms vlm=%s edge=%s)",
                     name, priority, max_latency_ms, is_vlm, is_edge)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> InferenceRouter:
        """
        Build a router from a config dict.

        Config format:
            {
                "cloud": {"model": "claude-sonnet-4-20250514", "max_latency_ms": 5000, "priority": 0},
                "edge": {"model": "ollama/llama3:8b", "max_latency_ms": 1000, "priority": 1, "is_edge": true},
                "vlm": {"model": "gpt-4o", "is_vlm": true, "max_latency_ms": 8000},
            }
        """
        budget_config = config.pop("_budget", None)
        cache_config = config.pop("_cache", None)

        budget = None
        if budget_config:
            budget = TokenBudget(
                monthly_limit=budget_config.get("monthly_limit", 1_000_000),
                alert_at_pct=budget_config.get("alert_at_pct", 80.0),
            )

        cache = None
        if cache_config:
            cache = PlanCache(
                max_size=cache_config.get("max_size", 100),
                ttl_seconds=cache_config.get("ttl_seconds", 3600.0),
            )

        router = cls(token_budget=budget, plan_cache=cache)
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
                failure_threshold=tier_config.get("failure_threshold", 3),
                recovery_timeout=tier_config.get("recovery_timeout", 30.0),
                is_vlm=tier_config.get("is_vlm", False),
                is_edge=tier_config.get("is_edge", False),
            )
        return router

    # ------------------------------------------------------------------
    # Routing (IN-01: urgency forwarding)
    # ------------------------------------------------------------------

    def plan(self, task: str, available_skills: list[dict[str, Any]],
             capabilities: list[str], urgency: Urgency | str = Urgency.NORMAL,
             use_vlm: bool = False, skip_cache: bool = False,
             **kwargs: Any) -> list[dict[str, Any]]:
        """
        Route a planning request to the best available tier.

        IN-01: Urgency forwarding via urgency= parameter.
        IN-02: Respects circuit-breaker state per tier.
        IN-07: Checks plan cache before calling providers.
        IN-08: If use_vlm=True, prefer VLM-capable tiers.
        """
        if isinstance(urgency, str):
            urgency = Urgency(urgency)

        # IN-07: Check cache first (skip for HIGH urgency — freshness matters)
        if self._plan_cache and not skip_cache and urgency != Urgency.HIGH:
            cached = self._plan_cache.get(task, capabilities)
            if cached is not None:
                self._log_route("cache_hit", 0.0, True, urgency)
                return cached

        # IN-04: Check token budget
        if self._token_budget and self._token_budget.is_over_budget:
            logger.warning("Router: token budget exceeded, using rule-based fallback")
            return self._fallback_plan(task, available_skills, capabilities, urgency)

        # Find eligible tiers
        eligible = self._select_tiers(urgency, use_vlm)

        # Try each eligible tier
        for tier in eligible:
            # IN-02: Check per-tier budget before attempting
            if tier.budget is not None:
                try:
                    tier.budget.check()
                except BudgetExceeded:
                    logger.warning(
                        "Router: tier %s over budget, skipping to next tier",
                        tier.name,
                    )
                    emit_event("budget_exceeded", tier=tier.name)
                    continue

            result = self._try_tier(tier, task, available_skills, capabilities)
            if result is not None:
                # IN-07: Cache the result
                if self._plan_cache and not skip_cache:
                    self._plan_cache.put(task, capabilities, result)
                return result

        # All tiers failed — use rule-based fallback
        return self._fallback_plan(task, available_skills, capabilities, urgency)

    def _select_tiers(self, urgency: Urgency, use_vlm: bool = False) -> list[InferenceTier]:
        """Select eligible tiers based on urgency, health, and VLM requirement."""
        eligible = []
        for t in self._tiers:
            if urgency not in t.supports_urgency:
                continue
            if not t.health.is_healthy:
                continue
            if use_vlm and not t.is_vlm:
                continue
            eligible.append(t)

        # For HIGH urgency, prefer edge tiers (IN-10)
        if urgency == Urgency.HIGH:
            edge_tiers = [t for t in eligible if t.is_edge]
            if edge_tiers:
                return edge_tiers + [t for t in eligible if not t.is_edge]

        return eligible

    def _fallback_plan(
        self, task: str, available_skills: list[dict[str, Any]],
        capabilities: list[str], urgency: Urgency,
    ) -> list[dict[str, Any]]:
        """Use rule-based fallback when all tiers fail."""
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

    # ------------------------------------------------------------------
    # IN-03: Streaming plan support
    # ------------------------------------------------------------------

    def stream_plan(
        self, task: str, available_skills: list[dict[str, Any]],
        capabilities: list[str], urgency: Urgency | str = Urgency.NORMAL,
        **kwargs: Any,
    ) -> Generator[dict[str, Any], None, None]:
        """
        IN-03: Yield skill steps one at a time as they become available.

        For LLM providers that support streaming, this yields each skill
        step as it is parsed from the stream. For non-streaming providers,
        falls back to yielding all steps from a batch call.
        """
        if isinstance(urgency, str):
            urgency = Urgency(urgency)

        # Try streaming from eligible tiers
        eligible = self._select_tiers(urgency)

        for tier in eligible:
            try:
                provider = tier.provider
                # Check if provider supports streaming
                if hasattr(provider, "stream_plan"):
                    t0 = time.time()
                    steps_yielded = 0
                    for step in provider.stream_plan(task, available_skills, capabilities):
                        yield step
                        steps_yielded += 1
                    latency_ms = (time.time() - t0) * 1000
                    tier.health.record_success(latency_ms)
                    self._log_route(tier.name, latency_ms, True, urgency, note="streamed")
                    return
                else:
                    # Fall back to batch call and yield steps
                    result = self._try_tier(tier, task, available_skills, capabilities)
                    if result is not None:
                        for step in result:
                            yield step
                        return
            except Exception as e:
                tier.health.record_failure(str(e))
                continue

        # Fallback: rule-based
        result = self._fallback.plan(task, available_skills, capabilities)
        for step in result:
            yield step

    def _try_tier(
        self, tier: InferenceTier, task: str,
        available_skills: list[dict[str, Any]], capabilities: list[str],
    ) -> list[dict[str, Any]] | None:
        """Attempt a single tier. Returns None on failure."""
        t0 = time.time()
        try:
            result = tier.provider.plan(task, available_skills, capabilities)
            latency_ms = (time.time() - t0) * 1000

            # IN-04: Track token usage (estimate from result size)
            estimated_tokens = max(100, len(json.dumps(result)) * 2)
            self._token_budget.record(tier.name, input_tokens=estimated_tokens,
                                       output_tokens=estimated_tokens // 2)

            # IN-02: Record usage on per-tier budget
            if tier.budget is not None:
                tier.budget.record(tier.name, input_tokens=estimated_tokens,
                                   output_tokens=estimated_tokens // 2)

            # Check latency budget
            if latency_ms > tier.max_latency_ms:
                logger.warning(
                    "Router: tier %s responded in %.0fms (budget: %.0fms) — "
                    "succeeded but slow",
                    tier.name, latency_ms, tier.max_latency_ms,
                )
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
    # IN-02: Circuit-breaker management
    # ------------------------------------------------------------------

    def get_circuit_state(self, tier_name: str) -> CircuitState | None:
        """Get the circuit-breaker state for a tier."""
        for t in self._tiers:
            if t.name == tier_name:
                return t.health.circuit_state
        return None

    def reset_circuit(self, tier_name: str) -> bool:
        """Manually reset a tier's circuit-breaker to CLOSED."""
        for t in self._tiers:
            if t.name == tier_name:
                t.health.reset()
                logger.info("Circuit-breaker %s manually reset to CLOSED", tier_name)
                return True
        return False

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
            report["is_vlm"] = t.is_vlm
            report["is_edge"] = t.is_edge
            tiers.append(report)

        result: dict[str, Any] = {
            "tiers": tiers,
            "fallback": self._fallback_health.to_dict(),
            "total_routes": len(self._route_log),
            "tier_count": len(self._tiers),
        }

        # IN-04: Token budget
        result["token_budget"] = self._token_budget.to_dict()

        # IN-07: Cache stats
        if self._plan_cache:
            result["plan_cache"] = self._plan_cache.to_dict()

        return result

    def connectivity_check(self) -> dict[str, bool]:
        """Quick check: which tiers are currently reachable."""
        status = {}
        for tier in self._tiers:
            status[tier.name] = tier.health.is_healthy
        status["rule_fallback"] = True  # always available
        return status

    # ------------------------------------------------------------------
    # IN-04: Token budget access
    # ------------------------------------------------------------------

    @property
    def token_budget(self) -> TokenBudget:
        return self._token_budget

    def get_budget_status(self) -> dict[str, Any]:
        """Return budget status for dashboard exposure."""
        status = self._token_budget.to_dict()
        # Include per-tier budgets
        tier_budgets = {}
        for t in self._tiers:
            if t.budget is not None:
                tier_budgets[t.name] = t.budget.to_dict()
        if tier_budgets:
            status["tier_budgets"] = tier_budgets
        return status

    # ------------------------------------------------------------------
    # IN-07: Plan cache access
    # ------------------------------------------------------------------

    @property
    def plan_cache_stats(self) -> dict[str, Any] | None:
        if self._plan_cache:
            return self._plan_cache.to_dict()
        return None

    def invalidate_cache(self) -> int:
        """Clear the entire plan cache."""
        if self._plan_cache:
            return self._plan_cache.invalidate()
        return 0

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

    # ------------------------------------------------------------------
    # IN-08: VLM vision routing
    # ------------------------------------------------------------------

    def route_vision(self, image_data: bytes, prompt: str) -> str:
        """
        Route a vision query to the best available VLM tier.

        Falls back to the VLMRouter (MockVLMAdapter) if no VLM tier is
        configured or all VLM tiers are unhealthy.

        Args:
            image_data: Raw image bytes (JPEG, PNG, etc.).
            prompt:     Question or instruction about the image.

        Returns:
            String response from the VLM.
        """
        from apyrobo.inference.vlm import VLMRouter, LiteLLMVLMAdapter  # local import to avoid cycle

        # Find the best healthy VLM tier
        vlm_tiers = [t for t in self._tiers if t.is_vlm and t.health.is_healthy]
        if vlm_tiers:
            tier = min(vlm_tiers, key=lambda t: t.health.avg_latency_ms)
            adapter = LiteLLMVLMAdapter(model=tier.provider.model if hasattr(tier.provider, "model") else "gpt-4o")
            router = VLMRouter(adapter=adapter)
        else:
            logger.debug("route_vision: no healthy VLM tier — using mock adapter")
            router = VLMRouter()

        t0 = time.time()
        try:
            result = router.route_vision(image_data, prompt)
            self._log_route(
                "vlm", (time.time() - t0) * 1000, True, note="vision"
            )
            return result
        except Exception as exc:
            self._log_route(
                "vlm", (time.time() - t0) * 1000, False, error=str(exc), note="vision"
            )
            raise

    def __repr__(self) -> str:
        healthy = sum(1 for t in self._tiers if t.health.is_healthy)
        return (
            f"<InferenceRouter tiers={len(self._tiers)} "
            f"healthy={healthy}/{len(self._tiers)} "
            f"routes={len(self._route_log)} "
            f"budget={self._token_budget.usage_pct:.0f}%>"
        )
