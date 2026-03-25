"""
Skill Retry Policies — configurable backoff strategies with circuit breaker.

Classes:
    RetryStrategy     — enum of available backoff strategies
    RetryPolicy       — dataclass configuring retry behaviour
    CircuitBreaker    — open/half-open/closed state machine
    CircuitOpenError  — raised when circuit is open
    RetryExecutor     — wraps any callable with retry + circuit breaker logic
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    NONE = "none"
    FIXED = "fixed"           # fixed delay between retries
    EXPONENTIAL = "exponential"   # 2^attempt * base_delay
    JITTER = "jitter"         # exponential + random jitter


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    base_delay: float = 1.0   # seconds
    max_delay: float = 60.0
    jitter_range: float = 0.5
    retryable_errors: tuple = field(default_factory=lambda: (Exception,))

    def delay_for(self, attempt: int) -> float:
        """Return the delay in seconds before *attempt* (1-indexed)."""
        if self.strategy == RetryStrategy.NONE:
            return 0.0
        if self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = (2 ** (attempt - 1)) * self.base_delay
        else:  # JITTER
            delay = (2 ** (attempt - 1)) * self.base_delay
            delay += random.uniform(0, self.jitter_range)
        return min(delay, self.max_delay)


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit breaker is open."""


class CircuitBreaker:
    """
    Open after *failure_threshold* consecutive failures; half-open after
    *recovery_timeout* seconds; closed again on the first success.

    States:
        closed    — normal operation, calls pass through
        open      — calls are rejected immediately (CircuitOpenError)
        half-open — one probe call allowed; success → closed, failure → open
    """

    _STATE_CLOSED = "closed"
    _STATE_OPEN = "open"
    _STATE_HALF_OPEN = "half-open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._state = self._STATE_CLOSED

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def state(self) -> str:
        """Return current state string: "closed", "open", or "half-open"."""
        self._maybe_transition_to_half_open()
        return self._state

    def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute *fn* if the circuit allows it.

        Raises:
            CircuitOpenError: if the circuit is open and recovery timeout
                has not elapsed.
        """
        self._maybe_transition_to_half_open()

        if self._state == self._STATE_OPEN:
            raise CircuitOpenError(
                f"Circuit breaker is open (opened at {self._opened_at:.1f})"
            )

        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure()
            raise exc

    def reset(self) -> None:
        """Force the circuit breaker back to closed state."""
        self._state = self._STATE_CLOSED
        self._consecutive_failures = 0
        self._opened_at = None
        logger.debug("CircuitBreaker: manually reset to closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_transition_to_half_open(self) -> None:
        if (
            self._state == self._STATE_OPEN
            and self._opened_at is not None
            and (time.time() - self._opened_at) >= self._recovery_timeout
        ):
            self._state = self._STATE_HALF_OPEN
            logger.info("CircuitBreaker: open → half-open (probe allowed)")

    def _on_success(self) -> None:
        if self._state == self._STATE_HALF_OPEN:
            logger.info("CircuitBreaker: half-open → closed (probe succeeded)")
        self._state = self._STATE_CLOSED
        self._consecutive_failures = 0
        self._opened_at = None

    def _on_failure(self) -> None:
        self._consecutive_failures += 1
        logger.warning(
            "CircuitBreaker: failure %d/%d",
            self._consecutive_failures,
            self._failure_threshold,
        )
        if (
            self._state == self._STATE_HALF_OPEN
            or self._consecutive_failures >= self._failure_threshold
        ):
            self._state = self._STATE_OPEN
            self._opened_at = time.time()
            logger.error(
                "CircuitBreaker: opened after %d consecutive failures",
                self._consecutive_failures,
            )


class RetryExecutor:
    """
    Wrap any callable with retry + circuit breaker logic.

    Each attempt is logged with its duration.  The circuit breaker (if
    provided) gates every attempt and tracks successes/failures.
    """

    def __init__(
        self,
        policy: RetryPolicy,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.policy = policy
        self.breaker = breaker

    # ------------------------------------------------------------------
    # Synchronous execution
    # ------------------------------------------------------------------

    def execute(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute *fn* with retry and circuit-breaker logic (synchronous).

        Raises the last exception if all attempts are exhausted or the
        circuit breaker opens.
        """
        last_exc: Exception | None = None

        for attempt in range(1, self.policy.max_attempts + 1):
            start = time.time()
            try:
                if self.breaker is not None:
                    result = self.breaker.call(fn, *args, **kwargs)
                else:
                    result = fn(*args, **kwargs)
                duration = time.time() - start
                logger.debug(
                    "RetryExecutor: attempt %d/%d succeeded in %.3fs",
                    attempt, self.policy.max_attempts, duration,
                )
                return result
            except CircuitOpenError:
                raise
            except Exception as exc:
                duration = time.time() - start
                last_exc = exc
                if not isinstance(exc, self.policy.retryable_errors):
                    logger.warning(
                        "RetryExecutor: attempt %d/%d non-retryable error in %.3fs: %s",
                        attempt, self.policy.max_attempts, duration, exc,
                    )
                    raise
                logger.warning(
                    "RetryExecutor: attempt %d/%d failed in %.3fs: %s",
                    attempt, self.policy.max_attempts, duration, exc,
                )
                if attempt < self.policy.max_attempts:
                    delay = self.policy.delay_for(attempt)
                    if delay > 0:
                        time.sleep(delay)

        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------
    # Asynchronous execution
    # ------------------------------------------------------------------

    async def execute_async(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute *fn* (sync or async) with retry and circuit-breaker logic.

        Delays between retries use ``asyncio.sleep`` so the event loop
        is not blocked.
        """
        last_exc: Exception | None = None

        for attempt in range(1, self.policy.max_attempts + 1):
            start = time.time()
            try:
                if self.breaker is not None:
                    # CircuitBreaker.call is synchronous; wrap if fn is async
                    if asyncio.iscoroutinefunction(fn):
                        self.breaker._maybe_transition_to_half_open()
                        if self.breaker.state() == CircuitBreaker._STATE_OPEN:
                            raise CircuitOpenError("Circuit breaker is open")
                        try:
                            result = await fn(*args, **kwargs)
                            self.breaker._on_success()
                        except Exception as exc:
                            self.breaker._on_failure()
                            raise exc
                    else:
                        result = self.breaker.call(fn, *args, **kwargs)
                else:
                    if asyncio.iscoroutinefunction(fn):
                        result = await fn(*args, **kwargs)
                    else:
                        result = fn(*args, **kwargs)
                duration = time.time() - start
                logger.debug(
                    "RetryExecutor(async): attempt %d/%d succeeded in %.3fs",
                    attempt, self.policy.max_attempts, duration,
                )
                return result
            except CircuitOpenError:
                raise
            except Exception as exc:
                duration = time.time() - start
                last_exc = exc
                if not isinstance(exc, self.policy.retryable_errors):
                    raise
                logger.warning(
                    "RetryExecutor(async): attempt %d/%d failed in %.3fs: %s",
                    attempt, self.policy.max_attempts, duration, exc,
                )
                if attempt < self.policy.max_attempts:
                    delay = self.policy.delay_for(attempt)
                    if delay > 0:
                        await asyncio.sleep(delay)

        assert last_exc is not None
        raise last_exc
