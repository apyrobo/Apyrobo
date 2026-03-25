"""Tests for apyrobo.skills.retry"""
from __future__ import annotations

import asyncio
import time
import pytest

from apyrobo.skills.retry import (
    RetryStrategy,
    RetryPolicy,
    CircuitBreaker,
    CircuitOpenError,
    RetryExecutor,
)


# ---------------------------------------------------------------------------
# RetryPolicy / RetryStrategy
# ---------------------------------------------------------------------------

class TestRetryPolicy:
    def test_none_strategy_zero_delay(self):
        p = RetryPolicy(strategy=RetryStrategy.NONE)
        assert p.delay_for(1) == 0.0
        assert p.delay_for(5) == 0.0

    def test_fixed_strategy_constant_delay(self):
        p = RetryPolicy(strategy=RetryStrategy.FIXED, base_delay=2.0)
        assert p.delay_for(1) == 2.0
        assert p.delay_for(3) == 2.0

    def test_exponential_strategy_doubles(self):
        p = RetryPolicy(strategy=RetryStrategy.EXPONENTIAL, base_delay=1.0)
        # 2^0 * 1.0, 2^1 * 1.0, 2^2 * 1.0
        assert p.delay_for(1) == 1.0
        assert p.delay_for(2) == 2.0
        assert p.delay_for(3) == 4.0

    def test_exponential_capped_by_max_delay(self):
        p = RetryPolicy(strategy=RetryStrategy.EXPONENTIAL, base_delay=1.0, max_delay=3.0)
        assert p.delay_for(10) == 3.0

    def test_jitter_strategy_within_range(self):
        p = RetryPolicy(strategy=RetryStrategy.JITTER, base_delay=1.0, jitter_range=0.5)
        for _ in range(20):
            d = p.delay_for(1)
            assert 1.0 <= d <= 1.5 + 1e-9  # base + jitter_range

    def test_jitter_strategy_capped_by_max_delay(self):
        p = RetryPolicy(strategy=RetryStrategy.JITTER, base_delay=1.0, max_delay=2.0)
        for _ in range(20):
            assert p.delay_for(10) <= 2.0

    def test_default_retryable_errors_is_exception(self):
        p = RetryPolicy()
        assert issubclass(Exception, p.retryable_errors)


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker()
        assert cb.state() == "closed"

    def test_successful_call_stays_closed(self):
        cb = CircuitBreaker()
        result = cb.call(lambda: 42)
        assert result == 42
        assert cb.state() == "closed"

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert cb.state() == "open"

    def test_open_circuit_raises_circuit_open_error(self):
        cb = CircuitBreaker(failure_threshold=1)
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert cb.state() == "open"
        with pytest.raises(CircuitOpenError):
            cb.call(lambda: 42)

    def test_transitions_to_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        assert cb.state() == "open"
        time.sleep(0.06)
        assert cb.state() == "half-open"

    def test_half_open_success_closes_circuit(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        time.sleep(0.06)
        cb.call(lambda: "ok")
        assert cb.state() == "closed"

    def test_half_open_failure_reopens_circuit(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        time.sleep(0.06)
        assert cb.state() == "half-open"
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("y")))
        assert cb.state() == "open"

    def test_reset_closes_circuit(self):
        cb = CircuitBreaker(failure_threshold=1)
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        cb.reset()
        assert cb.state() == "closed"
        assert cb.call(lambda: 99) == 99

    def test_success_resets_consecutive_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        # 2 failures then a success — should NOT open
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        cb.call(lambda: "ok")
        assert cb.state() == "closed"


# ---------------------------------------------------------------------------
# RetryExecutor — synchronous
# ---------------------------------------------------------------------------

class TestRetryExecutorSync:
    def test_success_on_first_attempt(self):
        policy = RetryPolicy(max_attempts=3, strategy=RetryStrategy.NONE)
        executor = RetryExecutor(policy)
        result = executor.execute(lambda: "done")
        assert result == "done"

    def test_retries_until_success(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("transient")
            return "ok"

        policy = RetryPolicy(max_attempts=3, strategy=RetryStrategy.NONE)
        executor = RetryExecutor(policy)
        result = executor.execute(flaky)
        assert result == "ok"
        assert calls["n"] == 3

    def test_raises_after_max_attempts(self):
        policy = RetryPolicy(max_attempts=2, strategy=RetryStrategy.NONE)
        executor = RetryExecutor(policy)
        with pytest.raises(ValueError, match="always"):
            executor.execute(lambda: (_ for _ in ()).throw(ValueError("always")))

    def test_non_retryable_error_raises_immediately(self):
        calls = {"n": 0}

        def boom():
            calls["n"] += 1
            raise TypeError("non-retryable")

        policy = RetryPolicy(
            max_attempts=5,
            strategy=RetryStrategy.NONE,
            retryable_errors=(ValueError,),
        )
        executor = RetryExecutor(policy)
        with pytest.raises(TypeError):
            executor.execute(boom)
        assert calls["n"] == 1

    def test_with_circuit_breaker(self):
        cb = CircuitBreaker(failure_threshold=2)
        policy = RetryPolicy(max_attempts=3, strategy=RetryStrategy.NONE)
        executor = RetryExecutor(policy, breaker=cb)
        # 2 failures open the circuit; 3rd attempt raises CircuitOpenError
        with pytest.raises((RuntimeError, CircuitOpenError)):
            executor.execute(lambda: (_ for _ in ()).throw(RuntimeError("x")))

    def test_circuit_open_error_not_retried(self):
        cb = CircuitBreaker(failure_threshold=1)
        # max_attempts=1 so only one attempt is made — RuntimeError propagates
        policy = RetryPolicy(max_attempts=1, strategy=RetryStrategy.NONE)
        executor = RetryExecutor(policy, breaker=cb)
        # Force circuit open with a single attempt
        with pytest.raises(RuntimeError):
            executor.execute(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        assert cb.state() == "open"
        with pytest.raises(CircuitOpenError):
            executor.execute(lambda: "probe")

    def test_exponential_delays_applied(self):
        calls = {"n": 0, "times": []}

        def failing():
            calls["times"].append(time.time())
            calls["n"] += 1
            raise ValueError("x")

        policy = RetryPolicy(
            max_attempts=3,
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=0.05,
        )
        executor = RetryExecutor(policy)
        with pytest.raises(ValueError):
            executor.execute(failing)
        assert calls["n"] == 3
        # First gap should be ~0.05s, second ~0.1s
        gaps = [calls["times"][i+1] - calls["times"][i] for i in range(2)]
        assert gaps[0] >= 0.04
        assert gaps[1] >= 0.08


# ---------------------------------------------------------------------------
# RetryExecutor — async
# ---------------------------------------------------------------------------

class TestRetryExecutorAsync:
    def test_async_success(self):
        async def run():
            policy = RetryPolicy(max_attempts=2, strategy=RetryStrategy.NONE)
            executor = RetryExecutor(policy)
            return await executor.execute_async(lambda: "async-ok")

        assert asyncio.run(run()) == "async-ok"

    def test_async_coroutine_success(self):
        async def async_fn():
            return "coro-ok"

        async def run():
            policy = RetryPolicy(max_attempts=2, strategy=RetryStrategy.NONE)
            executor = RetryExecutor(policy)
            return await executor.execute_async(async_fn)

        assert asyncio.run(run()) == "coro-ok"

    def test_async_retries_on_failure(self):
        calls = {"n": 0}

        async def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("transient")
            return "recovered"

        async def run():
            policy = RetryPolicy(max_attempts=3, strategy=RetryStrategy.NONE)
            executor = RetryExecutor(policy)
            return await executor.execute_async(flaky)

        assert asyncio.run(run()) == "recovered"
        assert calls["n"] == 3

    def test_async_raises_after_max_attempts(self):
        async def always_fail():
            raise ValueError("boom")

        async def run():
            policy = RetryPolicy(max_attempts=2, strategy=RetryStrategy.NONE)
            executor = RetryExecutor(policy)
            await executor.execute_async(always_fail)

        with pytest.raises(ValueError, match="boom"):
            asyncio.run(run())
