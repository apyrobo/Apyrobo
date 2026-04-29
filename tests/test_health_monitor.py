"""Tests for ConnectionHealth — connection health monitor for ROS2Adapter.

All tests use mock adapters and short timeouts so the suite finishes in < 2s.
"""
from __future__ import annotations

import threading
import time

import pytest

from apyrobo.core.health import ConnectionHealth


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class FakeAdapter:
    """Minimal adapter stub — tracks connect() calls."""

    def __init__(self, name: str = "test_robot") -> None:
        self.robot_name = name
        self.connect_calls = 0

    def connect(self) -> None:
        self.connect_calls += 1


def _health(
    adapter: FakeAdapter | None = None,
    *,
    timeout: float = 0.05,
    backoff_base: float = 0.01,
    backoff_max: float = 0.08,
    max_retries: int | None = None,
    check_interval: float = 0.01,
    reconnect_verify_timeout: float = 0.05,
) -> tuple[ConnectionHealth, FakeAdapter]:
    """Create a ConnectionHealth with fast timings for tests."""
    if adapter is None:
        adapter = FakeAdapter()
    h = ConnectionHealth(
        adapter,
        timeout_seconds=timeout,
        backoff_base=backoff_base,
        backoff_max=backoff_max,
        max_retries=max_retries,
        _check_interval=check_interval,
        _reconnect_verify_timeout=reconnect_verify_timeout,
    )
    return h, adapter


# ---------------------------------------------------------------------------
# is_healthy transitions
# ---------------------------------------------------------------------------

class TestIsHealthy:
    def test_starts_healthy(self):
        h, _ = _health()
        assert h.is_healthy is True

    def test_becomes_unhealthy_after_odom_timeout(self):
        h, _ = _health(timeout=0.05, check_interval=0.01, max_retries=0)
        fired = threading.Event()
        h.on_disconnect(fired.set)
        h.start()
        assert fired.wait(timeout=1.0), "disconnect not detected"
        assert h.is_healthy is False
        h.stop()

    def test_recovers_healthy_after_reconnect(self):
        h, adapter = _health(
            timeout=0.05,
            check_interval=0.01,
            backoff_base=0.01,
            reconnect_verify_timeout=0.08,
        )
        reconnected = threading.Event()
        h.on_reconnect(reconnected.set)

        original_connect = adapter.connect

        def connect_and_feed_odom():
            original_connect()
            h.record_odom()

        adapter.connect = connect_and_feed_odom

        h.start()
        assert reconnected.wait(timeout=1.5), "reconnect not detected"
        assert h.is_healthy is True
        h.stop()


# ---------------------------------------------------------------------------
# on_disconnect callbacks
# ---------------------------------------------------------------------------

class TestDisconnectCallback:
    def test_fires_on_odom_timeout(self):
        h, _ = _health(timeout=0.05, check_interval=0.01, max_retries=0)
        fired = threading.Event()
        h.on_disconnect(fired.set)
        h.start()
        assert fired.wait(timeout=0.5)
        h.stop()

    def test_fires_exactly_once_per_disconnect_episode(self):
        h, _ = _health(timeout=0.05, check_interval=0.01, max_retries=0)
        called = []
        fired = threading.Event()

        def cb():
            called.append(1)
            fired.set()

        h.on_disconnect(cb)
        h.start()
        assert fired.wait(timeout=0.5)
        time.sleep(0.1)  # extra window to catch a spurious second fire
        h.stop()
        assert len(called) == 1

    def test_all_registered_callbacks_fire(self):
        h, _ = _health(timeout=0.05, check_interval=0.01, max_retries=0)
        results = []
        done = threading.Event()
        h.on_disconnect(lambda: results.append("a"))
        h.on_disconnect(lambda: results.append("b"))

        def last():
            results.append("c")
            done.set()

        h.on_disconnect(last)
        h.start()
        assert done.wait(timeout=0.5)
        h.stop()
        assert set(results) == {"a", "b", "c"}

    def test_does_not_fire_when_odom_is_current(self):
        h, _ = _health(timeout=0.5, check_interval=0.01)
        called = []
        h.on_disconnect(lambda: called.append(1))
        h.start()
        time.sleep(0.1)
        h.stop()
        assert len(called) == 0

    def test_faulty_callback_does_not_prevent_others(self):
        h, _ = _health(timeout=0.05, check_interval=0.01, max_retries=0)
        results = []
        done = threading.Event()
        h.on_disconnect(lambda: (_ for _ in ()).throw(RuntimeError("boom")))

        def ok_cb():
            results.append("ok")
            done.set()

        h.on_disconnect(ok_cb)
        h.start()
        assert done.wait(timeout=0.5)
        h.stop()
        assert "ok" in results


# ---------------------------------------------------------------------------
# on_reconnect callbacks
# ---------------------------------------------------------------------------

class TestReconnectCallback:
    def test_fires_after_odom_resumes(self):
        h, adapter = _health(
            timeout=0.05,
            check_interval=0.01,
            backoff_base=0.01,
            reconnect_verify_timeout=0.08,
        )
        reconnected = threading.Event()
        h.on_reconnect(reconnected.set)

        original_connect = adapter.connect

        def connect_and_feed_odom():
            original_connect()
            h.record_odom()

        adapter.connect = connect_and_feed_odom

        h.start()
        assert reconnected.wait(timeout=1.5)
        h.stop()

    def test_does_not_fire_before_disconnect(self):
        h, _ = _health(timeout=10.0, check_interval=0.01)
        called = []
        h.on_reconnect(lambda: called.append(1))
        h.start()
        time.sleep(0.05)
        h.stop()
        assert len(called) == 0

    def test_reconnect_resets_healthy_flag(self):
        h, adapter = _health(
            timeout=0.05,
            check_interval=0.01,
            backoff_base=0.01,
            reconnect_verify_timeout=0.08,
        )
        reconnected = threading.Event()
        h.on_reconnect(reconnected.set)

        original_connect = adapter.connect

        def connect_and_feed_odom():
            original_connect()
            h.record_odom()

        adapter.connect = connect_and_feed_odom

        h.start()
        reconnected.wait(timeout=1.5)
        h.stop()
        assert h.is_healthy is True


# ---------------------------------------------------------------------------
# on_give_up callbacks
# ---------------------------------------------------------------------------

class TestGiveUp:
    def test_fires_when_max_retries_exhausted(self):
        h, _ = _health(
            timeout=0.05,
            check_interval=0.01,
            backoff_base=0.01,
            backoff_max=0.05,
            max_retries=2,
            reconnect_verify_timeout=0.02,
        )
        given_up = threading.Event()
        h.on_give_up(given_up.set)
        h.start()
        assert given_up.wait(timeout=1.5), "give_up not fired"
        h.stop()

    def test_does_not_fire_when_max_retries_is_none(self):
        h, _ = _health(
            timeout=0.05,
            check_interval=0.01,
            backoff_base=0.01,
            backoff_max=0.05,
            max_retries=None,
            reconnect_verify_timeout=0.02,
        )
        given_up = threading.Event()
        h.on_give_up(given_up.set)
        h.start()
        time.sleep(0.2)  # long enough for several attempts; give_up must NOT fire
        h.stop()
        assert not given_up.is_set()

    def test_connect_called_exactly_max_retries_times(self):
        adapter = FakeAdapter()
        h, _ = _health(
            adapter=adapter,
            timeout=0.05,
            check_interval=0.01,
            backoff_base=0.01,
            backoff_max=0.05,
            max_retries=3,
            reconnect_verify_timeout=0.02,
        )
        given_up = threading.Event()
        h.on_give_up(given_up.set)
        h.start()
        given_up.wait(timeout=1.5)
        h.stop()
        assert adapter.connect_calls == 3


# ---------------------------------------------------------------------------
# Backoff behaviour
# ---------------------------------------------------------------------------

class TestBackoff:
    def test_delay_doubles_each_attempt(self):
        h, _ = _health(backoff_base=1.0, backoff_max=1000.0)
        # Average over many samples to wash out ±20% jitter
        avg = lambda attempt: sum(h._backoff_delay(attempt) for _ in range(60)) / 60
        a0, a1, a2, a3 = avg(0), avg(1), avg(2), avg(3)
        assert 0.80 <= a0 <= 1.20
        assert 1.60 <= a1 <= 2.40
        assert 3.20 <= a2 <= 4.80
        assert 6.40 <= a3 <= 9.60

    def test_delay_capped_at_backoff_max(self):
        h, _ = _health(backoff_base=1.0, backoff_max=5.0)
        for attempt in range(10, 20):
            assert h._backoff_delay(attempt) <= 5.0 * 1.21  # max + 21% margin

    def test_jitter_stays_within_20_percent(self):
        h, _ = _health(backoff_base=1.0, backoff_max=1000.0)
        samples = [h._backoff_delay(0) for _ in range(300)]
        assert min(samples) >= 0.79   # 1.0 * (1 − 0.20) − floating-point slack
        assert max(samples) <= 1.21   # 1.0 * (1 + 0.20) + floating-point slack

    def test_delay_never_negative(self):
        h, _ = _health(backoff_base=0.0, backoff_max=0.0)
        assert h._backoff_delay(0) == 0.0
        assert h._backoff_delay(99) == 0.0


# ---------------------------------------------------------------------------
# record_odom
# ---------------------------------------------------------------------------

class TestRecordOdom:
    def test_updates_last_odom_time(self):
        h, _ = _health()
        before = h._last_odom_time
        time.sleep(0.02)
        h.record_odom()
        assert h._last_odom_time > before

    def test_regular_odom_keeps_monitor_healthy(self):
        h, _ = _health(timeout=0.1, check_interval=0.01)
        h.start()
        for _ in range(10):
            h.record_odom()
            time.sleep(0.01)
        h.stop()
        assert h.is_healthy is True

    def test_thread_safe_concurrent_record(self):
        h, _ = _health()
        errors = []

        def writer():
            for _ in range(200):
                try:
                    h.record_odom()
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_stop_terminates_thread(self):
        h, _ = _health(timeout=10.0)
        h.start()
        assert h._monitor_thread is not None and h._monitor_thread.is_alive()
        h.stop()
        assert h._monitor_thread is None

    def test_thread_is_daemon(self):
        h, _ = _health()
        h.start()
        assert h._monitor_thread.daemon is True
        h.stop()

    def test_start_is_idempotent(self):
        h, _ = _health()
        h.start()
        first_thread = h._monitor_thread
        h.start()
        assert h._monitor_thread is first_thread  # same thread, not a new one
        h.stop()

    def test_stop_without_start_is_safe(self):
        h, _ = _health()
        h.stop()  # must not raise

    def test_stop_is_idempotent(self):
        h, _ = _health()
        h.start()
        h.stop()
        h.stop()  # second stop must not raise

    def test_stop_during_reconnect_exits_cleanly(self):
        h, _ = _health(
            timeout=0.05,
            check_interval=0.01,
            backoff_base=0.05,
            max_retries=None,
        )
        disconnected = threading.Event()
        h.on_disconnect(disconnected.set)
        h.start()
        disconnected.wait(timeout=1.0)
        h.stop()
        assert h._monitor_thread is None
