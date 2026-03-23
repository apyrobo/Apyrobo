"""
Tests for connection resilience added to CapabilityAdapter (AD-06).
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch, call

import pytest

from apyrobo.core.adapters import CapabilityAdapter, MockAdapter, GazeboAdapter
from apyrobo.core.schemas import AdapterState, RobotCapability


# ---------------------------------------------------------------------------
# Minimal concrete adapter for testing base-class behaviour
# ---------------------------------------------------------------------------

class _ConcreteAdapter(CapabilityAdapter):
    """Minimal subclass to test base-class reconnect logic."""

    def __init__(self, robot_name: str, connect_fails: int = 0, **kwargs):
        super().__init__(robot_name, **kwargs)
        self._connect_fails_remaining = connect_fails
        self._connect_calls = 0

    def get_capabilities(self) -> RobotCapability:
        from apyrobo.core.schemas import Capability, CapabilityType
        return RobotCapability(
            robot_id=self.robot_name,
            name=self.robot_name,
            capabilities=[Capability(capability_type=CapabilityType.NAVIGATE, name="navigate_to")],
        )

    def move(self, x: float, y: float, speed=None) -> None:
        pass

    def stop(self) -> None:
        pass

    def connect(self) -> None:
        self._connect_calls += 1
        if self._connect_fails_remaining > 0:
            self._connect_fails_remaining -= 1
            raise ConnectionRefusedError("Simulated connection failure")
        super().connect()


# ---------------------------------------------------------------------------
# Disconnect lifecycle
# ---------------------------------------------------------------------------

class TestDisconnectLifecycle:
    def test_disconnect_sets_state(self):
        adapter = MockAdapter("bot")
        assert adapter.is_connected
        adapter.disconnect()
        assert not adapter.is_connected
        assert adapter.state == AdapterState.DISCONNECTED

    def test_disconnect_records_time(self):
        adapter = MockAdapter("bot")
        before = time.time()
        adapter.disconnect()
        after = time.time()
        assert adapter._last_disconnect_time is not None
        assert before <= adapter._last_disconnect_time <= after

    def test_disconnect_does_not_record_time_if_already_disconnected(self):
        adapter = _ConcreteAdapter("bot")
        # Starts DISCONNECTED
        assert not adapter.is_connected
        assert adapter._last_disconnect_time is None
        adapter.disconnect()
        assert adapter._last_disconnect_time is None

    def test_on_disconnect_callback_called(self):
        adapter = MockAdapter("bot")
        cb = MagicMock()
        adapter.on_disconnect(cb)
        adapter.disconnect()
        cb.assert_called_once()

    def test_multiple_disconnect_callbacks(self):
        adapter = MockAdapter("bot")
        cbs = [MagicMock() for _ in range(3)]
        for cb in cbs:
            adapter.on_disconnect(cb)
        adapter.disconnect()
        for cb in cbs:
            cb.assert_called_once()

    def test_disconnect_callback_exception_does_not_propagate(self):
        adapter = MockAdapter("bot")
        adapter.on_disconnect(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        # Should not raise
        adapter.disconnect()

    def test_on_reconnect_callback_registered(self):
        adapter = MockAdapter("bot")
        cb = MagicMock()
        adapter.on_reconnect(cb)
        assert cb in adapter._reconnect_handlers


# ---------------------------------------------------------------------------
# reconnect_with_backoff — success paths
# ---------------------------------------------------------------------------

class TestReconnectWithBackoff:
    def test_already_connected_returns_true_immediately(self):
        adapter = MockAdapter("bot")
        assert adapter.is_connected
        result = adapter.reconnect_with_backoff(max_attempts=3)
        assert result is True

    def test_reconnect_success_on_first_attempt(self):
        adapter = _ConcreteAdapter("bot", connect_fails=0)
        adapter.disconnect()  # go offline
        with patch("time.sleep"):  # skip actual sleeping
            result = adapter.reconnect_with_backoff(max_attempts=3, initial_delay=0.01)
        assert result is True
        assert adapter.is_connected

    def test_reconnect_success_after_failures(self):
        # Fails twice, then succeeds on 3rd attempt
        adapter = _ConcreteAdapter("bot", connect_fails=2)
        adapter._state = AdapterState.DISCONNECTED
        with patch("time.sleep"):
            result = adapter.reconnect_with_backoff(
                max_attempts=5,
                initial_delay=0.01,
                backoff_factor=2.0,
            )
        assert result is True
        assert adapter.is_connected
        assert adapter._connect_calls == 3

    def test_reconnect_increments_attempt_counter(self):
        adapter = _ConcreteAdapter("bot", connect_fails=1)
        adapter._state = AdapterState.DISCONNECTED
        with patch("time.sleep"):
            adapter.reconnect_with_backoff(max_attempts=3, initial_delay=0.01)
        assert adapter._reconnect_attempts >= 2

    def test_reconnect_failure_sets_error_state(self):
        adapter = _ConcreteAdapter("bot", connect_fails=999)
        adapter._state = AdapterState.DISCONNECTED
        with patch("time.sleep"):
            result = adapter.reconnect_with_backoff(max_attempts=3, initial_delay=0.01)
        assert result is False
        assert adapter.state == AdapterState.ERROR

    def test_reconnect_calls_reconnect_handler_on_success(self):
        adapter = _ConcreteAdapter("bot", connect_fails=0)
        adapter._state = AdapterState.DISCONNECTED
        cb = MagicMock()
        adapter.on_reconnect(cb)
        with patch("time.sleep"):
            adapter.reconnect_with_backoff(max_attempts=3, initial_delay=0.01)
        cb.assert_called_once()

    def test_reconnect_does_not_call_reconnect_handler_on_failure(self):
        adapter = _ConcreteAdapter("bot", connect_fails=999)
        adapter._state = AdapterState.DISCONNECTED
        cb = MagicMock()
        adapter.on_reconnect(cb)
        with patch("time.sleep"):
            adapter.reconnect_with_backoff(max_attempts=2, initial_delay=0.01)
        cb.assert_not_called()

    def test_backoff_delays_increase(self):
        adapter = _ConcreteAdapter("bot", connect_fails=999)
        adapter._state = AdapterState.DISCONNECTED
        sleep_calls = []
        with patch("time.sleep", side_effect=lambda d: sleep_calls.append(d)):
            adapter.reconnect_with_backoff(
                max_attempts=4,
                initial_delay=1.0,
                backoff_factor=2.0,
            )
        # Delays should be increasing (capped at max_delay)
        assert sleep_calls == [1.0, 2.0, 4.0]

    def test_backoff_capped_at_max_delay(self):
        adapter = _ConcreteAdapter("bot", connect_fails=999)
        adapter._state = AdapterState.DISCONNECTED
        sleep_calls = []
        with patch("time.sleep", side_effect=lambda d: sleep_calls.append(d)):
            adapter.reconnect_with_backoff(
                max_attempts=5,
                initial_delay=10.0,
                backoff_factor=4.0,
                max_delay=15.0,
            )
        # After first sleep (10.0), 10*4=40 > 15 → capped
        assert all(d <= 15.0 for d in sleep_calls)


# ---------------------------------------------------------------------------
# Observability events
# ---------------------------------------------------------------------------

class TestObservabilityEvents:
    def test_disconnect_emits_event(self):
        from apyrobo.observability import on_event, clear_event_handlers
        events = []
        clear_event_handlers()
        on_event(lambda e: events.append(e))

        adapter = MockAdapter("bot")
        adapter.disconnect()

        clear_event_handlers()
        disconnect_events = [e for e in events if e.event_type == "adapter.disconnect"]
        assert len(disconnect_events) == 1
        assert disconnect_events[0].data["robot"] == "bot"

    def test_reconnect_emits_event(self):
        from apyrobo.observability import on_event, clear_event_handlers
        events = []
        clear_event_handlers()
        on_event(lambda e: events.append(e))

        adapter = _ConcreteAdapter("bot", connect_fails=0)
        adapter._state = AdapterState.DISCONNECTED
        with patch("time.sleep"):
            adapter.reconnect_with_backoff(max_attempts=2, initial_delay=0.01)

        clear_event_handlers()
        reconnect_events = [e for e in events if e.event_type == "adapter.reconnect"]
        assert len(reconnect_events) == 1
        assert reconnect_events[0].data["robot"] == "bot"
