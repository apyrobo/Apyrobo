"""Tests for digital twin sync."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from apyrobo.sim.twin import (
    DigitalTwinSync,
    MockPhysicalSource,
    TwinState,
    TwinSyncConfig,
)


class MockSimAdapter:
    """Minimal sim adapter for testing."""

    def __init__(self) -> None:
        self._state: dict = {}
        self.set_calls: list[dict] = []

    def get_state(self, fields: list[str]) -> dict:
        return {f: self._state.get(f, 0.0) for f in fields}

    def set_state(self, state: dict) -> None:
        self._state.update(state)
        self.set_calls.append(dict(state))


def make_twin(
    fields=None,
    bidirectional=False,
    seed=42,
) -> DigitalTwinSync:
    fields = fields or ["x", "y", "z"]
    config = TwinSyncConfig(
        robot_id="bot-1",
        sim_adapter="mock",
        sync_interval_ms=100,
        state_fields=fields,
        bidirectional=bidirectional,
    )
    physical = MockPhysicalSource("bot-1", seed=seed)
    sim = MockSimAdapter()
    return DigitalTwinSync(config, physical, sim)


class TestSyncOnce:
    def test_sync_once_returns_twin_state(self):
        twin = make_twin()
        state = twin.sync_once()
        assert isinstance(state, TwinState)
        assert state.robot_id == "bot-1"

    def test_sync_once_has_physical_and_sim_state(self):
        twin = make_twin(["x", "y"])
        state = twin.sync_once()
        assert "x" in state.physical_state
        assert "y" in state.sim_state

    def test_sync_once_sim_updated_from_physical(self):
        twin = make_twin(["x"])
        state = twin.sync_once()
        # After sync, sim state should equal physical state
        assert state.sim_state["x"] == pytest.approx(state.physical_state["x"], abs=1e-9)

    def test_sync_once_drift_computed(self):
        twin = make_twin(["x", "y"])
        state = twin.sync_once()
        assert "x" in state.drift
        assert "y" in state.drift

    def test_sync_once_timestamp_set(self):
        twin = make_twin()
        before = datetime.utcnow()
        state = twin.sync_once()
        after = datetime.utcnow()
        assert before <= state.timestamp <= after

    def test_synced_true_after_successful_sync(self):
        twin = make_twin(["x"])
        state = twin.sync_once()
        assert state.synced is True  # drift should be ~0 after applying physical → sim


class TestDriftComputation:
    def test_drift_zero_when_states_equal(self):
        twin = make_twin()
        physical = {"x": 1.0, "y": 2.0}
        sim = {"x": 1.0, "y": 2.0}
        drift = twin._compute_drift(physical, sim)
        assert drift["x"] == pytest.approx(0.0)
        assert drift["y"] == pytest.approx(0.0)

    def test_drift_nonzero_when_states_differ(self):
        twin = make_twin()
        physical = {"x": 3.0}
        sim = {"x": 1.0}
        drift = twin._compute_drift(physical, sim)
        assert drift["x"] == pytest.approx(2.0)

    def test_drift_none_for_non_numeric_fields(self):
        twin = make_twin()
        drift = twin._compute_drift({"mode": "auto"}, {"mode": "manual"})
        assert drift["mode"] is None


class TestSyncHistory:
    def test_history_starts_empty(self):
        twin = make_twin()
        assert twin.get_sync_history() == []

    def test_history_grows_with_syncs(self):
        twin = make_twin()
        for _ in range(5):
            twin.sync_once()
        assert len(twin.get_sync_history()) == 5

    def test_history_respects_n_limit(self):
        twin = make_twin()
        for _ in range(10):
            twin.sync_once()
        history = twin.get_sync_history(n=3)
        assert len(history) == 3

    def test_history_returns_most_recent(self):
        twin = make_twin()
        for _ in range(5):
            twin.sync_once()
        history = twin.get_sync_history(n=2)
        full = twin.get_sync_history()
        assert history == full[-2:]

    def test_get_twin_state_raises_before_sync(self):
        twin = make_twin()
        with pytest.raises(RuntimeError, match="No sync performed"):
            twin.get_twin_state()

    def test_get_twin_state_returns_last(self):
        twin = make_twin()
        state = twin.sync_once()
        assert twin.get_twin_state() is state


class TestBidirectional:
    def test_bidirectional_applies_sim_to_physical(self):
        twin = make_twin(["x"], bidirectional=True)
        # Seed the sim with a known value before sync
        twin._sim._state["x"] = 99.0
        twin.sync_once()
        # physical source should have received the sim's commanded state
        assert "x" in twin._physical._commanded_state

    def test_unidirectional_does_not_apply_to_physical(self):
        twin = make_twin(["x"], bidirectional=False)
        twin._sim._state["x"] = 99.0
        twin.sync_once()
        assert twin._physical._commanded_state == {}


class TestMockPhysicalSource:
    def test_read_state_returns_requested_fields(self):
        src = MockPhysicalSource()
        state = src.read_state(["a", "b", "c"])
        assert set(state.keys()) == {"a", "b", "c"}

    def test_read_state_values_are_numeric(self):
        src = MockPhysicalSource()
        state = src.read_state(["x", "y"])
        for v in state.values():
            assert isinstance(v, float)

    def test_apply_commands_overrides_readings(self):
        src = MockPhysicalSource()
        src.apply_commands({"x": 42.0})
        state = src.read_state(["x"])
        assert state["x"] == 42.0
