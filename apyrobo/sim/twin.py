"""
Digital twin sync — mirror physical robot state into a simulation in real-time.

Supports unidirectional (physical → sim) and bidirectional sync.
Uses asyncio for the background sync loop.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class TwinSyncConfig:
    robot_id: str
    sim_adapter: str
    sync_interval_ms: int = 100
    state_fields: list[str] = field(default_factory=list)
    bidirectional: bool = False


@dataclass
class TwinState:
    robot_id: str
    timestamp: datetime
    physical_state: dict
    sim_state: dict
    drift: dict
    synced: bool


class DigitalTwinSync:
    """Continuously synchronise physical robot state to/from a simulation."""

    def __init__(
        self,
        config: TwinSyncConfig,
        physical_source: Any,
        sim_adapter: Any,
    ) -> None:
        self.config = config
        self._physical = physical_source
        self._sim = sim_adapter
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._history: deque[TwinState] = deque(maxlen=1000)
        self._last_state: Optional[TwinState] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start async background sync loop."""
        if self._running:
            return
        self._running = True
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._sync_loop())

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    async def _sync_loop(self) -> None:
        interval = self.config.sync_interval_ms / 1000.0
        while self._running:
            try:
                self.sync_once()
            except Exception:
                pass
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Sync logic
    # ------------------------------------------------------------------

    def sync_once(self) -> TwinState:
        fields = self.config.state_fields

        physical_state = self._physical.read_state(fields)
        sim_state = self._sim.get_state(fields)

        self._apply_physical_to_sim(physical_state)
        updated_sim_state = self._sim.get_state(fields)

        if self.config.bidirectional:
            self._apply_sim_to_physical(updated_sim_state)

        drift = self._compute_drift(physical_state, updated_sim_state)
        synced = all(abs(v) < 1e-6 for v in drift.values() if isinstance(v, (int, float)))

        state = TwinState(
            robot_id=self.config.robot_id,
            timestamp=datetime.utcnow(),
            physical_state=physical_state,
            sim_state=updated_sim_state,
            drift=drift,
            synced=synced,
        )
        self._history.append(state)
        self._last_state = state
        return state

    def _compute_drift(self, physical: dict, sim: dict) -> dict:
        drift = {}
        all_keys = set(physical) | set(sim)
        for k in all_keys:
            pv = physical.get(k)
            sv = sim.get(k)
            if isinstance(pv, (int, float)) and isinstance(sv, (int, float)):
                drift[k] = pv - sv
            else:
                drift[k] = None  # non-numeric fields: no numeric drift
        return drift

    def _apply_physical_to_sim(self, state: dict) -> None:
        self._sim.set_state(state)

    def _apply_sim_to_physical(self, state: dict) -> None:
        if hasattr(self._physical, "apply_commands"):
            self._physical.apply_commands(state)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_twin_state(self) -> TwinState:
        if self._last_state is None:
            raise RuntimeError("No sync performed yet; call sync_once() first")
        return self._last_state

    def get_sync_history(self, n: int = 10) -> list[TwinState]:
        history = list(self._history)
        return history[-n:]


class MockPhysicalSource:
    """Synthetic sensor source for testing — generates deterministic readings."""

    def __init__(self, robot_id: str = "mock-robot", seed: int = 42) -> None:
        self.robot_id = robot_id
        self._rng = random.Random(seed)
        self._commanded_state: dict = {}

    def read_state(self, fields: list[str]) -> dict:
        state = {}
        for f in fields:
            if f in self._commanded_state:
                state[f] = self._commanded_state[f]
            else:
                state[f] = round(self._rng.uniform(-1.0, 1.0), 4)
        return state

    def apply_commands(self, commands: dict) -> None:
        self._commanded_state.update(commands)
