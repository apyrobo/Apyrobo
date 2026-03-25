"""Digital twin sync — bidirectional physical↔simulation state mirroring."""
from __future__ import annotations
import asyncio, logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TwinSyncConfig:
    robot_id: str
    sim_adapter: str = "mujoco"
    sync_interval_ms: int = 100
    state_fields: list[str] = field(default_factory=lambda: ["qpos", "qvel", "time"])
    bidirectional: bool = False


@dataclass
class TwinState:
    robot_id: str
    timestamp: datetime
    physical_state: dict
    sim_state: dict
    drift: dict
    synced: bool


class MockPhysicalSource:
    """Generates synthetic sensor readings for testing."""

    def __init__(self, robot_id: str = "mock-robot") -> None:
        self.robot_id = robot_id
        self._state: dict = {"qpos": [0.0, 0.0, 0.0], "qvel": [0.0, 0.0, 0.0], "time": 0.0, "speed": 0.0}
        self._tick = 0

    def get_state(self) -> dict:
        self._tick += 1
        self._state["time"] = float(self._tick) * 0.1
        self._state["qpos"] = [float(self._tick) * 0.01] * 3
        return dict(self._state)

    def apply_command(self, command: dict) -> None:
        for k, v in command.items():
            if k in self._state:
                self._state[k] = v


class DigitalTwinSync:
    def __init__(self, config: TwinSyncConfig, physical_source: Any, sim_adapter: Any) -> None:
        self.config = config
        self._physical = physical_source
        self._sim = sim_adapter
        self._history: list[TwinState] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._sync_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def get_twin_state(self) -> Optional[TwinState]:
        return self._history[-1] if self._history else None

    async def sync_once(self) -> TwinState:
        physical_state = self._physical.get_state()
        sim_state = self._sim.get_state() if hasattr(self._sim, "get_state") else {}
        drift = self._compute_drift(physical_state, sim_state)
        self._apply_physical_to_sim(physical_state)
        if self.config.bidirectional:
            self._apply_sim_to_physical(sim_state)
        state = TwinState(
            robot_id=self.config.robot_id,
            timestamp=datetime.now(timezone.utc),
            physical_state=physical_state,
            sim_state=sim_state,
            drift=drift,
            synced=True,
        )
        self._history.append(state)
        if len(self._history) > 1000:
            self._history.pop(0)
        return state

    def _compute_drift(self, physical: dict, sim: dict) -> dict:
        drift = {}
        for f in self.config.state_fields:
            pv = physical.get(f)
            sv = sim.get(f)
            if pv is None or sv is None:
                continue
            if isinstance(pv, (int, float)) and isinstance(sv, (int, float)):
                drift[f] = float(pv) - float(sv)
            elif isinstance(pv, list) and isinstance(sv, list):
                drift[f] = [float(a) - float(b) for a, b in zip(pv, sv)]
        return drift

    def _apply_physical_to_sim(self, state: dict) -> None:
        if hasattr(self._sim, "set_state"):
            qpos = state.get("qpos", [])
            qvel = state.get("qvel", [])
            if qpos or qvel:
                try:
                    self._sim.set_state(qpos, qvel)
                except Exception:
                    pass

    def _apply_sim_to_physical(self, state: dict) -> None:
        if hasattr(self._physical, "apply_command"):
            self._physical.apply_command(state)

    def get_sync_history(self, n: int = 10) -> list[TwinState]:
        return list(self._history[-n:])

    async def _sync_loop(self) -> None:
        interval = self.config.sync_interval_ms / 1000.0
        while self._running:
            try:
                await self.sync_once()
                await asyncio.sleep(interval)
            except Exception as exc:
                logger.error("Sync error: %s", exc)
                await asyncio.sleep(interval)
