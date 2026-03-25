"""MuJoCo adapter and mock for apyrobo sim module."""
from __future__ import annotations
from typing import Any


class MockMuJoCoAdapter:
    """Lightweight mock MuJoCo adapter for testing digital twin sync.

    Provides get_state / set_state so DigitalTwinSync can exercise
    both directions of the physical<->sim bridge without a real MuJoCo
    installation.
    """

    def __init__(self) -> None:
        self._qpos: list[float] = [0.0, 0.0, 0.0]
        self._qvel: list[float] = [0.0, 0.0, 0.0]
        self._time: float = 0.0
        self._connected = False

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    def get_state(self) -> dict:
        return {
            "qpos": list(self._qpos),
            "qvel": list(self._qvel),
            "time": self._time,
        }

    def set_state(self, qpos: list[float], qvel: list[float]) -> None:
        if qpos:
            self._qpos = list(qpos)
        if qvel:
            self._qvel = list(qvel)

    def step(self, dt: float = 0.01) -> None:
        self._time += dt

    def is_connected(self) -> bool:
        return self._connected
