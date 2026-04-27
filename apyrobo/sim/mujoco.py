"""MuJoCo physics simulation adapter for apyrobo."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import mujoco  # type: ignore

    _MUJOCO_AVAILABLE = True
except ImportError:
    mujoco = None  # type: ignore
    _MUJOCO_AVAILABLE = False


@dataclass
class MuJoCoConfig:
    model_path: str = ""
    timestep: float = 0.002
    max_steps: int = 1000
    render_width: int = 640
    render_height: int = 480


class MuJoCoAdapter:
    """MuJoCo physics adapter. Falls back to stub when mujoco not installed."""

    def __init__(self, model_path: str = "", **kwargs: Any) -> None:
        cfg_fields = {"timestep", "max_steps", "render_width", "render_height"}
        cfg_kwargs = {k: v for k, v in kwargs.items() if k in cfg_fields}
        self.config = MuJoCoConfig(model_path=model_path, **cfg_kwargs)
        self._model: Any = None
        self._data: Any = None
        self._connected = False
        self._stub_state: dict = {"qpos": [], "qvel": [], "time": 0.0}

    async def connect(self) -> None:
        if not _MUJOCO_AVAILABLE:
            logger.warning("mujoco not installed — running in stub mode")
            self._connected = True
            return
        if self.config.model_path:
            try:
                self._model = mujoco.MjModel.from_xml_path(self.config.model_path)
                self._data = mujoco.MjData(self._model)
                logger.info("MuJoCo model loaded: %s", self.config.model_path)
            except Exception as exc:
                logger.warning("Failed to load MuJoCo model (%s) — stub mode", exc)
        self._connected = True

    async def disconnect(self) -> None:
        self._model = None
        self._data = None
        self._connected = False

    def step(self, n_steps: int = 1) -> None:
        if self._data is not None and _MUJOCO_AVAILABLE:
            for _ in range(n_steps):
                mujoco.mj_step(self._model, self._data)
        else:
            self._stub_state["time"] = (
                self._stub_state["time"] + self.config.timestep * n_steps
            )

    def get_state(self) -> dict:
        if self._data is not None and _MUJOCO_AVAILABLE:
            return {
                "qpos": self._data.qpos.tolist(),
                "qvel": self._data.qvel.tolist(),
                "time": float(self._data.time),
            }
        return dict(self._stub_state)

    def set_state(self, qpos: Any, qvel: Any) -> None:
        if self._data is not None and _MUJOCO_AVAILABLE:
            self._data.qpos[:] = qpos
            self._data.qvel[:] = qvel
        else:
            self._stub_state["qpos"] = list(qpos) if hasattr(qpos, "__iter__") else [qpos]
            self._stub_state["qvel"] = list(qvel) if hasattr(qvel, "__iter__") else [qvel]

    def reset(self) -> None:
        if self._data is not None and _MUJOCO_AVAILABLE:
            mujoco.mj_resetData(self._model, self._data)
        else:
            self._stub_state = {"qpos": [], "qvel": [], "time": 0.0}

    def render(self) -> Optional[bytes]:
        return None  # headless environments; renderer requires display

    def execute_skill(self, skill_name: str, params: dict) -> dict:
        dispatchers = {
            "move_to": self._skill_move_to,
            "grasp": self._skill_grasp,
            "release": self._skill_release,
            "navigate": self._skill_navigate,
        }
        fn = dispatchers.get(skill_name)
        if fn is None:
            return {"status": "error", "message": f"Unknown skill: {skill_name}"}
        return fn(params)

    def _skill_move_to(self, params: dict) -> dict:
        self.step(10)
        return {"status": "ok", "action": "move_to", "target": params.get("target")}

    def _skill_grasp(self, params: dict) -> dict:
        self.step(5)
        return {"status": "ok", "action": "grasp", "object": params.get("object")}

    def _skill_release(self, params: dict) -> dict:
        self.step(5)
        return {"status": "ok", "action": "release"}

    def _skill_navigate(self, params: dict) -> dict:
        self.step(20)
        return {"status": "ok", "action": "navigate", "goal": params.get("goal")}


class MockMuJoCoAdapter(MuJoCoAdapter):
    """Pure-Python mock adapter — no MuJoCo required."""

    def __init__(self, model_path: str = "", **kwargs: Any) -> None:
        super().__init__(model_path, **kwargs)
        self._step_count = 0

    async def connect(self) -> None:
        self._connected = True
        self._stub_state = {"qpos": [0.0, 0.0, 0.0], "qvel": [0.0, 0.0, 0.0], "time": 0.0}

    def step(self, n_steps: int = 1) -> None:
        self._step_count += n_steps
        self._stub_state["time"] = (
            self._stub_state["time"] + self.config.timestep * n_steps
        )
        for i in range(len(self._stub_state["qpos"])):
            self._stub_state["qpos"][i] += (
                self._stub_state["qvel"][i] * self.config.timestep * n_steps
            )

    def set_state(self, qpos: Any, qvel: Any) -> None:
        self._stub_state["qpos"] = list(qpos) if hasattr(qpos, "__iter__") else [qpos]
        self._stub_state["qvel"] = list(qvel) if hasattr(qvel, "__iter__") else [qvel]

    def reset(self) -> None:
        self._stub_state = {"qpos": [0.0, 0.0, 0.0], "qvel": [0.0, 0.0, 0.0], "time": 0.0}
        self._step_count = 0

    def is_connected(self) -> bool:
        return self._connected
