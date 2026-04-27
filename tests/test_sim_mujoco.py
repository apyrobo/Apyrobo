"""Tests for MuJoCo sim adapter."""
import asyncio
import pytest
from apyrobo.sim.mujoco import MuJoCoAdapter, MockMuJoCoAdapter, MuJoCoConfig


@pytest.fixture
def mock_adapter():
    return MockMuJoCoAdapter()


@pytest.fixture
def connected_adapter(mock_adapter):
    asyncio.get_event_loop().run_until_complete(mock_adapter.connect())
    return mock_adapter


def test_config_defaults():
    cfg = MuJoCoConfig()
    assert cfg.timestep == 0.002
    assert cfg.max_steps == 1000
    assert cfg.render_width == 640


def test_mock_connect_sets_connected(mock_adapter):
    asyncio.get_event_loop().run_until_complete(mock_adapter.connect())
    assert mock_adapter._connected is True


def test_mock_initial_state(connected_adapter):
    state = connected_adapter.get_state()
    assert "qpos" in state
    assert "qvel" in state
    assert "time" in state


def test_mock_step_advances_time(connected_adapter):
    t0 = connected_adapter.get_state()["time"]
    connected_adapter.step(10)
    assert connected_adapter.get_state()["time"] > t0


def test_mock_step_count(connected_adapter):
    connected_adapter.step(5)
    assert connected_adapter._step_count == 5


def test_mock_set_state(connected_adapter):
    connected_adapter.set_state([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    assert connected_adapter.get_state()["qpos"] == [1.0, 2.0, 3.0]


def test_mock_reset(connected_adapter):
    connected_adapter.step(50)
    connected_adapter.reset()
    assert connected_adapter.get_state()["time"] == 0.0


def test_mock_disconnect(connected_adapter):
    asyncio.get_event_loop().run_until_complete(connected_adapter.disconnect())
    assert connected_adapter._connected is False


def test_skill_move_to(connected_adapter):
    r = connected_adapter.execute_skill("move_to", {"target": {"x": 1.0}})
    assert r["status"] == "ok"
    assert r["action"] == "move_to"


def test_skill_grasp(connected_adapter):
    r = connected_adapter.execute_skill("grasp", {"object": "cube"})
    assert r["status"] == "ok"
    assert r["object"] == "cube"


def test_skill_release(connected_adapter):
    assert connected_adapter.execute_skill("release", {})["status"] == "ok"


def test_skill_navigate(connected_adapter):
    assert connected_adapter.execute_skill("navigate", {"goal": [5.0, 5.0]})["status"] == "ok"


def test_unknown_skill(connected_adapter):
    assert connected_adapter.execute_skill("fly", {})["status"] == "error"


def test_render_returns_none_or_bytes(connected_adapter):
    r = connected_adapter.render()
    assert r is None or isinstance(r, bytes)


def test_stub_mode_without_mujoco():
    adapter = MuJoCoAdapter(model_path="nonexistent.xml")
    asyncio.get_event_loop().run_until_complete(adapter.connect())
    assert adapter._connected is True
    assert "time" in adapter.get_state()
