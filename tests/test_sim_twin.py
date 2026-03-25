"""Tests for digital twin sync."""
import asyncio
import pytest
from apyrobo.sim.twin import DigitalTwinSync, TwinSyncConfig, TwinState, MockPhysicalSource
from apyrobo.sim.mujoco import MockMuJoCoAdapter


@pytest.fixture
def physical():
    return MockPhysicalSource("robot-test")


@pytest.fixture
def sim():
    adapter = MockMuJoCoAdapter()
    asyncio.get_event_loop().run_until_complete(adapter.connect())
    return adapter


@pytest.fixture
def twin(physical, sim):
    config = TwinSyncConfig(robot_id="robot-test")
    return DigitalTwinSync(config, physical, sim)


def test_config_defaults():
    cfg = TwinSyncConfig(robot_id="r1")
    assert cfg.sync_interval_ms == 100
    assert cfg.bidirectional is False


def test_mock_physical_state(physical):
    state = physical.get_state()
    assert "qpos" in state
    assert "time" in state


def test_mock_physical_ticks(physical):
    s1 = physical.get_state()
    s2 = physical.get_state()
    assert s2["time"] > s1["time"]


def test_sync_once(twin):
    state = asyncio.get_event_loop().run_until_complete(twin.sync_once())
    assert isinstance(state, TwinState)
    assert state.synced is True
    assert state.robot_id == "robot-test"


def test_sync_once_drift_computed(twin):
    state = asyncio.get_event_loop().run_until_complete(twin.sync_once())
    assert isinstance(state.drift, dict)


def test_sync_history_grows(twin):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(twin.sync_once())
    loop.run_until_complete(twin.sync_once())
    loop.run_until_complete(twin.sync_once())
    history = twin.get_sync_history(10)
    assert len(history) == 3


def test_sync_history_limit(twin):
    loop = asyncio.get_event_loop()
    for _ in range(5):
        loop.run_until_complete(twin.sync_once())
    history = twin.get_sync_history(3)
    assert len(history) == 3


def test_get_twin_state_none_initially(physical, sim):
    config = TwinSyncConfig(robot_id="r1")
    twin = DigitalTwinSync(config, physical, sim)
    assert twin.get_twin_state() is None


def test_get_twin_state_after_sync(twin):
    asyncio.get_event_loop().run_until_complete(twin.sync_once())
    state = twin.get_twin_state()
    assert state is not None


def test_bidirectional_mode(physical, sim):
    config = TwinSyncConfig(robot_id="r-bidir", bidirectional=True)
    twin = DigitalTwinSync(config, physical, sim)
    state = asyncio.get_event_loop().run_until_complete(twin.sync_once())
    assert state.synced is True


def test_compute_drift_scalar(twin, physical, sim):
    state = asyncio.get_event_loop().run_until_complete(twin.sync_once())
    # drift should be dict
    assert isinstance(state.drift, dict)


def test_start_stop(twin):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(twin.start())
    loop.run_until_complete(asyncio.sleep(0.05))
    loop.run_until_complete(twin.stop())
    assert twin._running is False
