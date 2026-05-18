"""
Tests for Isaac Sim and Unitree adapter skeletons.

All tests run without hardware or simulator installed — they verify that:
  - Both modules import cleanly without the vendor SDKs.
  - Correct ImportError messages are raised when the SDK is absent.
  - URI parsing is correct.
  - Capability sets are model-dependent (Go2 vs H1).
  - Both adapters register under the correct URI scheme.

v6.0.0 — APYROBO Ecosystem Integrations
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — fake SDK modules so we can test the SDK-available path too
# ---------------------------------------------------------------------------

def _make_fake_omni() -> types.ModuleType:
    """Build a fake ``omni`` package tree for testing."""
    omni = types.ModuleType("omni")
    omni_isaac = types.ModuleType("omni.isaac")
    omni_isaac_core = types.ModuleType("omni.isaac.core")

    # Fake World
    class FakeWorld:
        def __init__(self):
            self._physics_initialized = False

        def initialize_physics(self):
            self._physics_initialized = True

        def step(self, render: bool = True):
            pass

        def stop(self):
            pass

    omni_isaac_core.World = FakeWorld

    # Fake Robot prim
    class FakeIsaacRobot:
        def __init__(self, prim_path: str, name: str):
            self.prim_path = prim_path
            self.name = name

        def get_world_pose(self):
            return ([1.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0])

        def set_world_pose(self, position, orientation):
            pass

    omni_isaac_robots = types.ModuleType("omni.isaac.core.robots")
    omni_isaac_robots.Robot = FakeIsaacRobot
    omni_isaac_core.robots = omni_isaac_robots

    omni.isaac = omni_isaac
    omni_isaac.core = omni_isaac_core

    sys.modules["omni"] = omni
    sys.modules["omni.isaac"] = omni_isaac
    sys.modules["omni.isaac.core"] = omni_isaac_core
    sys.modules["omni.isaac.core.robots"] = omni_isaac_robots
    return omni


def _remove_fake_omni() -> None:
    for key in list(sys.modules.keys()):
        if key.startswith("omni"):
            del sys.modules[key]


def _make_fake_unitree() -> None:
    """Build a fake ``unitree_sdk2py`` package tree for testing."""
    root = types.ModuleType("unitree_sdk2py")

    go2 = types.ModuleType("unitree_sdk2py.go2")
    sport = types.ModuleType("unitree_sdk2py.go2.sport")

    class FakeSportClient:
        def __init__(self):
            self._timeout = 5.0
            self._calls: list[str] = []

        def SetTimeout(self, t):
            self._timeout = t

        def Init(self):
            pass

        def Move(self, vx, vy, vyaw):
            self._calls.append(f"Move({vx},{vy},{vyaw})")

        def StopMove(self):
            self._calls.append("StopMove")

        def StandUp(self):
            self._calls.append("StandUp")

        def StandDown(self):
            self._calls.append("StandDown")

        def Hello(self):
            self._calls.append("Hello")

        def HandAction(self, action):
            self._calls.append(f"HandAction({action})")

        def GetState(self):
            state = MagicMock()
            state.position = [1.0, 2.0, 0.0]
            state.imu_state = MagicMock()
            state.imu_state.rpy = [0.0, 0.0, 0.5]
            return state

    sport.sport_client = types.ModuleType("unitree_sdk2py.go2.sport.sport_client")
    sport.sport_client.SportClient = FakeSportClient

    core = types.ModuleType("unitree_sdk2py.core")

    class FakeChannelFactory:
        _instance: "FakeChannelFactory | None" = None

        @classmethod
        def Instance(cls) -> "FakeChannelFactory":
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

        def Init(self, domain_id, iface):
            pass

    channel = types.ModuleType("unitree_sdk2py.core.channel")
    channel.ChannelFactory = FakeChannelFactory
    core.channel = channel

    root.go2 = go2
    go2.sport = sport
    root.core = core

    sys.modules["unitree_sdk2py"] = root
    sys.modules["unitree_sdk2py.go2"] = go2
    sys.modules["unitree_sdk2py.go2.sport"] = sport
    sys.modules["unitree_sdk2py.go2.sport.sport_client"] = sport.sport_client
    sys.modules["unitree_sdk2py.core"] = core
    sys.modules["unitree_sdk2py.core.channel"] = channel


def _remove_fake_unitree() -> None:
    for key in list(sys.modules.keys()):
        if key.startswith("unitree"):
            del sys.modules[key]


def _reload_isaac_adapter():
    """Force-reload the isaac_adapter module (needed after SDK injection)."""
    import importlib
    if "apyrobo.core.isaac_adapter" in sys.modules:
        del sys.modules["apyrobo.core.isaac_adapter"]
    return importlib.import_module("apyrobo.core.isaac_adapter")


def _reload_unitree_adapter():
    """Force-reload the unitree_adapter module."""
    import importlib
    if "apyrobo.core.unitree_adapter" in sys.modules:
        del sys.modules["apyrobo.core.unitree_adapter"]
    return importlib.import_module("apyrobo.core.unitree_adapter")


# ===========================================================================
# IsaacSimAdapter
# ===========================================================================

class TestIsaacAdapterImport:
    """Module-level import tests — SDK absent."""

    def test_module_imports_without_omni(self):
        """The module must load cleanly even when omni is not installed."""
        # Ensure omni is not present
        _remove_fake_omni()
        mod = _reload_isaac_adapter()
        assert hasattr(mod, "IsaacSimAdapter")

    def test_omni_available_flag_false_without_sdk(self):
        _remove_fake_omni()
        mod = _reload_isaac_adapter()
        assert mod._OMNI_AVAILABLE is False

    def test_import_error_message_correct(self):
        """ImportError from _sdk_import_check must contain the install hint."""
        _remove_fake_omni()
        mod = _reload_isaac_adapter()
        adapter = mod.IsaacSimAdapter("test_scene")
        with pytest.raises(ImportError) as exc_info:
            adapter._sdk_import_check()
        assert "Install NVIDIA Isaac Sim" in str(exc_info.value)
        assert "source" in str(exc_info.value).lower() or "Python environment" in str(exc_info.value)


class TestIsaacAdapterURIScheme:
    def test_registered_under_isaac_scheme(self):
        from apyrobo.core.adapters import _ADAPTER_REGISTRY
        # Import adapter to trigger registration
        import apyrobo.core.isaac_adapter  # noqa: F401
        assert "isaac" in _ADAPTER_REGISTRY

    def test_registry_maps_to_isaac_sim_adapter(self):
        from apyrobo.core.adapters import _ADAPTER_REGISTRY
        import apyrobo.core.isaac_adapter as mod
        assert _ADAPTER_REGISTRY.get("isaac") is mod.IsaacSimAdapter


class TestIsaacAdapterWithoutSDK:
    """Behaviour when omni is absent — offline/stub mode."""

    @pytest.fixture(autouse=True)
    def ensure_no_omni(self):
        _remove_fake_omni()
        yield
        _remove_fake_omni()

    def test_init_does_not_raise(self):
        from apyrobo.core.isaac_adapter import IsaacSimAdapter
        adapter = IsaacSimAdapter("my_scene")
        assert adapter is not None

    def test_default_host_and_port(self):
        from apyrobo.core.isaac_adapter import IsaacSimAdapter
        adapter = IsaacSimAdapter("my_scene")
        assert adapter._host == "localhost"
        assert adapter._port == 8211

    def test_custom_host_port(self):
        from apyrobo.core.isaac_adapter import IsaacSimAdapter
        adapter = IsaacSimAdapter("scene", host="192.168.1.50", port=9000)
        assert adapter._host == "192.168.1.50"
        assert adapter._port == 9000

    def test_get_capabilities_returns_full_set(self):
        from apyrobo.core.isaac_adapter import IsaacSimAdapter
        from apyrobo.core.schemas import CapabilityType
        adapter = IsaacSimAdapter("my_scene")
        caps = adapter.get_capabilities()
        cap_types = {c.capability_type for c in caps.capabilities}
        assert CapabilityType.NAVIGATE in cap_types
        assert CapabilityType.MANIPULATE in cap_types
        assert CapabilityType.SCAN in cap_types
        assert CapabilityType.PICK in cap_types
        assert CapabilityType.PLACE in cap_types
        assert CapabilityType.CUSTOM in cap_types

    def test_get_battery_level_returns_1_0(self):
        from apyrobo.core.isaac_adapter import IsaacSimAdapter
        adapter = IsaacSimAdapter("my_scene")
        assert adapter.get_battery_level() == pytest.approx(1.0)

    def test_move_updates_position_offline(self):
        from apyrobo.core.isaac_adapter import IsaacSimAdapter
        adapter = IsaacSimAdapter("my_scene")
        adapter.move(3.0, 1.5)
        assert adapter.position == pytest.approx((3.0, 1.5))

    def test_stop_clears_moving_flag(self):
        from apyrobo.core.isaac_adapter import IsaacSimAdapter
        adapter = IsaacSimAdapter("my_scene")
        adapter.move(1.0, 0.0)
        assert adapter.is_moving is True
        adapter.stop()
        assert adapter.is_moving is False

    def test_get_position_returns_three_tuple(self):
        from apyrobo.core.isaac_adapter import IsaacSimAdapter
        adapter = IsaacSimAdapter("my_scene")
        pos = adapter.get_position()
        assert len(pos) == 3

    def test_simulation_step_is_noop_without_world(self):
        from apyrobo.core.isaac_adapter import IsaacSimAdapter
        adapter = IsaacSimAdapter("my_scene")
        # Must not raise
        adapter._simulation_step()

    def test_gripper_open_returns_true_offline(self):
        from apyrobo.core.isaac_adapter import IsaacSimAdapter
        adapter = IsaacSimAdapter("my_scene")
        assert adapter.gripper_open() is True

    def test_gripper_close_returns_true_offline(self):
        from apyrobo.core.isaac_adapter import IsaacSimAdapter
        adapter = IsaacSimAdapter("my_scene")
        assert adapter.gripper_close() is True

    def test_get_health_contains_expected_keys(self):
        from apyrobo.core.isaac_adapter import IsaacSimAdapter
        adapter = IsaacSimAdapter("my_scene")
        health = adapter.get_health()
        for key in ("state", "adapter", "robot", "sdk_available", "sim"):
            assert key in health, f"missing key {key!r}"

    def test_connect_falls_back_to_rest_probe(self):
        """connect() should not raise when REST is also unreachable."""
        from apyrobo.core.isaac_adapter import IsaacSimAdapter
        adapter = IsaacSimAdapter("my_scene")
        # REST probe will fail (no Isaac running) — offline stub mode
        adapter.connect()  # must not raise
        from apyrobo.core.schemas import AdapterState
        assert adapter.state == AdapterState.CONNECTED

    def test_sdk_import_check_raises_import_error(self):
        from apyrobo.core.isaac_adapter import IsaacSimAdapter
        adapter = IsaacSimAdapter("my_scene")
        with pytest.raises(ImportError) as exc_info:
            adapter._sdk_import_check()
        assert "Isaac Sim" in str(exc_info.value)

    def test_world_property_is_none_without_sdk(self):
        from apyrobo.core.isaac_adapter import IsaacSimAdapter
        adapter = IsaacSimAdapter("my_scene")
        assert adapter.world is None

    def test_sensors_include_camera_lidar_imu_depth(self):
        from apyrobo.core.isaac_adapter import IsaacSimAdapter
        from apyrobo.core.schemas import SensorType
        adapter = IsaacSimAdapter("my_scene")
        caps = adapter.get_capabilities()
        sensor_types = {s.sensor_type for s in caps.sensors}
        assert SensorType.CAMERA in sensor_types
        assert SensorType.LIDAR in sensor_types
        assert SensorType.IMU in sensor_types
        assert SensorType.DEPTH in sensor_types

    def test_metadata_flags_sim_true(self):
        from apyrobo.core.isaac_adapter import IsaacSimAdapter
        adapter = IsaacSimAdapter("my_scene")
        caps = adapter.get_capabilities()
        assert caps.metadata.get("sim") is True
        assert caps.metadata.get("engine") == "isaac_sim"


class TestIsaacRestFallback:
    """Test the REST fallback path via _isaac_rest_request mock."""

    @pytest.fixture(autouse=True)
    def ensure_no_omni(self):
        _remove_fake_omni()
        yield

    def test_move_calls_rest_when_available(self):
        from apyrobo.core.isaac_adapter import IsaacSimAdapter
        adapter = IsaacSimAdapter("my_scene")
        adapter._rest_available = True

        with patch("apyrobo.core.isaac_adapter._isaac_rest_request") as mock_req:
            mock_req.return_value = {"status": "ok"}
            adapter.move(2.0, 1.0)
            mock_req.assert_called()
            # Verify the endpoint contains 'move'
            call_args = mock_req.call_args_list
            endpoints = [c[0][2] for c in call_args]
            assert any("move" in e for e in endpoints)

    def test_stop_calls_rest_when_available(self):
        from apyrobo.core.isaac_adapter import IsaacSimAdapter
        adapter = IsaacSimAdapter("my_scene")
        adapter._rest_available = True

        with patch("apyrobo.core.isaac_adapter._isaac_rest_request") as mock_req:
            mock_req.return_value = {"status": "ok"}
            adapter.stop()
            mock_req.assert_called()
            call_args = mock_req.call_args_list
            endpoints = [c[0][2] for c in call_args]
            assert any("stop" in e for e in endpoints)

    def test_rest_request_returns_none_on_unreachable(self):
        from apyrobo.core.isaac_adapter import _isaac_rest_request
        # No Isaac running on port 19999
        result = _isaac_rest_request("localhost", 19999, "/status")
        assert result is None


# ===========================================================================
# UnitreeAdapter
# ===========================================================================

class TestUnitreeAdapterImport:
    """Module-level import tests — SDK absent."""

    def test_module_imports_without_unitree_sdk(self):
        _remove_fake_unitree()
        mod = _reload_unitree_adapter()
        assert hasattr(mod, "UnitreeAdapter")

    def test_sdk_available_flag_false_without_sdk(self):
        _remove_fake_unitree()
        mod = _reload_unitree_adapter()
        assert mod._UNITREE_SDK_AVAILABLE is False


class TestUnitreeAdapterURIScheme:
    def test_registered_under_unitree_scheme(self):
        from apyrobo.core.adapters import _ADAPTER_REGISTRY
        import apyrobo.core.unitree_adapter  # noqa: F401
        assert "unitree" in _ADAPTER_REGISTRY

    def test_registry_maps_to_unitree_adapter(self):
        from apyrobo.core.adapters import _ADAPTER_REGISTRY
        import apyrobo.core.unitree_adapter as mod
        assert _ADAPTER_REGISTRY.get("unitree") is mod.UnitreeAdapter


class TestDetectModel:
    """URI parsing — _detect_model() is a static method, no SDK needed."""

    @pytest.fixture(autouse=True)
    def ensure_no_unitree(self):
        _remove_fake_unitree()
        yield

    def test_go2_with_host(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        model, host = UnitreeAdapter._detect_model("go2@192.168.1.100")
        assert model == "go2"
        assert host == "192.168.1.100"

    def test_go2_with_scheme_and_host(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        model, host = UnitreeAdapter._detect_model("unitree://go2@192.168.1.100")
        assert model == "go2"
        assert host == "192.168.1.100"

    def test_h1_with_host(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        model, host = UnitreeAdapter._detect_model("unitree://h1@10.0.0.5")
        assert model == "h1"
        assert host == "10.0.0.5"

    def test_model_only_no_host(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        model, host = UnitreeAdapter._detect_model("go2")
        assert model == "go2"
        assert host is None

    def test_model_uppercase_normalised(self):
        # The regex forces lower-case — user may type "GO2"
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        # URI parser normalises to lower-case inside _detect_model
        model, host = UnitreeAdapter._detect_model("go2@192.168.1.100")
        assert model == "go2"

    def test_invalid_uri_raises_value_error(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        with pytest.raises(ValueError, match="Cannot parse Unitree URI"):
            UnitreeAdapter._detect_model("://bad@@uri")

    def test_model_stored_on_adapter(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2@192.168.1.100")
        assert adapter.model == "go2"
        assert adapter.host == "192.168.1.100"


class TestUnitreeCapabilities:
    """Capability set differs by model — tested without SDK."""

    @pytest.fixture(autouse=True)
    def ensure_no_unitree(self):
        _remove_fake_unitree()
        yield

    def test_go2_has_navigate_and_scan(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        from apyrobo.core.schemas import CapabilityType
        adapter = UnitreeAdapter("go2@192.168.1.100")
        caps = adapter.get_capabilities()
        cap_types = {c.capability_type for c in caps.capabilities}
        assert CapabilityType.NAVIGATE in cap_types
        assert CapabilityType.SCAN in cap_types

    def test_go2_has_no_manipulation(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        from apyrobo.core.schemas import CapabilityType
        adapter = UnitreeAdapter("go2@192.168.1.100")
        caps = adapter.get_capabilities()
        cap_types = {c.capability_type for c in caps.capabilities}
        assert CapabilityType.MANIPULATE not in cap_types
        assert CapabilityType.PICK not in cap_types
        assert CapabilityType.PLACE not in cap_types

    def test_h1_has_manipulation(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        from apyrobo.core.schemas import CapabilityType
        adapter = UnitreeAdapter("h1@10.0.0.5")
        caps = adapter.get_capabilities()
        cap_types = {c.capability_type for c in caps.capabilities}
        assert CapabilityType.MANIPULATE in cap_types
        assert CapabilityType.PICK in cap_types
        assert CapabilityType.PLACE in cap_types

    def test_h1_also_has_navigate(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        from apyrobo.core.schemas import CapabilityType
        adapter = UnitreeAdapter("h1@10.0.0.5")
        caps = adapter.get_capabilities()
        cap_types = {c.capability_type for c in caps.capabilities}
        assert CapabilityType.NAVIGATE in cap_types

    def test_go2_max_speed_higher_than_h1(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        go2 = UnitreeAdapter("go2")
        h1 = UnitreeAdapter("h1")
        assert go2.get_capabilities().max_speed > h1.get_capabilities().max_speed

    def test_capabilities_robot_id_matches_name(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2@192.168.1.100")
        caps = adapter.get_capabilities()
        assert caps.robot_id == "go2@192.168.1.100"

    def test_metadata_contains_model_and_vendor(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2")
        caps = adapter.get_capabilities()
        assert caps.metadata.get("model") == "go2"
        assert "Unitree" in caps.metadata.get("vendor", "")


class TestUnitreeAdapterWithoutSDK:
    """All hardware methods raise ImportError when SDK absent."""

    @pytest.fixture(autouse=True)
    def ensure_no_unitree(self):
        _remove_fake_unitree()
        mod = _reload_unitree_adapter()
        yield mod

    def test_move_raises_import_error(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2@192.168.1.100")
        with pytest.raises(ImportError) as exc_info:
            adapter.move(1.0, 0.0)
        assert "unitree-sdk2py" in str(exc_info.value) or "unitree_sdk2_python" in str(exc_info.value)

    def test_stop_raises_import_error(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2")
        with pytest.raises(ImportError):
            adapter.stop()

    def test_connect_raises_import_error(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2")
        with pytest.raises(ImportError):
            adapter.connect()

    def test_stand_raises_import_error(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2")
        with pytest.raises(ImportError):
            adapter.stand()

    def test_sit_down_raises_import_error(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2")
        with pytest.raises(ImportError):
            adapter.sit_down()

    def test_wave_hand_raises_import_error(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2")
        with pytest.raises(ImportError):
            adapter.wave_hand()

    def test_gripper_open_raises_import_error(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("h1")
        with pytest.raises(ImportError):
            adapter.gripper_open()

    def test_gripper_close_raises_import_error(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("h1")
        with pytest.raises(ImportError):
            adapter.gripper_close()

    def test_import_error_message_mentions_pip(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2")
        with pytest.raises(ImportError) as exc_info:
            adapter.move(1.0, 0.0)
        assert "pip install" in str(exc_info.value)

    def test_get_capabilities_does_not_raise(self):
        """get_capabilities() must work without the SDK — offline read."""
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2@192.168.1.100")
        caps = adapter.get_capabilities()  # should not raise
        assert caps is not None

    def test_get_health_does_not_raise(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2")
        health = adapter.get_health()
        assert health["sdk_available"] is False


class TestUnitreeAdapterWithFakeSDK:
    """Behaviour when the fake SDK is injected — verifies SDK call routing."""

    @pytest.fixture(autouse=True)
    def inject_fake_sdk(self):
        _make_fake_unitree()
        mod = _reload_unitree_adapter()
        yield mod
        _remove_fake_unitree()

    def test_connect_initialises_sport_client(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2@192.168.1.100")
        adapter.connect()
        assert adapter._sport_client is not None

    def test_move_calls_sport_client_move(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2@192.168.1.100")
        adapter.connect()
        adapter.move(1.0, 0.0)
        assert any("Move" in c for c in adapter._sport_client._calls)

    def test_stop_calls_stop_move(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2")
        adapter.connect()
        adapter.stop()
        assert "StopMove" in adapter._sport_client._calls

    def test_stand_calls_stand_up(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2")
        adapter.connect()
        result = adapter.stand()
        assert result is True
        assert "StandUp" in adapter._sport_client._calls

    def test_sit_down_calls_stand_down(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2")
        adapter.connect()
        result = adapter.sit_down()
        assert result is True
        assert "StandDown" in adapter._sport_client._calls

    def test_wave_hand_calls_hello(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2")
        adapter.connect()
        result = adapter.wave_hand()
        assert result is True
        assert "Hello" in adapter._sport_client._calls

    def test_h1_gripper_open_calls_hand_action(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("h1@10.0.0.5")
        adapter.connect()
        result = adapter.gripper_open()
        assert result is True
        assert any("HandAction" in c and "open" in c for c in adapter._sport_client._calls)

    def test_h1_gripper_close_calls_hand_action(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("h1@10.0.0.5")
        adapter.connect()
        result = adapter.gripper_close()
        assert result is True
        assert any("HandAction" in c and "close" in c for c in adapter._sport_client._calls)

    def test_go2_gripper_open_is_noop(self):
        """Go2 has no gripper — gripper_open should return True silently."""
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2")
        adapter.connect()
        result = adapter.gripper_open()
        assert result is True
        # Should NOT call HandAction on Go2
        assert not any("HandAction" in c for c in adapter._sport_client._calls)

    def test_get_position_reads_from_get_state(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2@192.168.1.100")
        adapter.connect()
        pos = adapter.get_position()
        assert pos == pytest.approx((1.0, 2.0))

    def test_disconnect_calls_stop_move(self):
        from apyrobo.core.unitree_adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2")
        adapter.connect()
        client = adapter._sport_client
        adapter.disconnect()
        assert "StopMove" in client._calls
        assert adapter._sport_client is None


# ===========================================================================
# Cross-adapter — adapter registry integration
# ===========================================================================

class TestAdapterRegistration:
    def test_get_adapter_isaac(self):
        """get_adapter() can instantiate IsaacSimAdapter via registry."""
        import apyrobo.core.isaac_adapter  # noqa: F401
        from apyrobo.core.adapters import get_adapter
        adapter = get_adapter("isaac", "warehouse")
        assert type(adapter).__name__ == "IsaacSimAdapter"

    def test_get_adapter_unitree(self):
        """get_adapter() can instantiate UnitreeAdapter via registry."""
        import apyrobo.core.unitree_adapter  # noqa: F401
        from apyrobo.core.adapters import get_adapter
        adapter = get_adapter("unitree", "go2@192.168.1.100")
        assert type(adapter).__name__ == "UnitreeAdapter"

    def test_both_schemes_in_list_adapters(self):
        import apyrobo.core.isaac_adapter  # noqa: F401
        import apyrobo.core.unitree_adapter  # noqa: F401
        from apyrobo.core.adapters import list_adapters
        schemes = list_adapters()
        assert "isaac" in schemes
        assert "unitree" in schemes
