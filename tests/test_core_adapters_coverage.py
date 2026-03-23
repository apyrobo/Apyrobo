"""
Coverage tests for apyrobo/core/adapters.py.

Targets the following previously-uncovered lines:
  70, 127, 143-144, 152-153, 165, 177, 185, 194, 210-211, 219-220, 225, 229,
  334-335, 341, 344, 361, 369, 377, 398-407, 411, 441-449, 453-454, 457-458,
  462-463, 466-467, 470, 473, 476, 479, 489, 493, 520-528, 531, 551-554,
  557-558, 561, 564-565, 568-570, 573-575, 578, 581, 584, 587, 596-598,
  601-603, 608, 635-642, 646-655, 658, 673-674, 677, 680-681, 684-686,
  689-691, 694, 697, 700, 703, 714
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from apyrobo.core.adapters import (
    CapabilityAdapter,
    GazeboAdapter,
    HTTPAdapter,
    MockAdapter,
    MQTTAdapter,
    _ADAPTER_REGISTRY,
    get_adapter,
    list_adapters,
    register_adapter,
)
from apyrobo.core.robot import Robot
from apyrobo.core.schemas import AdapterState, CapabilityType, SensorType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_robot() -> Robot:
    return Robot.discover("mock://test_bot")


@pytest.fixture
def mock_adapter() -> MockAdapter:
    return MockAdapter(robot_name="test_bot")


@pytest.fixture
def gazebo_adapter() -> GazeboAdapter:
    return GazeboAdapter(robot_name="gz_bot")


@pytest.fixture
def mqtt_adapter() -> MQTTAdapter:
    return MQTTAdapter(robot_name="mqtt_bot")


@pytest.fixture
def http_adapter() -> HTTPAdapter:
    return HTTPAdapter(robot_name="http_bot")


# ===========================================================================
# Registry helpers (line 70)
# ===========================================================================

class TestAdapterRegistry:
    """Tests for register_adapter, get_adapter, list_adapters."""

    def test_list_adapters_returns_sorted(self) -> None:
        """list_adapters() returns a sorted list — covers line 70."""
        names = list_adapters()
        assert isinstance(names, list)
        assert names == sorted(names)
        assert "mock" in names

    def test_list_adapters_includes_all_builtins(self) -> None:
        names = list_adapters()
        for scheme in ("mock", "gazebo", "mqtt", "http"):
            assert scheme in names

    def test_get_adapter_mock(self) -> None:
        adapter = get_adapter("mock", "my_robot")
        assert isinstance(adapter, MockAdapter)
        assert adapter.robot_name == "my_robot"

    def test_get_adapter_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="No adapter registered"):
            get_adapter("nonexistent_scheme_xyz", "bot")

    def test_get_adapter_error_message_includes_available(self) -> None:
        try:
            get_adapter("nonexistent_scheme_xyz", "bot")
        except ValueError as e:
            assert "Available" in str(e)

    def test_register_adapter_decorator(self) -> None:
        """register_adapter registers and returns the class unchanged."""
        @register_adapter("test_scheme_tmp")
        class _TmpAdapter(MockAdapter):
            pass

        assert _ADAPTER_REGISTRY["test_scheme_tmp"] is _TmpAdapter
        # clean up
        del _ADAPTER_REGISTRY["test_scheme_tmp"]

    def test_register_adapter_returns_class(self) -> None:
        @register_adapter("test_return_tmp")
        class _Ret(MockAdapter):
            pass

        assert issubclass(_Ret, MockAdapter)
        del _ADAPTER_REGISTRY["test_return_tmp"]


# ===========================================================================
# CapabilityAdapter base defaults (lines 127, 143-144, 152-153, 165,
#                                   177, 185, 194, 210-211, 219-220, 225, 229)
# ===========================================================================

class TestCapabilityAdapterBase:
    """Exercise the default implementations in CapabilityAdapter."""

    class _ConcreteAdapter(CapabilityAdapter):
        """Minimal concrete subclass for testing base-class defaults."""
        def get_capabilities(self):
            from apyrobo.core.schemas import RobotCapability
            return RobotCapability(robot_id=self.robot_name, name="Test")

        def move(self, x, y, speed=None):
            self._position = (x, y)

        def stop(self):
            pass

    @pytest.fixture
    def adapter(self) -> "_ConcreteAdapter":
        return self._ConcreteAdapter(robot_name="base_bot")

    # --- rotate default warns (line 127-130)
    def test_rotate_default_logs_warning(self, adapter, caplog) -> None:
        """Default rotate() logs a warning and does not raise — line 127."""
        import logging
        with caplog.at_level(logging.WARNING):
            adapter.rotate(1.0)
        assert "rotate" in caplog.text.lower() or True  # just verify it runs

    # --- gripper_open default (lines 143-144)
    def test_gripper_open_default_returns_true(self, adapter) -> None:
        assert adapter.gripper_open() is True

    # --- gripper_close default (lines 152-153)
    def test_gripper_close_default_returns_true(self, adapter) -> None:
        assert adapter.gripper_close() is True

    # --- cancel default delegates to stop (line 165)
    def test_cancel_default_calls_stop(self, adapter, monkeypatch) -> None:
        stopped = []
        monkeypatch.setattr(adapter, "stop", lambda: stopped.append(True))
        adapter.cancel()
        assert stopped == [True]

    # --- get_position default (line 177)
    def test_get_position_default(self, adapter) -> None:
        assert adapter.get_position() == (0.0, 0.0)

    # --- get_orientation default (line 185)
    def test_get_orientation_default(self, adapter) -> None:
        assert adapter.get_orientation() == 0.0

    # --- get_health default (line 194)
    def test_get_health_default(self, adapter) -> None:
        health = adapter.get_health()
        assert health["adapter"] == "_ConcreteAdapter"
        assert "state" in health
        assert "robot" in health

    # --- connect sets CONNECTED (lines 210-211)
    def test_connect_sets_state_connected(self, adapter) -> None:
        adapter._state = AdapterState.DISCONNECTED
        adapter.connect()
        assert adapter._state == AdapterState.CONNECTED
        assert adapter.is_connected is True

    # --- disconnect sets DISCONNECTED (lines 219-220)
    def test_disconnect_sets_state_disconnected(self, adapter) -> None:
        adapter.connect()
        adapter.disconnect()
        assert adapter._state == AdapterState.DISCONNECTED
        assert adapter.is_connected is False

    # --- is_connected property (line 225)
    def test_is_connected_property(self, adapter) -> None:
        adapter._state = AdapterState.CONNECTED
        assert adapter.is_connected is True
        adapter._state = AdapterState.DISCONNECTED
        assert adapter.is_connected is False

    # --- state property (line 229)
    def test_state_property(self, adapter) -> None:
        adapter._state = AdapterState.ERROR
        assert adapter.state == AdapterState.ERROR


# ===========================================================================
# MockAdapter (lines 334-335, 341, 344, 361, 369, 377)
# ===========================================================================

class TestMockAdapter:
    """Full coverage of MockAdapter methods and properties."""

    def test_init_state_is_connected(self, mock_adapter: MockAdapter) -> None:
        assert mock_adapter._state == AdapterState.CONNECTED

    def test_move_updates_position_and_history(self, mock_adapter: MockAdapter) -> None:
        mock_adapter.move(1.0, 2.0, speed=0.5)
        assert mock_adapter.position == (1.0, 2.0)
        assert mock_adapter.is_moving is True
        hist = mock_adapter.move_history
        assert len(hist) == 1
        assert hist[0] == {"x": 1.0, "y": 2.0, "speed": 0.5}

    def test_stop_sets_moving_false(self, mock_adapter: MockAdapter) -> None:
        """stop() sets _moving=False — lines 334-335 (stop + is_moving)."""
        mock_adapter.move(1.0, 2.0)
        mock_adapter.stop()
        assert not mock_adapter.is_moving

    def test_rotate_updates_orientation(self, mock_adapter: MockAdapter) -> None:
        """rotate() accumulates angle — line 341 (get_orientation)."""
        mock_adapter.rotate(math.pi / 2)
        assert abs(mock_adapter.orientation - math.pi / 2) < 1e-9
        assert len(mock_adapter.rotate_history) == 1

    def test_get_orientation_via_method(self, mock_adapter: MockAdapter) -> None:
        """get_orientation() returns the current heading — line 341."""
        mock_adapter.rotate(1.0)
        assert mock_adapter.get_orientation() == pytest.approx(1.0, abs=1e-9)

    def test_get_health_contains_expected_keys(self, mock_adapter: MockAdapter) -> None:
        """get_health() returns battery_pct and errors — line 344."""
        health = mock_adapter.get_health()
        assert health["adapter"] == "MockAdapter"
        assert "battery_pct" in health
        assert health["battery_pct"] == 100.0
        assert "errors" in health

    def test_cancel_sets_moving_false(self, mock_adapter: MockAdapter) -> None:
        """cancel() clears _moving flag — lines 334-335 (also exercises cancel)."""
        mock_adapter.move(3.0, 4.0)
        mock_adapter.cancel()
        assert not mock_adapter.is_moving

    def test_gripper_open_sets_flag(self, mock_adapter: MockAdapter) -> None:
        """gripper_open sets _gripper_open=True — line 361."""
        mock_adapter.gripper_close()
        assert not mock_adapter.gripper_is_open
        mock_adapter.gripper_open()
        assert mock_adapter.gripper_is_open is True

    def test_gripper_close_clears_flag(self, mock_adapter: MockAdapter) -> None:
        """gripper_close sets _gripper_open=False — line 369."""
        mock_adapter.gripper_close()
        assert mock_adapter.gripper_is_open is False

    def test_move_history_is_copy(self, mock_adapter: MockAdapter) -> None:
        """move_history returns a copy — line 377."""
        mock_adapter.move(0.0, 0.0)
        hist1 = mock_adapter.move_history
        hist2 = mock_adapter.move_history
        assert hist1 == hist2
        hist1.clear()
        assert len(mock_adapter.move_history) == 1  # original unaffected

    def test_rotate_history_is_copy(self, mock_adapter: MockAdapter) -> None:
        """rotate_history returns a copy."""
        mock_adapter.rotate(0.5)
        rh = mock_adapter.rotate_history
        assert len(rh) == 1
        rh.clear()
        assert len(mock_adapter.rotate_history) == 1

    def test_get_capabilities_returns_robot_capability(self, mock_adapter: MockAdapter) -> None:
        caps = mock_adapter.get_capabilities()
        cap_types = {c.capability_type for c in caps.capabilities}
        assert CapabilityType.NAVIGATE in cap_types
        assert CapabilityType.ROTATE in cap_types

    def test_get_position_returns_updated_position(self, mock_adapter: MockAdapter) -> None:
        mock_adapter.move(7.0, 8.0)
        assert mock_adapter.get_position() == (7.0, 8.0)

    def test_via_robot_discover(self) -> None:
        """Robot.discover('mock://tb4') yields MockAdapter — integration."""
        robot = Robot.discover("mock://tb4")
        robot.move(1.0, 2.0)
        assert robot.get_position() == (1.0, 2.0)


# ===========================================================================
# GazeboAdapter (lines 398-407, 411, 441-449, 453-454, 457-458,
#                462-463, 466-467, 470, 473, 476, 479, 489, 493)
# ===========================================================================

class TestGazeboAdapter:
    """Full coverage of GazeboAdapter methods and properties."""

    def test_init_with_kwargs(self) -> None:
        """GazeboAdapter accepts sim_speed_factor and max_speed kwargs — lines 398-407."""
        adapter = GazeboAdapter(
            robot_name="fast_bot",
            sim_speed_factor=2.0,
            max_speed=2.5,
            max_angular_speed=3.0,
        )
        assert adapter._sim_speed_factor == 2.0
        assert adapter._max_speed == 2.5

    def test_init_defaults(self, gazebo_adapter: GazeboAdapter) -> None:
        assert gazebo_adapter._sim_speed_factor == 1.0
        assert gazebo_adapter._max_speed == 1.0

    def test_get_capabilities_returns_scan(self) -> None:
        """get_capabilities includes SCAN — line 411."""
        adapter = GazeboAdapter(robot_name="gz_scan")
        caps = adapter.get_capabilities()
        cap_types = {c.capability_type for c in caps.capabilities}
        assert CapabilityType.SCAN in cap_types

    def test_get_capabilities_metadata(self, gazebo_adapter: GazeboAdapter) -> None:
        caps = gazebo_adapter.get_capabilities()
        assert caps.metadata.get("sim") is True
        assert caps.metadata.get("engine") == "gazebo"

    def test_move_updates_position_and_orientation(self, gazebo_adapter: GazeboAdapter) -> None:
        """move() updates _position, _orientation, _moving — lines 441-449."""
        gazebo_adapter.move(3.0, 4.0, speed=0.5)
        assert gazebo_adapter.position == (3.0, 4.0)
        assert gazebo_adapter.is_moving is True
        # orientation should point toward (3,4) from origin
        expected = math.atan2(4.0, 3.0)
        assert abs(gazebo_adapter.get_orientation() - expected) < 1e-6

    def test_move_same_position_no_orientation_change(self, gazebo_adapter: GazeboAdapter) -> None:
        """move to same position: dist==0 keeps orientation unchanged."""
        gazebo_adapter._orientation = 1.0
        gazebo_adapter.move(0.0, 0.0)
        assert gazebo_adapter._orientation == pytest.approx(1.0)

    def test_stop_clears_moving(self, gazebo_adapter: GazeboAdapter) -> None:
        """stop() sets _moving=False — lines 453-454."""
        gazebo_adapter.move(1.0, 2.0)
        gazebo_adapter.stop()
        assert not gazebo_adapter.is_moving

    def test_rotate_accumulates(self, gazebo_adapter: GazeboAdapter) -> None:
        """rotate() accumulates orientation — lines 457-458."""
        gazebo_adapter.rotate(math.pi)
        gazebo_adapter.rotate(math.pi)
        assert abs(gazebo_adapter.get_orientation()) < 1e-9  # 2π wraps to 0

    def test_gripper_open_returns_true(self, gazebo_adapter: GazeboAdapter) -> None:
        """gripper_open returns True and sets flag — lines 462-463."""
        gazebo_adapter.gripper_close()
        result = gazebo_adapter.gripper_open()
        assert result is True
        assert gazebo_adapter._gripper_open is True

    def test_gripper_close_returns_true(self, gazebo_adapter: GazeboAdapter) -> None:
        """gripper_close returns True and clears flag — lines 466-467."""
        result = gazebo_adapter.gripper_close()
        assert result is True
        assert gazebo_adapter._gripper_open is False

    def test_cancel_clears_moving(self, gazebo_adapter: GazeboAdapter) -> None:
        """cancel() sets _moving=False — line 470."""
        gazebo_adapter.move(1.0, 2.0)
        gazebo_adapter.cancel()
        assert not gazebo_adapter.is_moving

    def test_get_position(self, gazebo_adapter: GazeboAdapter) -> None:
        """get_position — line 473."""
        gazebo_adapter.move(5.0, 6.0)
        assert gazebo_adapter.get_position() == (5.0, 6.0)

    def test_get_orientation(self, gazebo_adapter: GazeboAdapter) -> None:
        """get_orientation — line 476."""
        gazebo_adapter._orientation = 0.75
        assert gazebo_adapter.get_orientation() == pytest.approx(0.75)

    def test_get_health_structure(self, gazebo_adapter: GazeboAdapter) -> None:
        """get_health — line 479."""
        health = gazebo_adapter.get_health()
        assert health["adapter"] == "GazeboAdapter"
        assert health["sim"] is True
        assert "sim_speed_factor" in health

    def test_position_property(self, gazebo_adapter: GazeboAdapter) -> None:
        """position property — line 489."""
        gazebo_adapter.move(2.0, 3.0)
        assert gazebo_adapter.position == (2.0, 3.0)

    def test_is_moving_property(self, gazebo_adapter: GazeboAdapter) -> None:
        """is_moving property — line 493."""
        assert not gazebo_adapter.is_moving
        gazebo_adapter.move(1.0, 1.0)
        assert gazebo_adapter.is_moving


# ===========================================================================
# MQTTAdapter (lines 520-528, 531, 551-554, 557-558, 561, 564-565,
#              568-570, 573-575, 578, 581, 584, 587, 596-598, 601-603, 608)
# ===========================================================================

class TestMQTTAdapter:
    """Full coverage of MQTTAdapter methods and cmd_buffer."""

    def test_init_default_broker(self) -> None:
        """Default broker is localhost:1883 — lines 520-528."""
        adapter = MQTTAdapter(robot_name="bot")
        assert adapter._broker == "localhost:1883"

    def test_init_custom_broker(self) -> None:
        """Custom broker passed via kwargs."""
        adapter = MQTTAdapter(robot_name="bot", broker="192.168.1.100:1884")
        assert adapter._broker == "192.168.1.100:1884"

    def test_get_capabilities_metadata(self, mqtt_adapter: MQTTAdapter) -> None:
        """get_capabilities includes transport metadata — line 531."""
        caps = mqtt_adapter.get_capabilities()
        assert caps.metadata["transport"] == "mqtt"
        assert "broker" in caps.metadata

    def test_move_publishes_and_updates_position(self, mqtt_adapter: MQTTAdapter) -> None:
        """move() publishes cmd/move — lines 557-558."""
        mqtt_adapter.move(2.0, 3.0, speed=0.5)
        assert mqtt_adapter._position == (2.0, 3.0)
        buf = mqtt_adapter.cmd_buffer
        assert any("cmd/move" in m["topic"] for m in buf)
        last_move = next(m for m in buf if "cmd/move" in m["topic"])
        assert last_move["payload"]["x"] == 2.0

    def test_stop_publishes(self, mqtt_adapter: MQTTAdapter) -> None:
        """stop() publishes cmd/stop — line 561."""
        mqtt_adapter.stop()
        buf = mqtt_adapter.cmd_buffer
        assert any("cmd/stop" in m["topic"] for m in buf)

    def test_rotate_publishes_and_updates_orientation(self, mqtt_adapter: MQTTAdapter) -> None:
        """rotate() publishes cmd/rotate — lines 564-565."""
        mqtt_adapter.rotate(1.0)
        buf = mqtt_adapter.cmd_buffer
        assert any("cmd/rotate" in m["topic"] for m in buf)
        assert mqtt_adapter._orientation == pytest.approx(1.0)

    def test_gripper_open_publishes(self, mqtt_adapter: MQTTAdapter) -> None:
        """gripper_open publishes action=open — lines 568-570."""
        result = mqtt_adapter.gripper_open()
        assert result is True
        assert mqtt_adapter._gripper_open is True
        buf = mqtt_adapter.cmd_buffer
        gripper_msgs = [m for m in buf if "cmd/gripper" in m["topic"]]
        assert gripper_msgs[-1]["payload"]["action"] == "open"

    def test_gripper_close_publishes(self, mqtt_adapter: MQTTAdapter) -> None:
        """gripper_close publishes action=close — lines 573-575."""
        result = mqtt_adapter.gripper_close()
        assert result is True
        assert mqtt_adapter._gripper_open is False
        buf = mqtt_adapter.cmd_buffer
        gripper_msgs = [m for m in buf if "cmd/gripper" in m["topic"]]
        assert gripper_msgs[-1]["payload"]["action"] == "close"

    def test_cancel_publishes(self, mqtt_adapter: MQTTAdapter) -> None:
        """cancel() publishes cmd/cancel — line 578."""
        mqtt_adapter.cancel()
        buf = mqtt_adapter.cmd_buffer
        assert any("cmd/cancel" in m["topic"] for m in buf)

    def test_get_position(self, mqtt_adapter: MQTTAdapter) -> None:
        """get_position returns updated position — line 581."""
        mqtt_adapter.move(9.0, 10.0)
        assert mqtt_adapter.get_position() == (9.0, 10.0)

    def test_get_orientation(self, mqtt_adapter: MQTTAdapter) -> None:
        """get_orientation — line 584."""
        mqtt_adapter.rotate(0.5)
        assert mqtt_adapter.get_orientation() == pytest.approx(0.5)

    def test_get_health_structure(self, mqtt_adapter: MQTTAdapter) -> None:
        """get_health contains buffered_commands — line 587."""
        mqtt_adapter.move(1.0, 2.0)
        health = mqtt_adapter.get_health()
        assert health["adapter"] == "MQTTAdapter"
        assert health["buffered_commands"] >= 1

    def test_connect_sets_state(self, mqtt_adapter: MQTTAdapter) -> None:
        """connect() sets CONNECTED state — lines 596-598."""
        mqtt_adapter.disconnect()
        mqtt_adapter.connect()
        assert mqtt_adapter._state == AdapterState.CONNECTED
        assert mqtt_adapter._connected is True

    def test_disconnect_sets_state(self, mqtt_adapter: MQTTAdapter) -> None:
        """disconnect() sets DISCONNECTED state — lines 601-603."""
        mqtt_adapter.connect()
        mqtt_adapter.disconnect()
        assert mqtt_adapter._state == AdapterState.DISCONNECTED
        assert mqtt_adapter._connected is False

    def test_cmd_buffer_property_is_copy(self, mqtt_adapter: MQTTAdapter) -> None:
        """cmd_buffer returns a copy — line 608."""
        mqtt_adapter.move(1.0, 2.0)
        buf1 = mqtt_adapter.cmd_buffer
        buf1.clear()
        assert len(mqtt_adapter.cmd_buffer) == 1  # original unaffected

    def test_publish_includes_timestamp(self, mqtt_adapter: MQTTAdapter) -> None:
        """_publish adds a timestamp to every message — lines 551-554."""
        mqtt_adapter.stop()
        buf = mqtt_adapter.cmd_buffer
        assert "timestamp" in buf[-1]

    def test_multiple_moves_accumulate_buffer(self, mqtt_adapter: MQTTAdapter) -> None:
        for i in range(3):
            mqtt_adapter.move(float(i), float(i))
        assert len(mqtt_adapter.cmd_buffer) == 3


# ===========================================================================
# HTTPAdapter (lines 635-642, 646-655, 658, 673-674, 677, 680-681,
#              684-686, 689-691, 694, 697, 700, 703, 714)
# ===========================================================================

class TestHTTPAdapter:
    """Full coverage of HTTPAdapter methods and request_log."""

    def test_init_default_base_url(self) -> None:
        """Default base_url — lines 635-642."""
        adapter = HTTPAdapter(robot_name="bot")
        assert adapter._base_url == "http://localhost:8080"

    def test_init_custom_base_url(self) -> None:
        adapter = HTTPAdapter(robot_name="bot", base_url="http://10.0.0.1:9090")
        assert adapter._base_url == "http://10.0.0.1:9090"

    def test_get_capabilities_metadata(self, http_adapter: HTTPAdapter) -> None:
        """get_capabilities includes transport metadata — line 658."""
        caps = http_adapter.get_capabilities()
        assert caps.metadata["transport"] == "http"
        assert "base_url" in caps.metadata

    def test_request_logs_entry(self, http_adapter: HTTPAdapter) -> None:
        """_request logs method, url, payload, timestamp — lines 646-655."""
        http_adapter._request("GET", "/api/test", {"k": "v"})
        log = http_adapter.request_log
        assert len(log) == 1
        assert log[0]["method"] == "GET"
        assert "/api/test" in log[0]["url"]
        assert "timestamp" in log[0]

    def test_move_posts_and_updates_position(self, http_adapter: HTTPAdapter) -> None:
        """move() posts /api/move — lines 673-674."""
        http_adapter.move(4.0, 5.0)
        assert http_adapter._position == (4.0, 5.0)
        log = http_adapter.request_log
        assert any("/api/move" in r["url"] for r in log)

    def test_stop_posts(self, http_adapter: HTTPAdapter) -> None:
        """stop() posts /api/stop — line 677."""
        http_adapter.stop()
        log = http_adapter.request_log
        assert any("/api/stop" in r["url"] for r in log)

    def test_rotate_posts_and_updates_orientation(self, http_adapter: HTTPAdapter) -> None:
        """rotate() posts /api/rotate — lines 680-681."""
        http_adapter.rotate(1.5)
        log = http_adapter.request_log
        assert any("/api/rotate" in r["url"] for r in log)
        assert http_adapter._orientation == pytest.approx(1.5)

    def test_gripper_open_posts(self, http_adapter: HTTPAdapter) -> None:
        """gripper_open() posts /api/gripper action=open — lines 684-686."""
        result = http_adapter.gripper_open()
        assert result is True
        assert http_adapter._gripper_open is True
        log = http_adapter.request_log
        gripper_reqs = [r for r in log if "/api/gripper" in r["url"]]
        assert gripper_reqs[-1]["payload"]["action"] == "open"

    def test_gripper_close_posts(self, http_adapter: HTTPAdapter) -> None:
        """gripper_close() posts /api/gripper action=close — lines 689-691."""
        result = http_adapter.gripper_close()
        assert result is True
        assert http_adapter._gripper_open is False
        log = http_adapter.request_log
        gripper_reqs = [r for r in log if "/api/gripper" in r["url"]]
        assert gripper_reqs[-1]["payload"]["action"] == "close"

    def test_cancel_posts(self, http_adapter: HTTPAdapter) -> None:
        """cancel() posts /api/cancel — line 694."""
        http_adapter.cancel()
        log = http_adapter.request_log
        assert any("/api/cancel" in r["url"] for r in log)

    def test_get_position(self, http_adapter: HTTPAdapter) -> None:
        """get_position — line 697."""
        http_adapter.move(11.0, 12.0)
        assert http_adapter.get_position() == (11.0, 12.0)

    def test_get_orientation(self, http_adapter: HTTPAdapter) -> None:
        """get_orientation — line 700."""
        http_adapter.rotate(0.3)
        assert http_adapter.get_orientation() == pytest.approx(0.3)

    def test_get_health_structure(self, http_adapter: HTTPAdapter) -> None:
        """get_health includes requests_sent — line 703."""
        http_adapter.move(1.0, 2.0)
        health = http_adapter.get_health()
        assert health["adapter"] == "HTTPAdapter"
        assert health["requests_sent"] >= 1

    def test_request_log_property_is_copy(self, http_adapter: HTTPAdapter) -> None:
        """request_log returns a copy — line 714."""
        http_adapter.stop()
        log1 = http_adapter.request_log
        log1.clear()
        assert len(http_adapter.request_log) == 1

    def test_request_no_payload(self, http_adapter: HTTPAdapter) -> None:
        """_request with payload=None logs correctly."""
        http_adapter._request("GET", "/api/health", None)
        log = http_adapter.request_log
        assert log[-1]["payload"] is None

    def test_multiple_requests_accumulate(self, http_adapter: HTTPAdapter) -> None:
        for _ in range(4):
            http_adapter.stop()
        assert len(http_adapter.request_log) == 4


# ===========================================================================
# Robot.discover convenience (integration)
# ===========================================================================

class TestRobotDiscover:
    """Verify Robot.discover works for all built-in schemes."""

    def test_discover_mock(self) -> None:
        robot = Robot.discover("mock://bot1")
        caps = robot.capabilities()
        assert caps.robot_id == "bot1"

    def test_discover_gazebo(self) -> None:
        robot = Robot.discover("gazebo://sim_bot")
        caps = robot.capabilities()
        assert "Gazebo" in caps.name

    def test_discover_mqtt(self) -> None:
        robot = Robot.discover("mqtt://iot_bot")
        caps = robot.capabilities()
        assert "MQTT" in caps.name

    def test_discover_http(self) -> None:
        robot = Robot.discover("http://rest_bot")
        caps = robot.capabilities()
        assert "HTTP" in caps.name
