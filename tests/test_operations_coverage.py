"""
Targeted tests for operations.py — BatteryMonitor, MapManager,
TeleoperationBridge, WebhookEmitter, ScheduledTaskRunner,
OperationsApiServer, FleetDashboard, and helpers.
"""

from __future__ import annotations

import json
import threading
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from apyrobo.operations import (
    BatteryMonitor,
    MapManager,
    TeleoperationBridge,
    WebhookEmitter,
    WebhookTarget,
)


# ---------------------------------------------------------------------------
# BatteryMonitor
# ---------------------------------------------------------------------------

class TestBatteryMonitor:
    def test_initial_state(self) -> None:
        bm = BatteryMonitor("tb4")
        assert bm.percentage == 100.0
        assert bm.status == "ok"
        assert bm.is_charging is False

    def test_update_percentage(self) -> None:
        bm = BatteryMonitor("tb4")
        bm.update(percentage=50.0)
        assert bm.percentage == 50.0

    def test_update_voltage(self) -> None:
        bm = BatteryMonitor("tb4")
        bm.update(voltage=12.5)
        assert bm.voltage == pytest.approx(12.5)

    def test_update_clamps_above_100(self) -> None:
        bm = BatteryMonitor("tb4")
        bm.update(percentage=150.0)
        assert bm.percentage == 100.0

    def test_update_clamps_below_0(self) -> None:
        bm = BatteryMonitor("tb4")
        bm.update(percentage=-5.0)
        assert bm.percentage == 0.0

    def test_status_low(self) -> None:
        bm = BatteryMonitor("tb4", low_threshold=20.0)
        bm.update(percentage=15.0)
        assert bm.status == "low"

    def test_status_critical(self) -> None:
        bm = BatteryMonitor("tb4", critical_threshold=10.0)
        bm.update(percentage=5.0)
        assert bm.status == "critical"

    def test_status_charging(self) -> None:
        bm = BatteryMonitor("tb4")
        bm.update(percentage=5.0, is_charging=True)
        assert bm.status == "charging"

    def test_estimated_range(self) -> None:
        bm = BatteryMonitor("tb4", meters_per_percent=2.0)
        bm.update(percentage=50.0)
        assert bm.estimated_range_m == pytest.approx(100.0)

    def test_can_complete_trip_ok(self) -> None:
        bm = BatteryMonitor("tb4", meters_per_percent=3.0)
        bm.update(percentage=100.0)  # 300m range
        assert bm.can_complete_trip(distance_m=50.0) is True

    def test_can_complete_trip_too_far(self) -> None:
        bm = BatteryMonitor("tb4", meters_per_percent=1.0)
        bm.update(percentage=5.0)  # 5m range
        assert bm.can_complete_trip(distance_m=50.0) is False

    def test_can_complete_trip_with_position(self) -> None:
        bm = BatteryMonitor("tb4", dock_position=(0.0, 0.0), meters_per_percent=5.0)
        bm.update(percentage=100.0)  # 500m range
        assert bm.can_complete_trip(distance_m=20.0, robot_position=(10.0, 0.0)) is True

    def test_on_threshold_callback_low(self) -> None:
        events = []
        bm = BatteryMonitor("tb4", low_threshold=20.0)
        bm.on_threshold(lambda lvl, pct: events.append(lvl))
        bm.update(percentage=15.0)
        assert "low" in events

    def test_on_threshold_callback_critical(self) -> None:
        events = []
        bm = BatteryMonitor("tb4", critical_threshold=10.0)
        bm.on_threshold(lambda lvl, pct: events.append(lvl))
        bm.update(percentage=5.0)
        assert "critical" in events

    def test_return_to_dock_not_triggered_when_ok(self) -> None:
        bm = BatteryMonitor("tb4")
        called = []
        bm.set_return_to_dock_callback(lambda: called.append(True))
        result = bm.evaluate_return_to_dock()
        assert result is False
        assert called == []

    def test_return_to_dock_triggered_when_critical(self) -> None:
        bm = BatteryMonitor("tb4", critical_threshold=10.0)
        bm.update(percentage=5.0)
        called = []
        bm.set_return_to_dock_callback(lambda: called.append(True))
        result = bm.evaluate_return_to_dock()
        assert result is True
        assert len(called) == 1

    def test_to_dict(self) -> None:
        bm = BatteryMonitor("tb4")
        d = bm.to_dict()
        assert d["robot_id"] == "tb4"
        assert "percentage" in d
        assert "status" in d

    def test_repr(self) -> None:
        bm = BatteryMonitor("tb4")
        assert "tb4" in repr(bm)

    def test_attach_ros2_no_ros(self) -> None:
        bm = BatteryMonitor("tb4")
        node = MagicMock()
        result = bm.attach_ros2(node)
        # Either returns False (no ROS) or True — just shouldn't crash
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# MapManager
# ---------------------------------------------------------------------------

class TestMapManager:
    def test_empty_maps(self) -> None:
        mm = MapManager()
        assert mm.available_maps == []
        assert mm.active_map is None

    def test_register_and_set_active(self) -> None:
        mm = MapManager()
        mm.register("floor1", "/maps/floor1.yaml", floor=0)
        mm.set_active("floor1")
        assert mm.active_map_name == "floor1"

    def test_active_map_returns_dict(self) -> None:
        mm = MapManager()
        mm.register("w1", "/maps/w1.yaml", floor=1, metadata={"desc": "warehouse"})
        mm.set_active("w1")
        m = mm.active_map
        assert m is not None
        assert m["floor"] == 1

    def test_set_active_unknown_raises(self) -> None:
        mm = MapManager()
        with pytest.raises(ValueError):
            mm.set_active("nonexistent")

    def test_get_floor_map(self) -> None:
        mm = MapManager()
        mm.register("f0", "/maps/f0.yaml", floor=0)
        mm.register("f1", "/maps/f1.yaml", floor=1)
        m = mm.get_floor_map(1)
        assert m is not None
        assert m["name"] == "f1"

    def test_get_floor_map_not_found(self) -> None:
        mm = MapManager()
        assert mm.get_floor_map(99) is None

    def test_available_maps_list(self) -> None:
        mm = MapManager()
        mm.register("a", "/a.yaml")
        mm.register("b", "/b.yaml")
        maps = mm.available_maps
        assert "a" in maps
        assert "b" in maps

    def test_discover_maps_from_dir(self, tmp_path: Path) -> None:
        # Create some yaml files
        (tmp_path / "warehouse.yaml").write_text("map: data")
        (tmp_path / "office.yaml").write_text("map: data")
        mm = MapManager(maps_dir=tmp_path)
        assert "warehouse" in mm.available_maps
        assert "office" in mm.available_maps

    def test_repr(self) -> None:
        mm = MapManager()
        assert "MapManager" in repr(mm)

    def test_load_map_ros2_no_ros2(self) -> None:
        mm = MapManager()
        # ros2 not installed — should not crash
        result = mm.load_map_ros2("/maps/test.yaml")
        assert isinstance(result, bool)

    def test_save_map_ros2_no_ros2(self, tmp_path: Path) -> None:
        mm = MapManager()
        result = mm.save_map_ros2("test_map", str(tmp_path))
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# TeleoperationBridge
# ---------------------------------------------------------------------------

class TestTeleoperationBridge:
    def test_initial_state(self) -> None:
        teleop = TeleoperationBridge("tb4")
        assert teleop.is_active is False
        assert teleop.operator is None
        assert teleop.command_count == 0

    def test_enable(self) -> None:
        teleop = TeleoperationBridge("tb4")
        teleop.enable("alice")
        assert teleop.is_active is True
        assert teleop.operator == "alice"

    def test_disable(self) -> None:
        teleop = TeleoperationBridge("tb4")
        teleop.enable("alice")
        teleop.disable()
        assert teleop.is_active is False
        assert teleop.operator is None

    def test_send_velocity_not_enabled(self) -> None:
        teleop = TeleoperationBridge("tb4")
        result = teleop.send_velocity(0.3, 0.0)
        assert result is False

    def test_send_velocity_no_callback(self) -> None:
        teleop = TeleoperationBridge("tb4")
        teleop.enable()
        result = teleop.send_velocity(0.3, 0.0)
        assert result is False

    def test_send_velocity_with_callback(self) -> None:
        calls = []
        teleop = TeleoperationBridge("tb4")
        teleop.set_velocity_callback(lambda lin, ang: calls.append((lin, ang)))
        teleop.enable()
        result = teleop.send_velocity(0.5, 0.1)
        assert result is True
        assert calls == [(0.5, 0.1)]
        assert teleop.command_count == 1

    def test_disable_sends_zero_velocity(self) -> None:
        calls = []
        teleop = TeleoperationBridge("tb4")
        teleop.set_velocity_callback(lambda lin, ang: calls.append((lin, ang)))
        teleop.enable()
        teleop.send_velocity(0.3, 0.0)
        teleop.disable()
        # Last call should be (0.0, 0.0) to stop the robot
        assert calls[-1] == (0.0, 0.0)

    def test_to_dict(self) -> None:
        teleop = TeleoperationBridge("tb4")
        d = teleop.to_dict()
        assert d["robot_id"] == "tb4"
        assert "mode" in d

    def test_repr(self) -> None:
        teleop = TeleoperationBridge("tb4")
        assert "tb4" in repr(teleop)


# ---------------------------------------------------------------------------
# WebhookEmitter
# ---------------------------------------------------------------------------

class TestWebhookEmitter:
    def test_add_and_emit_callback(self) -> None:
        events = []
        we = WebhookEmitter()
        we.add_callback("logger", lambda e: events.append(e["event_type"]))
        we.emit("task_completed", task_id="t1")
        assert "task_completed" in events

    def test_remove_target(self) -> None:
        events = []
        we = WebhookEmitter()
        we.add_callback("logger", lambda e: events.append(e))
        we.remove_target("logger")
        we.emit("test_event")
        assert events == []

    def test_emit_multiple_events(self) -> None:
        events = []
        we = WebhookEmitter()
        we.add_callback("cb", lambda e: events.append(e["event_type"]))
        we.emit("event_a")
        we.emit("event_b")
        assert events == ["event_a", "event_b"]


class TestWebhookTarget:
    def test_should_receive_all_events(self) -> None:
        wt = WebhookTarget("all", "http://example.com")
        assert wt.should_receive("anything") is True

    def test_should_receive_filtered(self) -> None:
        wt = WebhookTarget("filtered", "http://example.com", events=["task_completed"])
        assert wt.should_receive("task_completed") is True
        assert wt.should_receive("battery_low") is False

    def test_should_receive_disabled(self) -> None:
        wt = WebhookTarget("disabled", "http://example.com")
        wt.enabled = False
        assert wt.should_receive("anything") is False

    def test_repr(self) -> None:
        wt = WebhookTarget("test", "http://example.com")
        assert "test" in repr(wt)


# ===========================================================================
# NEW test classes — extending coverage for missing lines
# ===========================================================================

# ---------------------------------------------------------------------------
# BatteryMonitor — extended coverage
# ---------------------------------------------------------------------------

class TestBatteryMonitorExtended:
    """Cover lines 144-145, 165-177, 245, 256, 285 and history/edge cases."""

    def test_attach_ros2_with_mock_rclpy(self) -> None:
        """Lines 144-145, 165-177: ROS2 subscriber setup via mocked rclpy."""
        import apyrobo.operations as ops_module

        fake_rclpy = MagicMock()
        fake_qos_profile = MagicMock()
        fake_battery_state_cls = MagicMock()
        fake_reliability_policy = MagicMock()
        fake_durability_policy = MagicMock()

        node = MagicMock()
        sub = MagicMock()
        node.create_subscription.return_value = sub

        original_has_ros2 = ops_module._HAS_ROS2

        try:
            # Inject mock ROS2 symbols into the module
            ops_module._HAS_ROS2 = True
            ops_module.BatteryState = fake_battery_state_cls  # type: ignore[attr-defined]
            ops_module.QoSProfile = fake_qos_profile  # type: ignore[attr-defined]
            ops_module.ReliabilityPolicy = fake_reliability_policy  # type: ignore[attr-defined]
            ops_module.DurabilityPolicy = fake_durability_policy  # type: ignore[attr-defined]
            fake_reliability_policy.BEST_EFFORT = "best_effort_value"
            fake_reliability_policy.RELIABLE = "reliable_value"
            fake_durability_policy.VOLATILE = "volatile_value"

            bm = BatteryMonitor("tb4")
            result = bm.attach_ros2(node, topic="/battery_state")

            assert result is True
            node.create_subscription.assert_called_once()
            assert bm._ros_node is node
        finally:
            ops_module._HAS_ROS2 = original_has_ros2
            # Clean up injected attributes
            for attr in ("BatteryState", "QoSProfile", "ReliabilityPolicy", "DurabilityPolicy"):
                if hasattr(ops_module, attr):
                    try:
                        delattr(ops_module, attr)
                    except AttributeError:
                        pass

    def test_attach_ros2_reliable_qos(self) -> None:
        """Covers the reliability='reliable' branch in attach_ros2."""
        import apyrobo.operations as ops_module

        node = MagicMock()
        node.create_subscription.return_value = MagicMock()

        original_has_ros2 = ops_module._HAS_ROS2
        try:
            ops_module._HAS_ROS2 = True
            ops_module.BatteryState = MagicMock()  # type: ignore[attr-defined]

            fake_rel = MagicMock()
            fake_rel.RELIABLE = "rel_value"
            fake_rel.BEST_EFFORT = "be_value"
            ops_module.ReliabilityPolicy = fake_rel  # type: ignore[attr-defined]

            fake_dur = MagicMock()
            fake_dur.VOLATILE = "vol_value"
            ops_module.DurabilityPolicy = fake_dur  # type: ignore[attr-defined]
            ops_module.QoSProfile = MagicMock()  # type: ignore[attr-defined]

            bm = BatteryMonitor("tb4")
            result = bm.attach_ros2(node, reliability="reliable")
            assert result is True
        finally:
            ops_module._HAS_ROS2 = original_has_ros2

    def test_ros2_battery_callback_fires_update(self) -> None:
        """Lines 168-173: internal _cb processes a ROS2 BatteryState message."""
        import apyrobo.operations as ops_module

        node = MagicMock()
        captured_cb = {}

        def fake_create_subscription(msg_type, topic, callback, qos):
            captured_cb["fn"] = callback
            return MagicMock()

        node.create_subscription.side_effect = fake_create_subscription

        original_has_ros2 = ops_module._HAS_ROS2
        try:
            ops_module._HAS_ROS2 = True
            ops_module.BatteryState = MagicMock()  # type: ignore[attr-defined]
            ops_module.QoSProfile = MagicMock()  # type: ignore[attr-defined]

            fake_rel = MagicMock()
            fake_rel.BEST_EFFORT = "be"
            fake_rel.RELIABLE = "rel"
            ops_module.ReliabilityPolicy = fake_rel  # type: ignore[attr-defined]

            fake_dur = MagicMock()
            fake_dur.VOLATILE = "vol"
            ops_module.DurabilityPolicy = fake_dur  # type: ignore[attr-defined]

            bm = BatteryMonitor("tb4", critical_threshold=10.0)
            dock_called = []
            bm.set_return_to_dock_callback(lambda: dock_called.append(True))

            bm.attach_ros2(node)

            # Simulate a BatteryState message with percentage=0.5 (i.e. 50% in 0-1 scale)
            msg = MagicMock()
            msg.percentage = 0.05   # 5% — should trigger critical
            msg.voltage = 11.8
            msg.power_supply_status = 0  # not charging
            captured_cb["fn"](msg)

            # percentage should be scaled: 0.05 * 100 = 5%
            assert bm.percentage == pytest.approx(5.0)
            assert bm.status == "critical"
        finally:
            ops_module._HAS_ROS2 = original_has_ros2

    def test_ros2_battery_callback_already_percent_scale(self) -> None:
        """Covers branch where percentage > 1.0 (already in percent scale, no scaling)."""
        import apyrobo.operations as ops_module

        node = MagicMock()
        captured_cb = {}

        def fake_create_subscription(msg_type, topic, callback, qos):
            captured_cb["fn"] = callback
            return MagicMock()

        node.create_subscription.side_effect = fake_create_subscription

        original_has_ros2 = ops_module._HAS_ROS2
        try:
            ops_module._HAS_ROS2 = True
            ops_module.BatteryState = MagicMock()  # type: ignore[attr-defined]
            ops_module.QoSProfile = MagicMock()  # type: ignore[attr-defined]

            fake_rel = MagicMock()
            fake_rel.BEST_EFFORT = "be"
            fake_rel.RELIABLE = "rel"
            ops_module.ReliabilityPolicy = fake_rel  # type: ignore[attr-defined]

            fake_dur = MagicMock()
            fake_dur.VOLATILE = "vol"
            ops_module.DurabilityPolicy = fake_dur  # type: ignore[attr-defined]

            bm = BatteryMonitor("tb4")
            bm.attach_ros2(node)

            msg = MagicMock()
            msg.percentage = 75.0   # already in percent scale, multiplied by 1.0
            msg.voltage = 12.5
            msg.power_supply_status = 1  # charging
            captured_cb["fn"](msg)

            assert bm.percentage == pytest.approx(75.0)
            assert bm.status == "charging"
        finally:
            ops_module._HAS_ROS2 = original_has_ros2

    def test_can_complete_trip_without_position_uses_distance_as_dock_dist(self) -> None:
        """Line 245: dock_dist defaults to distance_m when no robot_position given."""
        # With 50m distance and meters_per_percent=2, percentage=100 → range=200m
        # total_needed = 50 + 50 = 100, with 20% margin = 120m required; 200 >= 120 → True
        bm = BatteryMonitor("tb4", meters_per_percent=2.0)
        bm.update(percentage=100.0)
        assert bm.can_complete_trip(distance_m=50.0) is True

    def test_can_complete_trip_without_position_fails_when_not_enough(self) -> None:
        """Line 245 (False branch): not enough range without position."""
        bm = BatteryMonitor("tb4", meters_per_percent=0.5)
        bm.update(percentage=10.0)  # range = 5m
        assert bm.can_complete_trip(distance_m=50.0) is False

    def test_evaluate_return_to_dock_critical_no_callback(self) -> None:
        """Line 256: critical battery but no callback set — returns False."""
        bm = BatteryMonitor("tb4", critical_threshold=10.0)
        bm.update(percentage=5.0)
        # No callback registered — should return False
        result = bm.evaluate_return_to_dock()
        assert result is False

    def test_evaluate_return_to_dock_when_charging_not_triggered(self) -> None:
        """Critical AND charging — should NOT trigger dock return."""
        bm = BatteryMonitor("tb4", critical_threshold=10.0)
        bm.update(percentage=5.0, is_charging=True)
        called = []
        bm.set_return_to_dock_callback(lambda: called.append(True))
        result = bm.evaluate_return_to_dock()
        assert result is False
        assert called == []

    def test_callback_exception_is_swallowed(self) -> None:
        """Lines 142-145: _notify swallows callback exceptions."""
        def bad_cb(level: str, pct: float) -> None:
            raise RuntimeError("callback error")

        bm = BatteryMonitor("tb4", low_threshold=20.0)
        bm.on_threshold(bad_cb)
        # Should not raise
        bm.update(percentage=15.0)

    def test_multiple_callbacks_all_called(self) -> None:
        """All registered threshold callbacks are invoked."""
        results: list[str] = []
        bm = BatteryMonitor("tb4", low_threshold=20.0)
        bm.on_threshold(lambda lvl, pct: results.append("cb1"))
        bm.on_threshold(lambda lvl, pct: results.append("cb2"))
        bm.update(percentage=15.0)
        assert results == ["cb1", "cb2"]

    def test_to_dict_with_voltage(self) -> None:
        bm = BatteryMonitor("tb4", dock_position=(1.0, 2.0))
        bm.update(percentage=60.0, voltage=12.3)
        d = bm.to_dict()
        assert d["voltage"] == pytest.approx(12.3)
        assert d["dock_position"] == [1.0, 2.0]
        assert d["estimated_range_m"] == pytest.approx(60.0 * 2.0)

    def test_update_is_charging_none_unchanged(self) -> None:
        """Passing is_charging=None should not alter the current value."""
        bm = BatteryMonitor("tb4")
        bm.update(is_charging=True)
        bm.update(is_charging=None)
        assert bm.is_charging is True

    def test_attach_ros2_returns_false_when_no_ros2(self) -> None:
        import apyrobo.operations as ops_module
        original = ops_module._HAS_ROS2
        try:
            ops_module._HAS_ROS2 = False
            bm = BatteryMonitor("tb4")
            result = bm.attach_ros2(MagicMock())
            assert result is False
        finally:
            ops_module._HAS_ROS2 = original


# ---------------------------------------------------------------------------
# MapManager — extended coverage
# ---------------------------------------------------------------------------

class TestMapManagerExtended:
    """Cover lines related to load_map_ros2, save_map_ros2, discover, metadata."""

    def test_register_stores_metadata(self) -> None:
        mm = MapManager()
        mm.register("lab", "/maps/lab.yaml", floor=2, metadata={"zone": "restricted"})
        m = mm._maps["lab"]
        assert m["metadata"]["zone"] == "restricted"
        assert m["floor"] == 2

    def test_register_default_metadata_is_empty_dict(self) -> None:
        mm = MapManager()
        mm.register("plain", "/maps/plain.yaml")
        assert mm._maps["plain"]["metadata"] == {}

    def test_active_map_none_when_no_map_set(self) -> None:
        mm = MapManager()
        assert mm.active_map is None
        assert mm.active_map_name is None

    def test_active_map_name_after_set_active(self) -> None:
        mm = MapManager()
        mm.register("floor0", "/maps/f0.yaml")
        mm.set_active("floor0")
        assert mm.active_map_name == "floor0"

    def test_set_active_raises_with_helpful_message(self) -> None:
        mm = MapManager()
        mm.register("a", "/a.yaml")
        with pytest.raises(ValueError, match="nonexistent"):
            mm.set_active("nonexistent")

    def test_get_floor_map_returns_first_match(self) -> None:
        mm = MapManager()
        mm.register("g1", "/g1.yaml", floor=3)
        mm.register("g2", "/g2.yaml", floor=3)
        m = mm.get_floor_map(3)
        # Returns one of the two floor-3 maps
        assert m is not None
        assert m["floor"] == 3

    def test_discover_maps_skips_nonexistent_dir(self) -> None:
        """MapManager with a non-existent maps_dir doesn't raise."""
        mm = MapManager(maps_dir="/does/not/exist")
        assert mm.available_maps == []

    def test_discover_maps_ignores_non_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "map1.yaml").write_text("map: x")
        (tmp_path / "readme.txt").write_text("ignore me")
        mm = MapManager(maps_dir=tmp_path)
        assert "map1" in mm.available_maps
        assert "readme" not in mm.available_maps

    def test_discover_maps_does_not_overwrite_registered(self, tmp_path: Path) -> None:
        """Auto-discovered maps don't clobber manually registered ones."""
        (tmp_path / "special.yaml").write_text("map: data")
        mm = MapManager()
        mm.register("special", "/custom/special.yaml", floor=5)
        mm._maps_dir = tmp_path
        mm._discover_maps()
        # The manually registered entry should still be present
        assert mm._maps["special"]["yaml_path"] == "/custom/special.yaml"

    def test_load_map_ros2_calls_subprocess(self) -> None:
        """load_map_ros2 issues subprocess.run and returns True on success."""
        mm = MapManager()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = mm.load_map_ros2("/maps/test.yaml")
        assert result is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "map_server/load_map" in " ".join(args)

    def test_load_map_ros2_returns_false_on_exception(self) -> None:
        mm = MapManager()
        with patch("subprocess.run", side_effect=FileNotFoundError("ros2 not found")):
            result = mm.load_map_ros2("/maps/test.yaml")
        assert result is False

    def test_save_map_ros2_calls_subprocess(self, tmp_path: Path) -> None:
        mm = MapManager()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = mm.save_map_ros2("my_map", str(tmp_path))
        assert result is True
        mock_run.assert_called_once()

    def test_save_map_ros2_without_output_dir(self) -> None:
        mm = MapManager()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = mm.save_map_ros2("my_map")
        assert result is True

    def test_save_map_ros2_returns_false_on_exception(self) -> None:
        mm = MapManager()
        with patch("subprocess.run", side_effect=OSError("no ros2")):
            result = mm.save_map_ros2("my_map", "/tmp")
        assert result is False

    def test_repr_shows_count_and_active(self) -> None:
        mm = MapManager()
        mm.register("f0", "/f0.yaml")
        mm.register("f1", "/f1.yaml")
        mm.set_active("f0")
        r = repr(mm)
        assert "2" in r
        assert "f0" in r

    def test_available_maps_is_sorted_or_ordered(self) -> None:
        mm = MapManager()
        mm.register("z_map", "/z.yaml")
        mm.register("a_map", "/a.yaml")
        maps = mm.available_maps
        assert set(maps) == {"z_map", "a_map"}


# ---------------------------------------------------------------------------
# TeleoperationBridge — extended coverage
# ---------------------------------------------------------------------------

class TestTeleoperationBridgeExtended:
    """Cover lines 339-351 (ROS2 publisher), command log growth, mode transitions."""

    def test_mode_starts_autonomous(self) -> None:
        from apyrobo.operations import TeleoperationMode
        teleop = TeleoperationBridge("spot")
        assert teleop.mode == TeleoperationMode.AUTONOMOUS

    def test_mode_switches_to_teleop_on_enable(self) -> None:
        from apyrobo.operations import TeleoperationMode
        teleop = TeleoperationBridge("spot")
        teleop.enable("bob")
        assert teleop.mode == TeleoperationMode.TELEOP

    def test_mode_returns_to_autonomous_on_disable(self) -> None:
        from apyrobo.operations import TeleoperationMode
        teleop = TeleoperationBridge("spot")
        teleop.enable("bob")
        teleop.disable()
        assert teleop.mode == TeleoperationMode.AUTONOMOUS

    def test_enable_default_operator(self) -> None:
        teleop = TeleoperationBridge("tb4")
        teleop.enable()  # no operator_id
        assert teleop.operator == "operator"

    def test_command_log_grows_with_each_send(self) -> None:
        teleop = TeleoperationBridge("tb4")
        teleop.set_velocity_callback(lambda lin, ang: None)
        teleop.enable()
        teleop.send_velocity(0.1, 0.0)
        teleop.send_velocity(0.2, 0.1)
        teleop.send_velocity(0.0, -0.1)
        assert teleop.command_count == 3

    def test_command_log_includes_operator(self) -> None:
        teleop = TeleoperationBridge("tb4")
        teleop.set_velocity_callback(lambda lin, ang: None)
        teleop.enable("charlie")
        teleop.send_velocity(0.3, 0.0)
        assert teleop._command_log[0]["operator"] == "charlie"

    def test_disable_without_callback_does_not_raise(self) -> None:
        teleop = TeleoperationBridge("tb4")
        teleop.enable("dave")
        # No callback set — disable should still work
        teleop.disable()
        assert teleop.is_active is False

    def test_attach_ros2_publisher_no_ros2(self) -> None:
        """Lines 339-341: returns False when ROS2 unavailable."""
        import apyrobo.operations as ops_module
        original = ops_module._HAS_ROS2
        try:
            ops_module._HAS_ROS2 = False
            teleop = TeleoperationBridge("tb4")
            result = teleop.attach_ros2_publisher(MagicMock())
            assert result is False
        finally:
            ops_module._HAS_ROS2 = original

    def test_attach_ros2_publisher_with_mock_ros2(self) -> None:
        """Lines 342-351: ROS2 publisher setup with mocked rclpy."""
        import apyrobo.operations as ops_module

        original = ops_module._HAS_ROS2
        try:
            ops_module._HAS_ROS2 = True

            # Mock the Twist class
            fake_twist_instance = MagicMock()
            fake_twist_instance.linear = MagicMock()
            fake_twist_instance.angular = MagicMock()

            fake_twist_cls = MagicMock(return_value=fake_twist_instance)
            ops_module.Twist = fake_twist_cls  # type: ignore[attr-defined]

            node = MagicMock()
            pub = MagicMock()
            node.create_publisher.return_value = pub

            teleop = TeleoperationBridge("tb4")
            result = teleop.attach_ros2_publisher(node, topic="/cmd_vel")

            assert result is True
            node.create_publisher.assert_called_once()

            # Verify the callback publishes a Twist message
            teleop._velocity_callback(0.5, 0.2)  # type: ignore[misc]
            pub.publish.assert_called_once()
        finally:
            ops_module._HAS_ROS2 = original

    def test_to_dict_reflects_active_state(self) -> None:
        teleop = TeleoperationBridge("tb4")
        teleop.set_velocity_callback(lambda lin, ang: None)
        teleop.enable("eve")
        teleop.send_velocity(0.1, 0.0)
        d = teleop.to_dict()
        assert d["is_active"] is True
        assert d["operator"] == "eve"
        assert d["commands_sent"] == 1

    def test_thread_safety_concurrent_enable_disable(self) -> None:
        """TeleoperationBridge uses a lock — concurrent enable/disable should not crash."""
        teleop = TeleoperationBridge("tb4")
        errors = []

        def toggle() -> None:
            try:
                for _ in range(20):
                    teleop.enable("worker")
                    teleop.disable()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=toggle) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


# ---------------------------------------------------------------------------
# WebhookEmitter — extended HTTP / retry coverage
# ---------------------------------------------------------------------------

class TestWebhookEmitterExtended:
    """Cover lines 510-516, 522-544, 548-560, 565, 570, 574, 578, 581."""

    def test_add_target_registers_webhook_target(self) -> None:
        we = WebhookEmitter()
        we.add_target("my_hook", "http://example.com/hook", events=["task_done"])
        assert "my_hook" in we._targets
        target = we._targets["my_hook"]
        assert target.url == "http://example.com/hook"
        assert "task_done" in target.events  # type: ignore[operator]

    def test_target_count_includes_both_targets_and_callbacks(self) -> None:
        we = WebhookEmitter()
        we.add_target("t1", "http://a.com")
        we.add_callback("cb1", lambda e: None)
        assert we.target_count == 2

    def test_event_log_grows_on_emit(self) -> None:
        we = WebhookEmitter()
        we.emit("event_a")
        we.emit("event_b")
        log = we.event_log
        assert len(log) == 2
        assert log[0]["event_type"] == "event_a"
        assert log[1]["event_type"] == "event_b"

    def test_event_log_is_a_copy(self) -> None:
        """event_log property returns a copy, not the internal list."""
        we = WebhookEmitter()
        we.emit("x")
        log = we.event_log
        log.clear()
        assert len(we.event_log) == 1

    def test_emit_callback_exception_does_not_stop_others(self) -> None:
        """Lines 508-511: exceptions in callbacks are swallowed."""
        results = []

        def bad_cb(event: dict) -> None:
            raise ValueError("bad")

        def good_cb(event: dict) -> None:
            results.append(event["event_type"])

        we = WebhookEmitter()
        we.add_callback("bad", bad_cb)
        we.add_callback("good", good_cb)
        we.emit("test_event")
        assert "test_event" in results

    def test_send_http_success_on_first_attempt(self) -> None:
        """Lines 522-533: successful HTTP POST on first attempt."""
        we = WebhookEmitter(retry_count=3, retry_backoff_s=0.0)
        target = WebhookTarget("test", "http://example.com/hook")

        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            with patch("urllib.request.Request") as mock_req:
                payload = {"event_type": "task_done", "timestamp": 1000.0, "data": {}}
                we._send_http(target, payload)

        assert target.failure_count == 0
        assert target.last_success is not None

    def test_send_http_retries_on_failure_then_succeeds(self) -> None:
        """Retry logic: fails twice, succeeds on third attempt."""
        we = WebhookEmitter(retry_count=3, retry_backoff_s=0.0)
        target = WebhookTarget("retry_hook", "http://flaky.example.com")

        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        call_count = [0]

        def fake_urlopen(req, timeout):
            call_count[0] += 1
            if call_count[0] < 3:
                raise OSError("connection refused")
            return mock_response

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with patch("time.sleep"):  # don't actually sleep
                payload = {"event_type": "ping", "timestamp": 0.0, "data": {}}
                we._send_http(target, payload)

        assert call_count[0] == 3
        assert target.last_success is not None
        assert target.failure_count == 0  # reset on success

    def test_send_http_all_retries_fail_increments_failure_count(self) -> None:
        """Lines 534-540: all retries exhaust; failure_count increments each attempt."""
        we = WebhookEmitter(retry_count=3, retry_backoff_s=0.0)
        target = WebhookTarget("broken_hook", "http://broken.example.com")

        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            with patch("time.sleep"):
                payload = {"event_type": "fail", "timestamp": 0.0, "data": {}}
                we._send_http(target, payload)

        assert target.failure_count == 3

    def test_send_http_disables_target_after_5_failures(self) -> None:
        """Lines 542-544: target is disabled when failure_count >= 5."""
        we = WebhookEmitter(retry_count=3, retry_backoff_s=0.0)
        target = WebhookTarget("fragile", "http://dead.example.com")
        target.failure_count = 3  # already has 3 failures

        with patch("urllib.request.urlopen", side_effect=OSError("dead")):
            with patch("time.sleep"):
                payload = {"event_type": "x", "timestamp": 0.0, "data": {}}
                we._send_http(target, payload)

        # failure_count is now 3 + 3 = 6 → disabled
        assert target.enabled is False

    def test_emit_spawns_thread_for_http_target(self) -> None:
        """Lines 514-518: HTTP targets are dispatched in daemon threads."""
        we = WebhookEmitter()
        we.add_target("hook", "http://example.com")

        threads_created = []

        original_thread_init = threading.Thread.__init__

        def patched_init(self_t, target=None, args=(), **kwargs):
            threads_created.append(target)
            original_thread_init(self_t, target=target, args=args, **kwargs)

        with patch.object(threading.Thread, "__init__", patched_init), \
             patch.object(threading.Thread, "start", MagicMock()):
            we.emit("some_event")

        assert any(threads_created)

    def test_emit_does_not_dispatch_to_filtered_out_targets(self) -> None:
        """HTTP target only receives events it's subscribed to."""
        sent = []

        we = WebhookEmitter()
        we.add_target("selective", "http://example.com", events=["task_completed"])

        with patch.object(threading.Thread, "start", lambda self: sent.append(True)):
            with patch.object(threading.Thread, "__init__", lambda self, target, args, daemon: None):
                we.emit("unrelated_event")

        assert sent == []

    def test_format_slack_known_event(self) -> None:
        """Lines 546-560: format_slack produces correct Slack payload."""
        we = WebhookEmitter()
        payload = we.format_slack("task_completed", task_id="t1", robot="tb4")
        assert "text" in payload
        assert "task_completed" in payload["text"]
        assert ":white_check_mark:" in payload["text"]
        assert "task_id" in payload["text"]

    def test_format_slack_unknown_event(self) -> None:
        """Fallback icon for unknown event types."""
        we = WebhookEmitter()
        payload = we.format_slack("mystery_event", foo="bar")
        assert ":robot_face:" in payload["text"]

    def test_format_slack_battery_low(self) -> None:
        we = WebhookEmitter()
        payload = we.format_slack("battery_low", level=12)
        assert ":battery:" in payload["text"]

    def test_format_slack_task_failed(self) -> None:
        we = WebhookEmitter()
        payload = we.format_slack("task_failed")
        assert ":x:" in payload["text"]

    def test_format_slack_safety_violation(self) -> None:
        we = WebhookEmitter()
        payload = we.format_slack("safety_violation")
        assert ":rotating_light:" in payload["text"]

    def test_format_slack_teleop_started(self) -> None:
        we = WebhookEmitter()
        payload = we.format_slack("teleop_started")
        assert ":joystick:" in payload["text"]

    def test_add_slack_target_registers(self) -> None:
        """Line 565: add_slack_target delegates to add_target."""
        we = WebhookEmitter()
        we.add_slack_target("slack", "https://hooks.slack.com/abc", events=["task_done"])
        assert "slack" in we._targets

    def test_add_teams_target_registers(self) -> None:
        """Line 570: add_teams_target delegates to add_target."""
        we = WebhookEmitter()
        we.add_teams_target("teams", "https://outlook.office.com/webhook/abc")
        assert "teams" in we._targets

    def test_repr_shows_target_count_and_event_count(self) -> None:
        """Line 581: __repr__ includes target count and event count."""
        we = WebhookEmitter()
        we.add_target("t1", "http://a.com")
        we.emit("some_event")
        r = repr(we)
        assert "WebhookEmitter" in r

    def test_remove_target_removes_http_target(self) -> None:
        we = WebhookEmitter()
        we.add_target("t1", "http://a.com")
        we.remove_target("t1")
        assert "t1" not in we._targets

    def test_remove_nonexistent_target_does_not_raise(self) -> None:
        we = WebhookEmitter()
        we.remove_target("no_such_target")  # should not raise


# ---------------------------------------------------------------------------
# _parse_cron_to_seconds  (lines 589-627)
# ---------------------------------------------------------------------------

class TestParseCronToSeconds:
    """Cover all branches of the cron-expression parser."""

    @pytest.fixture(autouse=True)
    def import_fn(self):
        from apyrobo.operations import _parse_cron_to_seconds
        self.parse = _parse_cron_to_seconds

    def test_every_n_minutes(self) -> None:
        assert self.parse("*/5 * * * *") == pytest.approx(5 * 60)

    def test_every_1_minute_star_format(self) -> None:
        assert self.parse("*/1 * * * *") == pytest.approx(60)

    def test_every_n_hours(self) -> None:
        assert self.parse("0 */2 * * *") == pytest.approx(2 * 3600)

    def test_every_6_hours(self) -> None:
        assert self.parse("0 */6 * * *") == pytest.approx(6 * 3600)

    def test_daily_at_hour(self) -> None:
        assert self.parse("0 9 * * *") == pytest.approx(86400)

    def test_every_minute_star_star(self) -> None:
        assert self.parse("* * * * *") == pytest.approx(60)

    def test_fallback_unrecognised(self) -> None:
        assert self.parse("15 4 * * 1") == pytest.approx(3600)

    def test_wrong_field_count_fallback(self) -> None:
        assert self.parse("* * *") == pytest.approx(3600)

    def test_invalid_minute_number_fallback(self) -> None:
        assert self.parse("*/abc * * * *") == pytest.approx(3600)

    def test_invalid_hour_number_fallback(self) -> None:
        assert self.parse("0 */xyz * * *") == pytest.approx(3600)


# ---------------------------------------------------------------------------
# ScheduledTaskRunner  (lines 630-731)
# ---------------------------------------------------------------------------

class TestScheduledTaskRunner:
    """Cover ScheduledTaskRunner.add_interval_job, add_task, start/stop, _execute."""

    @pytest.fixture(autouse=True)
    def import_cls(self):
        from apyrobo.operations import ScheduledTaskRunner
        self.Runner = ScheduledTaskRunner

    def test_add_interval_job_registers_job(self) -> None:
        runner = self.Runner()
        runner.add_interval_job("heartbeat", 60.0, lambda: None)
        assert len(runner._jobs) == 1
        assert runner._jobs[0]["name"] == "heartbeat"
        assert runner._jobs[0]["mode"] == "fn"

    def test_add_task_registers_agent_job(self) -> None:
        runner = self.Runner()
        robot = MagicMock()
        agent = MagicMock()
        runner.add_task("cleanup", "*/10 * * * *", "run diagnostics", robot, agent)
        assert len(runner._jobs) == 1
        job = runner._jobs[0]
        assert job["name"] == "cleanup"
        assert job["mode"] == "agent"
        assert job["interval_s"] == pytest.approx(10 * 60)

    def test_start_spawns_thread(self) -> None:
        runner = self.Runner()
        runner.start()
        assert runner._thread is not None
        assert runner._thread.is_alive()
        runner.stop()

    def test_start_idempotent(self) -> None:
        """Calling start() twice should not spawn a second thread."""
        runner = self.Runner()
        runner.start()
        first_thread = runner._thread
        runner.start()
        assert runner._thread is first_thread
        runner.stop()

    def test_stop_terminates_thread(self) -> None:
        runner = self.Runner()
        runner.start()
        assert runner._thread is not None and runner._thread.is_alive()
        runner.stop()
        # After stop, thread should finish within timeout
        runner._thread.join(timeout=3)
        assert not runner._thread.is_alive()

    def test_execute_fn_mode_calls_function(self) -> None:
        """_execute invokes the registered fn for fn-mode jobs."""
        from unittest.mock import patch as _patch

        results = []
        runner = self.Runner()
        job = {
            "name": "test_job",
            "mode": "fn",
            "fn": lambda: results.append("ran"),
            "interval_s": 60.0,
            "next": 0.0,
        }
        with _patch("apyrobo.observability.emit_event"):
            runner._execute(job, time.time())
        assert results == ["ran"]

    def test_execute_fn_mode_exception_is_logged(self) -> None:
        """_execute catches exceptions in fn-mode jobs."""
        from unittest.mock import patch as _patch

        runner = self.Runner()
        job = {
            "name": "bad_job",
            "mode": "fn",
            "fn": lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            "interval_s": 60.0,
            "next": 0.0,
        }
        with _patch("apyrobo.observability.emit_event"):
            runner._execute(job, time.time())  # should not raise

    def test_execute_agent_mode_calls_agent(self) -> None:
        """_execute calls agent.execute in agent-mode jobs."""
        from unittest.mock import patch as _patch

        agent = MagicMock()
        robot = MagicMock()
        result = MagicMock()
        result.status = "completed"
        agent.execute.return_value = result

        runner = self.Runner()
        job = {
            "name": "patrol",
            "mode": "agent",
            "task": "patrol the facility",
            "robot": robot,
            "agent": agent,
            "interval_s": 3600.0,
            "next": 0.0,
        }
        with _patch("apyrobo.observability.emit_event"):
            runner._execute(job, time.time())

        agent.execute.assert_called_once_with("patrol the facility", robot)

    def test_execute_agent_mode_stores_result_in_state_store(self) -> None:
        """_execute writes result to StateStore when available."""
        from unittest.mock import patch as _patch

        agent = MagicMock()
        robot = MagicMock()
        result = MagicMock()
        result.status = "completed"
        agent.execute.return_value = result

        store = MagicMock()
        store.set = MagicMock()

        runner = self.Runner(state_store=store)
        job = {
            "name": "report",
            "mode": "agent",
            "task": "generate report",
            "robot": robot,
            "agent": agent,
            "interval_s": 86400.0,
            "next": 0.0,
        }
        with _patch("apyrobo.observability.emit_event"):
            runner._execute(job, time.time())

        store.set.assert_called_once()

    def test_execute_agent_mode_exception_is_logged(self) -> None:
        """_execute catches exceptions in agent-mode jobs."""
        from unittest.mock import patch as _patch

        agent = MagicMock()
        agent.execute.side_effect = RuntimeError("agent crashed")
        robot = MagicMock()

        runner = self.Runner()
        job = {
            "name": "broken",
            "mode": "agent",
            "task": "do something",
            "robot": robot,
            "agent": agent,
            "interval_s": 60.0,
            "next": 0.0,
        }
        with _patch("apyrobo.observability.emit_event"):
            runner._execute(job, time.time())  # should not raise

    def test_interval_job_fires_when_due(self) -> None:
        """Integration: an overdue job fires within a short polling window."""
        from unittest.mock import patch as _patch

        results = []
        runner = self.Runner()
        runner.add_interval_job("tick", 0.01, lambda: results.append(1))

        with _patch("apyrobo.observability.emit_event"):
            runner.start()
            time.sleep(0.12)
            runner.stop()

        assert len(results) >= 1


# ---------------------------------------------------------------------------
# OperationsApiServer  (lines 734-937)
# ---------------------------------------------------------------------------

class TestOperationsApiServer:
    """Cover OperationsApiServer REST endpoints via direct HTTP calls."""

    @pytest.fixture(autouse=True)
    def import_cls(self, monkeypatch):
        from apyrobo.operations import OperationsApiServer
        self.Server = OperationsApiServer
        # Patch socket.getfqdn to avoid slow DNS reverse lookups on macOS
        monkeypatch.setattr("socket.getfqdn", lambda addr="": addr or "localhost")

    def _make_server(self, **kwargs) -> "OperationsApiServer":  # type: ignore[name-defined]
        import socket
        # Find a free port
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        srv = self.Server(host="127.0.0.1", port=port, **kwargs)
        srv.start()
        time.sleep(0.05)
        return srv

    def _get(self, srv, path: str, headers=None) -> tuple[int, dict]:
        import urllib.request as u
        req = u.Request(f"http://127.0.0.1:{srv.port}{path}", headers=headers or {})
        try:
            with u.urlopen(req, timeout=3) as resp:
                return resp.status, json.loads(resp.read())
        except Exception as exc:
            # urllib raises for non-200; parse from exception
            if hasattr(exc, "code"):
                return exc.code, json.loads(exc.read())
            raise

    def _post(self, srv, path: str, body: dict, headers=None) -> tuple[int, dict]:
        import urllib.request as u
        data = json.dumps(body).encode()
        h = {"Content-Type": "application/json", "Content-Length": str(len(data))}
        if headers:
            h.update(headers)
        req = u.Request(f"http://127.0.0.1:{srv.port}{path}", data=data, headers=h, method="POST")
        try:
            with u.urlopen(req, timeout=3) as resp:
                return resp.status, json.loads(resp.read())
        except Exception as exc:
            if hasattr(exc, "code"):
                return exc.code, json.loads(exc.read())
            raise

    def _delete(self, srv, path: str, headers=None) -> tuple[int, dict]:
        import urllib.request as u
        req = u.Request(
            f"http://127.0.0.1:{srv.port}{path}",
            headers=headers or {},
            method="DELETE",
        )
        try:
            with u.urlopen(req, timeout=3) as resp:
                return resp.status, json.loads(resp.read())
        except Exception as exc:
            if hasattr(exc, "code"):
                return exc.code, json.loads(exc.read())
            raise

    def test_health_endpoint(self) -> None:
        srv = self._make_server()
        try:
            code, body = self._get(srv, "/health")
            assert code == 200
            assert body["status"] == "ok"
        finally:
            srv.stop()

    def test_robots_endpoint_empty(self) -> None:
        srv = self._make_server()
        try:
            code, body = self._get(srv, "/robots")
            assert code == 200
            assert "robots" in body
        finally:
            srv.stop()

    def test_robots_endpoint_with_static_list(self) -> None:
        srv = self._make_server()
        srv.set_robots([{"id": "tb4", "name": "TurtleBot"}])
        try:
            code, body = self._get(srv, "/robots")
            assert code == 200
            assert len(body["robots"]) == 1
            assert body["robots"][0]["id"] == "tb4"
        finally:
            srv.stop()

    def test_unknown_get_path_returns_404(self) -> None:
        srv = self._make_server()
        try:
            code, body = self._get(srv, "/unknown")
            assert code == 404
        finally:
            srv.stop()

    def test_post_task_returns_202(self) -> None:
        srv = self._make_server()
        try:
            code, body = self._post(srv, "/tasks", {"task": "move to A", "robot_id": "tb4"})
            assert code == 202
            assert "task_id" in body
            assert body["status"] == "queued"
        finally:
            srv.stop()

    def test_post_task_without_robot_id(self) -> None:
        srv = self._make_server()
        try:
            code, body = self._post(srv, "/tasks", {"task": "scan area"})
            assert code == 202
        finally:
            srv.stop()

    def test_post_to_unknown_path_returns_404(self) -> None:
        srv = self._make_server()
        try:
            code, body = self._post(srv, "/unknown_post", {})
            assert code == 404
        finally:
            srv.stop()

    def test_get_task_status_after_post(self) -> None:
        srv = self._make_server()
        try:
            _, post_body = self._post(srv, "/tasks", {"task": "patrol"})
            task_id = post_body["task_id"]
            time.sleep(0.05)
            code, body = self._get(srv, f"/tasks/{task_id}")
            assert code == 200
            assert "task_id" in body
        finally:
            srv.stop()

    def test_get_unknown_task_returns_404(self) -> None:
        srv = self._make_server()
        try:
            code, body = self._get(srv, "/tasks/no_such_task_xyz")
            assert code == 404
        finally:
            srv.stop()

    def test_delete_task(self) -> None:
        srv = self._make_server()
        try:
            _, post_body = self._post(srv, "/tasks", {"task": "fetch item"})
            task_id = post_body["task_id"]
            time.sleep(0.05)
            code, body = self._delete(srv, f"/tasks/{task_id}")
            assert code == 200
            assert body["status"] == "cancelled"
        finally:
            srv.stop()

    def test_delete_unknown_task_returns_404(self) -> None:
        srv = self._make_server()
        try:
            code, body = self._delete(srv, "/tasks/no_such_xyz")
            assert code == 404
        finally:
            srv.stop()

    def test_delete_non_task_path_returns_404(self) -> None:
        srv = self._make_server()
        try:
            code, body = self._delete(srv, "/health")
            assert code == 404
        finally:
            srv.stop()

    def test_auth_manager_rejects_missing_key(self) -> None:
        auth = MagicMock()
        auth.authenticate.return_value = None
        srv = self._make_server(auth_manager=auth)
        try:
            code, body = self._get(srv, "/health")
            assert code == 401
        finally:
            srv.stop()

    def test_auth_manager_accepts_valid_key(self) -> None:
        auth = MagicMock()
        auth.authenticate.return_value = {"user": "admin"}
        srv = self._make_server(auth_manager=auth)
        try:
            code, body = self._get(srv, "/health", headers={"X-API-Key": "secret"})
            assert code == 200
        finally:
            srv.stop()

    def test_robots_from_swarm_bus(self) -> None:
        bus = MagicMock()
        bus.robot_ids = ["tb4", "spot"]
        cap = MagicMock()
        cap.model_dump.return_value = {"skills": ["pick"]}
        bus.get_capabilities.return_value = cap

        srv = self._make_server(swarm_bus=bus)
        try:
            code, body = self._get(srv, "/robots")
            assert code == 200
            assert len(body["robots"]) == 2
        finally:
            srv.stop()

    def test_robots_from_swarm_bus_capabilities_fallback(self) -> None:
        """Covers model_dump → dict() fallback path."""
        bus = MagicMock()
        bus.robot_ids = ["tb4"]

        cap = MagicMock(spec=[])  # no model_dump attribute
        cap.dict = MagicMock(return_value={"skills": []})
        bus.get_capabilities.return_value = cap

        srv = self._make_server(swarm_bus=bus)
        try:
            code, body = self._get(srv, "/robots")
            assert code == 200
        finally:
            srv.stop()

    def test_robots_from_swarm_bus_capabilities_exception(self) -> None:
        """get_capabilities raises — capabilities defaults to {}."""
        bus = MagicMock()
        bus.robot_ids = ["tb4"]
        bus.get_capabilities.side_effect = Exception("unavailable")

        srv = self._make_server(swarm_bus=bus)
        try:
            code, body = self._get(srv, "/robots")
            assert code == 200
            assert body["robots"][0]["capabilities"] == {}
        finally:
            srv.stop()

    def test_run_task_background_no_agent(self) -> None:
        """_run_task_background completes with status 'completed' when no agent."""
        srv = self._make_server()
        task_id = "test_bg_001"
        srv._tasks[task_id] = {
            "task_id": task_id,
            "task": "move",
            "robot_id": None,
            "status": "queued",
        }
        srv._run_task_background(task_id, "move", None)
        assert srv._tasks[task_id]["status"] == "completed"

    def test_run_task_background_with_store_begin_complete(self) -> None:
        """StateStore begin_task and complete_task are called."""
        store = MagicMock()
        store.begin_task = MagicMock()
        store.complete_task = MagicMock()

        srv = self._make_server(state_store=store)
        task_id = "bg_store_test"
        srv._tasks[task_id] = {
            "task_id": task_id,
            "task": "diagnose",
            "robot_id": "tb4",
            "status": "queued",
        }
        srv._run_task_background(task_id, "diagnose", "tb4")
        store.begin_task.assert_called_once()
        store.complete_task.assert_called_once()

    def test_run_task_background_agent_exception_marks_failed(self) -> None:
        """Exception in agent.execute sets status to 'failed'."""
        agent = MagicMock()
        agent.execute.side_effect = RuntimeError("agent error")
        bus = MagicMock()
        robot = MagicMock()
        bus.get_robot.return_value = robot
        bus.robot_ids = ["tb4"]

        srv = self._make_server(agent=agent, swarm_bus=bus)
        task_id = "bg_fail_test"
        srv._tasks[task_id] = {
            "task_id": task_id,
            "task": "crash_task",
            "robot_id": "tb4",
            "status": "queued",
        }
        srv._run_task_background(task_id, "crash_task", "tb4")
        assert srv._tasks[task_id]["status"] == "failed"
        assert "error" in srv._tasks[task_id]

    def test_get_task_status_falls_back_to_state_store(self) -> None:
        """_get_task_status queries StateStore when task not in memory."""
        entry = MagicMock()
        entry.to_dict.return_value = {"task_id": "remote_001", "status": "completed"}

        store = MagicMock()
        store.get_task.return_value = entry

        srv = self._make_server(state_store=store)
        result = srv._get_task_status("remote_001")
        assert result is not None
        assert result["task_id"] == "remote_001"

    def test_get_task_status_returns_none_when_not_found(self) -> None:
        store = MagicMock()
        store.get_task.return_value = None

        srv = self._make_server(state_store=store)
        result = srv._get_task_status("totally_unknown")
        assert result is None

    def test_stop_shuts_down_server(self) -> None:
        srv = self._make_server()
        srv.stop()
        # Calling stop again should not raise
        srv.stop()


# ---------------------------------------------------------------------------
# FleetDashboard  (lines 940-961)
# ---------------------------------------------------------------------------

class TestFleetDashboard:
    """Cover FleetDashboard.update_robot, update_task, snapshot."""

    @pytest.fixture(autouse=True)
    def import_cls(self):
        from apyrobo.operations import FleetDashboard
        self.Dashboard = FleetDashboard

    def test_initial_snapshot_is_empty(self) -> None:
        fd = self.Dashboard()
        snap = fd.snapshot()
        assert snap["robots"] == {}
        assert snap["tasks"] == {}
        assert snap["events"] == []

    def test_update_robot(self) -> None:
        fd = self.Dashboard()
        fd.update_robot("tb4", {"status": "idle", "battery": 80})
        snap = fd.snapshot()
        assert "tb4" in snap["robots"]
        assert snap["robots"]["tb4"]["battery"] == 80

    def test_update_task(self) -> None:
        fd = self.Dashboard()
        fd.update_task("task_001", {"status": "running", "skill": "pick"})
        snap = fd.snapshot()
        assert "task_001" in snap["tasks"]
        assert snap["tasks"]["task_001"]["status"] == "running"

    def test_events_list_grows(self) -> None:
        fd = self.Dashboard()
        fd.update_robot("r1", {"s": "ok"})
        fd.update_task("t1", {"s": "done"})
        snap = fd.snapshot()
        assert len(snap["events"]) == 2

    def test_events_capped_at_200(self) -> None:
        fd = self.Dashboard()
        for i in range(250):
            fd.update_robot(f"r{i}", {"i": i})
        snap = fd.snapshot()
        assert len(snap["events"]) == 200

    def test_snapshot_top_level_is_a_copy(self) -> None:
        """Adding a new key to the snapshot robots dict does not affect internal state."""
        fd = self.Dashboard()
        fd.update_robot("tb4", {"status": "idle"})
        snap = fd.snapshot()
        # Add a brand-new robot key to the returned copy
        snap["robots"]["new_robot"] = {"status": "injected"}
        snap2 = fd.snapshot()
        assert "new_robot" not in snap2["robots"]

    def test_robot_status_overwritten_on_update(self) -> None:
        fd = self.Dashboard()
        fd.update_robot("tb4", {"status": "idle"})
        fd.update_robot("tb4", {"status": "busy"})
        snap = fd.snapshot()
        assert snap["robots"]["tb4"]["status"] == "busy"

    def test_event_contains_type_and_timestamp(self) -> None:
        fd = self.Dashboard()
        fd.update_robot("tb4", {"status": "ok"})
        snap = fd.snapshot()
        evt = snap["events"][0]
        assert evt["type"] == "robot"
        assert evt["robot_id"] == "tb4"
        assert "t" in evt

    def test_task_event_type_is_task(self) -> None:
        fd = self.Dashboard()
        fd.update_task("t1", {"status": "queued"})
        snap = fd.snapshot()
        assert snap["events"][0]["type"] == "task"
        assert snap["events"][0]["task_id"] == "t1"
