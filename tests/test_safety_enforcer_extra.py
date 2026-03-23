"""
Extra coverage tests for apyrobo/safety/enforcer.py.

Targets missing lines not already covered by test_safety/test_enforcer.py:
  - SpeedProfile.__repr__
  - SafetyAuditEntry.to_dict
  - SafetyPolicy.from_ref, __repr__
  - SafetyEnforcer: speed clamping with speed_profile,
    dynamic collision zones via WorldState,
    human proximity enforcement,
    disconnect (cancel timers),
    _record_audit with state_store,
    escalation with webhook_emitter,
    battery check: no monitor, trip check, below minimum,
    check_battery method,
    add/remove collision zone audit,
    update_world_state,
    set_battery_monitor,
    rotate with negative speed clamping,
    robot_id property,
    __repr__,
    FormalConstraintExporter.to_uppaal,
    FormalConstraintExporter.to_tlaplus with zones,
    FormalConstraintExporter.to_dict,
    point_in_zone edge cases,
    cancel pass-through.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import SafetyPolicyRef
from apyrobo.safety.enforcer import (
    DEFAULT_POLICY,
    POLICY_REGISTRY,
    STRICT_POLICY,
    EscalationTimeout,
    FormalConstraintExporter,
    SafetyAuditEntry,
    SafetyEnforcer,
    SafetyPolicy,
    SafetyViolation,
    SpeedProfile,
)
from apyrobo.sensors.pipeline import DetectedObject, WorldState
from apyrobo.operations import BatteryMonitor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_robot() -> Robot:
    return Robot.discover("mock://enforcer_extra")


@pytest.fixture
def enforcer(mock_robot: Robot) -> SafetyEnforcer:
    return SafetyEnforcer(mock_robot, policy="default")


# ===========================================================================
# SpeedProfile
# ===========================================================================

class TestSpeedProfileExtra:
    def test_repr(self) -> None:
        profile = SpeedProfile(ramp_up_s=1.5, ramp_down_s=0.75)
        r = repr(profile)
        assert "SpeedProfile" in r
        assert "1.5" in r
        assert "0.75" in r

    def test_ramp_up_zero_elapsed(self) -> None:
        profile = SpeedProfile(ramp_up_s=1.0, min_speed=0.01)
        speed = profile.compute(requested=1.0, elapsed=0.0)
        assert speed == pytest.approx(profile.min_speed)

    def test_ramp_up_half_elapsed(self) -> None:
        profile = SpeedProfile(ramp_up_s=2.0, min_speed=0.01)
        speed = profile.compute(requested=1.0, elapsed=1.0)
        # elapsed/ramp_up_s = 0.5, so speed = 0.5*1.0 = 0.5
        assert speed == pytest.approx(0.5)

    def test_ramp_up_full(self) -> None:
        profile = SpeedProfile(ramp_up_s=1.0, min_speed=0.01)
        speed = profile.compute(requested=1.0, elapsed=5.0)
        assert speed == pytest.approx(1.0)

    def test_ramp_down_near_goal(self) -> None:
        profile = SpeedProfile(ramp_up_s=0.01, ramp_down_s=1.0, min_speed=0.01)
        # Very small remaining distance -> ramp down should reduce speed
        speed = profile.compute(requested=1.0, elapsed=10.0, remaining_dist=0.05)
        assert speed < 1.0

    def test_ramp_down_far_from_goal(self) -> None:
        profile = SpeedProfile(ramp_up_s=0.01, ramp_down_s=0.5, min_speed=0.01)
        # Large remaining distance -> no ramp down effect
        speed = profile.compute(requested=1.0, elapsed=10.0, remaining_dist=100.0)
        assert speed == pytest.approx(1.0)

    def test_min_speed_clamping(self) -> None:
        profile = SpeedProfile(ramp_up_s=100.0, min_speed=0.1)
        speed = profile.compute(requested=1.0, elapsed=0.0)
        assert speed >= profile.min_speed

    def test_min_ramp_up_clamped(self) -> None:
        # ramp_up_s below 0.01 should be clamped
        profile = SpeedProfile(ramp_up_s=0.0)
        assert profile.ramp_up_s == pytest.approx(0.01)

    def test_min_ramp_down_clamped(self) -> None:
        profile = SpeedProfile(ramp_down_s=0.0)
        assert profile.ramp_down_s == pytest.approx(0.01)

    def test_remaining_dist_none(self) -> None:
        profile = SpeedProfile(ramp_up_s=0.01)
        # remaining_dist=None should not trigger ramp down
        speed = profile.compute(requested=1.0, elapsed=5.0, remaining_dist=None)
        assert speed == pytest.approx(1.0)

    def test_remaining_dist_zero_skips_ramp_down(self) -> None:
        profile = SpeedProfile(ramp_up_s=0.01, ramp_down_s=1.0)
        # remaining_dist=0 should skip ramp down (not > 0)
        speed = profile.compute(requested=1.0, elapsed=10.0, remaining_dist=0.0)
        assert speed == pytest.approx(1.0)


# ===========================================================================
# SafetyAuditEntry
# ===========================================================================

class TestSafetyAuditEntry:
    def test_to_dict(self) -> None:
        entry = SafetyAuditEntry(
            event_type="violation",
            robot_id="tb4",
            details={"type": "speed_clamped"},
            policy_name="default",
        )
        d = entry.to_dict()
        assert d["event_type"] == "violation"
        assert d["robot_id"] == "tb4"
        assert d["details"] == {"type": "speed_clamped"}
        assert d["policy_name"] == "default"
        assert "timestamp" in d

    def test_timestamp_auto_set(self) -> None:
        before = time.time()
        entry = SafetyAuditEntry(event_type="e", robot_id="r", details={})
        after = time.time()
        assert before <= entry.timestamp <= after

    def test_explicit_timestamp(self) -> None:
        t = 12345.0
        entry = SafetyAuditEntry(event_type="e", robot_id="r", details={}, timestamp=t)
        assert entry.timestamp == t


# ===========================================================================
# SafetyPolicy
# ===========================================================================

class TestSafetyPolicyExtra:
    def test_repr(self) -> None:
        policy = SafetyPolicy(name="test", max_speed=0.5, max_angular_speed=1.0)
        r = repr(policy)
        assert "SafetyPolicy" in r
        assert "test" in r
        assert "0.5" in r

    def test_from_ref(self) -> None:
        ref = SafetyPolicyRef(
            policy_name="ref_policy",
            max_speed=0.3,
            collision_zones=[{"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}],
            human_proximity_limit=1.5,
        )
        policy = SafetyPolicy.from_ref(ref)
        assert policy.name == "ref_policy"
        assert policy.max_speed == pytest.approx(0.3)
        assert len(policy.collision_zones) == 1
        assert policy.human_proximity_limit == pytest.approx(1.5)

    def test_from_ref_defaults(self) -> None:
        ref = SafetyPolicyRef()  # all defaults
        policy = SafetyPolicy.from_ref(ref)
        assert policy.max_speed == pytest.approx(1.5)
        assert policy.human_proximity_limit == pytest.approx(0.5)

    def test_policy_registry(self) -> None:
        assert "default" in POLICY_REGISTRY
        assert "strict" in POLICY_REGISTRY

    def test_default_policy_values(self) -> None:
        assert DEFAULT_POLICY.max_speed == pytest.approx(1.5)
        assert DEFAULT_POLICY.name == "default"

    def test_strict_policy_values(self) -> None:
        assert STRICT_POLICY.max_speed <= 1.0
        assert STRICT_POLICY.name == "strict"
        assert STRICT_POLICY.speed_profile is not None


# ===========================================================================
# SafetyEnforcer: basic properties and pass-throughs
# ===========================================================================

class TestSafetyEnforcerProperties:
    def test_robot_property(self, enforcer: SafetyEnforcer, mock_robot: Robot) -> None:
        assert enforcer.robot is mock_robot

    def test_policy_property(self, enforcer: SafetyEnforcer) -> None:
        assert enforcer.policy.name == "default"

    def test_robot_id_property(self, enforcer: SafetyEnforcer, mock_robot: Robot) -> None:
        assert enforcer.robot_id == mock_robot.robot_id

    def test_repr(self, enforcer: SafetyEnforcer) -> None:
        r = repr(enforcer)
        assert "SafetyEnforcer" in r
        assert "default" in r

    def test_cancel_pass_through(self, enforcer: SafetyEnforcer) -> None:
        # Should not raise
        enforcer.cancel()

    def test_get_orientation_pass_through(self, enforcer: SafetyEnforcer) -> None:
        ori = enforcer.get_orientation()
        assert isinstance(ori, float)

    def test_get_health_pass_through(self, enforcer: SafetyEnforcer) -> None:
        health = enforcer.get_health()
        assert isinstance(health, dict)

    def test_connect_pass_through(self, mock_robot: Robot) -> None:
        enforcer = SafetyEnforcer(mock_robot, policy="default")
        enforcer.connect()

    def test_disconnect_cancels_timers(self, enforcer: SafetyEnforcer) -> None:
        # Start a move to create timers (use coords outside the collision zone x_max=1,y_max=1)
        enforcer.move(x=5.0, y=5.0, speed=0.5)
        enforcer.disconnect()

    def test_unknown_policy_name_uses_default(self, mock_robot: Robot) -> None:
        enforcer = SafetyEnforcer(mock_robot, policy="unknown_policy_xyz")
        assert enforcer.policy is DEFAULT_POLICY

    def test_custom_policy_object(self, mock_robot: Robot) -> None:
        custom = SafetyPolicy(name="my_custom", max_speed=0.2)
        enforcer = SafetyEnforcer(mock_robot, policy=custom)
        assert enforcer.policy.name == "my_custom"
        assert enforcer.policy.max_speed == pytest.approx(0.2)


# ===========================================================================
# Speed enforcement
# ===========================================================================

class TestSpeedEnforcementExtra:
    def test_speed_clamped_recorded_in_interventions(self, enforcer: SafetyEnforcer) -> None:
        # Use coords outside the collision zone (which covers 0-1 on both axes)
        enforcer.move(x=5.0, y=5.0, speed=999.0)
        assert any(i["type"] == "speed_clamped" for i in enforcer.interventions)
        # Check fields
        clamp = next(i for i in enforcer.interventions if i["type"] == "speed_clamped")
        assert clamp["requested"] == pytest.approx(999.0)
        assert clamp["enforced"] == pytest.approx(enforcer.policy.max_speed)

    def test_speed_clamped_recorded_in_audit(self, enforcer: SafetyEnforcer) -> None:
        enforcer.move(x=5.0, y=5.0, speed=999.0)
        types = [e.event_type for e in enforcer.audit_log]
        assert "intervention" in types

    def test_no_speed_no_clamping(self, enforcer: SafetyEnforcer) -> None:
        enforcer.move(x=5.0, y=5.0, speed=None)
        clamp_events = [i for i in enforcer.interventions if i["type"] == "speed_clamped"]
        assert len(clamp_events) == 0

    def test_speed_with_profile_applied(self, mock_robot: Robot) -> None:
        profile = SpeedProfile(ramp_up_s=1.0, min_speed=0.05)
        policy = SafetyPolicy(name="profile_test", max_speed=2.0, speed_profile=profile)
        enforcer = SafetyEnforcer(mock_robot, policy=policy)
        # Speed within limit, but profile should still be applied (elapsed=0 so min_speed)
        enforcer.move(x=1.0, y=1.0, speed=1.0)
        # Should not raise; profile computation runs

    def test_rotate_angular_speed_negative_clamped(self, enforcer: SafetyEnforcer) -> None:
        # Negative speed beyond limit
        enforcer.rotate(angle_rad=1.0, speed=-100.0)
        clamp = next(i for i in enforcer.interventions if i["type"] == "angular_speed_clamped")
        assert clamp["enforced"] < 0  # Should preserve sign
        assert abs(clamp["enforced"]) == pytest.approx(enforcer.policy.max_angular_speed)

    def test_rotate_within_limit_no_clamp(self, enforcer: SafetyEnforcer) -> None:
        enforcer.rotate(angle_rad=0.5, speed=0.5)
        clamp_events = [i for i in enforcer.interventions if i["type"] == "angular_speed_clamped"]
        assert len(clamp_events) == 0

    def test_rotate_none_speed_no_clamp(self, enforcer: SafetyEnforcer) -> None:
        enforcer.rotate(angle_rad=0.5, speed=None)
        clamp_events = [i for i in enforcer.interventions if i["type"] == "angular_speed_clamped"]
        assert len(clamp_events) == 0


# ===========================================================================
# Collision zone enforcement
# ===========================================================================

class TestCollisionZoneEnforcementExtra:
    def test_static_zone_blocks_move(self, mock_robot: Robot) -> None:
        zone = {"x_min": 0.0, "x_max": 5.0, "y_min": 0.0, "y_max": 5.0}
        policy = SafetyPolicy(name="zone_test", collision_zones=[zone])
        enforcer = SafetyEnforcer(mock_robot, policy=policy)
        with pytest.raises(SafetyViolation, match="collision zone"):
            enforcer.move(x=2.5, y=2.5)

    def test_static_zone_violation_recorded(self, mock_robot: Robot) -> None:
        zone = {"x_min": 0.0, "x_max": 5.0, "y_min": 0.0, "y_max": 5.0}
        policy = SafetyPolicy(name="zone_test", collision_zones=[zone])
        enforcer = SafetyEnforcer(mock_robot, policy=policy)
        try:
            enforcer.move(x=2.5, y=2.5)
        except SafetyViolation:
            pass
        assert any(v["type"] == "collision_zone" for v in enforcer.violations)

    def test_point_on_zone_boundary_rejected(self, mock_robot: Robot) -> None:
        zone = {"x_min": 0.0, "x_max": 5.0, "y_min": 0.0, "y_max": 5.0}
        policy = SafetyPolicy(name="zone_test", collision_zones=[zone])
        enforcer = SafetyEnforcer(mock_robot, policy=policy)
        with pytest.raises(SafetyViolation):
            enforcer.move(x=0.0, y=0.0)  # exactly on boundary

    def test_move_outside_zone_allowed(self, mock_robot: Robot) -> None:
        zone = {"x_min": 0.0, "x_max": 5.0, "y_min": 0.0, "y_max": 5.0}
        policy = SafetyPolicy(name="zone_test", collision_zones=[zone])
        enforcer = SafetyEnforcer(mock_robot, policy=policy)
        enforcer.move(x=10.0, y=10.0)  # should not raise

    def test_add_zone_creates_audit(self, mock_robot: Robot) -> None:
        # Use a fresh isolated policy so we don't mutate the shared DEFAULT_POLICY
        policy = SafetyPolicy(name="add_zone_test")
        enforcer = SafetyEnforcer(mock_robot, policy=policy)
        zone = {"x_min": 0.0, "x_max": 2.0, "y_min": 0.0, "y_max": 2.0}
        enforcer.add_collision_zone(zone)
        assert any(e.event_type == "zone_added" for e in enforcer.audit_log)
        assert len(enforcer.policy.collision_zones) == 1

    def test_remove_zone_valid_index(self, mock_robot: Robot) -> None:
        zone = {"x_min": 0.0, "x_max": 2.0, "y_min": 0.0, "y_max": 2.0}
        policy = SafetyPolicy(name="r", collision_zones=[zone])
        enforcer = SafetyEnforcer(mock_robot, policy=policy)
        removed = enforcer.remove_collision_zone(0)
        assert removed is not None
        assert len(enforcer.policy.collision_zones) == 0
        assert any(e.event_type == "zone_removed" for e in enforcer.audit_log)

    def test_remove_zone_invalid_index_returns_none(self, enforcer: SafetyEnforcer) -> None:
        result = enforcer.remove_collision_zone(99)
        assert result is None

    def test_point_in_zone_helper(self) -> None:
        zone = {"x_min": 0.0, "x_max": 5.0, "y_min": 0.0, "y_max": 5.0}
        assert SafetyEnforcer._point_in_zone(2.5, 2.5, zone) is True
        assert SafetyEnforcer._point_in_zone(10.0, 10.0, zone) is False
        assert SafetyEnforcer._point_in_zone(0.0, 0.0, zone) is True  # boundary
        assert SafetyEnforcer._point_in_zone(5.0, 5.0, zone) is True  # upper boundary

    def test_point_in_zone_with_partial_keys(self) -> None:
        # Missing keys default to -inf/+inf, so everything is "inside"
        assert SafetyEnforcer._point_in_zone(100.0, 100.0, {}) is True


# ===========================================================================
# Dynamic collision zones (SF-07 / WorldState)
# ===========================================================================

class TestDynamicCollisionZones:
    def _make_world_state(self, obstacles: list | None = None, detections: list | None = None) -> Any:
        ws = MagicMock(spec=WorldState)
        ws.obstacles = obstacles or []
        ws.detected_objects = detections or []
        return ws

    def test_dynamic_zone_blocks_move(self, mock_robot: Robot) -> None:
        # Create a WorldState obstacle at (2, 2) with radius 1.0
        # Use a fresh policy with no static zones to avoid test contamination
        obs = MagicMock()
        obs.x = 2.0
        obs.y = 2.0
        obs.radius = 1.5
        obs.source = "lidar"
        ws = self._make_world_state(obstacles=[obs])
        policy = SafetyPolicy(name="dyn_zone_test")
        enforcer = SafetyEnforcer(mock_robot, policy=policy, world_state=ws)
        with pytest.raises(SafetyViolation, match="dynamic obstacle zone"):
            enforcer.move(x=2.0, y=2.0)

    def test_dynamic_zone_violation_recorded(self, mock_robot: Robot) -> None:
        obs = MagicMock()
        obs.x = 2.0
        obs.y = 2.0
        obs.radius = 1.5
        obs.source = "lidar"
        ws = self._make_world_state(obstacles=[obs])
        policy = SafetyPolicy(name="dyn_zone_rec")
        enforcer = SafetyEnforcer(mock_robot, policy=policy, world_state=ws)
        try:
            enforcer.move(x=2.0, y=2.0)
        except SafetyViolation:
            pass
        assert any(v["type"] == "dynamic_collision_zone" for v in enforcer.violations)

    def test_no_world_state_no_dynamic_check(self, mock_robot: Robot) -> None:
        # Without world_state, dynamic checks are skipped
        # Use a fresh policy with no static zones to avoid test contamination
        policy = SafetyPolicy(name="no_ws_test")
        enforcer = SafetyEnforcer(mock_robot, policy=policy, world_state=None)
        enforcer.move(x=2.0, y=2.0)  # should not raise

    def test_update_world_state(self, enforcer: SafetyEnforcer) -> None:
        ws = self._make_world_state()
        enforcer.update_world_state(ws)
        assert enforcer._world_state is ws

    def test_world_state_obstacles_none_returns_empty(self, mock_robot: Robot) -> None:
        ws = self._make_world_state(obstacles=[])
        policy = SafetyPolicy(name="empty_obs_test")
        enforcer = SafetyEnforcer(mock_robot, policy=policy, world_state=ws)
        zones = enforcer._get_dynamic_zones()
        assert zones == []


# ===========================================================================
# Human proximity enforcement (SF-02)
# ===========================================================================

class TestHumanProximityExtra:
    def _human_ws(self, hx: float, hy: float, label: str = "person") -> Any:
        obj = MagicMock(spec=DetectedObject)
        obj.x = hx
        obj.y = hy
        obj.label = label
        obj.object_id = "h1"
        ws = MagicMock(spec=WorldState)
        ws.obstacles = []
        ws.detected_objects = [obj]
        return ws

    def test_human_too_close_raises(self, mock_robot: Robot) -> None:
        ws = self._human_ws(hx=2.0, hy=2.0)
        policy = SafetyPolicy(name="prox", human_proximity_limit=1.0)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, world_state=ws)
        # Target (2.0, 2.0) is exactly where the human is -> distance = 0
        with pytest.raises(SafetyViolation, match="Human detected"):
            enforcer.move(x=2.0, y=2.0)

    def test_human_violation_recorded(self, mock_robot: Robot) -> None:
        ws = self._human_ws(hx=2.0, hy=2.0)
        policy = SafetyPolicy(name="prox", human_proximity_limit=1.0)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, world_state=ws)
        try:
            enforcer.move(x=2.0, y=2.0)
        except SafetyViolation:
            pass
        assert any(v["type"] == "human_proximity" for v in enforcer.violations)

    def test_human_far_enough_allowed(self, mock_robot: Robot) -> None:
        ws = self._human_ws(hx=10.0, hy=10.0)
        policy = SafetyPolicy(name="prox", human_proximity_limit=1.0)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, world_state=ws)
        # Target (2.0, 2.0) is far from human at (10, 10)
        enforcer.move(x=2.0, y=2.0)  # should not raise

    def test_pedestrian_label_triggers_proximity(self, mock_robot: Robot) -> None:
        ws = self._human_ws(hx=2.0, hy=2.0, label="pedestrian")
        policy = SafetyPolicy(name="prox", human_proximity_limit=1.0)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, world_state=ws)
        with pytest.raises(SafetyViolation):
            enforcer.move(x=2.0, y=2.0)

    def test_non_human_label_ignored(self, mock_robot: Robot) -> None:
        ws = self._human_ws(hx=2.0, hy=2.0, label="box")
        policy = SafetyPolicy(name="prox", human_proximity_limit=1.0)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, world_state=ws)
        enforcer.move(x=2.0, y=2.0)  # "box" is not a human, should not raise

    def test_human_label_case_insensitive(self, mock_robot: Robot) -> None:
        ws = self._human_ws(hx=2.0, hy=2.0, label="HUMAN")
        policy = SafetyPolicy(name="prox", human_proximity_limit=1.0)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, world_state=ws)
        with pytest.raises(SafetyViolation):
            enforcer.move(x=2.0, y=2.0)


# ===========================================================================
# Audit log and StateStore persistence (SF-04)
# ===========================================================================

class TestAuditLogPersistence:
    def test_audit_log_populated_on_violation(self, mock_robot: Robot) -> None:
        zone = {"x_min": 0.0, "x_max": 5.0, "y_min": 0.0, "y_max": 5.0}
        policy = SafetyPolicy(name="audit_test", collision_zones=[zone])
        enforcer = SafetyEnforcer(mock_robot, policy=policy)
        try:
            enforcer.move(x=2.5, y=2.5)
        except SafetyViolation:
            pass
        assert len(enforcer.audit_log) > 0

    def test_audit_log_persisted_to_state_store(self, mock_robot: Robot) -> None:
        state_store = MagicMock()
        state_store.get.return_value = []
        zone = {"x_min": 0.0, "x_max": 5.0, "y_min": 0.0, "y_max": 5.0}
        policy = SafetyPolicy(name="audit_persist", collision_zones=[zone])
        enforcer = SafetyEnforcer(mock_robot, policy=policy, state_store=state_store)
        try:
            enforcer.move(x=2.5, y=2.5)
        except SafetyViolation:
            pass
        state_store.set.assert_called()

    def test_audit_log_state_store_error_swallowed(self, mock_robot: Robot) -> None:
        state_store = MagicMock()
        state_store.get.return_value = []
        state_store.set.side_effect = RuntimeError("store down")
        # Use a fresh policy with no zones to avoid contamination from other tests
        policy = SafetyPolicy(name="store_err_test")
        enforcer = SafetyEnforcer(mock_robot, policy=policy, state_store=state_store)
        # Should not raise despite state_store failing
        enforcer.move(x=1.0, y=1.0, speed=999.0)

    def test_audit_log_is_copy(self, enforcer: SafetyEnforcer) -> None:
        log = enforcer.audit_log
        # Modifying the returned list does not affect internal state
        log.append(MagicMock())
        assert len(enforcer.audit_log) == 0


# ===========================================================================
# Battery monitor (SF-10)
# ===========================================================================

class TestBatteryMonitorExtra:
    def _make_monitor(self, pct: float = 80.0, can_trip: bool = True) -> Any:
        monitor = MagicMock(spec=BatteryMonitor)
        monitor.percentage = pct
        monitor.can_complete_trip.return_value = can_trip
        monitor.estimated_range_m = 1000.0
        monitor.status = "normal"
        return monitor

    def test_check_battery_no_monitor(self, enforcer: SafetyEnforcer) -> None:
        result = enforcer.check_battery(10.0)
        assert result["available"] is True
        assert result["reason"] == "no_monitor"

    def test_check_battery_with_monitor(self, mock_robot: Robot) -> None:
        monitor = self._make_monitor(pct=50.0, can_trip=True)
        policy = SafetyPolicy(name="batt", min_battery_pct=20.0)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, battery_monitor=monitor)
        result = enforcer.check_battery(5.0)
        assert result["available"] is True
        assert result["battery_pct"] == pytest.approx(50.0)

    def test_check_battery_trip_not_possible(self, mock_robot: Robot) -> None:
        monitor = self._make_monitor(pct=50.0, can_trip=False)
        policy = SafetyPolicy(name="batt", min_battery_pct=20.0)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, battery_monitor=monitor)
        result = enforcer.check_battery(100.0)
        assert result["available"] is False

    def test_check_battery_below_minimum(self, mock_robot: Robot) -> None:
        monitor = self._make_monitor(pct=10.0, can_trip=True)
        policy = SafetyPolicy(name="batt", min_battery_pct=20.0)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, battery_monitor=monitor)
        result = enforcer.check_battery(1.0)
        assert result["above_minimum"] is False

    def test_move_battery_trip_not_possible_raises(self, mock_robot: Robot) -> None:
        monitor = self._make_monitor(pct=50.0, can_trip=False)
        policy = SafetyPolicy(name="batt", min_battery_pct=5.0)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, battery_monitor=monitor)
        with pytest.raises(SafetyViolation, match="Battery too low"):
            enforcer.move(x=100.0, y=100.0)

    def test_move_battery_below_minimum_raises(self, mock_robot: Robot) -> None:
        monitor = self._make_monitor(pct=5.0, can_trip=True)
        policy = SafetyPolicy(name="batt", min_battery_pct=20.0)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, battery_monitor=monitor)
        with pytest.raises(SafetyViolation, match="below minimum"):
            enforcer.move(x=1.0, y=1.0)

    def test_move_battery_violation_recorded(self, mock_robot: Robot) -> None:
        monitor = self._make_monitor(pct=5.0, can_trip=True)
        policy = SafetyPolicy(name="batt", min_battery_pct=20.0)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, battery_monitor=monitor)
        try:
            enforcer.move(x=1.0, y=1.0)
        except SafetyViolation:
            pass
        assert any(v["type"] == "battery_below_minimum" for v in enforcer.violations)

    def test_set_battery_monitor(self, enforcer: SafetyEnforcer) -> None:
        monitor = self._make_monitor()
        enforcer.set_battery_monitor(monitor)
        assert enforcer._battery_monitor is monitor


# ===========================================================================
# Escalation (SF-03) with webhook
# ===========================================================================

class TestEscalationExtra:
    def test_escalation_with_webhook(self, mock_robot: Robot) -> None:
        webhook = MagicMock()
        policy = SafetyPolicy(name="esc", escalation_timeout=2.0)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, webhook_emitter=webhook)

        def ack_later() -> None:
            time.sleep(0.05)
            enforcer.acknowledge_escalation()

        threading.Thread(target=ack_later, daemon=True).start()
        result = enforcer.escalate("test reason", context={"ctx": "data"})
        assert result is True
        webhook.emit.assert_called()

    def test_escalation_no_webhook(self, mock_robot: Robot) -> None:
        policy = SafetyPolicy(name="esc", escalation_timeout=0.05)
        enforcer = SafetyEnforcer(mock_robot, policy=policy)
        result = enforcer.escalate("no webhook")
        assert result is False

    def test_is_escalation_pending_true_during(self, mock_robot: Robot) -> None:
        policy = SafetyPolicy(name="esc", escalation_timeout=5.0)
        enforcer = SafetyEnforcer(mock_robot, policy=policy)

        pending_status: list[bool] = []

        def check_pending() -> None:
            time.sleep(0.05)
            pending_status.append(enforcer.is_escalation_pending)
            enforcer.acknowledge_escalation()

        threading.Thread(target=check_pending, daemon=True).start()
        enforcer.escalate("test")
        assert True in pending_status

    def test_escalation_sets_pending_false_after(self, mock_robot: Robot) -> None:
        policy = SafetyPolicy(name="esc", escalation_timeout=0.05)
        enforcer = SafetyEnforcer(mock_robot, policy=policy)
        enforcer.escalate("timeout test")
        assert enforcer.is_escalation_pending is False


# ===========================================================================
# Move timeout (SF-01)
# ===========================================================================

class TestMoveTimeoutExtra:
    def test_move_timeout_recorded_in_interventions(self, mock_robot: Robot) -> None:
        policy = SafetyPolicy(name="timeout", move_timeout=0.05)
        enforcer = SafetyEnforcer(mock_robot, policy=policy)
        enforcer.move(x=10.0, y=10.0, speed=0.5)
        time.sleep(0.2)
        assert any(i["type"] == "move_timeout" for i in enforcer.interventions)

    def test_stop_cancels_timer(self, mock_robot: Robot) -> None:
        policy = SafetyPolicy(name="stoptimer", move_timeout=5.0)
        enforcer = SafetyEnforcer(mock_robot, policy=policy)
        enforcer.move(x=5.0, y=5.0, speed=0.5)
        enforcer.stop()
        # Timer should be cancelled — no timeout intervention after stop
        time.sleep(0.1)
        timeout_events = [i for i in enforcer.interventions if i["type"] == "move_timeout"]
        assert len(timeout_events) == 0


# ===========================================================================
# Policy hot-swap (SF-11)
# ===========================================================================

class TestPolicySwapExtra:
    def test_swap_by_name_strict(self, enforcer: SafetyEnforcer) -> None:
        old = enforcer.swap_policy("strict")
        assert old.name == "default"
        assert enforcer.policy.name == "strict"

    def test_swap_by_name_unknown_uses_default(self, enforcer: SafetyEnforcer) -> None:
        enforcer.swap_policy("default")
        enforcer.swap_policy("ghost_policy")
        assert enforcer.policy is DEFAULT_POLICY

    def test_swap_by_object(self, enforcer: SafetyEnforcer) -> None:
        new_policy = SafetyPolicy(name="fast", max_speed=3.0)
        old = enforcer.swap_policy(new_policy)
        assert enforcer.policy.max_speed == pytest.approx(3.0)

    def test_swap_creates_audit_entry(self, enforcer: SafetyEnforcer) -> None:
        enforcer.swap_policy("strict")
        assert any(e.event_type == "policy_swap" for e in enforcer.audit_log)

    def test_swap_audit_fields(self, enforcer: SafetyEnforcer) -> None:
        enforcer.swap_policy("strict")
        entry = next(e for e in enforcer.audit_log if e.event_type == "policy_swap")
        assert entry.details["old_policy"] == "default"
        assert entry.details["new_policy"] == "strict"


# ===========================================================================
# FormalConstraintExporter (SF-12)
# ===========================================================================

class TestFormalConstraintExporterExtra:
    def test_tlaplus_contains_policy_name(self) -> None:
        policy = SafetyPolicy(name="mytest", max_speed=0.5)
        exp = FormalConstraintExporter(policy)
        tla = exp.to_tlaplus()
        assert "SafetyPolicy_mytest" in tla

    def test_tlaplus_with_collision_zones(self) -> None:
        zone = {"x_min": 0.0, "x_max": 5.0, "y_min": 0.0, "y_max": 5.0}
        policy = SafetyPolicy(name="zones", collision_zones=[zone])
        exp = FormalConstraintExporter(policy)
        tla = exp.to_tlaplus()
        assert "x_min" in tla.lower() or "x_min" in tla

    def test_tlaplus_no_zones(self) -> None:
        policy = SafetyPolicy(name="nozone")
        exp = FormalConstraintExporter(policy)
        tla = exp.to_tlaplus()
        assert "MODULE SafetyPolicy_nozone" in tla

    def test_uppaal_contains_safety_enforcer(self) -> None:
        exp = FormalConstraintExporter(DEFAULT_POLICY)
        xml = exp.to_uppaal()
        assert "SafetyEnforcer" in xml

    def test_uppaal_contains_max_speed(self) -> None:
        exp = FormalConstraintExporter(DEFAULT_POLICY)
        xml = exp.to_uppaal()
        assert "MAX_SPEED" in xml

    def test_to_dict_structure(self) -> None:
        exp = FormalConstraintExporter(DEFAULT_POLICY)
        d = exp.to_dict()
        assert d["policy_name"] == "default"
        assert "constraints" in d
        constraints = d["constraints"]
        # Key names use suffixes like _ms or _rads — check at least one speed key exists
        speed_keys = [k for k in constraints if "speed" in k.lower()]
        assert len(speed_keys) > 0

    def test_to_dict_with_custom_policy(self) -> None:
        policy = SafetyPolicy(
            name="custom_export",
            max_speed=0.3,
            max_angular_speed=0.5,
            human_proximity_limit=2.0,
        )
        exp = FormalConstraintExporter(policy)
        d = exp.to_dict()
        constraints = d["constraints"]
        # Find the max_speed value regardless of exact key suffix
        speed_val = next(
            (v for k, v in constraints.items() if "max_speed" in k),
            None,
        )
        assert speed_val is not None
        assert speed_val == pytest.approx(0.3)
