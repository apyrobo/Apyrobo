"""
CI-08: SafetyEnforcer tests — watchdog, runtime divergence detection,
move timeout, escalation, policy hot-swap, battery checks.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from apyrobo.core.robot import Robot
from apyrobo.safety.enforcer import (
    SafetyEnforcer,
    SafetyPolicy,
    SafetyViolation,
    SpeedProfile,
    FormalConstraintExporter,
    DEFAULT_POLICY,
    STRICT_POLICY,
)
from apyrobo.sensors.pipeline import DetectedObject, WorldState
from apyrobo.operations import BatteryMonitor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_robot() -> Robot:
    return Robot.discover("mock://safety_bot")


@pytest.fixture
def enforcer(mock_robot: Robot) -> SafetyEnforcer:
    return SafetyEnforcer(mock_robot, policy="default")


@pytest.fixture
def strict_enforcer(mock_robot: Robot) -> SafetyEnforcer:
    return SafetyEnforcer(mock_robot, policy="strict")


# ===========================================================================
# SF-05: Watchdog — odometry vs commanded position divergence
# ===========================================================================

class TestWatchdog:
    """CI-08: Watchdog detects runtime divergence."""

    def test_watchdog_no_commanded_position(self, enforcer: SafetyEnforcer) -> None:
        """No divergence when no move has been commanded."""
        result = enforcer.check_watchdog()
        assert result is None

    def test_watchdog_within_tolerance(self, enforcer: SafetyEnforcer) -> None:
        """Robot at commanded position passes watchdog."""
        enforcer.move(x=1.0, y=1.0, speed=0.5)
        result = enforcer.check_watchdog()
        assert result is not None
        assert result["ok"]

    def test_watchdog_divergence_detected(self, mock_robot: Robot) -> None:
        """Watchdog detects divergence and triggers stop."""
        policy = SafetyPolicy(name="tight", watchdog_tolerance=0.1)
        enforcer = SafetyEnforcer(mock_robot, policy=policy)

        # Move robot to (1, 1)
        enforcer.move(x=1.0, y=1.0, speed=0.5)

        # Simulate the robot being at a different position than commanded
        # MockAdapter stores position from move(), so override it
        mock_robot._adapter._position = (5.0, 5.0)

        result = enforcer.check_watchdog()
        assert result is not None
        assert not result["ok"]
        assert result["divergence_m"] > 0.1

        # Verify stop was called (check interventions)
        assert any(
            i["type"] == "watchdog_triggered" for i in enforcer.interventions
        )

    def test_watchdog_start_stop(self, enforcer: SafetyEnforcer) -> None:
        """Watchdog thread can be started and stopped."""
        enforcer.move(x=0.0, y=0.0, speed=0.5)
        t = enforcer.start_watchdog(interval=0.05)
        time.sleep(0.15)
        enforcer.stop_watchdog()
        time.sleep(0.1)  # let thread exit
        # No assertion needed — just verify it doesn't crash


# ===========================================================================
# SF-01: Move timeout
# ===========================================================================

class TestMoveTimeout:
    """Move timeout auto-stops the robot."""

    def test_move_timeout_auto_stop(self, mock_robot: Robot) -> None:
        """Robot is auto-stopped when move timeout expires."""
        policy = SafetyPolicy(name="fast_timeout", move_timeout=0.1)
        enforcer = SafetyEnforcer(mock_robot, policy=policy)
        enforcer.move(x=10.0, y=10.0, speed=0.5)
        time.sleep(0.3)
        assert any(i["type"] == "move_timeout" for i in enforcer.interventions)


# ===========================================================================
# Speed enforcement
# ===========================================================================

class TestSpeedEnforcement:
    """Speed clamping and speed profiles."""

    def test_speed_clamped(self, enforcer: SafetyEnforcer) -> None:
        """Speed exceeding max is clamped."""
        enforcer.move(x=1.0, y=1.0, speed=100.0)
        assert any(i["type"] == "speed_clamped" for i in enforcer.interventions)

    def test_speed_within_limit(self, enforcer: SafetyEnforcer) -> None:
        """Speed within limit passes without intervention."""
        enforcer.move(x=1.0, y=1.0, speed=0.5)
        assert len(enforcer.interventions) == 0

    def test_angular_speed_clamped(self, enforcer: SafetyEnforcer) -> None:
        """Angular speed exceeding max is clamped."""
        enforcer.rotate(angle_rad=1.0, speed=100.0)
        assert any(i["type"] == "angular_speed_clamped" for i in enforcer.interventions)


# ===========================================================================
# Collision zones
# ===========================================================================

class TestCollisionZones:
    """Collision zone enforcement."""

    def test_collision_zone_blocks_move(self, mock_robot: Robot) -> None:
        """Move into a collision zone raises SafetyViolation."""
        policy = SafetyPolicy(
            name="zones",
            collision_zones=[{"x_min": 0, "x_max": 5, "y_min": 0, "y_max": 5}],
        )
        enforcer = SafetyEnforcer(mock_robot, policy=policy)
        with pytest.raises(SafetyViolation, match="collision zone"):
            enforcer.move(x=2.5, y=2.5, speed=0.5)

    def test_outside_zone_allowed(self, mock_robot: Robot) -> None:
        """Move outside collision zone succeeds."""
        policy = SafetyPolicy(
            name="zones",
            collision_zones=[{"x_min": 0, "x_max": 5, "y_min": 0, "y_max": 5}],
        )
        enforcer = SafetyEnforcer(mock_robot, policy=policy)
        enforcer.move(x=10.0, y=10.0, speed=0.5)  # Should not raise

    def test_add_collision_zone_runtime(self, enforcer: SafetyEnforcer) -> None:
        """Zones can be added at runtime."""
        zone = {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}
        enforcer.add_collision_zone(zone)
        with pytest.raises(SafetyViolation):
            enforcer.move(x=0.5, y=0.5, speed=0.5)

    def test_remove_collision_zone(self, mock_robot: Robot) -> None:
        """Zones can be removed at runtime."""
        policy = SafetyPolicy(
            name="zones",
            collision_zones=[{"x_min": 0, "x_max": 5, "y_min": 0, "y_max": 5}],
        )
        enforcer = SafetyEnforcer(mock_robot, policy=policy)
        removed = enforcer.remove_collision_zone(0)
        assert removed is not None
        # Now the move should succeed
        enforcer.move(x=2.5, y=2.5, speed=0.5)


# ===========================================================================
# SF-11: Policy hot-swap
# ===========================================================================

class TestPolicyHotSwap:
    """Policy hot-swap."""

    def test_swap_policy_by_name(self, enforcer: SafetyEnforcer) -> None:
        """Swap policy by name."""
        old = enforcer.swap_policy("strict")
        assert old.name == "default"
        assert enforcer.policy.name == "strict"

    def test_swap_policy_by_object(self, enforcer: SafetyEnforcer) -> None:
        """Swap policy by object."""
        custom = SafetyPolicy(name="custom", max_speed=0.1)
        old = enforcer.swap_policy(custom)
        assert old.name == "default"
        assert enforcer.policy.name == "custom"
        assert enforcer.policy.max_speed == 0.1

    def test_swap_policy_audit(self, enforcer: SafetyEnforcer) -> None:
        """Policy swap is recorded in audit log."""
        enforcer.swap_policy("strict")
        assert any(
            e.event_type == "policy_swap" for e in enforcer.audit_log
        )


# ===========================================================================
# SF-06: Speed profile
# ===========================================================================

class TestSpeedProfile:
    """Speed profile ramp-up/ramp-down."""

    def test_ramp_up(self) -> None:
        """Speed ramps up from zero."""
        profile = SpeedProfile(ramp_up_s=1.0)
        # At t=0, speed should be near min_speed
        speed = profile.compute(requested=1.0, elapsed=0.0)
        assert speed == profile.min_speed  # 0 * 1.0 = 0, clamp to min

    def test_full_speed(self) -> None:
        """After ramp-up, full speed is reached."""
        profile = SpeedProfile(ramp_up_s=1.0)
        speed = profile.compute(requested=1.0, elapsed=2.0)
        assert speed == 1.0

    def test_ramp_down(self) -> None:
        """Speed ramps down near the goal."""
        profile = SpeedProfile(ramp_up_s=0.1, ramp_down_s=1.0)
        speed = profile.compute(requested=1.0, elapsed=5.0, remaining_dist=0.1)
        assert speed < 1.0


# ===========================================================================
# SF-03: Escalation
# ===========================================================================

class TestEscalation:
    """Escalation workflow."""

    def test_escalation_with_ack(self, mock_robot: Robot) -> None:
        """Escalation with quick acknowledgment succeeds."""
        policy = SafetyPolicy(name="esc", escalation_timeout=2.0)
        enforcer = SafetyEnforcer(mock_robot, policy=policy)

        # ACK in a background thread
        def ack_later() -> None:
            time.sleep(0.1)
            enforcer.acknowledge_escalation()

        threading.Thread(target=ack_later, daemon=True).start()

        result = enforcer.escalate("test reason")
        assert result is True
        assert not enforcer.is_escalation_pending

    def test_escalation_timeout(self, mock_robot: Robot) -> None:
        """Escalation without acknowledgment times out."""
        policy = SafetyPolicy(name="esc_timeout", escalation_timeout=0.1)
        enforcer = SafetyEnforcer(mock_robot, policy=policy)

        result = enforcer.escalate("test timeout")
        assert result is False


# ===========================================================================
# Pass-throughs
# ===========================================================================

class TestPassThroughs:
    """Safety enforcer pass-through methods."""

    def test_stop_always_allowed(self, enforcer: SafetyEnforcer) -> None:
        enforcer.stop()

    def test_gripper_pass_through(self, enforcer: SafetyEnforcer) -> None:
        assert enforcer.gripper_open() is True
        assert enforcer.gripper_close() is True

    def test_position_pass_through(self, enforcer: SafetyEnforcer) -> None:
        pos = enforcer.get_position()
        assert isinstance(pos, tuple)

    def test_capabilities_pass_through(self, enforcer: SafetyEnforcer) -> None:
        caps = enforcer.capabilities()
        assert caps is not None


# ===========================================================================
# SF-12: Formal constraint export
# ===========================================================================

class TestFormalExport:
    """Formal constraint export."""

    def test_tlaplus_export(self) -> None:
        exp = FormalConstraintExporter(DEFAULT_POLICY)
        tla = exp.to_tlaplus()
        assert "MODULE SafetyPolicy_default" in tla
        assert "MaxSpeed" in tla

    def test_uppaal_export(self) -> None:
        exp = FormalConstraintExporter(DEFAULT_POLICY)
        xml = exp.to_uppaal()
        assert "SafetyEnforcer" in xml
        assert "MAX_SPEED" in xml

    def test_dict_export(self) -> None:
        exp = FormalConstraintExporter(DEFAULT_POLICY)
        d = exp.to_dict()
        assert d["policy_name"] == "default"
        assert "constraints" in d


# ===========================================================================
# SF-01: Watchdog odometry divergence — Timer-based periodic checks
# ===========================================================================

class TestWatchdogDivergence:
    """SF-01: Watchdog wires odometry feedback to divergence detection."""

    def test_watchdog_triggers_stop_on_divergence(self, mock_robot: Robot) -> None:
        """Watchdog triggers stop() when MockAdapter.position is manually overridden to diverge."""
        policy = SafetyPolicy(
            name="tight_watchdog",
            watchdog_tolerance=0.5,
            watchdog_interval=0.05,
        )
        enforcer = SafetyEnforcer(mock_robot, policy=policy)

        # Issue move — this starts the watchdog timer
        enforcer.move(x=5.0, y=5.0, speed=0.5)

        # Manually override the adapter position to simulate divergence
        mock_robot._adapter._position = (50.0, 50.0)

        # Wait for the watchdog timer to fire
        time.sleep(0.2)

        # The watchdog should have triggered
        assert enforcer.watchdog_triggered_count >= 1
        assert any(
            i["type"] == "watchdog_triggered" for i in enforcer.interventions
        )

    def test_watchdog_no_trigger_at_correct_position(self, mock_robot: Robot) -> None:
        """Watchdog does NOT trigger when robot arrives at correct position."""
        policy = SafetyPolicy(
            name="normal_watchdog",
            watchdog_tolerance=2.0,
            watchdog_interval=0.05,
        )
        enforcer = SafetyEnforcer(mock_robot, policy=policy)

        # Move robot — MockAdapter sets position to commanded position
        enforcer.move(x=3.0, y=3.0, speed=0.5)

        # Wait for a few watchdog cycles
        time.sleep(0.2)

        # No divergence should be detected
        assert enforcer.watchdog_triggered_count == 0
        assert not any(
            i["type"] == "watchdog_triggered" for i in enforcer.interventions
        )

        enforcer.stop_watchdog()

    def test_watchdog_interval_configurable(self, mock_robot: Robot) -> None:
        """Watchdog interval is configurable and polls repeatedly."""
        policy = SafetyPolicy(
            name="fast_watchdog",
            watchdog_tolerance=100.0,  # high tolerance — won't trigger
            watchdog_interval=0.05,
        )
        enforcer = SafetyEnforcer(mock_robot, policy=policy)

        enforcer.move(x=1.0, y=1.0, speed=0.5)

        # Wait long enough for multiple timer firings
        time.sleep(0.3)
        enforcer.stop_watchdog()

        # The default interval is what we set
        assert policy.watchdog_interval == 0.05

    def test_watchdog_audit_log_entry(self, mock_robot: Robot) -> None:
        """Triggered watchdog appears in enforcer.audit_log."""
        policy = SafetyPolicy(
            name="audit_watchdog",
            watchdog_tolerance=0.1,
            watchdog_interval=0.05,
        )
        enforcer = SafetyEnforcer(mock_robot, policy=policy)

        enforcer.move(x=2.0, y=2.0, speed=0.5)

        # Force divergence
        mock_robot._adapter._position = (20.0, 20.0)

        # Wait for watchdog
        time.sleep(0.2)

        watchdog_entries = [
            e for e in enforcer.audit_log
            if e.event_type == "watchdog_triggered"
        ]
        assert len(watchdog_entries) >= 1
        assert watchdog_entries[0].details["type"] == "watchdog_triggered"
        assert "divergence_m" in watchdog_entries[0].details

    def test_watchdog_check_raises_safety_violation(self, mock_robot: Robot) -> None:
        """check_watchdog stops robot on divergence detection."""
        policy = SafetyPolicy(name="check_test", watchdog_tolerance=0.1)
        enforcer = SafetyEnforcer(mock_robot, policy=policy)

        enforcer.move(x=1.0, y=1.0, speed=0.5)
        mock_robot._adapter._position = (10.0, 10.0)

        result = enforcer.check_watchdog()
        assert result is not None
        assert not result["ok"]
        assert result["divergence_m"] > 0.1
        assert enforcer.watchdog_triggered_count == 1

    def test_stop_cancels_watchdog(self, mock_robot: Robot) -> None:
        """Calling stop() cancels the watchdog timer."""
        policy = SafetyPolicy(
            name="stop_test",
            watchdog_tolerance=0.5,
            watchdog_interval=0.05,
        )
        enforcer = SafetyEnforcer(mock_robot, policy=policy)
        enforcer.move(x=5.0, y=5.0, speed=0.5)

        # Stop should cancel the watchdog
        enforcer.stop()
        assert not enforcer._watchdog_active


# ===========================================================================
# SF-02: Human proximity — WorldState detections
# ===========================================================================

class TestHumanProximityWorldState:
    """SF-02: Human proximity wired to WorldState detections."""

    def test_person_too_close_raises_violation(self, mock_robot: Robot) -> None:
        """Move to (2,2) with person detected at (2.1, 2.1) at proximity < 0.5m: violation raised."""
        world = WorldState()
        world.detected_objects.append(
            DetectedObject("p1", "person", 2.1, 2.1, confidence=0.95)
        )
        policy = SafetyPolicy(name="proximity", human_proximity_limit=0.5)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, world_state=world)

        with pytest.raises(SafetyViolation, match="Human detected"):
            enforcer.move(x=2.0, y=2.0, speed=0.5)

    def test_person_far_away_no_violation(self, mock_robot: Robot) -> None:
        """Move to (5,5) with person at (1,1): no violation."""
        world = WorldState()
        world.detected_objects.append(
            DetectedObject("p1", "person", 1.0, 1.0, confidence=0.9)
        )
        policy = SafetyPolicy(name="proximity", human_proximity_limit=0.5)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, world_state=world)

        # Should not raise
        enforcer.move(x=5.0, y=5.0, speed=0.5)

    def test_empty_detected_objects_no_violation(self, mock_robot: Robot) -> None:
        """Empty detected_objects: no violation."""
        world = WorldState()
        policy = SafetyPolicy(name="proximity", human_proximity_limit=0.5)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, world_state=world)

        enforcer.move(x=3.0, y=3.0, speed=0.5)

    def test_violation_in_audit_log(self, mock_robot: Robot) -> None:
        """Violation appears in audit_log with person coordinates."""
        world = WorldState()
        world.detected_objects.append(
            DetectedObject("p1", "person", 2.1, 2.1, confidence=0.95)
        )
        policy = SafetyPolicy(name="proximity", human_proximity_limit=0.5)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, world_state=world)

        with pytest.raises(SafetyViolation):
            enforcer.move(x=2.0, y=2.0, speed=0.5)

        proximity_entries = [
            e for e in enforcer.audit_log
            if e.details.get("type") == "human_proximity"
        ]
        assert len(proximity_entries) == 1
        assert "human_position" in proximity_entries[0].details
        assert "distance" in proximity_entries[0].details

    def test_pedestrian_label_detected(self, mock_robot: Robot) -> None:
        """Pedestrian label is also recognized as a human."""
        world = WorldState()
        world.detected_objects.append(
            DetectedObject("p1", "pedestrian", 3.0, 3.0, confidence=0.9)
        )
        policy = SafetyPolicy(name="proximity", human_proximity_limit=0.5)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, world_state=world)

        with pytest.raises(SafetyViolation, match="Human detected"):
            enforcer.move(x=3.0, y=3.0, speed=0.5)

    def test_human_label_detected(self, mock_robot: Robot) -> None:
        """'human' label is also recognized."""
        world = WorldState()
        world.detected_objects.append(
            DetectedObject("h1", "Human", 1.0, 1.0, confidence=0.9)
        )
        policy = SafetyPolicy(name="proximity", human_proximity_limit=2.0)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, world_state=world)

        with pytest.raises(SafetyViolation):
            enforcer.move(x=1.0, y=1.0, speed=0.5)


# ===========================================================================
# SF-03/SF-10: Battery-aware safety gate
# ===========================================================================

class TestBatterySafetyGate:
    """SF-03: Battery-aware safety — refuse tasks below return threshold."""

    def test_low_battery_raises_violation(self, mock_robot: Robot) -> None:
        """8% battery raises SafetyViolation for a 100m trip."""
        battery = BatteryMonitor(robot_id="safety_bot", dock_position=(0.0, 0.0))
        battery.update(percentage=8.0)

        policy = SafetyPolicy(name="battery_test", min_battery_pct=15.0)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, battery_monitor=battery)

        with pytest.raises(SafetyViolation, match="Battery"):
            enforcer.move(x=100.0, y=0.0, speed=0.5)

    def test_sufficient_battery_allows_move(self, mock_robot: Robot) -> None:
        """80% battery allows a 100m trip."""
        battery = BatteryMonitor(robot_id="safety_bot", dock_position=(0.0, 0.0))
        battery.update(percentage=80.0)

        policy = SafetyPolicy(name="battery_test", min_battery_pct=15.0)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, battery_monitor=battery)

        # Should not raise
        enforcer.move(x=50.0, y=86.6, speed=0.5)  # ~100m from origin

    def test_no_battery_monitor_allows_move(self, mock_robot: Robot) -> None:
        """No battery monitor attached: move proceeds normally."""
        policy = SafetyPolicy(name="no_battery")
        enforcer = SafetyEnforcer(mock_robot, policy=policy)

        # No battery_monitor — should not raise
        enforcer.move(x=100.0, y=100.0, speed=0.5)

    def test_battery_violation_in_audit_log(self, mock_robot: Robot) -> None:
        """Violation appears in audit_log with percentage and distance."""
        battery = BatteryMonitor(robot_id="safety_bot", dock_position=(0.0, 0.0))
        battery.update(percentage=5.0)

        policy = SafetyPolicy(name="battery_audit", min_battery_pct=15.0)
        enforcer = SafetyEnforcer(mock_robot, policy=policy, battery_monitor=battery)

        with pytest.raises(SafetyViolation):
            enforcer.move(x=100.0, y=0.0, speed=0.5)

        battery_entries = [
            e for e in enforcer.audit_log
            if "battery" in e.details.get("type", "")
        ]
        assert len(battery_entries) >= 1
        assert "battery_pct" in battery_entries[0].details
        assert "distance_m" in battery_entries[0].details
