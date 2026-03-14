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
            i["type"] == "watchdog_estop" for i in enforcer.interventions
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
