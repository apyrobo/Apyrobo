"""
Safety Enforcer — hard constraints that no agent can bypass.

The SafetyEnforcer wraps a Robot instance and intercepts every command,
rejecting or clamping anything that violates the active safety policy.
It sits between the executor and the robot — invisible to agents.

Features (SF-01 through SF-12):
    SF-01: Move timeout via threading.Timer — auto stop() on expiry
    SF-02: Human proximity enforcement via WorldState detections
    SF-03: ESCALATE recovery — webhook + pause + wait for human ACK
    SF-04: Audit log — persist every violation/intervention to StateStore
    SF-05: Runtime watchdog — compare odometry vs commanded; e-stop on divergence
    SF-06: Speed profiles — ramp-up/ramp-down instead of instant max
    SF-07: Dynamic collision zones — load/update from WorldState at runtime
    SF-10: Battery-aware safety — refuse task if battery < return cost
    SF-11: Policy hot-swap — change policy without restarting executor
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Callable

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import SafetyPolicyRef, RobotCapability

logger = logging.getLogger(__name__)


class SafetyViolation(Exception):
    """Raised when a command violates a safety constraint."""
    pass


class EscalationTimeout(Exception):
    """Raised when human ACK is not received within the escalation timeout."""
    pass


# ---------------------------------------------------------------------------
# SF-06: Speed Profile
# ---------------------------------------------------------------------------

class SpeedProfile:
    """
    Ramp-up and ramp-down speed envelope.

    Instead of instantly applying max speed, the robot accelerates
    from zero to target speed over ramp_up_seconds, and decelerates
    over ramp_down_seconds before stopping.

    Usage:
        profile = SpeedProfile(ramp_up_s=1.0, ramp_down_s=0.5)
        effective = profile.compute(requested=1.5, elapsed=0.3, remaining_dist=0.2)
    """

    def __init__(
        self,
        ramp_up_s: float = 1.0,
        ramp_down_s: float = 0.5,
        min_speed: float = 0.05,
    ) -> None:
        self.ramp_up_s = max(0.01, ramp_up_s)
        self.ramp_down_s = max(0.01, ramp_down_s)
        self.min_speed = min_speed

    def compute(
        self,
        requested: float,
        elapsed: float = 0.0,
        remaining_dist: float | None = None,
    ) -> float:
        """
        Compute effective speed given current phase of motion.

        Args:
            requested: The target speed (already clamped by policy).
            elapsed: Seconds since the move command started.
            remaining_dist: If known, distance to goal (triggers ramp-down).

        Returns the effective speed to apply.
        """
        # Ramp-up phase
        ramp_up_factor = min(1.0, elapsed / self.ramp_up_s)
        speed = requested * ramp_up_factor

        # Ramp-down phase (if we know remaining distance)
        if remaining_dist is not None and remaining_dist > 0:
            # Decel distance = speed * ramp_down_s / 2 (constant decel)
            decel_dist = requested * self.ramp_down_s / 2.0
            if remaining_dist < decel_dist:
                ramp_down_factor = remaining_dist / decel_dist
                speed = min(speed, requested * ramp_down_factor)

        return max(self.min_speed, speed)

    def __repr__(self) -> str:
        return f"<SpeedProfile up={self.ramp_up_s}s down={self.ramp_down_s}s>"


# ---------------------------------------------------------------------------
# SF-04: Audit Entry
# ---------------------------------------------------------------------------

class SafetyAuditEntry:
    """A single safety audit record."""

    def __init__(
        self,
        event_type: str,
        robot_id: str,
        details: dict[str, Any],
        policy_name: str = "",
        timestamp: float | None = None,
    ) -> None:
        self.event_type = event_type  # violation, intervention, escalation, watchdog, etc.
        self.robot_id = robot_id
        self.details = details
        self.policy_name = policy_name
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "robot_id": self.robot_id,
            "details": self.details,
            "policy_name": self.policy_name,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Safety Policy
# ---------------------------------------------------------------------------

class SafetyPolicy:
    """
    A concrete safety policy with enforceable constraints.

    Constraints:
        - max_speed: Hard cap on movement speed (m/s)
        - max_angular_speed: Hard cap on rotation speed (rad/s)
        - collision_zones: List of rectangular zones the robot cannot enter
        - human_proximity_limit: Minimum distance to maintain from humans (m)
        - move_timeout: Max seconds a move command can run before auto-stop
        - watchdog_tolerance: Max divergence (m) between odometry and command
        - min_battery_pct: Minimum battery percentage to accept new tasks
        - speed_profile: Ramp-up/ramp-down speed envelope
    """

    def __init__(
        self,
        name: str = "default",
        max_speed: float = 1.5,
        max_angular_speed: float = 2.0,
        collision_zones: list[dict[str, float]] | None = None,
        human_proximity_limit: float = 0.5,
        move_timeout: float = 120.0,
        watchdog_tolerance: float = 2.0,
        min_battery_pct: float = 15.0,
        speed_profile: SpeedProfile | None = None,
        escalation_timeout: float = 300.0,
        watchdog_interval: float = 2.0,
    ) -> None:
        self.name = name
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        self.collision_zones = list(collision_zones or [])
        self.human_proximity_limit = human_proximity_limit
        self.move_timeout = move_timeout
        self.watchdog_tolerance = watchdog_tolerance
        self.min_battery_pct = min_battery_pct
        self.speed_profile = speed_profile
        self.escalation_timeout = escalation_timeout
        self.watchdog_interval = watchdog_interval

    @classmethod
    def from_ref(cls, ref: SafetyPolicyRef) -> SafetyPolicy:
        """Create a policy from a SafetyPolicyRef schema."""
        return cls(
            name=ref.policy_name,
            max_speed=ref.max_speed or 1.5,
            collision_zones=ref.collision_zones,
            human_proximity_limit=ref.human_proximity_limit or 0.5,
        )

    def __repr__(self) -> str:
        return (
            f"SafetyPolicy(name={self.name!r}, max_speed={self.max_speed}, "
            f"max_angular_speed={self.max_angular_speed}, "
            f"zones={len(self.collision_zones)}, proximity={self.human_proximity_limit}, "
            f"move_timeout={self.move_timeout}s)"
        )


# Default policies
DEFAULT_POLICY = SafetyPolicy(
    name="default", max_speed=1.5, max_angular_speed=2.0,
    human_proximity_limit=0.5, move_timeout=120.0,
)
STRICT_POLICY = SafetyPolicy(
    name="strict", max_speed=0.5, max_angular_speed=1.0,
    human_proximity_limit=1.0, move_timeout=60.0,
    speed_profile=SpeedProfile(ramp_up_s=2.0, ramp_down_s=1.0),
)

POLICY_REGISTRY: dict[str, SafetyPolicy] = {
    "default": DEFAULT_POLICY,
    "strict": STRICT_POLICY,
}


# ---------------------------------------------------------------------------
# Safety Enforcer
# ---------------------------------------------------------------------------

class SafetyEnforcer:
    """
    Wraps a Robot and enforces safety constraints on every command.

    Usage:
        enforcer = SafetyEnforcer(robot, policy="default")
        enforcer.move(x=2.0, y=3.0, speed=5.0)  # speed clamped to max
        enforcer.move(x=10.0, y=10.0)  # rejected if in collision zone

    The enforcer is transparent — it has the same API as Robot.
    """

    def __init__(
        self,
        robot: Robot,
        policy: str | SafetyPolicy = "default",
        world_state: Any = None,
        state_store: Any = None,
        webhook_emitter: Any = None,
        battery_monitor: Any = None,
    ) -> None:
        self._robot = robot
        if isinstance(policy, str):
            self._policy = POLICY_REGISTRY.get(policy, DEFAULT_POLICY)
        else:
            self._policy = policy
        self._violations: list[dict[str, Any]] = []
        self._interventions: list[dict[str, Any]] = []
        self._audit_log: list[SafetyAuditEntry] = []

        # SF-02/SF-07: WorldState integration
        self._world_state = world_state

        # SF-04: StateStore for persistent audit
        self._state_store = state_store

        # SF-03: Webhook for escalation
        self._webhook_emitter = webhook_emitter

        # SF-10: Battery monitor
        self._battery_monitor = battery_monitor

        # SF-01: Move timeout tracking
        self._move_timer: threading.Timer | None = None
        self._move_start_time: float | None = None
        self._lock = threading.Lock()

        # SF-05: Watchdog state
        self._last_commanded_position: tuple[float, float] | None = None
        self._watchdog_active = False
        self._watchdog_timer: threading.Timer | None = None
        self._watchdog_triggered_count = 0

        # SF-03: Escalation state
        self._escalation_event = threading.Event()
        self._escalation_pending = False

        logger.info("SafetyEnforcer active: %s", self._policy)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def robot(self) -> Robot:
        return self._robot

    @property
    def policy(self) -> SafetyPolicy:
        return self._policy

    @property
    def audit_log(self) -> list[SafetyAuditEntry]:
        return list(self._audit_log)

    # ------------------------------------------------------------------
    # SF-11: Policy hot-swap
    # ------------------------------------------------------------------

    def swap_policy(self, new_policy: str | SafetyPolicy) -> SafetyPolicy:
        """
        Hot-swap the active safety policy without restarting.

        Returns the previous policy.
        """
        old = self._policy
        if isinstance(new_policy, str):
            self._policy = POLICY_REGISTRY.get(new_policy, DEFAULT_POLICY)
        else:
            self._policy = new_policy
        self._record_audit("policy_swap", {
            "old_policy": old.name,
            "new_policy": self._policy.name,
        })
        logger.info(
            "SAFETY: Policy swapped from %s → %s", old.name, self._policy.name,
        )
        return old

    # ------------------------------------------------------------------
    # Pass-throughs
    # ------------------------------------------------------------------

    def capabilities(self, **kwargs: Any) -> RobotCapability:
        """Pass-through to robot capabilities."""
        return self._robot.capabilities(**kwargs)

    # ------------------------------------------------------------------
    # SF-01 + SF-02 + SF-06 + SF-07 + SF-10: Move with full enforcement
    # ------------------------------------------------------------------

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        """
        Move command with full safety enforcement stack.

        Enforcements (in order):
            1. Battery check (SF-10)
            2. Speed clamping + speed profile (SF-06)
            3. Static collision zone check
            4. Dynamic collision zone check (SF-07)
            5. Human proximity check (SF-02)
            6. Move timeout (SF-01)
        """
        # --- SF-10: Battery check ---
        if self._battery_monitor is not None:
            pos = self._robot.get_position()
            dx = x - pos[0]
            dy = y - pos[1]
            distance = math.sqrt(dx * dx + dy * dy)
            if not self._battery_monitor.can_complete_trip(distance, pos):
                violation = {
                    "type": "battery_insufficient",
                    "battery_pct": self._battery_monitor.percentage,
                    "distance_m": distance,
                }
                self._violations.append(violation)
                self._record_audit("violation", violation)
                raise SafetyViolation(
                    f"Battery too low ({self._battery_monitor.percentage:.0f}%) "
                    f"for {distance:.1f}m trip"
                )

            if self._battery_monitor.percentage < self._policy.min_battery_pct:
                violation = {
                    "type": "battery_below_minimum",
                    "battery_pct": self._battery_monitor.percentage,
                    "min_required": self._policy.min_battery_pct,
                }
                self._violations.append(violation)
                self._record_audit("violation", violation)
                raise SafetyViolation(
                    f"Battery ({self._battery_monitor.percentage:.0f}%) "
                    f"below minimum ({self._policy.min_battery_pct:.0f}%)"
                )

        # --- Speed enforcement + SF-06: Speed profile ---
        original_speed = speed
        if speed is not None and speed > self._policy.max_speed:
            speed = self._policy.max_speed
            intervention = {
                "type": "speed_clamped",
                "requested": original_speed,
                "enforced": speed,
                "max_allowed": self._policy.max_speed,
            }
            self._interventions.append(intervention)
            self._record_audit("intervention", intervention)
            logger.warning(
                "SAFETY: Speed clamped from %.2f to %.2f m/s (policy: %s)",
                original_speed, speed, self._policy.name,
            )

        if self._policy.speed_profile is not None and speed is not None:
            speed = self._policy.speed_profile.compute(
                requested=speed, elapsed=0.0,
            )

        # --- Static collision zone enforcement ---
        for zone in self._policy.collision_zones:
            if self._point_in_zone(x, y, zone):
                violation = {
                    "type": "collision_zone",
                    "requested_position": (x, y),
                    "zone": zone,
                }
                self._violations.append(violation)
                self._record_audit("violation", violation)
                logger.error(
                    "SAFETY: Command REJECTED — position (%.2f, %.2f) is inside "
                    "collision zone %s (policy: %s)",
                    x, y, zone, self._policy.name,
                )
                raise SafetyViolation(
                    f"Position ({x}, {y}) is inside collision zone: {zone}"
                )

        # --- SF-07: Dynamic collision zones from WorldState ---
        if self._world_state is not None:
            dynamic_zones = self._get_dynamic_zones()
            for zone in dynamic_zones:
                if self._point_in_zone(x, y, zone):
                    violation = {
                        "type": "dynamic_collision_zone",
                        "requested_position": (x, y),
                        "zone": zone,
                    }
                    self._violations.append(violation)
                    self._record_audit("violation", violation)
                    raise SafetyViolation(
                        f"Position ({x}, {y}) blocked by dynamic obstacle zone: {zone}"
                    )

        # --- SF-02: Human proximity enforcement ---
        if self._world_state is not None:
            self._check_human_proximity(x, y)

        # --- SF-05: Track commanded position for watchdog ---
        self._last_commanded_position = (x, y)
        self._move_start_time = time.time()

        # --- SF-01: Start move timeout timer ---
        self._cancel_move_timer()
        self._move_timer = threading.Timer(
            self._policy.move_timeout, self._on_move_timeout
        )
        self._move_timer.daemon = True
        self._move_timer.start()

        # --- SF-05: Start watchdog timer for periodic odometry check ---
        self._start_move_watchdog()

        # --- All checks passed — forward to robot ---
        self._robot.move(x=x, y=y, speed=speed)

    # ------------------------------------------------------------------
    # SF-01: Move timeout
    # ------------------------------------------------------------------

    def _on_move_timeout(self) -> None:
        """Called by Timer when a move exceeds the configured timeout."""
        logger.error(
            "SAFETY: Move timeout (%.0fs) — auto-stopping robot %s",
            self._policy.move_timeout, self._robot.robot_id,
        )
        intervention = {
            "type": "move_timeout",
            "timeout_s": self._policy.move_timeout,
            "commanded_position": self._last_commanded_position,
        }
        self._interventions.append(intervention)
        self._record_audit("intervention", intervention)
        try:
            self._robot.stop()
        except Exception as e:
            logger.error("SAFETY: Failed to stop robot on timeout: %s", e)

    def _cancel_move_timer(self) -> None:
        """Cancel any pending move timeout timer."""
        if self._move_timer is not None:
            self._move_timer.cancel()
            self._move_timer = None

    # ------------------------------------------------------------------
    # Rotate with angular speed enforcement
    # ------------------------------------------------------------------

    def rotate(self, angle_rad: float, speed: float | None = None) -> None:
        """Rotate command with angular speed enforcement."""
        original_speed = speed
        if speed is not None and abs(speed) > self._policy.max_angular_speed:
            speed = math.copysign(self._policy.max_angular_speed, speed)
            intervention = {
                "type": "angular_speed_clamped",
                "requested": original_speed,
                "enforced": speed,
                "max_allowed": self._policy.max_angular_speed,
            }
            self._interventions.append(intervention)
            self._record_audit("intervention", intervention)
            logger.warning(
                "SAFETY: Angular speed clamped from %.2f to %.2f rad/s (policy: %s)",
                original_speed, speed, self._policy.name,
            )
        self._robot.rotate(angle_rad=angle_rad, speed=speed)

    # ------------------------------------------------------------------
    # Stop / cancel
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Stop is always allowed — safety never prevents stopping."""
        self._cancel_move_timer()
        self._cancel_watchdog_timer()
        self._robot.stop()

    def cancel(self) -> None:
        """Cancel is always allowed."""
        self._cancel_move_timer()
        self._cancel_watchdog_timer()
        self._robot.cancel()

    # ------------------------------------------------------------------
    # Gripper pass-throughs
    # ------------------------------------------------------------------

    def gripper_open(self) -> bool:
        return self._robot.gripper_open()

    def gripper_close(self) -> bool:
        return self._robot.gripper_close()

    # ------------------------------------------------------------------
    # State query pass-throughs
    # ------------------------------------------------------------------

    def get_position(self) -> tuple[float, float]:
        return self._robot.get_position()

    def get_orientation(self) -> float:
        return self._robot.get_orientation()

    def get_health(self) -> dict[str, Any]:
        return self._robot.get_health()

    # ------------------------------------------------------------------
    # Lifecycle pass-throughs
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._robot.connect()

    def disconnect(self) -> None:
        self._cancel_move_timer()
        self._cancel_watchdog_timer()
        self._robot.disconnect()

    # ------------------------------------------------------------------
    # SF-02: Human proximity enforcement
    # ------------------------------------------------------------------

    def _check_human_proximity(self, target_x: float, target_y: float) -> None:
        """
        Check if any detected humans are too close to the target position.

        Uses WorldState.detected_objects with label 'person' or 'human'.
        """
        ws = self._world_state
        if ws is None:
            return

        human_labels = {"person", "human", "pedestrian"}
        limit = self._policy.human_proximity_limit

        for obj in ws.detected_objects:
            if obj.label.lower() not in human_labels:
                continue
            dist = math.sqrt((obj.x - target_x) ** 2 + (obj.y - target_y) ** 2)
            if dist < limit:
                violation = {
                    "type": "human_proximity",
                    "human_position": (obj.x, obj.y),
                    "target_position": (target_x, target_y),
                    "distance": dist,
                    "limit": limit,
                    "human_id": obj.object_id,
                }
                self._violations.append(violation)
                self._record_audit("violation", violation)
                logger.error(
                    "SAFETY: Human detected %.2fm from target (limit: %.2fm) — "
                    "REJECTING move to (%.2f, %.2f)",
                    dist, limit, target_x, target_y,
                )
                raise SafetyViolation(
                    f"Human detected {dist:.2f}m from target position "
                    f"({target_x}, {target_y}) — minimum distance is {limit}m"
                )

    # ------------------------------------------------------------------
    # SF-03: ESCALATE recovery
    # ------------------------------------------------------------------

    def escalate(self, reason: str, context: dict[str, Any] | None = None) -> bool:
        """
        Trigger an escalation: stop robot, send webhook, wait for human ACK.

        Returns True if human acknowledged within timeout, False otherwise.
        """
        self._robot.stop()
        self._cancel_move_timer()
        self._escalation_pending = True
        self._escalation_event.clear()

        escalation_data = {
            "type": "escalation",
            "reason": reason,
            "robot_id": self._robot.robot_id,
            "position": self._robot.get_position(),
            "context": context or {},
        }

        self._record_audit("escalation", escalation_data)

        # Send webhook notification
        if self._webhook_emitter is not None:
            self._webhook_emitter.emit("safety_escalation", **escalation_data)
            logger.warning(
                "SAFETY ESCALATION: %s — webhook sent, waiting for human ACK "
                "(timeout: %.0fs)", reason, self._policy.escalation_timeout,
            )
        else:
            logger.warning(
                "SAFETY ESCALATION: %s — no webhook configured, "
                "waiting for programmatic ACK (timeout: %.0fs)",
                reason, self._policy.escalation_timeout,
            )

        # Wait for human acknowledgment
        acked = self._escalation_event.wait(timeout=self._policy.escalation_timeout)
        self._escalation_pending = False

        if acked:
            self._record_audit("escalation_ack", {"reason": reason})
            logger.info("SAFETY: Escalation acknowledged — resuming")
            return True
        else:
            self._record_audit("escalation_timeout", {
                "reason": reason,
                "timeout_s": self._policy.escalation_timeout,
            })
            logger.error("SAFETY: Escalation timed out — robot remains stopped")
            return False

    def acknowledge_escalation(self) -> None:
        """Human operator acknowledges the escalation, allowing robot to resume."""
        self._escalation_event.set()

    @property
    def is_escalation_pending(self) -> bool:
        return self._escalation_pending

    # ------------------------------------------------------------------
    # SF-04: Audit log persistence
    # ------------------------------------------------------------------

    def _record_audit(self, event_type: str, details: dict[str, Any]) -> None:
        """Record an audit entry and persist to StateStore if available."""
        entry = SafetyAuditEntry(
            event_type=event_type,
            robot_id=self._robot.robot_id,
            details=details,
            policy_name=self._policy.name,
        )
        self._audit_log.append(entry)

        # Persist to StateStore
        if self._state_store is not None:
            try:
                existing = self._state_store.get("safety_audit_log", [])
                existing.append(entry.to_dict())
                self._state_store.set("safety_audit_log", existing)
            except Exception as e:
                logger.warning("Failed to persist audit entry: %s", e)

    # ------------------------------------------------------------------
    # SF-05: Runtime watchdog — odometry divergence detection
    # ------------------------------------------------------------------

    def _start_move_watchdog(self) -> None:
        """Start the periodic watchdog timer after a move command."""
        self._cancel_watchdog_timer()
        self._watchdog_active = True
        self._reschedule_watchdog()

    def _cancel_watchdog_timer(self) -> None:
        """Cancel any pending watchdog timer."""
        self._watchdog_active = False
        if self._watchdog_timer is not None:
            self._watchdog_timer.cancel()
            self._watchdog_timer = None

    def _reschedule_watchdog(self) -> None:
        """Schedule the next watchdog check using threading.Timer."""
        if not self._watchdog_active:
            return
        self._watchdog_timer = threading.Timer(
            self._policy.watchdog_interval, self._watchdog_check,
        )
        self._watchdog_timer.daemon = True
        self._watchdog_timer.start()

    def _watchdog_check(self) -> None:
        """
        Timer callback: compare actual odometry vs commanded position.

        If divergence exceeds tolerance, stop the robot, record audit,
        and raise SafetyViolation. Otherwise, reschedule for the next check.
        """
        if self._last_commanded_position is None:
            self._reschedule_watchdog()
            return

        actual = self._robot.get_position()
        cx, cy = self._last_commanded_position
        dist = math.sqrt((actual[0] - cx) ** 2 + (actual[1] - cy) ** 2)

        if dist > self._policy.watchdog_tolerance:
            self._watchdog_triggered_count += 1
            logger.error(
                "SAFETY WATCHDOG: Position divergence %.2fm exceeds tolerance "
                "%.2fm — E-STOP triggered",
                dist, self._policy.watchdog_tolerance,
            )
            self._robot.stop()
            self._cancel_move_timer()
            intervention = {
                "type": "watchdog_triggered",
                "divergence_m": dist,
                "tolerance_m": self._policy.watchdog_tolerance,
                "commanded": self._last_commanded_position,
                "actual": actual,
            }
            self._interventions.append(intervention)
            self._record_audit("watchdog_triggered", intervention)
            self._watchdog_active = False
            raise SafetyViolation(
                f"Divergence {dist:.2f}m > tolerance "
                f"{self._policy.watchdog_tolerance}m"
            )

        self._reschedule_watchdog()

    def check_watchdog(self) -> dict[str, Any] | None:
        """
        Compare actual odometry position against last commanded position.

        Returns a dict with divergence info, or None if everything is OK.
        Calls stop() if divergence exceeds watchdog_tolerance.
        """
        if self._last_commanded_position is None:
            return None

        actual = self._robot.get_position()
        cmd_x, cmd_y = self._last_commanded_position
        act_x, act_y = actual
        divergence = math.sqrt((act_x - cmd_x) ** 2 + (act_y - cmd_y) ** 2)

        result = {
            "commanded": self._last_commanded_position,
            "actual": actual,
            "divergence_m": divergence,
            "tolerance_m": self._policy.watchdog_tolerance,
            "ok": divergence <= self._policy.watchdog_tolerance,
        }

        if not result["ok"]:
            self._watchdog_triggered_count += 1
            logger.error(
                "SAFETY WATCHDOG: Position divergence %.2fm exceeds tolerance "
                "%.2fm — E-STOP triggered",
                divergence, self._policy.watchdog_tolerance,
            )
            intervention = {
                "type": "watchdog_triggered",
                "divergence_m": divergence,
                "tolerance_m": self._policy.watchdog_tolerance,
                "commanded": self._last_commanded_position,
                "actual": actual,
            }
            self._interventions.append(intervention)
            self._record_audit("watchdog_triggered", intervention)
            self._robot.stop()
            self._cancel_move_timer()
            self._cancel_watchdog_timer()

        return result

    def start_watchdog(self, interval: float | None = None) -> threading.Timer:
        """
        Start a periodic watchdog that checks odometry vs commanded position.

        Args:
            interval: Check interval in seconds. Defaults to policy watchdog_interval.

        Returns the Timer object so the caller can cancel it.
        """
        if interval is not None:
            self._policy.watchdog_interval = interval
        self._watchdog_active = True

        def _watchdog_loop() -> None:
            while self._watchdog_active:
                self.check_watchdog()
                time.sleep(self._policy.watchdog_interval)

        t = threading.Thread(target=_watchdog_loop, daemon=True)
        t.start()
        return t  # type: ignore[return-value]

    def stop_watchdog(self) -> None:
        """Stop the periodic watchdog."""
        self._cancel_watchdog_timer()

    @property
    def watchdog_triggered_count(self) -> int:
        """Number of times the watchdog has triggered an E-STOP."""
        return self._watchdog_triggered_count

    # ------------------------------------------------------------------
    # SF-07: Dynamic collision zones from WorldState
    # ------------------------------------------------------------------

    def _get_dynamic_zones(self) -> list[dict[str, float]]:
        """
        Convert WorldState obstacles into rectangular collision zones.

        Each obstacle at (x, y) with radius r becomes a zone:
            x_min=x-r, x_max=x+r, y_min=y-r, y_max=y+r
        """
        ws = self._world_state
        if ws is None:
            return []

        zones = []
        for obs in ws.obstacles:
            zones.append({
                "x_min": obs.x - obs.radius,
                "x_max": obs.x + obs.radius,
                "y_min": obs.y - obs.radius,
                "y_max": obs.y + obs.radius,
                "dynamic": True,
                "source": obs.source,
            })
        return zones

    def add_collision_zone(self, zone: dict[str, float]) -> None:
        """Add a collision zone at runtime (SF-07)."""
        self._policy.collision_zones.append(zone)
        self._record_audit("zone_added", zone)
        logger.info("SAFETY: Added collision zone: %s", zone)

    def remove_collision_zone(self, index: int) -> dict[str, float] | None:
        """Remove a collision zone by index."""
        if 0 <= index < len(self._policy.collision_zones):
            removed = self._policy.collision_zones.pop(index)
            self._record_audit("zone_removed", removed)
            return removed
        return None

    def update_world_state(self, world_state: Any) -> None:
        """Update the WorldState used for dynamic checks (SF-07)."""
        self._world_state = world_state

    # ------------------------------------------------------------------
    # SF-10: Battery-aware safety
    # ------------------------------------------------------------------

    def set_battery_monitor(self, monitor: Any) -> None:
        """Attach a BatteryMonitor for battery-aware safety checks."""
        self._battery_monitor = monitor

    def check_battery(self, distance_m: float = 0.0) -> dict[str, Any]:
        """
        Check battery status.

        Returns a dict with battery info and whether the task is allowed.
        """
        if self._battery_monitor is None:
            return {"available": True, "reason": "no_monitor"}

        pos = self._robot.get_position()
        can_trip = self._battery_monitor.can_complete_trip(distance_m, pos)
        above_min = self._battery_monitor.percentage >= self._policy.min_battery_pct

        return {
            "available": can_trip and above_min,
            "battery_pct": self._battery_monitor.percentage,
            "estimated_range_m": self._battery_monitor.estimated_range_m,
            "can_complete_trip": can_trip,
            "above_minimum": above_min,
            "min_required_pct": self._policy.min_battery_pct,
            "status": self._battery_monitor.status,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _point_in_zone(x: float, y: float, zone: dict[str, float]) -> bool:
        """Check if (x, y) falls inside a rectangular zone."""
        x_min = zone.get("x_min", float("-inf"))
        x_max = zone.get("x_max", float("inf"))
        y_min = zone.get("y_min", float("-inf"))
        y_max = zone.get("y_max", float("inf"))
        return x_min <= x <= x_max and y_min <= y <= y_max

    @property
    def violations(self) -> list[dict[str, Any]]:
        """All safety violations logged during this session."""
        return list(self._violations)

    @property
    def interventions(self) -> list[dict[str, Any]]:
        """All safety interventions (speed clamping, timeout, etc.)."""
        return list(self._interventions)

    @property
    def robot_id(self) -> str:
        return self._robot.robot_id

    def __repr__(self) -> str:
        return f"<SafetyEnforcer robot={self._robot.robot_id!r} policy={self._policy.name!r}>"


# ---------------------------------------------------------------------------
# SF-12: Formal Constraint Export (TLA+ / UPPAAL)
# ---------------------------------------------------------------------------

class FormalConstraintExporter:
    """
    Exports a SafetyPolicy as formal verification specs.

    Generates:
        - TLA+ specification for model checking
        - UPPAAL timed automaton XML for temporal verification

    These artifacts satisfy grant committees that require formal verification.

    Usage:
        exporter = FormalConstraintExporter(policy)
        tla_spec = exporter.to_tlaplus()
        uppaal_xml = exporter.to_uppaal()
    """

    def __init__(self, policy: SafetyPolicy) -> None:
        self._policy = policy

    def to_tlaplus(self) -> str:
        """Generate a TLA+ specification from the safety policy."""
        p = self._policy
        zones_tla = "{"
        for i, z in enumerate(p.collision_zones):
            zones_tla += (
                f'\n    [x_min |-> {z.get("x_min", "-Infinity")}, '
                f'x_max |-> {z.get("x_max", "Infinity")}, '
                f'y_min |-> {z.get("y_min", "-Infinity")}, '
                f'y_max |-> {z.get("y_max", "Infinity")}]'
            )
            if i < len(p.collision_zones) - 1:
                zones_tla += ","
        zones_tla += "\n  }" if p.collision_zones else "}"

        return f"""---- MODULE SafetyPolicy_{p.name} ----
\\* Auto-generated from APYROBO SafetyPolicy: {p.name}
\\* Generated for formal verification of safety constraints.

EXTENDS Integers, Reals, Sequences

CONSTANTS
    MaxSpeed,           \\* {p.max_speed} m/s
    MaxAngularSpeed,    \\* {p.max_angular_speed} rad/s
    HumanProximityLimit,\\* {p.human_proximity_limit} m
    MoveTimeout,        \\* {p.move_timeout} s
    WatchdogTolerance,  \\* {p.watchdog_tolerance} m
    MinBatteryPct       \\* {p.min_battery_pct} %

VARIABLES
    robotPos,       \\* <<x, y>> current robot position
    robotSpeed,     \\* current linear speed (m/s)
    angularSpeed,   \\* current angular speed (rad/s)
    batteryPct,     \\* current battery percentage
    moveActive,     \\* whether a move command is active
    moveStartTime,  \\* when the current move started
    commandedPos,   \\* target position of current move
    humans          \\* set of <<x, y>> positions of detected humans

CollisionZones == {zones_tla}

\\* --- Safety Invariants ---

SpeedInvariant ==
    /\\ robotSpeed <= MaxSpeed
    /\\ robotSpeed >= 0

AngularSpeedInvariant ==
    /\\ angularSpeed <= MaxAngularSpeed
    /\\ angularSpeed >= -MaxAngularSpeed

CollisionFreedom ==
    \\A zone \\in CollisionZones :
        ~(zone.x_min <= robotPos[1] /\\ robotPos[1] <= zone.x_max
          /\\ zone.y_min <= robotPos[2] /\\ robotPos[2] <= zone.y_max)

HumanSafety ==
    \\A h \\in humans :
        Sqrt((robotPos[1] - h[1])^2 + (robotPos[2] - h[2])^2)
            >= HumanProximityLimit

MoveTimeoutSafety ==
    moveActive =>
        (CurrentTime - moveStartTime) <= MoveTimeout

BatterySafety ==
    moveActive => batteryPct >= MinBatteryPct

WatchdogSafety ==
    moveActive =>
        Sqrt((robotPos[1] - commandedPos[1])^2
           + (robotPos[2] - commandedPos[2])^2)
            <= WatchdogTolerance

\\* --- Combined Safety Property ---

SafetyInvariant ==
    /\\ SpeedInvariant
    /\\ AngularSpeedInvariant
    /\\ CollisionFreedom
    /\\ HumanSafety
    /\\ MoveTimeoutSafety
    /\\ BatterySafety
    /\\ WatchdogSafety

====
"""

    def to_uppaal(self) -> str:
        """Generate a UPPAAL timed automaton XML from the safety policy."""
        p = self._policy
        zones_xml = ""
        for i, z in enumerate(p.collision_zones):
            zones_xml += (
                f'    <zone id="z{i}" '
                f'x_min="{z.get("x_min", 0)}" x_max="{z.get("x_max", 0)}" '
                f'y_min="{z.get("y_min", 0)}" y_max="{z.get("y_max", 0)}"/>\n'
            )

        return f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.1//EN'
    'http://www.it.uu.se/research/group/darts/uppaal/flat-1_2.dtd'>
<!-- Auto-generated from APYROBO SafetyPolicy: {p.name} -->
<nta>
  <declaration>
// Safety policy constants
const double MAX_SPEED = {p.max_speed};
const double MAX_ANGULAR_SPEED = {p.max_angular_speed};
const double HUMAN_PROXIMITY_LIMIT = {p.human_proximity_limit};
const int MOVE_TIMEOUT = {int(p.move_timeout)};
const double WATCHDOG_TOLERANCE = {p.watchdog_tolerance};
const double MIN_BATTERY_PCT = {p.min_battery_pct};

// Robot state
double robot_x, robot_y;
double robot_speed;
double angular_speed;
double battery_pct;
clock move_clock;
bool move_active;
  </declaration>

  <template>
    <name>SafetyEnforcer</name>
    <declaration>
// Local declarations
    </declaration>

    <location id="idle" x="0" y="0">
      <name>Idle</name>
      <label kind="invariant">!move_active</label>
    </location>

    <location id="moving" x="200" y="0">
      <name>Moving</name>
      <label kind="invariant">
        move_active &amp;&amp;
        move_clock &lt;= {int(p.move_timeout)} &amp;&amp;
        robot_speed &lt;= {p.max_speed} &amp;&amp;
        battery_pct &gt;= {p.min_battery_pct}
      </label>
    </location>

    <location id="stopped" x="400" y="0">
      <name>EmergencyStopped</name>
    </location>

    <init ref="idle"/>

    <transition>
      <source ref="idle"/>
      <target ref="moving"/>
      <label kind="guard">battery_pct &gt;= {p.min_battery_pct}</label>
      <label kind="assignment">move_active = true, move_clock = 0</label>
    </transition>

    <transition>
      <source ref="moving"/>
      <target ref="idle"/>
      <label kind="guard">true</label>
      <label kind="assignment">move_active = false</label>
    </transition>

    <transition>
      <source ref="moving"/>
      <target ref="stopped"/>
      <label kind="guard">move_clock &gt;= {int(p.move_timeout)}</label>
      <label kind="assignment">move_active = false, robot_speed = 0</label>
    </transition>

    <transition>
      <source ref="moving"/>
      <target ref="stopped"/>
      <label kind="guard">battery_pct &lt; {p.min_battery_pct}</label>
      <label kind="assignment">move_active = false, robot_speed = 0</label>
    </transition>

    <transition>
      <source ref="stopped"/>
      <target ref="idle"/>
      <label kind="assignment">robot_speed = 0</label>
    </transition>
  </template>

  <system>
    system SafetyEnforcer;
  </system>

  <queries>
    <query>
      <formula>A[] (robot_speed &lt;= {p.max_speed})</formula>
      <comment>Speed never exceeds maximum</comment>
    </query>
    <query>
      <formula>A[] (move_active imply move_clock &lt;= {int(p.move_timeout)})</formula>
      <comment>Move always terminates within timeout</comment>
    </query>
    <query>
      <formula>A[] (move_active imply battery_pct &gt;= {p.min_battery_pct})</formula>
      <comment>Robot never moves with insufficient battery</comment>
    </query>
  </queries>
</nta>
"""

    def to_dict(self) -> dict[str, Any]:
        """Export the policy constraints as a structured dict."""
        p = self._policy
        return {
            "policy_name": p.name,
            "constraints": {
                "max_speed_ms": p.max_speed,
                "max_angular_speed_rads": p.max_angular_speed,
                "human_proximity_limit_m": p.human_proximity_limit,
                "move_timeout_s": p.move_timeout,
                "watchdog_tolerance_m": p.watchdog_tolerance,
                "min_battery_pct": p.min_battery_pct,
                "collision_zones": p.collision_zones,
            },
            "has_speed_profile": p.speed_profile is not None,
            "escalation_timeout_s": p.escalation_timeout,
        }
