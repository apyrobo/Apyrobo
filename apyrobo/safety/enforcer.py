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

import copy
import logging
import math
import threading
import time
from collections.abc import Sequence
from typing import Any, Callable, NoReturn

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
        - collision_zones: List of rectangular zones the robot cannot enter;
          a zone with min > max or a bound that is not a number raises
          ``ValueError``
        - allowed_areas: Rectangles a move target must fall in; empty means
          anywhere outside the collision zones
        - human_proximity_limit: Minimum distance to maintain from humans (m)
        - move_timeout: Max seconds a move command can run before auto-stop
        - watchdog_tolerance: Max divergence (m) between odometry and command
        - min_battery_pct: Minimum battery percentage to accept new tasks
        - speed_profile: Ramp-up/ramp-down speed envelope, not applied to moves
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
        allowed_areas: list[dict[str, float]] | None = None,
    ) -> None:
        self.name = name
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        self.collision_zones = list(collision_zones or [])
        self.allowed_areas = list(allowed_areas or [])
        for zone in self.collision_zones:
            check_area(zone, "collision zone")
        for area in self.allowed_areas:
            check_area(area, "allowed area")
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

#: A gap between two areas narrower than this share of a segment is rounding.
_SPAN_TOLERANCE = 1e-9


def describe_area(area: dict[str, Any]) -> str:
    """A zone or allowed area in words: ``'dock' (x 2 to 4, y 2 to 4)``."""
    bounds = (
        f"(x {area.get('x_min', float('-inf')):g} to "
        f"{area.get('x_max', float('inf')):g}, "
        f"y {area.get('y_min', float('-inf')):g} to "
        f"{area.get('y_max', float('inf')):g})"
    )
    name = area.get("name")
    return f"'{name}' {bounds}" if name else bounds


_BOUND_DEFAULTS = (
    ("x_min", float("-inf")), ("x_max", float("inf")),
    ("y_min", float("-inf")), ("y_max", float("inf")),
)


def check_area(area: Any, what: str = "collision zone") -> None:
    """
    Raise ``ValueError`` for a zone or allowed area the checks cannot use.

    A bound may be left out (it is then open on that side), but a bound
    that is given must be a number other than NaN, and each min must not
    exceed its max. An inverted zone would otherwise block nothing.
    """
    if not isinstance(area, dict):
        raise ValueError(f"{what} must be a dict of x_min/x_max/y_min/y_max, got {area!r}")
    for key, _ in _BOUND_DEFAULTS:
        if key not in area:
            continue
        value = area[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
            raise ValueError(f"{what} {key} must be a number, got {value!r}")
    bounds = {key: area.get(key, default) for key, default in _BOUND_DEFAULTS}
    if bounds["x_min"] > bounds["x_max"] or bounds["y_min"] > bounds["y_max"]:
        raise ValueError(f"{what} has min > max: {area!r}")


def segment_enters_zone(
    start: tuple[float, float],
    end: tuple[float, float],
    zone: dict[str, Any],
) -> bool:
    """
    True if the straight line from ``start`` to ``end`` enters ``zone``.

    Liang-Barsky clipping against the rectangle, edges included as in
    ``SafetyEnforcer._point_in_zone``. A start strictly inside the zone is
    never "entering": a robot that is already in a zone may drive out of it.
    A start on the edge may leave, but not drive in across the zone.
    """
    x0, y0 = start
    if (
        zone.get("x_min", float("-inf")) < x0 < zone.get("x_max", float("inf"))
        and zone.get("y_min", float("-inf")) < y0 < zone.get("y_max", float("inf"))
    ):
        return False
    span = _clip_segment(start, end, zone)
    return span is not None and span[1] > 0.0


def _clip_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    rect: dict[str, Any],
) -> tuple[float, float] | None:
    """The stretch of ``start``->``end`` inside ``rect``, as fractions, or None."""
    x0, y0 = start
    dx, dy = end[0] - x0, end[1] - y0
    x_min = rect.get("x_min", float("-inf"))
    x_max = rect.get("x_max", float("inf"))
    y_min = rect.get("y_min", float("-inf"))
    y_max = rect.get("y_max", float("inf"))

    t_in, t_out = 0.0, 1.0
    for p, q in (
        (-dx, x0 - x_min), (dx, x_max - x0),
        (-dy, y0 - y_min), (dy, y_max - y0),
    ):
        if p == 0:
            if q < 0:
                return None
        elif p < 0:
            t_in = max(t_in, q / p)
        else:
            t_out = min(t_out, q / p)
        if t_in > t_out:
            return None
    return t_in, t_out


def route_leaves_areas(
    route: Sequence[tuple[float, float]],
    areas: Sequence[dict[str, Any]],
) -> tuple[float, float] | None:
    """
    Return the first point where ``route`` leaves the union of ``areas``, or None.

    Areas that touch or overlap count as one space, edges included. A route
    that starts outside every area may drive in: only the stretch after it
    first enters one is checked.
    """
    entered = False
    for start, end in zip(route, route[1:], strict=False):
        clipped = (_clip_segment(start, end, area) for area in areas)
        spans = sorted(span for span in clipped if span is not None)
        if not spans:
            if entered:
                return start
            continue
        reached = 0.0 if entered else spans[0][0]
        entered = True
        for t_in, t_out in spans:
            if t_in > reached + _SPAN_TOLERANCE:
                break
            reached = max(reached, t_out)
        if reached < 1.0 - _SPAN_TOLERANCE:
            return (
                start[0] + (end[0] - start[0]) * reached,
                start[1] + (end[1] - start[1]) * reached,
            )
    return None


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
        # The speed of the last command handed to the robot, after all limits.
        self.last_speed: float | None = None
        # How many moves were handed to the robot (rotations not counted).
        self.moves_sent = 0

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
        """Run a move through the safety stack; a refusal keeps its reason.

        The reason is stored on the violation because a caller going through
        SkillExecutor gets no exception text back.
        """
        try:
            self._enforced_move(x, y, speed)
        except SafetyViolation as exc:
            if self._violations:
                self._violations[-1].setdefault("message", str(exc))
            raise

    def _enforced_move(self, x: float, y: float, speed: float | None = None) -> None:
        """
        Move command with full safety enforcement stack.

        Enforcements (in order):
            1. Battery check (SF-10)
            2. Speed clamping + speed profile (SF-06)
            3. Static collision zone check (target)
            4. Allowed area check (target), when any are set
            5. Dynamic collision zone check (SF-07, target)
            6. Path check: the route must not enter a zone or leave the
               allowed areas. That is the planner's route on adapters that
               plan their own, refused if none comes back, and the straight
               line on the others
            7. Human proximity check (SF-02)
            8. Move timeout (SF-01)
        """
        self._reject_bad_numbers("move", x=x, y=y, speed=speed)
        if speed is not None and speed < 0:
            self._reject_bad_value("move", f"speed must be 0 or more, got {speed!r}")

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

        # --- Speed enforcement ---
        # A SpeedProfile is not applied here: a single move has no elapsed time
        # to ramp over, and elapsed=0 would pin every move at min_speed.
        original_speed = speed
        if not speed:
            # No speed (or 0) means "the adapter's default". Hand the robot an
            # explicit speed so that default can't exceed the policy: the
            # adapter's own default when it is within the limit, else the limit.
            # An adapter whose robot picks its own speed gets the limit.
            default = self._adapter_default_speed()
            limit = self._policy.max_speed
            speed = limit if default is None else min(default, limit)
            if default is not None and default > limit:
                intervention = {
                    "type": "speed_clamped",
                    "requested": None,
                    "default": default,
                    "enforced": speed,
                    "max_allowed": limit,
                }
                self._interventions.append(intervention)
                self._record_audit("intervention", intervention)
                logger.warning(
                    "SAFETY: Default speed %.2f lowered to %.2f m/s (policy: %s)",
                    default, speed, self._policy.name,
                )
        elif speed > self._policy.max_speed:
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
                    f"Position ({x:g}, {y:g}) is inside collision zone "
                    f"{describe_area(zone)}"
                )

        # --- Allowed areas: deny by default once any are set ---
        if self.outside_allowed_areas(x, y):
            violation = {
                "type": "outside_allowed_area",
                "requested_position": (x, y),
                "allowed_areas": list(self._policy.allowed_areas),
            }
            self._violations.append(violation)
            self._record_audit("violation", violation)
            logger.error(
                "SAFETY: Command REJECTED — position (%.2f, %.2f) is outside every "
                "allowed area (policy: %s)",
                x, y, self._policy.name,
            )
            raise SafetyViolation(
                f"Position ({x:g}, {y:g}) is outside every allowed area: "
                + ", ".join(describe_area(a) for a in self._policy.allowed_areas)
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
                        f"Position ({x:g}, {y:g}) blocked by dynamic obstacle zone "
                        f"{describe_area(zone)}"
                    )

        # --- Path check: a safe target does not make a safe route ---
        self._check_path(x, y)

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
        self.last_speed = speed
        self.moves_sent += 1
        self._robot.move(x=x, y=y, speed=speed)

    def _adapter_default_speed(self) -> float | None:
        """The robot's speed for a move sent without one, or None if unknown."""
        value = getattr(self._robot, "default_speed", None)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(value) or value <= 0:
            return None
        return float(value)

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
        self._reject_bad_numbers("rotate", angle_rad=angle_rad, speed=speed)
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
        self.last_speed = speed
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
                "type": "watchdog_estop",
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

    def route_fault(self, x: float, y: float) -> dict[str, Any] | None:
        """
        Return what is wrong with the route to (x, y), or None if it is clear.

        The answer is ``{"zone": <first zone entered>}`` or
        ``{"left_at": (x, y)}``, the point where the route leaves the allowed
        areas. The route is the planner's when the adapter plans its own,
        otherwise the straight line from the current position. With no zones
        and no allowed areas no route is fetched. Raises SafetyViolation when
        the route is needed and cannot be known: no plan came back, or the
        position is unreadable.
        """
        zones = list(self._policy.collision_zones)
        if self._world_state is not None:
            zones += self._get_dynamic_zones()
        areas = self._policy.allowed_areas
        if not zones and not areas:
            return None
        if getattr(self._robot, "plans_own_path", False) is True:
            route = self._planned_route(x, y)
        else:
            route = [self._read_position(), (x, y)]
        for start, end in zip(route, route[1:], strict=False):
            for zone in zones:
                if segment_enters_zone(start, end, zone):
                    return {"zone": zone}
        left_at = route_leaves_areas(route, areas)
        return None if left_at is None else {"left_at": left_at}

    def _planned_route(self, x: float, y: float) -> list[tuple[float, float]]:
        """The adapter's planned route as finite points, or SafetyViolation."""
        try:
            route = [(float(px), float(py)) for px, py in self._robot.plan_path(x, y)]
        except Exception as exc:
            self._reject_path(f"Cannot check the route: no plan from the robot ({exc})")
        if not route:
            self._reject_path("Cannot check the route: the planner returned an empty path")
        if not all(math.isfinite(v) for point in route for v in point):
            self._reject_path("Cannot check the route: the plan has non-finite points")
        return route

    def outside_allowed_areas(self, x: float, y: float) -> bool:
        """True if allowed areas are set and (x, y) is in none of them."""
        areas = self._policy.allowed_areas
        return bool(areas) and not any(self._point_in_zone(x, y, a) for a in areas)

    def _read_position(self) -> tuple[float, float]:
        """Current position as two finite floats, or SafetyViolation."""
        try:
            px, py = self._robot.get_position()
            start = (float(px), float(py))
        except Exception as exc:
            self._reject_path(f"Cannot check the route: position unavailable ({exc})")
        if not all(math.isfinite(v) for v in start):
            self._reject_path(f"Cannot check the route: position is {start}")
        return start

    def _reject_path(self, message: str) -> NoReturn:
        violation = {"type": "path_unchecked", "reason": message}
        self._violations.append(violation)
        self._record_audit("violation", violation)
        raise SafetyViolation(message)

    def move_problem(
        self, x: float, y: float, start: tuple[float, float] | None = None
    ) -> str | None:
        """Why move(x, y) would be refused by the zones, or None. Moves nothing.

        Checks the target against the no-go zones, the allowed areas and the
        dynamic zones, then the straight route from *start* (default: the
        current position), with the same wording move() uses. Records no
        violation. On an adapter that plans its own route the route is not
        checked here: only move() asks the planner for it. Battery and human
        proximity are not checked either.
        """
        for name, value in (("x", x), ("y", y)):
            if not math.isfinite(value):
                return f"Invalid move command: {name} must be a finite number, got {value!r}"
        for zone in self._policy.collision_zones:
            if self._point_in_zone(x, y, zone):
                return f"Position ({x:g}, {y:g}) is inside collision zone {describe_area(zone)}"
        areas = self._policy.allowed_areas
        if self.outside_allowed_areas(x, y):
            return f"Position ({x:g}, {y:g}) is outside every allowed area: " + ", ".join(
                describe_area(a) for a in areas
            )
        zones = list(self._policy.collision_zones)
        if self._world_state is not None:
            dynamic = self._get_dynamic_zones()
            for zone in dynamic:
                if self._point_in_zone(x, y, zone):
                    return (
                        f"Position ({x:g}, {y:g}) blocked by dynamic obstacle zone "
                        f"{describe_area(zone)}"
                    )
            zones += dynamic
        if (not zones and not areas) or getattr(self._robot, "plans_own_path", False) is True:
            return None
        if start is None:
            px, py = self._robot.get_position()
            start = (float(px), float(py))
        route_text = f"({start[0]:g}, {start[1]:g}) to ({x:g}, {y:g})"
        for zone in zones:
            if segment_enters_zone(start, (x, y), zone):
                return f"Path from {route_text} crosses collision zone {describe_area(zone)}"
        left_at = route_leaves_areas([start, (x, y)], areas)
        if left_at is not None:
            return (
                f"Path from {route_text} leaves the allowed areas at "
                f"({left_at[0]:g}, {left_at[1]:g}): "
                + ", ".join(describe_area(a) for a in areas)
            )
        return None

    def _check_path(self, x: float, y: float) -> None:
        """Refuse a move whose route enters a zone or leaves the allowed areas."""
        fault = self.route_fault(x, y)
        if fault is None:
            return
        start = self._read_position()
        route_text = f"({start[0]:g}, {start[1]:g}) to ({x:g}, {y:g})"
        violation: dict[str, Any]
        if "zone" in fault:
            zone = fault["zone"]
            violation = {"type": "collision_zone_path", "zone": zone}
            problem = f"crosses collision zone {describe_area(zone)}"
        else:
            px, py = fault["left_at"]
            areas = self._policy.allowed_areas
            violation = {
                "type": "allowed_area_path",
                "left_at": (px, py),
                "allowed_areas": list(areas),
            }
            problem = (
                f"leaves the allowed areas at ({px:g}, {py:g}): "
                + ", ".join(describe_area(a) for a in areas)
            )
        violation.update(start=start, requested_position=(x, y))
        self._violations.append(violation)
        self._record_audit("violation", violation)
        logger.error(
            "SAFETY: Command REJECTED — route %s %s (policy: %s)",
            route_text, problem, self._policy.name,
        )
        raise SafetyViolation(f"Path from {route_text} {problem}")

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

    def _own_policy(self) -> None:
        """Copy a built-in policy before editing it, so no other robot sees the edit."""
        if self._policy is POLICY_REGISTRY.get(self._policy.name):
            self._policy = copy.copy(self._policy)
            self._policy.collision_zones = list(self._policy.collision_zones)

    def add_collision_zone(self, zone: dict[str, float]) -> None:
        """Add a collision zone at runtime (SF-07).

        Raises ``ValueError`` for a zone with min > max or a bound that is
        not a number, rather than add a zone that blocks nothing.
        """
        check_area(zone, "collision zone")
        self._own_policy()
        self._policy.collision_zones.append(zone)
        self._record_audit("zone_added", zone)
        logger.info("SAFETY: Added collision zone: %s", zone)

    def remove_collision_zone(self, index: int) -> dict[str, float] | None:
        """Remove a collision zone by index."""
        if 0 <= index < len(self._policy.collision_zones):
            self._own_policy()
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

    def _reject_bad_numbers(self, command: str, **values: float | None) -> None:
        """Refuse NaN/inf inputs: NaN slips past every ``>`` and zone check."""
        for name, value in values.items():
            if value is None:
                continue
            try:
                ok = math.isfinite(value)
            except (TypeError, OverflowError):
                ok = False
            if not ok:
                self._reject_bad_value(command, f"{name} must be a finite number, got {value!r}")

    def _reject_bad_value(self, command: str, detail: str) -> None:
        violation = {"type": "invalid_command", "command": command, "detail": detail}
        self._violations.append(violation)
        self._record_audit("violation", violation)
        logger.error("SAFETY: Command REJECTED — %s %s (policy: %s)",
                     command, detail, self._policy.name)
        raise SafetyViolation(f"Invalid {command} command: {detail}")

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
            "speed_profile_applied_to_moves": False,
            "escalation_timeout_s": p.escalation_timeout,
        }
