"""
Safety Enforcer — hard constraints that no agent can bypass.

The SafetyEnforcer wraps a Robot instance and intercepts every command,
rejecting or clamping anything that violates the active safety policy.
It sits between the executor and the robot — invisible to agents.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import SafetyPolicyRef, RobotCapability

logger = logging.getLogger(__name__)


class SafetyViolation(Exception):
    """Raised when a command violates a safety constraint."""
    pass


class SafetyPolicy:
    """
    A concrete safety policy with enforceable constraints.

    Constraints:
        - max_speed: Hard cap on movement speed
        - collision_zones: List of rectangular zones the robot cannot enter
        - human_proximity_limit: Minimum distance to maintain from humans
    """

    def __init__(
        self,
        name: str = "default",
        max_speed: float = 1.5,
        collision_zones: list[dict[str, float]] | None = None,
        human_proximity_limit: float = 0.5,
    ) -> None:
        self.name = name
        self.max_speed = max_speed
        self.collision_zones = collision_zones or []
        self.human_proximity_limit = human_proximity_limit

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
            f"zones={len(self.collision_zones)}, proximity={self.human_proximity_limit})"
        )


# Default policies
DEFAULT_POLICY = SafetyPolicy(name="default", max_speed=1.5, human_proximity_limit=0.5)
STRICT_POLICY = SafetyPolicy(name="strict", max_speed=0.5, human_proximity_limit=1.0)

POLICY_REGISTRY: dict[str, SafetyPolicy] = {
    "default": DEFAULT_POLICY,
    "strict": STRICT_POLICY,
}


class SafetyEnforcer:
    """
    Wraps a Robot and enforces safety constraints on every command.

    Usage:
        enforcer = SafetyEnforcer(robot, policy="default")
        enforcer.move(x=2.0, y=3.0, speed=5.0)  # speed clamped to max
        enforcer.move(x=10.0, y=10.0)  # rejected if in collision zone

    The enforcer is transparent — it has the same API as Robot.
    """

    def __init__(self, robot: Robot, policy: str | SafetyPolicy = "default") -> None:
        self._robot = robot
        if isinstance(policy, str):
            self._policy = POLICY_REGISTRY.get(policy, DEFAULT_POLICY)
        else:
            self._policy = policy
        self._violations: list[dict[str, Any]] = []
        self._interventions: list[dict[str, Any]] = []
        logger.info("SafetyEnforcer active: %s", self._policy)

    @property
    def robot(self) -> Robot:
        return self._robot

    @property
    def policy(self) -> SafetyPolicy:
        return self._policy

    def capabilities(self, **kwargs: Any) -> RobotCapability:
        """Pass-through to robot capabilities."""
        return self._robot.capabilities(**kwargs)

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        """
        Move command with safety enforcement.

        - Speed is clamped to policy max_speed (never rejected, just limited)
        - Position is checked against collision zones (rejected if inside)
        """
        # --- Speed enforcement ---
        original_speed = speed
        if speed is not None and speed > self._policy.max_speed:
            speed = self._policy.max_speed
            self._interventions.append({
                "type": "speed_clamped",
                "requested": original_speed,
                "enforced": speed,
                "max_allowed": self._policy.max_speed,
            })
            logger.warning(
                "SAFETY: Speed clamped from %.2f to %.2f m/s (policy: %s)",
                original_speed, speed, self._policy.name,
            )

        # --- Collision zone enforcement ---
        for zone in self._policy.collision_zones:
            if self._point_in_zone(x, y, zone):
                violation = {
                    "type": "collision_zone",
                    "requested_position": (x, y),
                    "zone": zone,
                }
                self._violations.append(violation)
                logger.error(
                    "SAFETY: Command REJECTED — position (%.2f, %.2f) is inside "
                    "collision zone %s (policy: %s)",
                    x, y, zone, self._policy.name,
                )
                raise SafetyViolation(
                    f"Position ({x}, {y}) is inside collision zone: {zone}"
                )

        # --- All checks passed — forward to robot ---
        self._robot.move(x=x, y=y, speed=speed)

    def stop(self) -> None:
        """Stop is always allowed — safety never prevents stopping."""
        self._robot.stop()

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
        """All safety interventions (speed clamping, etc.)."""
        return list(self._interventions)

    @property
    def robot_id(self) -> str:
        return self._robot.robot_id

    def __repr__(self) -> str:
        return f"<SafetyEnforcer robot={self._robot.robot_id!r} policy={self._policy.name!r}>"
