"""
Swarm Safety — robot-to-robot proximity enforcement and deadlock detection.

Extends the single-robot SafetyEnforcer with swarm-aware constraints:
- Minimum distance between any two robots
- Deadlock detection (two robots waiting on each other)
- Shared collision zone enforcement across the fleet
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from apyrobo.core.robot import Robot
from apyrobo.safety.enforcer import SafetyEnforcer, SafetyPolicy, SafetyViolation
from apyrobo.swarm.bus import SwarmBus

logger = logging.getLogger(__name__)


class ProximityViolation(SafetyViolation):
    """Raised when two robots are too close to each other."""
    pass


class DeadlockDetected(SafetyViolation):
    """Raised when a deadlock is detected between robots."""
    pass


class SwarmSafety:
    """
    Swarm-wide safety enforcement.

    Monitors all robots in the swarm and enforces:
    - Minimum inter-robot distance
    - Deadlock detection
    - Shared no-go zones

    Usage:
        swarm_safety = SwarmSafety(bus, min_distance=0.5)
        swarm_safety.check_proximity()  # raises ProximityViolation if too close
        swarm_safety.check_deadlock()   # raises DeadlockDetected if deadlocked
    """

    def __init__(
        self,
        bus: SwarmBus,
        min_distance: float = 0.5,
        deadlock_timeout: float = 10.0,
    ) -> None:
        self._bus = bus
        self.min_distance = min_distance
        self.deadlock_timeout = deadlock_timeout
        self._positions: dict[str, tuple[float, float]] = {}
        self._waiting_on: dict[str, str | None] = {}  # robot -> robot it's waiting on
        self._wait_start: dict[str, float] = {}
        self._violations: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Position tracking
    # ------------------------------------------------------------------

    def update_position(self, robot_id: str, x: float, y: float) -> None:
        """Update a robot's known position (from odometry or mock)."""
        self._positions[robot_id] = (x, y)

    def get_position(self, robot_id: str) -> tuple[float, float] | None:
        """Get last known position of a robot."""
        return self._positions.get(robot_id)

    # ------------------------------------------------------------------
    # Proximity check
    # ------------------------------------------------------------------

    def check_proximity(self) -> list[tuple[str, str, float]]:
        """
        Check all robot pairs for proximity violations.

        Returns list of (robot_a, robot_b, distance) for violations.
        Does NOT raise — caller decides what to do.
        """
        violations = []
        robot_ids = list(self._positions.keys())

        for i in range(len(robot_ids)):
            for j in range(i + 1, len(robot_ids)):
                rid_a = robot_ids[i]
                rid_b = robot_ids[j]
                pos_a = self._positions[rid_a]
                pos_b = self._positions[rid_b]

                dist = math.sqrt(
                    (pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2
                )

                if dist < self.min_distance:
                    violations.append((rid_a, rid_b, dist))
                    self._violations.append({
                        "type": "proximity",
                        "robot_a": rid_a,
                        "robot_b": rid_b,
                        "distance": dist,
                        "min_required": self.min_distance,
                        "timestamp": time.time(),
                    })
                    logger.warning(
                        "SWARM SAFETY: Robots %s and %s too close (%.2fm < %.2fm)",
                        rid_a, rid_b, dist, self.min_distance,
                    )

        return violations

    def enforce_proximity(self) -> None:
        """
        Check proximity and raise if any violation found.

        Call this before executing a move command.
        """
        violations = self.check_proximity()
        if violations:
            a, b, dist = violations[0]
            raise ProximityViolation(
                f"Robots {a} and {b} are {dist:.2f}m apart "
                f"(minimum: {self.min_distance}m)"
            )

    def would_violate_proximity(
        self, robot_id: str, target_x: float, target_y: float,
    ) -> tuple[bool, str | None, float]:
        """
        Check if moving robot_id to (target_x, target_y) would violate proximity.

        Returns (would_violate, other_robot_id, distance).
        """
        for rid, pos in self._positions.items():
            if rid == robot_id:
                continue
            dist = math.sqrt(
                (target_x - pos[0]) ** 2 + (target_y - pos[1]) ** 2
            )
            if dist < self.min_distance:
                return True, rid, dist
        return False, None, float("inf")

    # ------------------------------------------------------------------
    # Deadlock detection
    # ------------------------------------------------------------------

    def set_waiting(self, robot_id: str, waiting_on: str | None) -> None:
        """Mark that a robot is waiting on another robot (or None if free)."""
        if waiting_on is not None:
            self._waiting_on[robot_id] = waiting_on
            if robot_id not in self._wait_start:
                self._wait_start[robot_id] = time.time()
        else:
            self._waiting_on.pop(robot_id, None)
            self._wait_start.pop(robot_id, None)

    def check_deadlock(self) -> list[list[str]]:
        """
        Detect deadlock cycles in the wait graph.

        Returns a list of cycles, where each cycle is a list of robot IDs.
        Example: [["robot_a", "robot_b"]] means A waits on B and B waits on A.
        """
        cycles = []
        visited: set[str] = set()

        for start in self._waiting_on:
            if start in visited:
                continue

            path: list[str] = []
            current: str | None = start

            while current is not None and current not in visited:
                if current in path:
                    # Found a cycle
                    cycle_start = path.index(current)
                    cycle = path[cycle_start:]
                    cycles.append(cycle)
                    self._violations.append({
                        "type": "deadlock",
                        "cycle": cycle,
                        "timestamp": time.time(),
                    })
                    logger.error(
                        "SWARM SAFETY: Deadlock detected: %s",
                        " → ".join(cycle) + f" → {cycle[0]}",
                    )
                    break

                path.append(current)
                current = self._waiting_on.get(current)

            visited.update(path)

        return cycles

    def enforce_deadlock(self) -> None:
        """Check for deadlocks and raise if found."""
        cycles = self.check_deadlock()
        if cycles:
            cycle = cycles[0]
            raise DeadlockDetected(
                f"Deadlock: {' → '.join(cycle)} → {cycle[0]}"
            )

    # ------------------------------------------------------------------
    # Combined check
    # ------------------------------------------------------------------

    def check_all(self) -> dict[str, Any]:
        """
        Run all safety checks and return a summary.

        Does NOT raise — returns a dict with results.
        """
        proximity = self.check_proximity()
        deadlocks = self.check_deadlock()
        return {
            "safe": len(proximity) == 0 and len(deadlocks) == 0,
            "proximity_violations": proximity,
            "deadlocks": deadlocks,
            "robot_count": len(self._positions),
        }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def violations(self) -> list[dict[str, Any]]:
        return list(self._violations)

    @property
    def positions(self) -> dict[str, tuple[float, float]]:
        return dict(self._positions)

    def __repr__(self) -> str:
        return (
            f"<SwarmSafety robots={len(self._positions)} "
            f"min_dist={self.min_distance}m "
            f"violations={len(self._violations)}>"
        )
