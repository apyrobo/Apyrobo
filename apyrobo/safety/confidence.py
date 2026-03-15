"""
Execution Confidence — estimates task success probability before execution.

This is the "capability contract" from the roadmap: before the robot moves,
APYROBO surfaces a confidence score and risk factors so the caller can
decide whether to proceed.

Factors:
    - Robot capabilities vs. required skills
    - Sensor availability
    - Battery / connectivity (when available)
    - Path clearance (from world state)
    - Historical success rate (SF-09: from StateStore data)

SF-08: ConfidenceEstimator gates execution — block if confidence < threshold.
SF-09: Historical success rate feeds into confidence from StateStore.
"""

from __future__ import annotations

import logging
from typing import Any

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import CapabilityType, RobotCapability
from apyrobo.skills.skill import Skill
from apyrobo.skills.executor import SkillGraph

logger = logging.getLogger(__name__)


class RiskFactor:
    """A single identified risk for a planned task."""

    def __init__(self, name: str, severity: float, description: str = "") -> None:
        self.name = name
        self.severity = severity  # 0.0 (no risk) to 1.0 (critical)
        self.description = description

    def __repr__(self) -> str:
        level = "LOW" if self.severity < 0.3 else "MED" if self.severity < 0.7 else "HIGH"
        return f"<Risk {self.name}: {level} ({self.severity:.1f})>"


class ConfidenceReport:
    """
    Pre-execution confidence assessment.

    Returned by ConfidenceEstimator.assess() before a task runs.
    """

    def __init__(
        self,
        confidence: float,
        risks: list[RiskFactor],
        can_proceed: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.confidence = confidence  # 0.0 to 1.0
        self.risks = risks
        self.can_proceed = can_proceed
        self.details = details or {}

    @property
    def risk_level(self) -> str:
        if self.confidence >= 0.8:
            return "low"
        elif self.confidence >= 0.5:
            return "medium"
        else:
            return "high"

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": round(self.confidence, 3),
            "risk_level": self.risk_level,
            "can_proceed": self.can_proceed,
            "risks": [
                {"name": r.name, "severity": r.severity, "description": r.description}
                for r in self.risks
            ],
            "details": self.details,
        }

    def __repr__(self) -> str:
        return (
            f"<ConfidenceReport {self.confidence:.0%} "
            f"risk={self.risk_level} proceed={self.can_proceed} "
            f"risks={len(self.risks)}>"
        )


class LowConfidenceError(Exception):
    """Raised when confidence is below threshold and execution is blocked (SF-08)."""

    def __init__(self, report: ConfidenceReport) -> None:
        self.report = report
        super().__init__(
            f"Confidence {report.confidence:.0%} below threshold — "
            f"risks: {[r.name for r in report.risks]}"
        )


class ConfidenceEstimator:
    """
    Estimates task success probability before execution.

    SF-08: Wire to executor — blocks execution if confidence < threshold.
    SF-09: Incorporates historical success rate from StateStore.

    Usage:
        estimator = ConfidenceEstimator()
        report = estimator.assess(graph, robot)
        if report.can_proceed:
            executor.execute_graph(graph)
        else:
            print(f"Too risky: {report.risks}")

    With executor gating (SF-08):
        estimator = ConfidenceEstimator(block_below=0.4)
        estimator.gate(graph, robot)  # raises LowConfidenceError if too risky
    """

    # Minimum confidence to auto-proceed
    PROCEED_THRESHOLD = 0.4

    def __init__(
        self,
        world_state: Any = None,
        state_store: Any = None,
        block_below: float | None = None,
    ) -> None:
        self._world_state = world_state
        self._state_store = state_store  # SF-09: for historical success rate
        self._block_threshold = block_below  # SF-08: raise if below this

    def assess(self, graph: SkillGraph, robot: Robot) -> ConfidenceReport:
        """
        Assess a skill graph against a robot's current state.

        Returns a ConfidenceReport with confidence score and risk factors.
        """
        caps = robot.capabilities()
        risks: list[RiskFactor] = []
        score = 1.0

        # --- Check 1: Capability coverage ---
        score, risks = self._check_capabilities(graph, caps, score, risks)

        # --- Check 2: Sensor availability ---
        score, risks = self._check_sensors(graph, caps, score, risks)

        # --- Check 3: Skill complexity ---
        score, risks = self._check_complexity(graph, score, risks)

        # --- Check 4: World state (obstacles, path) ---
        if self._world_state is not None:
            score, risks = self._check_world_state(graph, score, risks)

        # --- Check 5: Speed constraints ---
        score, risks = self._check_speed(graph, caps, score, risks)

        # --- Check 6 (SF-09): Historical success rate ---
        score, risks = self._check_historical_success(graph, score, risks)

        # Clamp
        score = max(0.0, min(1.0, score))
        can_proceed = score >= self.PROCEED_THRESHOLD

        report = ConfidenceReport(
            confidence=score,
            risks=risks,
            can_proceed=can_proceed,
            details={
                "skill_count": len(graph),
                "capability_types_needed": list(self._required_capabilities(graph)),
                "robot_capabilities": [c.capability_type.value for c in caps.capabilities],
                "sensor_count": len(caps.sensors),
            },
        )

        logger.info(
            "Confidence: %.0f%% risk=%s proceed=%s risks=%d",
            report.confidence * 100, report.risk_level, report.can_proceed, len(risks),
        )

        return report

    def gate(self, graph: SkillGraph, robot: Robot) -> ConfidenceReport:
        """
        SF-08: Assess and block execution if confidence is too low.

        Returns the report if confidence is sufficient.
        Raises LowConfidenceError if confidence < block_threshold.
        """
        report = self.assess(graph, robot)
        threshold = self._block_threshold if self._block_threshold is not None else self.PROCEED_THRESHOLD
        if report.confidence < threshold:
            raise LowConfidenceError(report)
        return report

    def _check_capabilities(
        self, graph: SkillGraph, caps: RobotCapability,
        score: float, risks: list[RiskFactor],
    ) -> tuple[float, list[RiskFactor]]:
        """Check that the robot has all required capabilities."""
        robot_caps = {c.capability_type for c in caps.capabilities}
        needed = self._required_capabilities(graph)

        for cap_type in needed:
            if cap_type == CapabilityType.CUSTOM:
                continue  # CUSTOM is always available
            if cap_type not in robot_caps:
                risks.append(RiskFactor(
                    name=f"missing_capability_{cap_type.value}",
                    severity=0.9,
                    description=f"Robot lacks required capability: {cap_type.value}",
                ))
                score -= 0.3

        return score, risks

    def _check_sensors(
        self, graph: SkillGraph, caps: RobotCapability,
        score: float, risks: list[RiskFactor],
    ) -> tuple[float, list[RiskFactor]]:
        """Check sensor availability for skills that need perception."""
        needed = self._required_capabilities(graph)
        has_camera = any(s.sensor_type.value == "camera" for s in caps.sensors)
        has_lidar = any(s.sensor_type.value == "lidar" for s in caps.sensors)

        if CapabilityType.PICK in needed or CapabilityType.PLACE in needed:
            if not has_camera:
                risks.append(RiskFactor(
                    name="no_camera_for_manipulation",
                    severity=0.6,
                    description="Pick/place skills need camera but none detected",
                ))
                score -= 0.15

        if CapabilityType.NAVIGATE in needed:
            if not has_lidar:
                risks.append(RiskFactor(
                    name="no_lidar_for_navigation",
                    severity=0.3,
                    description="Navigation without lidar — obstacle avoidance limited",
                ))
                score -= 0.05

        if len(caps.sensors) == 0:
            risks.append(RiskFactor(
                name="no_sensors",
                severity=0.7,
                description="No sensors detected — robot is flying blind",
            ))
            score -= 0.2

        return score, risks

    def _check_complexity(
        self, graph: SkillGraph,
        score: float, risks: list[RiskFactor],
    ) -> tuple[float, list[RiskFactor]]:
        """Penalise long skill chains (more steps = more failure points)."""
        skill_count = len(graph)
        if skill_count > 10:
            risks.append(RiskFactor(
                name="high_complexity",
                severity=0.5,
                description=f"Plan has {skill_count} skills — complex plans are riskier",
            ))
            score -= 0.1
        elif skill_count > 5:
            risks.append(RiskFactor(
                name="moderate_complexity",
                severity=0.2,
                description=f"Plan has {skill_count} skills",
            ))
            score -= 0.03

        return score, risks

    def _check_world_state(
        self, graph: SkillGraph,
        score: float, risks: list[RiskFactor],
    ) -> tuple[float, list[RiskFactor]]:
        """Check world state for obstacles in the path."""
        ws = self._world_state
        if ws is None:
            return score, risks

        # Check for nearby obstacles
        nearby = ws.obstacles_within(1.0)
        if len(nearby) > 3:
            risks.append(RiskFactor(
                name="crowded_environment",
                severity=0.4,
                description=f"{len(nearby)} obstacles within 1m — navigation may be difficult",
            ))
            score -= 0.1
        elif len(nearby) > 0:
            risks.append(RiskFactor(
                name="obstacles_nearby",
                severity=0.15,
                description=f"{len(nearby)} obstacle(s) within 1m",
            ))

        return score, risks

    def _check_speed(
        self, graph: SkillGraph, caps: RobotCapability,
        score: float, risks: list[RiskFactor],
    ) -> tuple[float, list[RiskFactor]]:
        """Check if any skill requests speed above robot max."""
        if caps.max_speed is None:
            return score, risks

        for skill in graph.get_execution_order():
            params = graph.get_parameters(skill.skill_id)
            speed = params.get("speed")
            if isinstance(speed, (int, float)) and speed > caps.max_speed:
                risks.append(RiskFactor(
                    name="speed_exceeded",
                    severity=0.2,
                    description=(
                        f"Skill {skill.name} requests {speed} m/s "
                        f"but robot max is {caps.max_speed} m/s (will be clamped)"
                    ),
                ))

        return score, risks

    def _check_historical_success(
        self, graph: SkillGraph,
        score: float, risks: list[RiskFactor],
    ) -> tuple[float, list[RiskFactor]]:
        """
        SF-09: Factor in historical success rate from StateStore.

        Looks at recent tasks in the StateStore to compute a success rate.
        A low success rate penalises the confidence score.
        """
        if self._state_store is None:
            return score, risks

        try:
            recent = self._state_store.get_recent_tasks(limit=50)
        except Exception:
            return score, risks

        if len(recent) < 3:
            # Not enough history to be meaningful
            return score, risks

        completed = sum(1 for t in recent if t.status == "completed")
        total = len(recent)
        success_rate = completed / total

        if success_rate < 0.5:
            risks.append(RiskFactor(
                name="low_historical_success",
                severity=0.6,
                description=(
                    f"Historical success rate is {success_rate:.0%} "
                    f"({completed}/{total} recent tasks) — system may be unreliable"
                ),
            ))
            score -= 0.15
        elif success_rate < 0.8:
            risks.append(RiskFactor(
                name="moderate_historical_success",
                severity=0.3,
                description=(
                    f"Historical success rate is {success_rate:.0%} "
                    f"({completed}/{total} recent tasks)"
                ),
            ))
            score -= 0.05

        return score, risks

    @staticmethod
    def _required_capabilities(graph: SkillGraph) -> set[CapabilityType]:
        """Collect all capability types needed by the graph."""
        return {
            skill.required_capability
            for skill in graph.get_execution_order()
        }
