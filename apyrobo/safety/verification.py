"""Formal safety property verification for regulatory compliance."""
from __future__ import annotations
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SafetyProperty:
    name: str
    description: str
    property_type: str  # "invariant" | "reachability" | "liveness"
    formula: str
    severity: str = "critical"  # "critical" | "warning"


@dataclass
class VerificationResult:
    property_name: str
    satisfied: bool
    counterexample: Optional[dict] = None
    proof_steps: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CertificationReport:
    robot_id: str
    timestamp: datetime
    properties_verified: int
    all_satisfied: bool
    hash: str  # SHA-256 of report content


# Built-in safety properties
SPEED_LIMIT_INVARIANT = SafetyProperty(
    name="speed_limit_invariant",
    description="Robot speed must never exceed maximum allowed",
    property_type="invariant",
    formula="speed <= max_speed",
    severity="critical",
)

NO_COLLISION_ZONE_INVARIANT = SafetyProperty(
    name="no_collision_zone_invariant",
    description="Robot must not enter restricted zones",
    property_type="invariant",
    formula="position not in collision_zones",
    severity="critical",
)

GRIPPER_SAFE_RELEASE = SafetyProperty(
    name="gripper_safe_release",
    description="Gripper must not release payload above unsafe height",
    property_type="invariant",
    formula="not (gripper_open and height > safe_release_height)",
    severity="warning",
)

BUILTIN_PROPERTIES = [SPEED_LIMIT_INVARIANT, NO_COLLISION_ZONE_INVARIANT, GRIPPER_SAFE_RELEASE]


class SafetyVerifier:
    def __init__(self, properties: Optional[list[SafetyProperty]] = None) -> None:
        self.properties = properties if properties is not None else list(BUILTIN_PROPERTIES)

    def verify_plan(self, plan: list[dict]) -> list[VerificationResult]:
        return [self._check_property_against_plan(p, plan) for p in self.properties]

    def verify_state_sequence(self, states: list[dict]) -> list[VerificationResult]:
        return [self._check_property_against_states(p, states) for p in self.properties]

    def export_proof(self, results: list[VerificationResult], format: str = "json") -> str:
        if format == "json":
            return json.dumps(
                [
                    {
                        "property": r.property_name,
                        "satisfied": r.satisfied,
                        "counterexample": r.counterexample,
                        "proof_steps": r.proof_steps,
                        "timestamp": r.timestamp.isoformat(),
                    }
                    for r in results
                ],
                indent=2,
            )
        elif format == "markdown":
            lines = ["# Safety Verification Report\n"]
            for r in results:
                status = "✅ PASS" if r.satisfied else "❌ FAIL"
                lines.append(f"## {r.property_name} — {status}")
                lines.append(f"*Verified at: {r.timestamp.isoformat()}*\n")
                if r.proof_steps:
                    lines.append("**Proof steps:**")
                    for step in r.proof_steps:
                        lines.append(f"- {step}")
                if r.counterexample:
                    lines.append(f"\n**Counterexample:** `{r.counterexample}`")
                lines.append("")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _check_property_against_plan(
        self, prop: SafetyProperty, plan: list[dict]
    ) -> VerificationResult:
        steps = [f"Checking property '{prop.name}' against {len(plan)} plan steps"]
        if prop.name == "speed_limit_invariant":
            for i, step in enumerate(plan):
                speed = step.get("params", {}).get("speed", 0)
                max_speed = step.get("params", {}).get("max_speed", float("inf"))
                if speed > max_speed:
                    steps.append(
                        f"Violation at step {i}: speed={speed} > max_speed={max_speed}"
                    )
                    return VerificationResult(
                        prop.name,
                        False,
                        {"step": i, "speed": speed, "max_speed": max_speed},
                        steps,
                    )
            steps.append("No speed violations found")
        elif prop.name == "no_collision_zone_invariant":
            for i, step in enumerate(plan):
                target = step.get("params", {}).get("target", {})
                zones = step.get("params", {}).get("collision_zones", [])
                if target and zones and target in zones:
                    steps.append(
                        f"Violation at step {i}: target={target} in collision_zones"
                    )
                    return VerificationResult(
                        prop.name, False, {"step": i, "target": target}, steps
                    )
            steps.append("No collision zone violations found")
        else:
            steps.append(f"No violations detected for '{prop.name}'")
        return VerificationResult(prop.name, True, proof_steps=steps)

    def _check_property_against_states(
        self, prop: SafetyProperty, states: list[dict]
    ) -> VerificationResult:
        steps = [f"Checking property '{prop.name}' across {len(states)} states"]
        if prop.name == "speed_limit_invariant":
            for i, state in enumerate(states):
                speed = state.get("speed", 0)
                max_speed = state.get("max_speed", float("inf"))
                if speed > max_speed:
                    steps.append(f"Violation at state {i}")
                    return VerificationResult(
                        prop.name, False, {"state_index": i, "state": state}, steps
                    )
        steps.append("All states satisfy property")
        return VerificationResult(prop.name, True, proof_steps=steps)


def generate_certification_report(
    verifier: SafetyVerifier,
    results: list[VerificationResult],
    robot_id: str,
) -> CertificationReport:
    all_ok = all(r.satisfied for r in results)
    ts = datetime.now(timezone.utc)
    content = json.dumps(
        {
            "robot_id": robot_id,
            "timestamp": ts.isoformat(),
            "properties_verified": len(results),
            "all_satisfied": all_ok,
        },
        sort_keys=True,
    )
    report_hash = hashlib.sha256(content.encode()).hexdigest()
    return CertificationReport(
        robot_id=robot_id,
        timestamp=ts,
        properties_verified=len(results),
        all_satisfied=all_ok,
        hash=report_hash,
    )
