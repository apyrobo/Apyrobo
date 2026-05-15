"""Inspection round — visit a list of checkpoints and record status at each.

Useful for facility monitoring, equipment checks, and safety audits.

Usage:
    python examples/workflows/inspection_round.py
    python examples/workflows/inspection_round.py --robot ros2://spot --config checkpoints.json
"""
import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from apyrobo import Robot, Agent

DEFAULT_CHECKPOINTS = [
    {"id": "cp-01", "name": "main entrance", "checks": ["door closed", "lights on"]},
    {"id": "cp-02", "name": "server room",   "checks": ["temperature normal", "no water"]},
    {"id": "cp-03", "name": "fire panel",    "checks": ["no active alarms"]},
    {"id": "cp-04", "name": "loading bay",   "checks": ["bay doors secured"]},
]


@dataclass
class InspectionRecord:
    checkpoint_id: str
    name: str
    timestamp: str
    checks_passed: list[str] = field(default_factory=list)
    notes: str = ""


def run_inspection(robot_uri: str, checkpoints: list[dict]) -> list[InspectionRecord]:
    robot = Robot.discover(robot_uri)
    agent = Agent(provider="rule")
    records: list[InspectionRecord] = []

    print(f"[inspection] starting on {robot_uri} — {len(checkpoints)} checkpoints")

    for cp in checkpoints:
        print(f"\n  → Navigating to {cp['name']} ({cp['id']})")
        agent.execute(task=f"navigate to {cp['name']}", robot=robot)

        ts = datetime.now(timezone.utc).isoformat()
        record = InspectionRecord(
            checkpoint_id=cp["id"],
            name=cp["name"],
            timestamp=ts,
            checks_passed=cp.get("checks", []),
        )
        records.append(record)
        print(f"     checks: {', '.join(record.checks_passed)}")
        time.sleep(0.2)

    return records


def main() -> None:
    p = argparse.ArgumentParser(description="Inspection round workflow")
    p.add_argument("--robot", default="mock://spot")
    p.add_argument("--config", default=None, help="JSON file with checkpoint list")
    p.add_argument("--output", default=None, help="Write JSON report to this file")
    args = p.parse_args()

    if args.config:
        checkpoints = json.loads(Path(args.config).read_text())
    else:
        checkpoints = DEFAULT_CHECKPOINTS

    records = run_inspection(args.robot, checkpoints)

    report = [
        {"id": r.checkpoint_id, "name": r.name, "timestamp": r.timestamp,
         "checks": r.checks_passed}
        for r in records
    ]

    print("\n--- Inspection Report ---")
    print(json.dumps(report, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"\nReport saved to {args.output}")

    print(f"\n[inspection] complete — {len(records)} checkpoints visited")


if __name__ == "__main__":
    main()
