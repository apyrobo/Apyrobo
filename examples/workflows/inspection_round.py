"""Inspection round — visit a list of checkpoints and record status at each.

Useful for facility monitoring, equipment checks, and safety audits.

The rule planner only reads coordinates, so each checkpoint carries ``at``:
its (x, y) in metres in the robot's map frame. Measure your own (for example
with ``ros2 topic echo /amcl_pose``) and replace the ones below, or pass a
``--config`` JSON list of ``{"id", "name", "at": [x, y], "checks"}``. The
checks are the list to look at on arrival; the robot does not run them.
A checkpoint that is not reached is marked so in the report and the script
exits 1.

Usage:
    python examples/workflows/inspection_round.py
    python examples/workflows/inspection_round.py --robot ros2://spot --config checkpoints.json
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from apyrobo import Agent, Robot
from apyrobo.core.schemas import TaskStatus

# "at" is (x, y) in metres, map frame. Example values: measure your own.
DEFAULT_CHECKPOINTS = [
    {"id": "cp-01", "name": "main entrance", "at": [0.5, 0.0],
     "checks": ["door closed", "lights on"]},
    {"id": "cp-02", "name": "server room", "at": [6.0, 2.0],
     "checks": ["temperature normal", "no water"]},
    {"id": "cp-03", "name": "fire panel", "at": [6.0, 5.5],
     "checks": ["no active alarms"]},
    {"id": "cp-04", "name": "loading bay", "at": [-3.0, 4.0],
     "checks": ["bay doors secured"]},
]


@dataclass
class InspectionRecord:
    checkpoint_id: str
    name: str
    timestamp: str
    reached: bool
    checks: list[str] = field(default_factory=list)
    notes: str = ""


def _target(cp: dict) -> tuple[float, float] | None:
    at = cp.get("at")
    if not isinstance(at, (list, tuple)) or len(at) != 2:
        return None
    try:
        return float(at[0]), float(at[1])
    except (TypeError, ValueError):
        return None


def run_inspection(robot_uri: str, checkpoints: list[dict]) -> list[InspectionRecord]:
    robot = Robot.discover(robot_uri)
    agent = Agent(provider="rule")
    records: list[InspectionRecord] = []

    print(f"[inspection] starting on {robot_uri} — {len(checkpoints)} checkpoints")

    for cp in checkpoints:
        target = _target(cp)
        if target is None:
            reached, notes = False, 'no "at": [x, y] for this checkpoint'
            print(f"\n  ✗ {cp['name']} ({cp['id']}) skipped: {notes}")
        else:
            x, y = target
            print(f"\n  → {cp['name']} ({cp['id']}): go to ({x}, {y})")
            result = agent.execute(task=f"go to ({x}, {y})", robot=robot)
            reached = result.status == TaskStatus.COMPLETED
            notes = "" if reached else (result.error or result.status.value)
            if not reached:
                print(f"     ✗ not reached: {notes}")

        record = InspectionRecord(
            checkpoint_id=cp["id"],
            name=cp["name"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            reached=reached,
            checks=cp.get("checks", []) if reached else [],
            notes=notes,
        )
        records.append(record)
        if reached:
            print(f"     checks to look at: {', '.join(record.checks)}")
        time.sleep(0.2)

    return records


def main() -> None:
    p = argparse.ArgumentParser(description="Inspection round workflow")
    p.add_argument("--robot", default="mock://spot")
    p.add_argument("--config", default=None, help="JSON file with checkpoint list")
    p.add_argument("--output", default=None, help="Write JSON report to this file")
    args = p.parse_args()

    checkpoints = (
        json.loads(Path(args.config).read_text()) if args.config else DEFAULT_CHECKPOINTS
    )

    records = run_inspection(args.robot, checkpoints)

    report = [
        {"id": r.checkpoint_id, "name": r.name, "timestamp": r.timestamp,
         "reached": r.reached, "checks": r.checks, "notes": r.notes}
        for r in records
    ]

    print("\n--- Inspection Report ---")
    print(json.dumps(report, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"\nReport saved to {args.output}")

    reached = sum(r.reached for r in records)
    if reached < len(records):
        print(f"\n[inspection] incomplete — {reached}/{len(records)} checkpoints reached")
        sys.exit(1)
    print(f"\n[inspection] complete — {len(records)} checkpoints reached")


if __name__ == "__main__":
    main()
