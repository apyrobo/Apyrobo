"""Delivery run — pick up packages and deliver them to multiple destinations.

Models a last-mile indoor delivery robot: collect each package from a
mailroom, deliver it to an office, return to the mailroom at the end.

The rule planner only reads coordinates, so every place is a named point in
metres in the robot's map frame. Measure your own and replace the ones below
before running on a real robot.

Usage:
    python examples/workflows/delivery_run.py
    python examples/workflows/delivery_run.py --robot ros2://turtlebot4 --packages 4
"""
import argparse
import sys
import time
from dataclasses import dataclass

from apyrobo import Agent, Robot
from apyrobo.core.schemas import TaskStatus

# name -> (x, y) in metres, map frame. Example values: measure your own.
MAILROOM = ("mailroom", (0.0, 0.0))

DELIVERIES = [
    {"package": "PKG-001", "destination": "office 101", "at": (3.0, 1.0)},
    {"package": "PKG-002", "destination": "office 204", "at": (6.0, 2.0)},
    {"package": "PKG-003", "destination": "lab 3B", "at": (6.0, -2.0)},
    {"package": "PKG-004", "destination": "reception", "at": (1.0, -3.0)},
    {"package": "PKG-005", "destination": "office 310", "at": (8.0, 4.0)},
]


@dataclass
class DeliveryResult:
    package: str
    destination: str
    delivered: bool
    notes: str = ""


def _xy(point: tuple[float, float]) -> str:
    return f"({point[0]}, {point[1]})"


def run_deliveries(robot_uri: str, num_packages: int) -> list[DeliveryResult]:
    robot = Robot.discover(robot_uri)
    agent = Agent(provider="rule")
    results: list[DeliveryResult] = []
    mailroom_name, mailroom = MAILROOM

    deliveries = DELIVERIES[:num_packages]
    print(f"[delivery] robot={robot_uri}  packages={len(deliveries)}")

    for i, d in enumerate(deliveries, 1):
        print(f"\n  [{i}/{len(deliveries)}] {d['package']}: {mailroom_name} → {d['destination']}")
        task = f"deliver from {_xy(mailroom)} to {_xy(d['at'])}"
        result = agent.execute(task=task, robot=robot)
        delivered = result.status == TaskStatus.COMPLETED
        notes = "" if delivered else (result.error or result.status.value)
        results.append(DeliveryResult(
            package=d["package"],
            destination=d["destination"],
            delivered=delivered,
            notes=notes,
        ))
        if not delivered:
            print(f"     ✗ not delivered: {notes}")
            break
        print("     ✓ delivered")
        time.sleep(0.1)

    print(f"\n  → Returning to {mailroom_name}")
    back = agent.execute(task=f"go to {_xy(mailroom)}", robot=robot)
    if back.status != TaskStatus.COMPLETED:
        print(f"     ✗ did not get back: {back.error or back.status.value}")

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-stop package delivery")
    p.add_argument("--robot", default="mock://turtlebot4")
    p.add_argument("--packages", type=int, default=3)
    args = p.parse_args()

    wanted = min(args.packages, len(DELIVERIES))
    results = run_deliveries(args.robot, wanted)

    delivered = sum(1 for r in results if r.delivered)
    print(f"\n[delivery] {delivered}/{wanted} delivered")
    if delivered < wanted:
        sys.exit(1)


if __name__ == "__main__":
    main()
