"""Delivery run — pick up packages and deliver them to multiple destinations.

Models a last-mile indoor delivery robot: collect from a mailroom,
deliver to offices, return for next batch.

Usage:
    python examples/workflows/delivery_run.py
    python examples/workflows/delivery_run.py --robot ros2://turtlebot4 --packages 4
"""
import argparse
import time
from dataclasses import dataclass

from apyrobo import Robot, Agent

MAILROOM = "mailroom"

DELIVERIES = [
    {"package": "PKG-001", "destination": "office 101"},
    {"package": "PKG-002", "destination": "office 204"},
    {"package": "PKG-003", "destination": "lab 3B"},
    {"package": "PKG-004", "destination": "reception"},
    {"package": "PKG-005", "destination": "office 310"},
]


@dataclass
class DeliveryResult:
    package: str
    destination: str
    delivered: bool
    notes: str = ""


def run_deliveries(robot_uri: str, num_packages: int) -> list[DeliveryResult]:
    robot = Robot.discover(robot_uri)
    agent = Agent(provider="rule")
    results: list[DeliveryResult] = []

    deliveries = DELIVERIES[:num_packages]
    print(f"[delivery] robot={robot_uri}  packages={len(deliveries)}")

    # Collect all packages from mailroom
    print(f"\n  → Collecting {len(deliveries)} package(s) from {MAILROOM}")
    agent.execute(task=f"navigate to {MAILROOM} and collect packages", robot=robot)

    # Deliver each
    for i, d in enumerate(deliveries, 1):
        print(f"\n  [{i}/{len(deliveries)}] delivering {d['package']} to {d['destination']}")
        task = f"navigate to {d['destination']} and deliver package"
        agent.execute(task=task, robot=robot)
        results.append(DeliveryResult(
            package=d["package"],
            destination=d["destination"],
            delivered=True,
        ))
        print(f"     ✓ delivered")
        time.sleep(0.1)

    # Return to mailroom
    print(f"\n  → Returning to {MAILROOM}")
    agent.execute(task=f"navigate to {MAILROOM}", robot=robot)

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-stop package delivery")
    p.add_argument("--robot", default="mock://turtlebot4")
    p.add_argument("--packages", type=int, default=3)
    args = p.parse_args()

    results = run_deliveries(args.robot, min(args.packages, len(DELIVERIES)))

    delivered = sum(1 for r in results if r.delivered)
    print(f"\n[delivery] complete — {delivered}/{len(results)} delivered")


if __name__ == "__main__":
    main()
