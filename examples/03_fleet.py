#!/usr/bin/env python3
"""
Fleet — coordinate multiple robots with load-balanced task assignment.

Demonstrates:
  - Registering multiple robots with FleetManager
  - Assigning tasks to the least-recently-used idle robot
  - Executing tasks on the assigned robot via Agent
  - Querying fleet status

The FleetManager tracks availability and picks the idle robot with the
oldest last-heartbeat (least recently used), providing simple round-robin
load balancing across the fleet.

Run from the repo root:
    python examples/03_fleet.py
"""

import time

from apyrobo import Robot, Agent
from apyrobo.fleet.manager import FleetManager, RobotInfo


FLEET_SIZE = 3

TASKS = [
    "navigate to dock 1",
    "deliver package from (0, 0) to (5, 3)",
    "navigate to dock 2",
    "deliver package from (2, 1) to (8, 6)",
    "navigate to dock 3",
]


def main() -> None:
    # ── 1. Build the fleet ───────────────────────────────────
    fleet = FleetManager()
    robots: dict[str, Robot] = {}

    print(f"Registering {FLEET_SIZE} robots...")
    for i in range(FLEET_SIZE):
        robot_id = f"robot-{i}"
        robot = Robot.discover(f"mock://{robot_id}")
        robots[robot_id] = robot

        fleet.register(RobotInfo(
            robot_id=robot_id,
            capabilities=["navigate", "pick", "place"],
        ))
        time.sleep(0.001)  # stagger so registration timestamps differ

    status = fleet.get_status()
    print(f"Fleet ready: {status['total']} robots ({status['idle']} idle)\n")

    # ── 2. Execute tasks across the fleet ────────────────────
    agent = Agent(provider="rule")

    print(f"Executing {len(TASKS)} tasks across fleet:\n")
    print(f"  {'Robot':<10} {'Task':<45} {'Result'}")
    print(f"  {'-'*10} {'-'*45} {'-'*10}")

    for task in TASKS:
        # Assign to least-recently-used idle robot
        robot_id = fleet.assign_task({
            "task_id": task,
            "required": ["navigate"],
        })

        if robot_id is None:
            print(f"  [no robot available for: {task!r}]")
            continue

        result = agent.execute(task, robot=robots[robot_id])

        # Mark complete and update heartbeat (sends robot to back of queue)
        fleet.complete_task(robot_id)
        fleet.heartbeat(robot_id)

        print(f"  {robot_id:<10} {task:<45} {result.status.value}")

    # ── 3. Final fleet status ────────────────────────────────
    print()
    status = fleet.get_status()
    print(f"Fleet summary: {status['total']} robots, "
          f"{status['idle']} idle, {status['busy']} busy")


if __name__ == "__main__":
    main()
