"""Charging cycle — monitor battery and autonomously dock when low.

The robot pauses its current task, navigates to the charging dock,
waits until charged, then returns to resume work.

The rule planner only reads coordinates, so the dock and the work spot are
named points in metres in the robot's map frame. Measure your own (for
example with ``ros2 topic echo /amcl_pose``) and replace the ones below
before running on a real robot. The battery level is simulated.

Usage:
    python examples/workflows/charging_cycle.py
    python examples/workflows/charging_cycle.py --robot ros2://turtlebot4 --threshold 20
"""
import argparse
import sys
import time

from apyrobo import Agent, Robot
from apyrobo.core.schemas import TaskStatus

# (x, y) in metres, map frame. Example values: measure your own.
DOCK = (-1.0, 1.0)
WORK_SPOT = (2.0, 0.0)
FULL_CHARGE_PCT = 95
POLL_INTERVAL_S = 2.0


def simulate_battery(step: int) -> float:
    """Mock battery level: starts at 18%, climbs after docking."""
    if step < 3:
        return 18.0 - step
    return min(FULL_CHARGE_PCT + 1, 30.0 + (step - 3) * 25)


def _drive(agent: Agent, robot: Robot, name: str, point: tuple[float, float]) -> bool:
    x, y = point
    result = agent.execute(task=f"go to ({x}, {y})", robot=robot)
    if result.status != TaskStatus.COMPLETED:
        print(f"  ✗ {name} not reached: {result.error or result.status.value}")
        return False
    return True


def charging_cycle(robot_uri: str, threshold: float, max_steps: int) -> bool:
    """Run one charge cycle; False as soon as the dock or work spot is not reached."""
    robot = Robot.discover(robot_uri)
    agent = Agent(provider="rule")

    print(f"[charging] monitoring {robot_uri}  threshold={threshold}%  dock={DOCK}")

    docked = False

    for step in range(max_steps):
        battery = simulate_battery(step)
        print(f"  step {step+1:02d}  battery={battery:.0f}%  docked={docked}")

        if not docked and battery < threshold:
            print(f"  ⚡ Battery low ({battery:.0f}%) — going to the charging dock {DOCK}")
            if not _drive(agent, robot, "charging dock", DOCK):
                print("\n[charging] stopped")
                return False
            docked = True
            print("  ⚡ Docked — charging …")

        elif docked and battery >= FULL_CHARGE_PCT:
            print(f"  ✓ Fully charged ({battery:.0f}%) — back to the work spot {WORK_SPOT}")
            if not _drive(agent, robot, "work spot", WORK_SPOT):
                print("\n[charging] stopped")
                return False
            docked = False
            print("  ✓ Ready to resume task")
            break

        time.sleep(POLL_INTERVAL_S * 0.1)  # speed up for demo

    if docked:
        print("\n[charging] still docked when the steps ran out")
        return False
    print("\n[charging] cycle complete")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Autonomous charging cycle")
    p.add_argument("--robot", default="mock://turtlebot4")
    p.add_argument("--threshold", type=float, default=20.0, help="Battery % to trigger docking")
    p.add_argument("--max-steps", type=int, default=10)
    args = p.parse_args()
    if not charging_cycle(args.robot, args.threshold, args.max_steps):
        sys.exit(1)


if __name__ == "__main__":
    main()
