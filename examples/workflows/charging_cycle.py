"""Charging cycle — monitor battery and autonomously dock when low.

The robot pauses its current task, navigates to the charging dock,
waits until charged, then returns to resume work.

Usage:
    python examples/workflows/charging_cycle.py
    python examples/workflows/charging_cycle.py --robot ros2://turtlebot4 --threshold 20
"""
import argparse
import time

from apyrobo import Robot, Agent

DOCK_LOCATION = "charging dock"
FULL_CHARGE_PCT = 95
POLL_INTERVAL_S = 2.0


def simulate_battery(step: int) -> float:
    """Mock battery level: starts at 18%, climbs after docking."""
    if step < 3:
        return 18.0 - step
    return min(FULL_CHARGE_PCT + 1, 30.0 + (step - 3) * 25)


def charging_cycle(robot_uri: str, threshold: float, max_steps: int) -> None:
    robot = Robot.discover(robot_uri)
    agent = Agent(provider="rule")

    print(f"[charging] monitoring {robot_uri}  threshold={threshold}%  dock={DOCK_LOCATION!r}")

    docked = False

    for step in range(max_steps):
        battery = simulate_battery(step)
        print(f"  step {step+1:02d}  battery={battery:.0f}%  docked={docked}")

        if not docked and battery < threshold:
            print(f"  ⚡ Battery low ({battery:.0f}%) — navigating to {DOCK_LOCATION}")
            agent.execute(task=f"navigate to {DOCK_LOCATION}", robot=robot)
            docked = True
            print("  ⚡ Docked — charging …")

        elif docked and battery >= FULL_CHARGE_PCT:
            print(f"  ✓ Fully charged ({battery:.0f}%) — undocking")
            agent.execute(task="undock from charging station", robot=robot)
            docked = False
            print("  ✓ Ready to resume task")
            break

        time.sleep(POLL_INTERVAL_S * 0.1)  # speed up for demo

    print("\n[charging] cycle complete")


def main() -> None:
    p = argparse.ArgumentParser(description="Autonomous charging cycle")
    p.add_argument("--robot", default="mock://turtlebot4")
    p.add_argument("--threshold", type=float, default=20.0, help="Battery % to trigger docking")
    p.add_argument("--max-steps", type=int, default=10)
    args = p.parse_args()
    charging_cycle(args.robot, args.threshold, args.max_steps)


if __name__ == "__main__":
    main()
