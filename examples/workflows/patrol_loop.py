"""Patrol loop — continuously navigate a fixed set of waypoints.

Works out of the box with mock:// robots. Swap the URI for ros2:// or
gazebo:// to run on real hardware or in simulation.

Usage:
    python examples/workflows/patrol_loop.py
    python examples/workflows/patrol_loop.py --robot ros2://turtlebot4 --rounds 10
"""
import argparse
import time

from apyrobo import Robot, Agent

WAYPOINTS = [
    "entrance",
    "corridor A",
    "lab room 1",
    "corridor B",
    "charging bay",
]


def patrol(robot_uri: str, rounds: int, pause_s: float) -> None:
    robot = Robot.discover(robot_uri)
    agent = Agent(provider="rule")

    print(f"[patrol] starting on {robot_uri} — {rounds} round(s), {len(WAYPOINTS)} waypoints")

    for round_num in range(1, rounds + 1):
        print(f"\n--- Round {round_num}/{rounds} ---")
        for wp in WAYPOINTS:
            task = f"navigate to {wp}"
            print(f"  → {task}")
            agent.execute(task=task, robot=robot)
            time.sleep(pause_s)

    print("\n[patrol] complete")


def main() -> None:
    p = argparse.ArgumentParser(description="Continuous patrol loop")
    p.add_argument("--robot", default="mock://turtlebot4")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--pause", type=float, default=0.5, help="Seconds between waypoints")
    args = p.parse_args()
    patrol(args.robot, args.rounds, args.pause)


if __name__ == "__main__":
    main()
