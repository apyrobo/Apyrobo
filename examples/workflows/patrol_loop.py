"""Patrol loop — continuously navigate a fixed set of waypoints.

Works out of the box with mock:// robots. Swap the URI for ros2:// to run on
a real ROS 2 robot or against a running Gazebo simulation.

The rule planner only reads coordinates, so each waypoint is a named point
in metres in the robot's map frame. Measure your own (for example with
``ros2 topic echo /amcl_pose``) and replace the ones below before running on
a real robot.

Usage:
    python examples/workflows/patrol_loop.py
    python examples/workflows/patrol_loop.py --robot ros2://turtlebot4 --rounds 10
"""
import argparse
import sys
import time

from apyrobo import Agent, Robot
from apyrobo.core.schemas import TaskStatus

# name -> (x, y) in metres, map frame. Example values: measure your own.
WAYPOINTS = {
    "entrance": (0.0, 0.0),
    "corridor A": (4.0, 0.0),
    "lab room 1": (4.0, 3.0),
    "corridor B": (0.0, 3.0),
    "charging bay": (-1.0, 1.0),
}


def patrol(robot_uri: str, rounds: int, pause_s: float) -> bool:
    """Run the patrol; False as soon as a waypoint is not reached."""
    robot = Robot.discover(robot_uri)
    agent = Agent(provider="rule")

    print(f"[patrol] starting on {robot_uri} — {rounds} round(s), {len(WAYPOINTS)} waypoints")

    for round_num in range(1, rounds + 1):
        print(f"\n--- Round {round_num}/{rounds} ---")
        for name, (x, y) in WAYPOINTS.items():
            task = f"go to ({x}, {y})"
            print(f"  → {name}: {task}")
            result = agent.execute(task=task, robot=robot)
            if result.status != TaskStatus.COMPLETED:
                print(f"  ✗ {name} not reached: {result.error or result.status.value}")
                print("\n[patrol] stopped")
                return False
            time.sleep(pause_s)

    print("\n[patrol] complete")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Continuous patrol loop")
    p.add_argument("--robot", default="mock://turtlebot4")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--pause", type=float, default=0.5, help="Seconds between waypoints")
    args = p.parse_args()
    if not patrol(args.robot, args.rounds, args.pause):
        sys.exit(1)


if __name__ == "__main__":
    main()
