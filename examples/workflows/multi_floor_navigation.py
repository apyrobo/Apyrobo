"""Multi-floor navigation — drive to the elevator, ride it, drive to a spot on the new floor.

The rule planner only reads coordinates, so the elevator and each floor's
destination are named points in metres in that floor's map frame. Measure
your own (for example with ``ros2 topic echo /amcl_pose``) and replace the
ones below before running on a real robot.

APYROBO has no elevator interface. The ride (call the car, board, press the
floor, load the new floor's map) is a hand-off: ``ride_elevator`` simulates
it on mock:// and, on any other robot, stops at the elevator and exits 1
until you wire in your building's elevator and your map server (with Nav2,
the ``/map_server/load_map`` service).

Usage:
    python examples/workflows/multi_floor_navigation.py
    python examples/workflows/multi_floor_navigation.py --from-floor 1 --to-floor 3
"""
import argparse
import sys
import time

from apyrobo import Agent, Robot
from apyrobo.core.schemas import TaskStatus

# (x, y) in metres. The shaft is vertical, so the elevator door is at the
# same point on every floor's map. Example values: measure your own.
ELEVATOR = (0.5, 4.0)

FLOOR_MAPS = {
    1: "floor_1_map",
    2: "floor_2_map",
    3: "floor_3_map",
    4: "floor_4_map",
}

# name, (x, y) on that floor's map
FLOOR_DESTINATIONS = {
    1: ("ground floor reception", (3.0, 1.0)),
    2: ("second floor office area", (6.0, 2.5)),
    3: ("third floor lab", (5.0, -1.5)),
    4: ("fourth floor rooftop access", (1.5, 7.0)),
}


def _drive(agent: Agent, robot: Robot, name: str, point: tuple[float, float]) -> bool:
    x, y = point
    result = agent.execute(task=f"go to ({x}, {y})", robot=robot)
    if result.status != TaskStatus.COMPLETED:
        print(f"  ✗ {name} not reached: {result.error or result.status.value}")
        return False
    return True


def ride_elevator(robot_uri: str, from_floor: int, to_floor: int) -> bool:
    """Ride from one floor to another. Replace with your building's integration."""
    new_map = FLOOR_MAPS.get(to_floor, f"floor_{to_floor}_map")
    if not robot_uri.startswith("mock://"):
        print("  ✗ no elevator integration: APYROBO cannot call an elevator or load "
              f"{new_map!r}; wire them into ride_elevator() first")
        return False
    print(f"  (simulated) call the car, ride {from_floor} → {to_floor}, load map {new_map!r}")
    time.sleep(0.1)
    return True


def navigate_multi_floor(robot_uri: str, from_floor: int, to_floor: int) -> bool:
    """Go from the elevator on one floor to the destination on another; False on any miss."""
    if to_floor not in FLOOR_DESTINATIONS:
        print(f"[multi_floor] no destination for floor {to_floor}; "
              f"known floors: {sorted(FLOOR_DESTINATIONS)}")
        return False
    name, point = FLOOR_DESTINATIONS[to_floor]
    robot = Robot.discover(robot_uri)
    agent = Agent(provider="rule")

    print(f"[multi_floor] robot={robot_uri}  floor {from_floor} → floor {to_floor}")
    print(f"  destination: {name} {point}")

    if from_floor != to_floor:
        print(f"\n  [1] Drive to the elevator {ELEVATOR} on floor {from_floor}")
        if not _drive(agent, robot, "elevator", ELEVATOR):
            print("\n[multi_floor] stopped")
            return False
        print("  [2] Ride the elevator")
        if not ride_elevator(robot_uri, from_floor, to_floor):
            print(f"\n[multi_floor] stopped at the elevator on floor {from_floor}")
            return False
        print(f"  [3] Drive to {name} {point}")
    if not _drive(agent, robot, name, point):
        print("\n[multi_floor] stopped")
        return False

    print(f"\n[multi_floor] arrived at {name!r} on floor {to_floor}")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-floor navigation workflow")
    p.add_argument("--robot", default="mock://husky")
    p.add_argument("--from-floor", type=int, default=1)
    p.add_argument("--to-floor", type=int, default=3)
    args = p.parse_args()
    if not navigate_multi_floor(args.robot, args.from_floor, args.to_floor):
        sys.exit(1)


if __name__ == "__main__":
    main()
