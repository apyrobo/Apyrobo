"""Multi-floor navigation — ride an elevator between floors to reach a destination.

Demonstrates map switching and elevator interaction for multi-level buildings.

Usage:
    python examples/workflows/multi_floor_navigation.py
    python examples/workflows/multi_floor_navigation.py --robot ros2://husky --from-floor 1 --to-floor 3
"""
import argparse
import time

from apyrobo import Robot, Agent

ELEVATOR_CALL_POINT = "elevator lobby"
ELEVATOR_INTERIOR = "inside elevator"

FLOOR_MAPS = {
    1: "floor_1_map",
    2: "floor_2_map",
    3: "floor_3_map",
    4: "floor_4_map",
}

FLOOR_DESTINATIONS = {
    1: "ground floor reception",
    2: "second floor office area",
    3: "third floor lab",
    4: "fourth floor rooftop access",
}


def navigate_multi_floor(
    robot_uri: str,
    from_floor: int,
    to_floor: int,
    final_destination: str,
) -> None:
    robot = Robot.discover(robot_uri)
    agent = Agent(provider="rule")

    print(f"[multi_floor] robot={robot_uri}  floor {from_floor} → floor {to_floor}")
    print(f"  destination: {final_destination!r}")

    if from_floor == to_floor:
        print(f"  already on floor {to_floor} — navigating directly")
        agent.execute(task=f"navigate to {final_destination}", robot=robot)
        return

    # Step 1: Navigate to elevator on current floor
    print(f"\n  [1] Navigate to {ELEVATOR_CALL_POINT} on floor {from_floor}")
    agent.execute(task=f"navigate to {ELEVATOR_CALL_POINT}", robot=robot)

    # Step 2: Call elevator
    direction = "up" if to_floor > from_floor else "down"
    print(f"  [2] Call elevator going {direction}")
    agent.execute(task=f"call elevator going {direction}", robot=robot)
    time.sleep(0.3)

    # Step 3: Board elevator
    print(f"  [3] Board elevator")
    agent.execute(task=f"navigate to {ELEVATOR_INTERIOR}", robot=robot)

    # Step 4: Select floor
    print(f"  [4] Press floor {to_floor} button")
    agent.execute(task=f"press button for floor {to_floor}", robot=robot)
    print(f"       … travelling to floor {to_floor} …")
    time.sleep(0.5)

    # Step 5: Exit elevator — switch to destination floor map
    new_map = FLOOR_MAPS.get(to_floor, f"floor_{to_floor}_map")
    print(f"  [5] Exit elevator on floor {to_floor} (map: {new_map})")
    agent.execute(task=f"exit elevator and load map {new_map}", robot=robot)

    # Step 6: Navigate to final destination on new floor
    print(f"  [6] Navigate to {final_destination}")
    agent.execute(task=f"navigate to {final_destination}", robot=robot)

    print(f"\n[multi_floor] arrived at {final_destination!r} on floor {to_floor}")


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-floor navigation workflow")
    p.add_argument("--robot", default="mock://husky")
    p.add_argument("--from-floor", type=int, default=1)
    p.add_argument("--to-floor", type=int, default=3)
    p.add_argument("--destination", default=None,
                   help="Final destination (defaults to floor's landmark)")
    args = p.parse_args()

    dest = args.destination or FLOOR_DESTINATIONS.get(args.to_floor, f"floor {args.to_floor}")
    navigate_multi_floor(args.robot, args.from_floor, args.to_floor, dest)


if __name__ == "__main__":
    main()
