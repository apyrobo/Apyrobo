"""Shelf restock — pick items from a depot and place them on store shelves.

Combines navigation and manipulation in a structured restock workflow.

Usage:
    python examples/workflows/shelf_restock.py
    python examples/workflows/shelf_restock.py --robot ros2://mir100 --items 5
"""
import argparse
import asyncio

from apyrobo import Robot, Agent
from apyrobo.moveit import MockMoveItAdapter

DEPOT = "storage depot"
SHELVES = ["shelf A1", "shelf B2", "shelf C3", "shelf D4", "shelf E5"]
ITEM_NAMES = ["cereal box", "soup can", "pasta pack", "coffee jar", "tea box"]


async def restock(robot_uri: str, num_items: int) -> None:
    robot = Robot.discover(robot_uri)
    agent = Agent(provider="rule")
    arm = MockMoveItAdapter()
    await arm.connect()

    items_to_restock = list(zip(ITEM_NAMES, SHELVES))[:num_items]
    print(f"[shelf_restock] robot={robot_uri}  items={num_items}")

    for i, (item, shelf) in enumerate(items_to_restock, 1):
        print(f"\n  [{i}/{num_items}] restocking {item!r} → {shelf}")

        # Navigate to depot
        print(f"    → navigate to {DEPOT}")
        agent.execute(task=f"navigate to {DEPOT}", robot=robot)

        # Pick item
        print(f"    → pick {item}")
        await arm.open_gripper()
        await arm.move_to_pose(x=0.5, y=0.0, z=0.4)
        await arm.close_gripper()

        # Navigate to shelf
        print(f"    → navigate to {shelf}")
        agent.execute(task=f"navigate to {shelf}", robot=robot)

        # Place item
        print(f"    → place {item} on {shelf}")
        await arm.move_to_pose(x=0.3, y=0.0, z=0.6)
        await arm.open_gripper()
        await arm.home_arm()

    await arm.disconnect()
    print(f"\n[shelf_restock] complete — {num_items} item(s) restocked")


def main() -> None:
    p = argparse.ArgumentParser(description="Shelf restock workflow")
    p.add_argument("--robot", default="mock://mir100")
    p.add_argument("--items", type=int, default=3, help="Number of items to restock")
    args = p.parse_args()
    asyncio.run(restock(args.robot, min(args.items, len(ITEM_NAMES))))


if __name__ == "__main__":
    main()
