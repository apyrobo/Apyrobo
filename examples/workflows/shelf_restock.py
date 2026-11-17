"""Shelf restock — pick items from a depot and place them on store shelves.

Combines navigation and manipulation: for each item, drive to the depot,
pick, drive to the item's shelf, place. The rule planner only reads
coordinates, so the depot and each shelf are named points in metres in the
robot's map frame. Measure your own (for example with
``ros2 topic echo /amcl_pose``) and replace the ones below.

The arm here is ``MockMoveItAdapter``, a simulated MoveIt arm, so this runs
on mock:// only. The gripper calls in ``apyrobo.moveit`` do not drive a real
gripper yet, so on any other robot the script stops before moving and exits
1: wire in your arm in ``connect_arm()`` first. A drive or arm step that
fails stops that item; the summary counts items restocked and not
restocked, and the script exits 1 if any was not.

Usage:
    python examples/workflows/shelf_restock.py
    python examples/workflows/shelf_restock.py --items 5
"""
import argparse
import asyncio
import sys

from apyrobo import Agent, Robot
from apyrobo.core.schemas import TaskStatus
from apyrobo.moveit import MockMoveItAdapter, MoveItAdapter

# (x, y) in metres, map frame. Example values: measure your own.
DEPOT = (0.0, -2.0)
# item, shelf name, (x, y) of the spot in front of that shelf
RESTOCK = [
    ("cereal box", "shelf A1", (2.0, 1.0)),
    ("soup can", "shelf B2", (2.0, 3.0)),
    ("pasta pack", "shelf C3", (4.0, 1.0)),
    ("coffee jar", "shelf D4", (4.0, 3.0)),
    ("tea box", "shelf E5", (6.0, 1.0)),
]
# (x, y, z) in metres, arm base frame.
PICK_POSE = (0.5, 0.0, 0.4)
PLACE_POSE = (0.3, 0.0, 0.6)


def connect_arm(robot_uri: str) -> MoveItAdapter | None:
    """The arm to drive. Replace with your MoveIt setup for a real arm."""
    if robot_uri.startswith("mock://"):
        return MockMoveItAdapter()
    print("  ✗ no arm integration: this example drives a simulated MoveIt arm only; "
          "wire your arm and gripper into connect_arm() first")
    return None


def _drive(agent: Agent, robot: Robot, name: str, point: tuple[float, float]) -> str | None:
    x, y = point
    result = agent.execute(task=f"go to ({x}, {y})", robot=robot)
    if result.status != TaskStatus.COMPLETED:
        return f"{name} not reached: {result.error or result.status.value}"
    return None


async def _arm(arm: MoveItAdapter, motions) -> str | None:
    for label, motion in motions:
        result = await motion()
        if not result.success:
            return f"{label} failed: {result.message or 'no reason given'}"
    return None


async def restock(robot_uri: str, num_items: int) -> tuple[list[str], list[str]]:
    """Restock the first ``num_items`` items; returns (restocked, not restocked)."""
    items = RESTOCK[:num_items]
    print(f"[shelf_restock] robot={robot_uri}  items={len(items)}  depot={DEPOT}")
    arm = connect_arm(robot_uri)
    if arm is None:
        return [], [item for item, _, _ in items]
    robot = Robot.discover(robot_uri)
    agent = Agent(provider="rule")
    await arm.connect()

    px, py, pz = PICK_POSE
    lx, ly, lz = PLACE_POSE
    pick = [
        ("open gripper", lambda: arm.open_gripper()),
        ("move to pick pose", lambda: arm.move_to_pose(x=px, y=py, z=pz)),
        ("close gripper", lambda: arm.close_gripper()),
    ]
    place = [
        ("move to place pose", lambda: arm.move_to_pose(x=lx, y=ly, z=lz)),
        ("open gripper", lambda: arm.open_gripper()),
        ("home arm", lambda: arm.home_arm()),
    ]

    done: list[str] = []
    missed: list[str] = []
    try:
        for i, (item, shelf, spot) in enumerate(items, 1):
            print(f"\n  [{i}/{len(items)}] {item!r} → {shelf} {spot}")
            error = (
                _drive(agent, robot, "depot", DEPOT)
                or await _arm(arm, pick)
                or _drive(agent, robot, shelf, spot)
                or await _arm(arm, place)
            )
            if error:
                print(f"    ✗ {item} not restocked: {error}")
                missed.append(item)
                await arm.home_arm()
            else:
                print(f"    ✓ {item} placed on {shelf}")
                done.append(item)
    finally:
        await arm.disconnect()
    return done, missed


def main() -> None:
    p = argparse.ArgumentParser(description="Shelf restock workflow")
    p.add_argument("--robot", default="mock://mir100")
    p.add_argument("--items", type=int, default=3, help=f"Number of items (1-{len(RESTOCK)})")
    args = p.parse_args()
    done, missed = asyncio.run(restock(args.robot, max(1, min(args.items, len(RESTOCK)))))
    print(f"\n[shelf_restock] {len(done)} restocked, {len(missed)} not restocked")
    if missed:
        sys.exit(1)


if __name__ == "__main__":
    main()
