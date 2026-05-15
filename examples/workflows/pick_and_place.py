"""Pick-and-place workflow — arm picks an object and places it at a target.

Demonstrates MoveIt 2 adapter integration for manipulation tasks.
Works with mock:// out of the box; swap for ros2:// with a real arm.

Usage:
    python examples/workflows/pick_and_place.py
    python examples/workflows/pick_and_place.py --robot ros2://ur5 --object "red box" --target "bin A"
"""
import argparse
import asyncio

from apyrobo import Robot, Agent
from apyrobo.moveit import MockMoveItAdapter


async def pick_and_place(robot_uri: str, obj: str, target: str) -> None:
    robot = Robot.discover(robot_uri)
    agent = Agent(provider="rule")
    arm = MockMoveItAdapter()
    await arm.connect()

    print(f"[pick_and_place] robot={robot_uri}  object={obj!r}  target={target!r}")

    # 1. Move to home position
    print("  1. Homing arm …")
    await arm.home_arm()

    # 2. Plan approach
    task = f"pick up {obj} and place it at {target}"
    print(f"  2. Planning: {task!r}")
    graph = agent.plan(task, robot)
    print(f"     {len(graph)} skills planned")

    # 3. Open gripper → approach → close → retreat → place → open
    print("  3. Open gripper")
    await arm.open_gripper()

    print("  4. Move to pick pose")
    await arm.move_to_pose(x=0.4, y=0.0, z=0.3)

    print("  5. Close gripper (grasp)")
    await arm.close_gripper()

    print("  6. Retreat")
    await arm.move_to_pose(x=0.4, y=0.0, z=0.5)

    print("  7. Move to place pose")
    await arm.move_to_pose(x=0.0, y=0.5, z=0.3)

    print("  8. Open gripper (release)")
    await arm.open_gripper()

    print("  9. Home arm")
    await arm.home_arm()

    await arm.disconnect()
    print("\n[pick_and_place] complete")


def main() -> None:
    p = argparse.ArgumentParser(description="Pick-and-place manipulation workflow")
    p.add_argument("--robot", default="mock://ur5")
    p.add_argument("--object", default="red box", dest="obj")
    p.add_argument("--target", default="bin A")
    args = p.parse_args()
    asyncio.run(pick_and_place(args.robot, args.obj, args.target))


if __name__ == "__main__":
    main()
