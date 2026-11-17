"""Pick-and-place workflow — arm picks an object and places it at a named spot.

Each step is one arm motion (open, move to a pose, close, retreat, move to
the place pose, release, home); a step that fails stops the run and the
script exits 1. Pick and place poses are (x, y, z) in metres in the arm's
base frame. Measure your own and replace the ones below.

The arm here is ``MockMoveItAdapter``, a simulated MoveIt arm, so this runs
on mock:// only. The gripper calls in ``apyrobo.moveit`` do not drive a real
gripper yet, so on any other robot the script stops before moving and exits
1: wire in your arm's MoveIt setup and gripper action in ``connect_arm()``
first.

Usage:
    python examples/workflows/pick_and_place.py
    python examples/workflows/pick_and_place.py --object "red box" --target "bin B"
"""
import argparse
import asyncio
import sys

from apyrobo.moveit import MockMoveItAdapter, MotionResult, MoveItAdapter

# (x, y, z) in metres, arm base frame. Example values: measure your own.
PICK_POSE = (0.4, 0.0, 0.3)
RETREAT_HEIGHT = 0.5
PLACE_POSES = {
    "bin A": (0.0, 0.5, 0.3),
    "bin B": (0.0, -0.5, 0.3),
    "tray": (0.3, 0.3, 0.2),
}


def connect_arm(robot_uri: str) -> MoveItAdapter | None:
    """The arm to drive. Replace with your MoveIt setup for a real arm."""
    if robot_uri.startswith("mock://"):
        return MockMoveItAdapter()
    print("  ✗ no arm integration: this example drives a simulated MoveIt arm only; "
          "wire your arm and gripper into connect_arm() first")
    return None


async def _step(label: str, motion) -> bool:
    print(f"  {label}")
    result: MotionResult = await motion
    if not result.success:
        print(f"  ✗ {label} failed: {result.message or 'no reason given'}")
        return False
    return True


async def pick_and_place(robot_uri: str, obj: str, target: str) -> bool:
    """Pick ``obj`` and place it at ``target``; False as soon as a step fails."""
    if target not in PLACE_POSES:
        print(f"[pick_and_place] no place pose for {target!r}; "
              f"known targets: {sorted(PLACE_POSES)}")
        return False
    print(f"[pick_and_place] robot={robot_uri}  object={obj!r}  target={target!r}")
    arm = connect_arm(robot_uri)
    if arm is None:
        print("\n[pick_and_place] stopped")
        return False
    await arm.connect()

    px, py, pz = PICK_POSE
    tx, ty, tz = PLACE_POSES[target]
    steps = [
        ("1. Home arm", lambda: arm.home_arm()),
        ("2. Open gripper", lambda: arm.open_gripper()),
        (f"3. Move to pick pose {PICK_POSE}", lambda: arm.move_to_pose(x=px, y=py, z=pz)),
        (f"4. Close gripper on {obj}", lambda: arm.close_gripper()),
        ("5. Retreat", lambda: arm.move_to_pose(x=px, y=py, z=RETREAT_HEIGHT)),
        (f"6. Move to {target} {PLACE_POSES[target]}", lambda: arm.move_to_pose(x=tx, y=ty, z=tz)),
        ("7. Open gripper (release)", lambda: arm.open_gripper()),
        ("8. Home arm", lambda: arm.home_arm()),
    ]
    try:
        for label, motion in steps:
            if not await _step(label, motion()):
                print("\n[pick_and_place] stopped")
                return False
    finally:
        await arm.disconnect()
    print(f"\n[pick_and_place] {obj} placed at {target}")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Pick-and-place manipulation workflow")
    p.add_argument("--robot", default="mock://ur5")
    p.add_argument("--object", default="red box", dest="obj")
    p.add_argument("--target", default="bin A", help=f"one of {sorted(PLACE_POSES)}")
    args = p.parse_args()
    if not asyncio.run(pick_and_place(args.robot, args.obj, args.target)):
        sys.exit(1)


if __name__ == "__main__":
    main()
