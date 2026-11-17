"""Quality inspection — pick parts at a station, check each one, sort into bins.

The robot drives to the inspection station (a point in metres in the map
frame: the rule planner only reads coordinates), then for each part the arm
picks it, holds it at the inspection pose, and places it in the pass or the
reject bin. Every drive and arm step is checked; a step that fails stops
the run and the script exits 1. A part found defective is a result, not a
failure.

The inspection itself is simulated: ``inspect_part()`` draws a seeded
random result. Replace it with your camera or vision-model check. The arm
here is ``MockMoveItAdapter``, a simulated MoveIt arm, so this runs on
mock:// only; on any other robot the script stops before moving and exits
1: wire in your arm in ``connect_arm()`` first.

Usage:
    python examples/workflows/quality_inspection.py
    python examples/workflows/quality_inspection.py --items 8 --seed 3
"""
import argparse
import asyncio
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

from apyrobo import Agent, Robot
from apyrobo.core.schemas import TaskStatus
from apyrobo.moveit import MockMoveItAdapter, MoveItAdapter

# (x, y) in metres, map frame. Example value: measure your own.
STATION = (3.0, 1.5)
# (x, y, z) in metres, arm base frame. Example values: measure your own.
PICK_POSE = (0.4, 0.0, 0.2)
INSPECTION_POSE = (0.3, 0.0, 0.45)
BIN_POSES = {
    "pass bin": (0.0, 0.45, 0.25),
    "reject bin": (0.0, -0.45, 0.25),
}


@dataclass
class InspectionResult:
    item_id: str
    timestamp: str
    passed: bool
    defect: str = ""
    scores: dict = field(default_factory=dict)


def inspect_part(item_id: str, rng: random.Random) -> InspectionResult:
    """Simulated check. Replace with your camera or vision-model call."""
    passed = rng.random() > 0.25
    defect = "" if passed else rng.choice(["scratch", "dent", "misalignment", "colour"])
    return InspectionResult(
        item_id=item_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        passed=passed,
        defect=defect,
        scores={"surface": round(rng.uniform(0.7, 1.0), 2)},
    )


def connect_arm(robot_uri: str) -> MoveItAdapter | None:
    """The arm to drive. Replace with your MoveIt setup for a real arm."""
    if robot_uri.startswith("mock://"):
        return MockMoveItAdapter()
    print("  ✗ no arm integration: this example drives a simulated MoveIt arm only; "
          "wire your arm and gripper into connect_arm() first")
    return None


async def _arm(arm: MoveItAdapter, motions) -> str | None:
    for label, motion in motions:
        result = await motion()
        if not result.success:
            return f"{label} failed: {result.message or 'no reason given'}"
    return None


async def quality_inspection(
    robot_uri: str, num_items: int, seed: int = 1
) -> tuple[list[InspectionResult], str | None]:
    """Inspect ``num_items`` parts; returns (results so far, error or None)."""
    print(f"[quality_inspection] robot={robot_uri}  items={num_items}  station={STATION}")
    arm = connect_arm(robot_uri)
    if arm is None:
        return [], "no arm integration"
    robot = Robot.discover(robot_uri)
    agent = Agent(provider="rule")

    x, y = STATION
    print(f"\n  → Drive to inspection station {STATION}")
    drive = agent.execute(task=f"go to ({x}, {y})", robot=robot)
    if drive.status != TaskStatus.COMPLETED:
        return [], f"inspection station not reached: {drive.error or drive.status.value}"

    rng = random.Random(seed)
    px, py, pz = PICK_POSE
    ix, iy, iz = INSPECTION_POSE
    results: list[InspectionResult] = []
    await arm.connect()
    try:
        for i in range(1, num_items + 1):
            item_id = f"PART-{i:04d}"
            print(f"\n  [{i}/{num_items}] {item_id}")
            error = await _arm(arm, [
                ("open gripper", lambda: arm.open_gripper()),
                ("move to pick pose", lambda: arm.move_to_pose(x=px, y=py, z=pz)),
                ("close gripper", lambda: arm.close_gripper()),
                ("move to inspection pose", lambda: arm.move_to_pose(x=ix, y=iy, z=iz)),
            ])
            if error:
                return results, f"{item_id}: {error}"

            result = inspect_part(item_id, rng)
            results.append(result)
            print("     PASS ✓" if result.passed else f"     FAIL ✗ ({result.defect})")

            bin_name = "pass bin" if result.passed else "reject bin"
            bx, by, bz = BIN_POSES[bin_name]
            error = await _arm(arm, [
                (f"move to {bin_name}",
                 lambda x=bx, y=by, z=bz: arm.move_to_pose(x=x, y=y, z=z)),
                ("open gripper", lambda: arm.open_gripper()),
                ("home arm", lambda: arm.home_arm()),
            ])
            if error:
                return results, f"{item_id}: {error}"
            print(f"     placed in {bin_name}")
    finally:
        await arm.disconnect()
    return results, None


def main() -> None:
    p = argparse.ArgumentParser(description="Production line quality inspection")
    p.add_argument("--robot", default="mock://franka_panda")
    p.add_argument("--items", type=int, default=6)
    p.add_argument("--seed", type=int, default=1, help="Seed for the simulated check")
    args = p.parse_args()
    results, error = asyncio.run(
        quality_inspection(args.robot, max(1, args.items), args.seed)
    )

    passed = sum(1 for r in results if r.passed)
    print("\n--- Quality Report ---")
    print(f"  Inspected: {len(results)}")
    print(f"  Passed:    {passed}")
    print(f"  Failed:    {len(results) - passed}")
    defects = [r.defect for r in results if not r.passed]
    if defects:
        print(f"  Defects:   {', '.join(defects)}")
    if error:
        print(f"\n[quality_inspection] stopped: {error}")
        sys.exit(1)
    print("\n[quality_inspection] all parts inspected and sorted")


if __name__ == "__main__":
    main()
