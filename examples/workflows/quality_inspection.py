"""Quality inspection — scan items on a production line and flag defects.

Uses the robot's camera feed and arm to inspect parts, logging pass/fail
for each item. Integrates with the hardware sensor pipeline.

Usage:
    python examples/workflows/quality_inspection.py
    python examples/workflows/quality_inspection.py --robot ros2://franka_panda --items 8
"""
import argparse
import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

from apyrobo import Robot, Agent
from apyrobo.moveit import MockMoveItAdapter

INSPECTION_STATION = "inspection station"
REJECT_BIN = "reject bin"
PASS_BIN = "pass bin"

INSPECTION_POSES = [
    {"x": 0.4, "y": -0.1, "z": 0.3},
    {"x": 0.4, "y":  0.0, "z": 0.3},
    {"x": 0.4, "y":  0.1, "z": 0.3},
]


@dataclass
class InspectionResult:
    item_id: str
    timestamp: str
    passed: bool
    defect: str = ""
    scores: dict = field(default_factory=dict)


def _mock_inspect(item_id: str) -> InspectionResult:
    """Simulate camera-based inspection (replace with real VLM call in production)."""
    passed = random.random() > 0.25  # 75% pass rate
    defect = "" if passed else random.choice(["scratch", "dent", "misalignment", "colour"])
    return InspectionResult(
        item_id=item_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        passed=passed,
        defect=defect,
        scores={"surface": round(random.uniform(0.7, 1.0), 2)},
    )


async def quality_inspection(robot_uri: str, num_items: int) -> None:
    robot = Robot.discover(robot_uri)
    agent = Agent(provider="rule")
    arm = MockMoveItAdapter()
    await arm.connect()

    results: list[InspectionResult] = []

    print(f"[quality_inspection] robot={robot_uri}  items={num_items}")

    # Navigate to inspection station
    print(f"\n  → Navigate to {INSPECTION_STATION}")
    agent.execute(task=f"navigate to {INSPECTION_STATION}", robot=robot)

    for i in range(1, num_items + 1):
        item_id = f"PART-{i:04d}"
        print(f"\n  [{i}/{num_items}] inspecting {item_id}")

        # Pick item from conveyor
        await arm.open_gripper()
        await arm.move_to_pose(**INSPECTION_POSES[i % len(INSPECTION_POSES)])
        await arm.close_gripper()

        # Inspect
        result = _mock_inspect(item_id)
        results.append(result)
        status = "PASS ✓" if result.passed else f"FAIL ✗ ({result.defect})"
        print(f"     {status}")

        # Route to correct bin
        target = PASS_BIN if result.passed else REJECT_BIN
        agent.execute(task=f"place item in {target}", robot=robot)
        await arm.open_gripper()
        await arm.home_arm()

    await arm.disconnect()

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print(f"\n--- Quality Report ---")
    print(f"  Total:  {len(results)}")
    print(f"  Passed: {passed}  ({100*passed/len(results):.0f}%)")
    print(f"  Failed: {failed}")
    if failed:
        defects = [r.defect for r in results if not r.passed]
        print(f"  Defects: {', '.join(defects)}")
    print("\n[quality_inspection] complete")


def main() -> None:
    p = argparse.ArgumentParser(description="Production line quality inspection")
    p.add_argument("--robot", default="mock://franka_panda")
    p.add_argument("--items", type=int, default=6)
    args = p.parse_args()
    asyncio.run(quality_inspection(args.robot, args.items))


if __name__ == "__main__":
    main()
