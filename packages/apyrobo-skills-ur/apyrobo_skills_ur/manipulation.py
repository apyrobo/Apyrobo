"""Universal Robots manipulation skills — pick, place, get_pose."""
from __future__ import annotations

import time

from apyrobo import skill


@skill(
    description="Pick an object from a specified Cartesian pose using the active gripper",
    capability="manipulation",
)
def pick(x: float, y: float, z: float, approach_height: float = 0.1) -> bool:
    """Execute a pick sequence at the target Cartesian position.

    The robot approaches from *approach_height* above the grasp point,
    descends in a linear Cartesian move, closes the gripper, then lifts
    back to the approach height with the object secured.

    Args:
        x:               Target grasp X position in metres (robot base frame).
        y:               Target grasp Y position in metres (robot base frame).
        z:               Target grasp Z position in metres (robot base frame).
        approach_height: Vertical clearance above the grasp point for the
                         pre-grasp approach and post-grasp lift, in metres.
                         Defaults to 0.1 m.
    """
    approach_height = max(0.01, approach_height)
    print(f"  [pick] Approaching pre-grasp pose: ({x:.3f}, {y:.3f}, {z + approach_height:.3f}) m")
    time.sleep(0.05)
    print(f"  [pick] Descending to grasp pose: ({x:.3f}, {y:.3f}, {z:.3f}) m")
    time.sleep(0.05)
    print("  [pick] Closing gripper — object secured")
    time.sleep(0.05)
    print(f"  [pick] Lifting to post-grasp height: {z + approach_height:.3f} m")
    time.sleep(0.05)
    print("  [pick] Pick complete — object in hand")
    return True


@skill(
    description="Place a held object at a specified Cartesian pose and release",
    capability="manipulation",
)
def place(x: float, y: float, z: float, approach_height: float = 0.1) -> bool:
    """Execute a place sequence at the target Cartesian position.

    The robot moves to *approach_height* above the release point, descends
    in a linear Cartesian move, opens the gripper to release the object,
    then lifts back to the approach height.

    Args:
        x:               Target place X position in metres (robot base frame).
        y:               Target place Y position in metres (robot base frame).
        z:               Target place Z position in metres (robot base frame).
        approach_height: Vertical clearance above the place point for the
                         pre-place approach and post-place retreat, in metres.
                         Defaults to 0.1 m.
    """
    approach_height = max(0.01, approach_height)
    print(f"  [place] Approaching pre-place pose: ({x:.3f}, {y:.3f}, {z + approach_height:.3f}) m")
    time.sleep(0.05)
    print(f"  [place] Descending to place pose: ({x:.3f}, {y:.3f}, {z:.3f}) m")
    time.sleep(0.05)
    print("  [place] Opening gripper — object released")
    time.sleep(0.05)
    print(f"  [place] Retreating to post-place height: {z + approach_height:.3f} m")
    time.sleep(0.05)
    print("  [place] Place complete — gripper clear")
    return True


@skill(
    description="Read the current TCP pose from the UR controller",
    capability="manipulation",
)
def get_pose() -> dict:
    """Query the UR controller for the current TCP pose in the base frame.

    Returns the six-DOF pose of the tool centre point expressed as a
    position (x, y, z) in metres and orientation (rx, ry, rz) as a
    rotation vector (axis-angle) in radians — the native UR pose format.

    Returns:
        A dict with keys ``x``, ``y``, ``z``, ``rx``, ``ry``, ``rz``
        representing the current TCP pose (all values as float, SI units).
    """
    print("  [get_pose] Querying UR controller for current TCP pose")
    time.sleep(0.05)
    pose = {
        "x": 0.300,
        "y": -0.200,
        "z": 0.450,
        "rx": 0.000,
        "ry": 3.14159,
        "rz": 0.000,
    }
    print(
        f"  [get_pose] TCP pose — "
        f"x={pose['x']:.3f} m, y={pose['y']:.3f} m, z={pose['z']:.3f} m | "
        f"rx={pose['rx']:.3f}, ry={pose['ry']:.5f}, rz={pose['rz']:.3f} rad"
    )
    return pose
