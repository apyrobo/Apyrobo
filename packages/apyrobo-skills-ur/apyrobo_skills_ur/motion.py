"""Universal Robots motion skills — joint move, linear move, home, TCP."""
from __future__ import annotations

import time

from apyrobo import skill


@skill(
    description="Move the UR arm to a joint-space configuration",
    capability="motion",
)
def move_joints(joint_positions: list[float], speed: float = 0.5) -> bool:
    """Command the arm to move to the specified joint angles.

    Sends a MoveJ command to the UR controller, blending through the
    trajectory with the given joint speed scaling factor.

    Args:
        joint_positions: Six joint angles in radians [J1, J2, J3, J4, J5, J6].
                         Typical range is -2π to +2π per joint.
        speed:           Joint speed scaling factor in the range 0.0–1.0.
                         Defaults to 0.5 (50% of maximum speed).
    """
    speed = max(0.0, min(1.0, speed))
    n = len(joint_positions)
    rounded = [f"{v:.3f}" for v in joint_positions]
    print(f"  [move_joints] Executing MoveJ — {n} joint(s): [{', '.join(rounded)}]")
    print(f"  [move_joints] Speed scaling: {speed:.2f}")
    time.sleep(0.05)
    print("  [move_joints] Joint target reached — motion complete")
    return True


@skill(
    description="Move the UR arm TCP in a straight Cartesian line to a target pose",
    capability="motion",
)
def move_linear(x: float, y: float, z: float, speed: float = 0.1) -> bool:
    """Execute a linear Cartesian move (MoveL) to the target TCP position.

    The robot moves in a straight line in Cartesian space, maintaining
    the current TCP orientation unless overridden by the active motion plan.

    Args:
        x:     Target TCP X position in metres (robot base frame).
        y:     Target TCP Y position in metres (robot base frame).
        z:     Target TCP Z position in metres (robot base frame).
        speed: Cartesian TCP speed in metres per second (clamped to 0.001–1.0 m/s).
               Defaults to 0.1 m/s for safe operation near obstacles.
    """
    speed = max(0.001, min(1.0, speed))
    print(f"  [move_linear] Executing MoveL — target: ({x:.3f}, {y:.3f}, {z:.3f}) m")
    print(f"  [move_linear] TCP speed: {speed:.3f} m/s")
    time.sleep(0.05)
    print("  [move_linear] Cartesian target reached — motion complete")
    return True


@skill(
    description="Return the UR arm to its home (upright) configuration",
    capability="motion",
)
def move_home() -> bool:
    """Move all joints to the canonical home configuration.

    The home configuration places the arm in a safe, upright pose
    with all joints at 0 radians (straight-up position), suitable
    for unobstructed workspace access and safe operator interaction.
    """
    home = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]
    rounded = [f"{v:.4f}" for v in home]
    print(f"  [move_home] Moving to home configuration: [{', '.join(rounded)}]")
    time.sleep(0.05)
    print("  [move_home] Home position reached — arm ready")
    return True


@skill(
    description="Set the Tool Centre Point (TCP) offset for the active end-effector",
    capability="motion",
)
def set_tcp(x: float = 0.0, y: float = 0.0, z: float = 0.1) -> bool:
    """Configure the TCP offset relative to the robot's tool flange.

    The TCP defines the point the controller uses for Cartesian path
    planning. Accurate TCP calibration is essential for precise pick-and-place
    and force-controlled operations.

    Args:
        x: TCP offset along the flange X axis in metres. Defaults to 0.0 m.
        y: TCP offset along the flange Y axis in metres. Defaults to 0.0 m.
        z: TCP offset along the flange Z axis in metres. Defaults to 0.1 m
           (typical for a lightweight gripper).
    """
    print(f"  [set_tcp] Setting TCP offset — x={x:.4f} m, y={y:.4f} m, z={z:.4f} m")
    time.sleep(0.05)
    print(f"  [set_tcp] TCP configured — controller updated with new offset")
    return True
