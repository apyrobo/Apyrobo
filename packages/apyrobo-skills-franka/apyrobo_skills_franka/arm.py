"""Franka Panda arm skills — Cartesian pose control and sweeps."""
from __future__ import annotations

import math
import time

from apyrobo import skill


@skill(
    description="Move the Franka Panda end-effector to a Cartesian pose",
    capability="manipulate",
)
def move_to_pose(
    x: float,
    y: float,
    z: float,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> bool:
    """Move the end-effector to the specified Cartesian pose.

    Plans and executes a joint-space trajectory that places the
    end-effector at position (*x*, *y*, *z*) with orientation defined
    by the (*roll*, *pitch*, *yaw*) Euler angles (radians).

    Args:
        x:     Target position along the X axis in metres.
        y:     Target position along the Y axis in metres.
        z:     Target position along the Z axis in metres.
        roll:  End-effector roll angle in radians (default 0.0).
        pitch: End-effector pitch angle in radians (default 0.0).
        yaw:   End-effector yaw angle in radians (default 0.0).
    """
    print(
        f"  [move_to_pose] Target position: ({x:.3f}, {y:.3f}, {z:.3f}) m"
    )
    print(
        f"  [move_to_pose] Target orientation: roll={roll:.3f} rad, "
        f"pitch={pitch:.3f} rad, yaw={yaw:.3f} rad"
    )
    print("  [move_to_pose] Planning trajectory via inverse kinematics")
    time.sleep(0.05)
    print("  [move_to_pose] Executing joint-space trajectory")
    time.sleep(0.05)
    print(
        f"  [move_to_pose] End-effector reached target pose "
        f"({x:.3f}, {y:.3f}, {z:.3f})"
    )
    return True


@skill(
    description="Return the Franka Panda arm to its default home configuration",
    capability="manipulate",
)
def move_home() -> bool:
    """Return the Franka Panda to its default home joint configuration.

    Sends all seven joints to their zero-ish home angles:
    [0, -π/4, 0, -3π/4, 0, π/2, π/4], which places the end-effector
    in a safe, upright position above the robot base.
    """
    # Franka's canonical "ready" configuration (radians)
    home_joints = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]
    joint_str = ", ".join(f"{j:.3f}" for j in home_joints)
    print("  [move_home] Commanding all joints to home configuration")
    print(f"  [move_home] Joint targets (rad): [{joint_str}]")
    time.sleep(0.05)
    print("  [move_home] Executing homing trajectory")
    time.sleep(0.05)
    print("  [move_home] Franka Panda is at home configuration")
    return True


@skill(
    description="Sweep the Franka Panda end-effector linearly between two Cartesian points",
    capability="manipulate",
)
def cartesian_sweep(
    start: list[float],
    end: list[float],
    steps: int = 10,
) -> bool:
    """Move the end-effector in a straight Cartesian line from *start* to *end*.

    Interpolates *steps* intermediate waypoints along the line segment
    connecting *start* [x, y, z] and *end* [x, y, z] and executes each
    sub-move sequentially.

    Args:
        start: Starting position as [x, y, z] in metres.
        end:   Ending position as [x, y, z] in metres.
        steps: Number of interpolation steps (default 10, minimum 2).
    """
    steps = max(2, steps)

    sx, sy, sz = float(start[0]), float(start[1]), float(start[2])
    ex, ey, ez = float(end[0]), float(end[1]), float(end[2])

    print(
        f"  [cartesian_sweep] Sweep from ({sx:.3f}, {sy:.3f}, {sz:.3f}) "
        f"to ({ex:.3f}, {ey:.3f}, {ez:.3f}) in {steps} steps"
    )

    for i in range(steps):
        alpha = i / (steps - 1)
        cx = sx + alpha * (ex - sx)
        cy = sy + alpha * (ey - sy)
        cz = sz + alpha * (ez - sz)
        print(
            f"  [cartesian_sweep] Step {i + 1}/{steps}: "
            f"({cx:.3f}, {cy:.3f}, {cz:.3f})"
        )
        time.sleep(0.05)

    print("  [cartesian_sweep] Cartesian sweep complete")
    return True
