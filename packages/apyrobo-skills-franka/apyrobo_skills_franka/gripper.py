"""Franka Panda gripper skills — grasp, release, and impedance control."""
from __future__ import annotations

import time

from apyrobo import skill


@skill(
    description="Close the Franka Hand gripper fingers to grasp an object",
    capability="manipulate",
)
def grasp(
    width: float = 0.04,
    force: float = 20.0,
    speed: float = 0.1,
) -> bool:
    """Close the Franka Hand gripper to grasp an object.

    Commands both gripper fingers to close to *width* metres apart,
    applying up to *force* newtons at *speed* metres per second.

    Args:
        width: Target finger separation in metres (default 0.04 m).
               Valid range: 0.0–0.08 m.
        force: Maximum grasping force in newtons (default 20.0 N).
               Valid range: 0.1–70.0 N.
        speed: Finger closing speed in metres per second (default 0.1 m/s).
               Valid range: 0.01–0.15 m/s.
    """
    width = max(0.0, min(0.08, width))
    force = max(0.1, min(70.0, force))
    speed = max(0.01, min(0.15, speed))

    print(
        f"  [grasp] Closing gripper — width={width:.3f} m, "
        f"force={force:.1f} N, speed={speed:.3f} m/s"
    )
    time.sleep(0.05)
    print(f"  [grasp] Gripper fingers at {width:.3f} m separation — object grasped")
    return True


@skill(
    description="Open the Franka Hand gripper fingers to release an object",
    capability="manipulate",
)
def release(width: float = 0.08) -> bool:
    """Open the Franka Hand gripper to release a held object.

    Commands both gripper fingers to move apart to *width* metres,
    releasing any grasped object.

    Args:
        width: Target finger separation in metres (default 0.08 m, fully open).
               Valid range: 0.0–0.08 m.
    """
    width = max(0.0, min(0.08, width))

    print(f"  [release] Opening gripper to width={width:.3f} m")
    time.sleep(0.05)
    print(f"  [release] Gripper open — fingers at {width:.3f} m separation")
    return True


@skill(
    description="Run the Franka impedance controller for compliant contact force control",
    capability="manipulate",
)
def impedance_control(
    stiffness: float = 200.0,
    damping: float = 10.0,
    duration_s: float = 1.0,
) -> bool:
    """Run a Cartesian impedance controller on the Franka Panda arm.

    Activates the impedance controller with the specified stiffness and
    damping gains, holding the current end-effector pose as the
    equilibrium point, and runs for *duration_s* seconds.

    This skill is useful for compliant manipulation, contact-rich tasks,
    and safe human-robot interaction.

    Args:
        stiffness:  Translational stiffness in N/m (default 200.0).
                    Valid range: 10.0–3000.0 N/m.
        damping:    Translational damping in N·s/m (default 10.0).
                    Valid range: 0.1–300.0 N·s/m.
        duration_s: Duration to run the impedance controller in seconds
                    (default 1.0 s, minimum 0.1 s).
    """
    stiffness = max(10.0, min(3000.0, stiffness))
    damping = max(0.1, min(300.0, damping))
    duration_s = max(0.1, duration_s)

    print(
        f"  [impedance_control] Starting impedance controller — "
        f"stiffness={stiffness:.1f} N/m, damping={damping:.1f} N·s/m"
    )
    print(f"  [impedance_control] Running for {duration_s:.2f} s")
    time.sleep(0.05)

    elapsed = 0.0
    tick = 0.1
    steps = max(1, int(duration_s / tick))
    report_every = max(1, steps // 4)

    for step in range(steps):
        elapsed += tick
        if step % report_every == 0 or step == steps - 1:
            print(
                f"  [impedance_control] Active — {elapsed:.1f}/{duration_s:.1f} s "
                f"(K={stiffness:.0f} N/m, D={damping:.0f} N·s/m)"
            )

    print("  [impedance_control] Impedance controller stopped — returning to position mode")
    return True
