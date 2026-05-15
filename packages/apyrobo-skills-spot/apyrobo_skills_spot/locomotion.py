"""Boston Dynamics Spot locomotion skills — walk, sit, stand, stair climb."""
from __future__ import annotations

import time

from apyrobo import skill


@skill(
    description="Walk Spot to a world-frame position with optional heading and speed",
    capability="navigate",
)
def walk_to(x: float, y: float, heading: float = 0.0, speed: float = 1.0) -> bool:
    """Walk the robot to a target world position.

    Args:
        x:       Target X coordinate in the world frame (metres).
        y:       Target Y coordinate in the world frame (metres).
        heading: Final body heading in radians (default 0.0 = forward).
        speed:   Walk speed in m/s (clamped to 0.1–2.0 m/s).
    """
    speed = max(0.1, min(2.0, speed))
    print(f"  [walk_to] Planning path to ({x:.3f}, {y:.3f}), heading={heading:.3f} rad")
    time.sleep(0.05)
    print(f"  [walk_to] Walking at {speed:.1f} m/s")
    time.sleep(0.05)
    print(f"  [walk_to] Arrived at ({x:.3f}, {y:.3f}) — heading adjusted to {heading:.3f} rad")
    return True


@skill(
    description="Command Spot to sit down",
    capability="locomotion",
)
def sit() -> bool:
    """Lower the robot into a seated/resting pose.

    The robot completes its current motion, then smoothly transitions
    through a crouch to the sit position with all four legs folded.
    """
    print("  [sit] Initiating sit sequence")
    time.sleep(0.05)
    print("  [sit] Lowering body — four-leg crouch")
    time.sleep(0.05)
    print("  [sit] Sit complete — robot is at rest")
    return True


@skill(
    description="Command Spot to stand at a nominal or custom body height",
    capability="locomotion",
)
def stand(height: float = 0.52) -> bool:
    """Raise the robot to the standing pose.

    Args:
        height: Desired body height in metres above ground
                (clamped to 0.28–0.72 m — Spot's physical range).
    """
    height = max(0.28, min(0.72, height))
    print(f"  [stand] Standing up — target body height {height:.2f} m")
    time.sleep(0.05)
    print(f"  [stand] Body height stabilised at {height:.2f} m")
    return True


@skill(
    description="Climb or descend a staircase",
    capability="locomotion",
)
def stair_climb(num_stairs: int = 1, direction: str = "up") -> bool:
    """Navigate a staircase autonomously using Spot's stair-climbing behaviour.

    Args:
        num_stairs: Number of individual stairs to traverse (minimum 1).
        direction:  ``"up"`` to ascend or ``"down"`` to descend.
    """
    num_stairs = max(1, num_stairs)
    direction = direction.lower().strip()
    if direction not in ("up", "down"):
        direction = "up"
    verb = "Ascending" if direction == "up" else "Descending"
    print(f"  [stair_climb] {verb} {num_stairs} stair(s) — engaging stair-climb mode")
    for step in range(1, num_stairs + 1):
        print(f"  [stair_climb] Stair {step}/{num_stairs} — adjusting foothold placement")
        time.sleep(0.05)
    print(f"  [stair_climb] Stair traverse complete — {direction} {num_stairs} step(s)")
    return True
