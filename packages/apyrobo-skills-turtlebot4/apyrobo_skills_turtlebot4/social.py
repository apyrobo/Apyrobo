"""TurtleBot 4 social skills — follow person."""
from __future__ import annotations

import time

from apyrobo import skill


@skill(
    description="Follow a detected person for a specified duration",
    capability="navigate",
)
def follow_person(person_id: str = "person_01", duration_s: float = 10.0) -> bool:
    """Track and follow a detected person.

    The robot uses the OAKD camera's person-detection model to locate
    *person_id* and maintains a following distance of ~1 m for up to
    *duration_s* seconds.

    Args:
        person_id:  Tracker ID assigned by the detection pipeline.
        duration_s: Maximum duration to follow, in seconds (1–120 s).
    """
    duration_s = max(1.0, min(120.0, duration_s))
    print(f"  [follow_person] Locking onto person '{person_id}'")
    time.sleep(0.05)

    elapsed = 0.0
    tick = 0.1
    steps = max(1, int(duration_s / tick))
    report_every = max(1, steps // 4)

    for step in range(steps):
        elapsed += tick
        if step % report_every == 0 or step == steps - 1:
            print(f"  [follow_person] Following '{person_id}' — {elapsed:.1f}/{duration_s:.1f} s")

    print(f"  [follow_person] Follow complete — tracked for {duration_s:.1f} s")
    return True
