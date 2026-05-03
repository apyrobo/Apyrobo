"""TurtleBot 4 navigation skills — patrol, dock, undock."""
from __future__ import annotations

import time

from apyrobo import skill


@skill(
    description="Patrol a defined area following waypoints",
    capability="navigate",
)
def patrol_area(waypoints: list | None = None, loops: int = 1) -> bool:
    """Navigate through a sequence of waypoints N times.

    Args:
        waypoints: List of (x, y) pairs, e.g. [[0,0],[1,0],[1,1]].
                   Defaults to a 1 m² square if not provided.
        loops:     Number of patrol circuits to complete.
    """
    if waypoints is None:
        waypoints = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

    print(f"  [patrol_area] Starting {loops} loop(s) over {len(waypoints)} waypoint(s)")
    for loop in range(1, loops + 1):
        for i, wp in enumerate(waypoints, 1):
            x, y = wp[0], wp[1]
            print(f"  [patrol_area] Loop {loop}/{loops} — waypoint {i}/{len(waypoints)}: ({x}, {y})")
            time.sleep(0.05)
    print("  [patrol_area] Patrol complete")
    return True


@skill(
    description="Navigate to and dock at a charging station",
    capability="navigate",
)
def dock(dock_station_id: str = "dock_01") -> bool:
    """Approach the charging dock and engage the docking mechanism.

    Args:
        dock_station_id: Identifier of the dock in the environment map.
    """
    print(f"  [dock] Navigating to dock station '{dock_station_id}'")
    time.sleep(0.05)
    print("  [dock] Aligning with dock IR beacon")
    time.sleep(0.05)
    print("  [dock] Docking — charging contact established")
    return True


@skill(
    description="Leave the charging station and resume normal operation",
    capability="navigate",
)
def undock() -> bool:
    """Disengage the charging dock and reverse to a safe operating distance."""
    print("  [undock] Disengaging charging contacts")
    time.sleep(0.05)
    print("  [undock] Reversing from dock — clear in 0.5 m")
    time.sleep(0.05)
    print("  [undock] Undock complete — robot is free to navigate")
    return True
