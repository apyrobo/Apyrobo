"""AGV navigation skills — navigate_to, follow_route, dock_to_station."""
from __future__ import annotations

import time

from apyrobo import skill


@skill(
    description="Navigate the AGV to a pose on a named map",
    capability="navigate",
)
def navigate_to(
    x: float,
    y: float,
    theta: float = 0.0,
    map_id: str = "default",
) -> bool:
    """Drive to a 2-D pose (x, y, theta) on the specified environment map.

    The AGV plans a collision-free path, executes it, and confirms
    arrival within the configured position tolerance.

    Args:
        x:      Target X position in metres relative to map origin.
        y:      Target Y position in metres relative to map origin.
        theta:  Target heading in radians (0 = positive X axis).
        map_id: Named map to use for path planning (default: ``"default"``).
    """
    print(f"  [navigate_to] Map='{map_id}' — planning path to ({x:.3f}, {y:.3f}, θ={theta:.3f} rad)")
    time.sleep(0.05)
    print(f"  [navigate_to] Executing path")
    time.sleep(0.05)
    print(f"  [navigate_to] Arrived at ({x:.3f}, {y:.3f}) on map '{map_id}'")
    return True


@skill(
    description="Execute a pre-programmed route by its identifier",
    capability="navigate",
)
def follow_route(route_id: str, loop: bool = False) -> bool:
    """Run a stored route mission, optionally repeating it indefinitely.

    The AGV loads the named route from its mission store and drives each
    waypoint in sequence. When *loop* is ``True`` the route repeats until
    an external stop command is received; in simulation a single pass is
    always executed.

    Args:
        route_id: Identifier of the pre-programmed route in the AGV mission store.
        loop:     If ``True``, repeat the route continuously (single pass in tests).
    """
    mode = "looping" if loop else "single-pass"
    print(f"  [follow_route] Loading route '{route_id}' — mode={mode}")
    time.sleep(0.05)
    print(f"  [follow_route] Executing route '{route_id}'")
    time.sleep(0.05)
    print(f"  [follow_route] Route '{route_id}' complete")
    return True


@skill(
    description="Perform precise docking at a named station",
    capability="navigate",
)
def dock_to_station(station_id: str, approach_speed: float = 0.2) -> bool:
    """Navigate to and precisely dock at the specified station.

    The AGV uses fiducial markers or reflector tape to achieve sub-centimetre
    docking accuracy. *approach_speed* is reduced further during the final
    docking phase for safe contact.

    Args:
        station_id:     Identifier of the docking station in the environment map.
        approach_speed: Approach speed in m/s (clamped to 0.05–0.5 m/s).
    """
    approach_speed = max(0.05, min(0.5, approach_speed))
    print(f"  [dock_to_station] Navigating to station '{station_id}'")
    time.sleep(0.05)
    print(f"  [dock_to_station] Approaching at {approach_speed:.2f} m/s — aligning fiducials")
    time.sleep(0.05)
    print(f"  [dock_to_station] Docked at '{station_id}' — contact confirmed")
    return True
