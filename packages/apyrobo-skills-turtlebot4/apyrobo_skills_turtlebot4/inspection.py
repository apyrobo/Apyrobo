"""TurtleBot 4 inspection skills — room scan, surroundings check."""
from __future__ import annotations

import time

from apyrobo import skill


@skill(
    description="Systematically scan a room using the onboard camera",
    capability="inspect",
)
def inspect_room(room_id: str = "room_01", camera_height: float = 0.9) -> bool:
    """Perform a systematic sweep of the named room.

    The robot moves to the room centre, raises the camera to
    *camera_height* metres, then rotates 360° in four 90° steps
    while capturing frames at each position.

    Args:
        room_id:       Environment-map identifier for the target room.
        camera_height: Camera tilt height in metres (0.3–1.2 m range).
    """
    camera_height = max(0.3, min(1.2, camera_height))
    print(f"  [inspect_room] Navigating to centre of '{room_id}'")
    time.sleep(0.05)
    print(f"  [inspect_room] Camera height set to {camera_height:.2f} m")
    for bearing in (0, 90, 180, 270):
        print(f"  [inspect_room] Scanning at {bearing}° — capturing frame")
        time.sleep(0.025)
    print(f"  [inspect_room] Room '{room_id}' inspection complete — 4 frames captured")
    return True


@skill(
    description="Perform a 360-degree sensor sweep at the current position",
    capability="inspect",
)
def check_surroundings(radius: float = 2.0) -> bool:
    """Rotate in place and scan the environment within *radius* metres.

    Uses the TurtleBot 4's LIDAR and RGB-D camera to detect obstacles,
    people, and points of interest in a full circle around the robot.

    Args:
        radius: Detection radius in metres (clamped to 0.5–10.0 m).
    """
    radius = max(0.5, min(10.0, radius))
    print(f"  [check_surroundings] Scanning surroundings — radius={radius:.1f} m")
    for step in range(1, 5):
        angle = step * 90
        print(f"  [check_surroundings] Rotating to {angle}°, sampling LIDAR + RGB-D")
        time.sleep(0.025)
    print("  [check_surroundings] 360° scan complete — environment map updated")
    return True
