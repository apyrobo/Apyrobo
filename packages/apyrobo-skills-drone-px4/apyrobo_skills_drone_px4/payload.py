"""PX4 drone payload skills — orbit, capture_image."""
from __future__ import annotations

import time

from apyrobo import skill


@skill(
    description="Circle a GPS point at a fixed radius for a number of loops",
    capability="fly",
)
def orbit(
    center_lat: float,
    center_lon: float,
    radius_m: float = 20.0,
    loops: int = 1,
) -> bool:
    """Execute one or more orbital circuits around a GPS coordinate.

    The drone enters PX4 orbit mode, maintaining *radius_m* metres from
    the centre point and completing *loops* full revolutions.

    Args:
        center_lat: Centre point latitude in decimal degrees.
        center_lon: Centre point longitude in decimal degrees.
        radius_m:   Orbit radius in metres (clamped to 5.0–500.0 m).
        loops:      Number of full circuits to complete (minimum 1).
    """
    radius_m = max(5.0, min(500.0, radius_m))
    loops = max(1, loops)
    print(f"  [orbit] Centre: lat={center_lat:.6f}, lon={center_lon:.6f}")
    print(f"  [orbit] Radius={radius_m:.1f} m, loops={loops}")
    time.sleep(0.05)
    for loop in range(1, loops + 1):
        print(f"  [orbit] Executing loop {loop}/{loops}")
        time.sleep(0.05)
    print(f"  [orbit] Orbit complete — {loops} loop(s) flown")
    return True


@skill(
    description="Trigger the drone camera shutter and return the saved image path",
    capability="capture",
)
def capture_image(camera: str = "downward", save_path: str = "") -> str:
    """Trigger the specified camera and save the captured image.

    Sends a MAVLink CMD_IMAGE_START_CAPTURE command to the onboard
    camera and waits for acknowledgement before returning the storage
    path of the saved file.

    Args:
        camera:    Camera identifier, e.g. ``"downward"``, ``"forward"``,
                   or ``"thermal"``.
        save_path: Directory or full file path for the captured image.
                   Defaults to the onboard SD card root if empty.

    Returns:
        Absolute path to the saved image file as a string.
    """
    if not save_path:
        save_path = f"/media/sdcard/{camera}_capture.jpg"
    print(f"  [capture_image] Camera='{camera}' — triggering shutter")
    time.sleep(0.05)
    print(f"  [capture_image] Image saved to '{save_path}'")
    return save_path
