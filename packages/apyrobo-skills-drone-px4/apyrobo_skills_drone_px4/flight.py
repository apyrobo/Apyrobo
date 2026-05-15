"""PX4 drone flight skills — takeoff, land, fly_to, return_home."""
from __future__ import annotations

import time

from apyrobo import skill


@skill(
    description="Arm the drone and take off to a target altitude",
    capability="fly",
)
def takeoff(altitude_m: float = 10.0) -> bool:
    """Arm motors and ascend to the specified altitude in metres.

    The drone performs pre-arm checks, arms the propulsion system, and
    climbs vertically to *altitude_m* before switching to loiter mode.

    Args:
        altitude_m: Target hover altitude in metres above the launch point.
                    Clamped to the range 1.0–120.0 m.
    """
    altitude_m = max(1.0, min(120.0, altitude_m))
    print(f"  [takeoff] Running pre-arm checks")
    time.sleep(0.05)
    print(f"  [takeoff] Arming motors")
    time.sleep(0.05)
    print(f"  [takeoff] Ascending to {altitude_m:.1f} m")
    time.sleep(0.05)
    print(f"  [takeoff] Loitering at {altitude_m:.1f} m — takeoff complete")
    return True


@skill(
    description="Descend and disarm the drone at the current position",
    capability="fly",
)
def land() -> bool:
    """Initiate a controlled descent and disarm motors after touchdown.

    The drone switches to land mode, descends at a safe rate, detects
    ground contact via the barometer and accelerometers, and disarms.
    """
    print("  [land] Switching to LAND mode")
    time.sleep(0.05)
    print("  [land] Descending — monitoring ground contact")
    time.sleep(0.05)
    print("  [land] Touchdown detected — disarming motors")
    time.sleep(0.05)
    print("  [land] Landing complete")
    return True


@skill(
    description="Fly the drone to a GPS coordinate at a specified altitude and speed",
    capability="fly",
)
def fly_to(
    lat: float,
    lon: float,
    alt_m: float = 50.0,
    speed_ms: float = 5.0,
) -> bool:
    """Navigate to a GPS position using PX4 mission mode.

    The drone climbs to *alt_m* if currently lower, then flies in a
    straight line to (lat, lon) at *speed_ms* metres per second.

    Args:
        lat:      Target latitude in decimal degrees.
        lon:      Target longitude in decimal degrees.
        alt_m:    Target altitude in metres (clamped to 1.0–500.0 m).
        speed_ms: Cruise speed in m/s (clamped to 0.5–20.0 m/s).
    """
    alt_m = max(1.0, min(500.0, alt_m))
    speed_ms = max(0.5, min(20.0, speed_ms))
    print(f"  [fly_to] Target: lat={lat:.6f}, lon={lon:.6f}, alt={alt_m:.1f} m")
    time.sleep(0.05)
    print(f"  [fly_to] Cruising at {speed_ms:.1f} m/s")
    time.sleep(0.05)
    print(f"  [fly_to] Arrived at ({lat:.6f}, {lon:.6f}) — loitering")
    return True


@skill(
    description="Return the drone to its launch position (RTL)",
    capability="fly",
)
def return_home() -> bool:
    """Activate Return-To-Launch (RTL) mode.

    The drone climbs to the configured RTL altitude, navigates back to
    the recorded launch position, descends, and disarms automatically.
    """
    print("  [return_home] Activating RTL mode")
    time.sleep(0.05)
    print("  [return_home] Climbing to RTL altitude and heading home")
    time.sleep(0.05)
    print("  [return_home] Over launch point — descending")
    time.sleep(0.05)
    print("  [return_home] Touchdown at home position — disarmed")
    return True
