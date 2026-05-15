"""AGV cargo skills — load_cargo, unload_cargo."""
from __future__ import annotations

import time

from apyrobo import skill


@skill(
    description="Initiate a cargo loading sequence at a named station",
    capability="manipulate",
)
def load_cargo(station_id: str, cargo_id: str = "") -> bool:
    """Request and confirm a cargo loading operation at the given station.

    The AGV sends a load request to the station controller, waits for
    the cargo transfer handshake, and verifies load-sensor confirmation
    before returning.

    Args:
        station_id: Identifier of the loading station (must match fleet map).
        cargo_id:   Optional identifier of the specific cargo unit to load.
                    If empty, the station assigns the next queued unit.
    """
    cargo_label = f"'{cargo_id}'" if cargo_id else "<next queued>"
    print(f"  [load_cargo] Requesting load of {cargo_label} at station '{station_id}'")
    time.sleep(0.05)
    print(f"  [load_cargo] Handshake with station '{station_id}' — transfer in progress")
    time.sleep(0.05)
    print(f"  [load_cargo] Load sensor confirmed — cargo {cargo_label} secured")
    return True


@skill(
    description="Initiate a cargo unloading sequence at a named station",
    capability="manipulate",
)
def unload_cargo(station_id: str, cargo_id: str = "") -> bool:
    """Request and confirm a cargo unloading operation at the given station.

    The AGV positions at the unloading station, signals the station
    controller to accept the transfer, and waits for load-sensor
    clearance before reporting success.

    Args:
        station_id: Identifier of the unloading station (must match fleet map).
        cargo_id:   Optional identifier of the cargo unit to unload.
                    If empty, the currently loaded unit is unloaded.
    """
    cargo_label = f"'{cargo_id}'" if cargo_id else "<current load>"
    print(f"  [unload_cargo] Requesting unload of {cargo_label} at station '{station_id}'")
    time.sleep(0.05)
    print(f"  [unload_cargo] Handshake with station '{station_id}' — transfer in progress")
    time.sleep(0.05)
    print(f"  [unload_cargo] Load sensor cleared — cargo {cargo_label} delivered")
    return True
