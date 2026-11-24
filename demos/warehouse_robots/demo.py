"""
Demo 2: Warehouse Multi-Robot Pick-and-Pack
============================================
Three specialized robots collaborate to fill orders:
  picker_bot  → PICK + NAVIGATE
  packer_bot  → PLACE + NAVIGATE
  hauler_bot  → HAUL + NAVIGATE (takes packed boxes to the dock)

APYROBO's TaskBus routes each step to the robot with the right capability,
and that robot's agent plans it. The bus plans; it does not drive the mock
robots. A step the agent refuses (for example a drive with no target) is
shown as not planned, with the agent's reason.

Run:
    pip install apyrobo
    python demo.py
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, __file__.rsplit("/demos/", 1)[0])  # repo root when cloned

from apyrobo import Agent, MockAdapter, Robot
from apyrobo.coordination.bus import MultiAgentCoordinator, TaskBus

# Recording pace: seconds between visible steps so demo videos play at a
# watchable speed (set by demos/*/record.sh; 0 = full speed). Paced time is
# tracked so the printed timing stats reflect real work, not the sleeps.
_PACE = float(os.environ.get("APYROBO_DEMO_PACE", "0") or 0)
_PACED_S = 0.0


def _pace() -> None:
    global _PACED_S
    if _PACE > 0:
        time.sleep(_PACE)
        _PACED_S += _PACE

# ---------------------------------------------------------------------------
# Order catalogue
# ---------------------------------------------------------------------------

ORDERS = [
    {"id": "ORD-001", "items": ["widget_A", "widget_B"], "dest": "dock_1"},
    {"id": "ORD-002", "items": ["gadget_X"], "dest": "dock_2"},
    {"id": "ORD-003", "items": ["widget_A", "gadget_Y", "gadget_Z"], "dest": "dock_1"},
    {"id": "ORD-004", "items": ["widget_C", "widget_D"], "dest": "dock_3"},
    {"id": "ORD-005", "items": ["gadget_X", "gadget_Y"], "dest": "dock_2"},
]

# Where things are on the warehouse floor (metres). The rule planner never
# guesses a target, so every drive names its coordinates.
SHELVES = {
    "widget_A": (2.0, 3.0),
    "widget_B": (2.0, 5.0),
    "widget_C": (2.0, 7.0),
    "widget_D": (2.0, 9.0),
    "gadget_X": (4.0, 3.0),
    "gadget_Y": (4.0, 5.0),
    "gadget_Z": (4.0, 7.0),
}
PACK_STATION = (6.0, 2.0)
DOCKS = {"dock_1": (12.0, 1.0), "dock_2": (12.0, 4.0), "dock_3": (12.0, 7.0)}


def _xy(point: tuple[float, float]) -> str:
    return f"({point[0]:g}, {point[1]:g})"


def _show(result, done: str) -> bool:
    """Print one step and return whether it was planned."""
    if result.success:
        skills = ", ".join(result.skills_planned)
        print(f"  │  ✓ [{result.agent_id}] {done}  (planned: {skills})")
    else:
        print(f"  │  ✗ [{result.agent_id}] not planned: {result.error}")
    return result.success

# ---------------------------------------------------------------------------
# Robot fleet setup
# ---------------------------------------------------------------------------

@dataclass
class WarehouseRobot:
    name: str
    role: str
    capabilities: list[str]


FLEET: list[WarehouseRobot] = [
    WarehouseRobot("picker_bot", "Picker",   ["PICK", "NAVIGATE"]),
    WarehouseRobot("packer_bot", "Packer",   ["PLACE", "NAVIGATE"]),
    WarehouseRobot("hauler_bot", "Hauler",   ["HAUL", "NAVIGATE"]),
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  APYROBO — Warehouse Multi-Robot Pick-and-Pack")
    print(f"  Fleet: {len(FLEET)} robots | Orders: {len(ORDERS)} | Backend: mock://")
    print("=" * 65)

    bus = TaskBus(timeout=10.0)

    # Spin up all three robots on the task bus
    coordinators: list[MultiAgentCoordinator] = []
    for wr in FLEET:
        robot = Robot(f"mock://{wr.name}", MockAdapter(wr.name))
        coord = MultiAgentCoordinator(
            Agent(provider="rule"), robot, bus,
            agent_id=wr.name,
            capabilities=wr.capabilities,
        )
        coord.start()
        coordinators.append(coord)

    time.sleep(0.05)  # let worker threads settle

    print(f"\n  Fleet online: {', '.join(wr.name for wr in FLEET)}")
    print(f"  Processing {len(ORDERS)} orders...\n")

    t_start = time.monotonic()
    orders_planned = 0
    items_processed = 0
    step_times: list[float] = []

    def dispatch(task: str, capability: str, **metadata):
        t0 = time.monotonic()
        result = bus.dispatch(task, required_capability=capability, metadata=metadata)
        step_times.append(time.monotonic() - t0)
        return result

    for order in ORDERS:
        order_t = time.monotonic()
        order_paced0 = _PACED_S
        print(f"  ┌─ {order['id']}  ({len(order['items'])} items → {order['dest']})")
        ok = True

        # Step 1: picker drives to each item's shelf
        for item in order["items"]:
            shelf = _xy(SHELVES[item])
            result = dispatch(f"go to {shelf}", "PICK", item=item, order=order["id"])
            ok &= _show(result, f"to {item!r} shelf at {shelf}")
            items_processed += 1
            _pace()

        # Step 2: packer consolidates items into a box
        result = dispatch("place_object", "PLACE", order=order["id"], items=order["items"])
        ok &= _show(result, f"pack box for {order['id']}")
        _pace()

        # Step 3: hauler takes the box from the pack station to the dock
        dock = _xy(DOCKS[order["dest"]])
        result = dispatch(
            f"deliver from {_xy(PACK_STATION)} to {dock}", "HAUL",
            dest=order["dest"], order=order["id"],
        )
        ok &= _show(result, f"deliver to {order['dest']} at {dock}")
        order_elapsed = time.monotonic() - order_t - (_PACED_S - order_paced0)
        verdict = "planned" if ok else "NOT fully planned"
        print(f"  └─ {order['id']} {verdict} in {order_elapsed*1000:.0f} ms\n")
        orders_planned += ok
        _pace()

    fleet_elapsed = time.monotonic() - t_start - _PACED_S

    for coord in coordinators:
        coord.stop()

    avg_step_ms = sum(step_times) / len(step_times) * 1000 if step_times else 0
    print("=" * 65)
    wall_ms = fleet_elapsed * 1000
    print(f"  {orders_planned}/{len(ORDERS)} orders planned in {wall_ms:.0f} ms wall-clock")
    print(f"  Items processed: {items_processed}")
    print(f"  Total dispatch calls: {len(step_times)} (avg {avg_step_ms:.0f} ms/call)")
    print(f"  Throughput: {orders_planned / fleet_elapsed:.0f} orders planned/sec")
    print("=" * 65)
    print()
    print("  Next steps:")
    print("  • Cap speed near people: `apyrobo policy add \"never exceed 0.5 m/s near humans\"`")
    print("  • Connect a real robot: `import apyrobo.core.unitree_adapter`, then")
    print("    `Robot.discover('unitree://go2@192.168.1.10')` (needs the Unitree SDK)")
    print("  • Add `bus.broadcast('emergency_stop')` for fleet-wide safety halt")
    print()


if __name__ == "__main__":
    main()
