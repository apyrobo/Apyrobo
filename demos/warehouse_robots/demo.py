"""
Demo 2: Warehouse Multi-Robot Pick-and-Pack
============================================
Three specialized robots collaborate to fill orders:
  picker_bot  → PICK + NAVIGATE
  packer_bot  → PLACE + NAVIGATE
  hauler_bot  → NAVIGATE (transports packed boxes)

APYROBO's TaskBus routes each step to the robot with the right capability.

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
    WarehouseRobot("hauler_bot", "Hauler",   ["NAVIGATE"]),
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  APYROBO — Warehouse Multi-Robot Pick-and-Pack")
    print(f"  Fleet: {len(FLEET)} robots | Orders: {len(ORDERS)} | Backend: mock://")
    print("=" * 65)

    agent = Agent(provider="rule")
    bus = TaskBus(timeout=10.0)

    # Spin up all three robots on the task bus
    coordinators: list[MultiAgentCoordinator] = []
    for wr in FLEET:
        robot = Robot(f"mock://{wr.name}", MockAdapter(wr.name))
        coord = MultiAgentCoordinator(
            agent, robot, bus,
            agent_id=wr.name,
            capabilities=wr.capabilities,
        )
        coord.start()
        coordinators.append(coord)

    time.sleep(0.05)  # let worker threads settle

    print(f"\n  Fleet online: {', '.join(wr.name for wr in FLEET)}")
    print(f"  Processing {len(ORDERS)} orders...\n")

    t_start = time.monotonic()
    orders_filled = 0
    items_processed = 0
    step_times: list[float] = []

    for order in ORDERS:
        order_t = time.monotonic()
        order_paced0 = _PACED_S
        print(f"  ┌─ {order['id']}  ({len(order['items'])} items → {order['dest']})")

        # Step 1: picker navigates to each item shelf and picks
        for item in order["items"]:
            t0 = time.monotonic()
            result = bus.dispatch(
                "navigate_to",
                required_capability="PICK",
                metadata={"item": item, "order": order["id"]},
            )
            dt = time.monotonic() - t0
            step_times.append(dt)
            icon = "✓" if result.success else "✗"
            print(f"  │  {icon} [picker_bot] picked {item!r}  ({dt*1000:.0f} ms)")
            items_processed += 1
            _pace()

        # Step 2: packer consolidates items into a box
        t0 = time.monotonic()
        result = bus.dispatch(
            "place_object",
            required_capability="PLACE",
            metadata={"order": order["id"], "items": order["items"]},
        )
        dt = time.monotonic() - t0
        step_times.append(dt)
        icon = "✓" if result.success else "✗"
        print(f"  │  {icon} [packer_bot] packed box for {order['id']}  ({dt*1000:.0f} ms)")
        _pace()

        # Step 3: hauler delivers to dock
        t0 = time.monotonic()
        result = bus.dispatch(
            "navigate_to",
            required_capability="NAVIGATE",
            metadata={"dest": order["dest"], "order": order["id"]},
        )
        dt = time.monotonic() - t0
        step_times.append(dt)
        icon = "✓" if result.success else "✗"
        order_elapsed = time.monotonic() - order_t - (_PACED_S - order_paced0)
        print(f"  │  {icon} [hauler_bot] delivered to {order['dest']}  ({dt*1000:.0f} ms)")
        print(f"  └─ {order['id']} filled in {order_elapsed*1000:.0f} ms\n")
        orders_filled += 1
        _pace()

    fleet_elapsed = time.monotonic() - t_start - _PACED_S

    for coord in coordinators:
        coord.stop()

    avg_step_ms = sum(step_times) / len(step_times) * 1000 if step_times else 0
    print("=" * 65)
    print(f"  All {orders_filled} orders filled in {fleet_elapsed*1000:.0f} ms wall-clock")
    print(f"  Items processed: {items_processed}")
    print(f"  Total dispatch calls: {len(step_times)} (avg {avg_step_ms:.0f} ms/call)")
    print(f"  Throughput: {orders_filled / fleet_elapsed:.0f} orders/sec")
    print("=" * 65)
    print()
    print("  Next steps:")
    print("  • Add `apyrobo policy add \"keep 1.5m away from humans\"` for safe co-working")
    print("  • Connect real robots: `Robot('unitree://go2@192.168.1.10', ...)`")
    print("  • Add `bus.broadcast('emergency_stop')` for fleet-wide safety halt")
    print()


if __name__ == "__main__":
    main()
