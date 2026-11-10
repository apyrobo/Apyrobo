"""
Demo 1: 10-Drone Coordinated Survey
====================================
10 drones survey a 1 km² grid in parallel using APYROBO's fleet coordination.
Each drone gets an assigned sector; results stream in as sectors complete.

Run:
    pip install apyrobo
    python demo.py                   # 10 drones
    python demo.py --fleet-size 100  # any fleet size, one 1 km² sector each
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

sys.path.insert(0, __file__.rsplit("/demos/", 1)[0])  # repo root when cloned

from apyrobo import Agent, MockAdapter, Robot, SkillExecutor

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
# Configuration
# ---------------------------------------------------------------------------
NUM_DRONES = 10         # default fleet size; change it with --fleet-size
GRID_COLUMNS = 5        # sectors per row of the survey grid
SURVEY_SPEED = 0.8      # m/s (within safe speed envelope)
SECTOR_SIZE_M = 1000    # metres per sector side

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class SurveyResult:
    drone_id: str
    sector: tuple[int, int]
    anomalies: list[str]
    duration_s: float
    coverage_pct: float = field(default=100.0)


def sector_to_coords(col: int, row: int) -> tuple[float, float]:
    return col * SECTOR_SIZE_M + SECTOR_SIZE_M / 2, row * SECTOR_SIZE_M + SECTOR_SIZE_M / 2


def survey_sector(drone_id: str, col: int, row: int, agent: Agent) -> SurveyResult:
    """Plan and execute a survey mission for one grid sector."""
    adapter = MockAdapter(drone_id)
    robot = Robot(f"mock://{drone_id}", adapter)
    executor = SkillExecutor(robot)

    cx, cy = sector_to_coords(col, row)
    task = f"navigate_to and survey sector ({col},{row}) at position ({cx:.0f},{cy:.0f})"
    t0 = time.monotonic()
    plan = agent.plan(task, robot)
    executor.execute_graph(plan)
    duration = time.monotonic() - t0

    # Simulate occasional anomaly detection (thermal hotspot, debris)
    anomalies: list[str] = []
    if random.random() < 0.2:
        anomaly_type = random.choice(["thermal hotspot", "debris field", "obstacle cluster"])
        ax, ay = cx + random.randint(-50, 50), cy + random.randint(-50, 50)
        anomalies.append(f"{anomaly_type} at ({ax:.0f}, {ay:.0f})")

    return SurveyResult(drone_id=drone_id, sector=(col, row), anomalies=anomalies,
                        duration_s=duration)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _fleet_size(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {n}")
    return n


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="APYROBO coordinated drone survey on mock://")
    parser.add_argument("--fleet-size", type=_fleet_size, default=NUM_DRONES,
                        help=f"number of drones, one 1 km² sector each (default {NUM_DRONES})")
    num_drones = parser.parse_args(argv).fleet_size

    random.seed(42)
    print("=" * 60)
    print(f"  APYROBO — {num_drones}-Drone Coordinated Survey")
    print(f"  Coverage: {num_drones} km² | Drones: {num_drones} | Backend: mock://")
    print("=" * 60)

    agent = Agent(provider="rule")

    # Assign one 1 km² sector per drone, GRID_COLUMNS sectors per row
    sectors = [(i % GRID_COLUMNS, i // GRID_COLUMNS) for i in range(num_drones)]

    id_width = max(2, len(str(num_drones - 1)))
    drone_ids = [f"drone_{i:0{id_width}d}" for i in range(num_drones)]
    assignments = list(zip(drone_ids, sectors, strict=True))

    print(f"\n  Launching {num_drones} drones simultaneously...\n")
    t_fleet_start = time.monotonic()
    results: list[SurveyResult] = []

    with ThreadPoolExecutor(max_workers=min(num_drones, 32)) as pool:
        futures = {
            pool.submit(survey_sector, did, col, row, agent): did
            for did, (col, row) in assignments
        }
        for future in as_completed(futures):
            r = future.result()
            status = "⚠ anomaly" if r.anomalies else "✓ clear"
            print(f"  [{r.drone_id}] sector {r.sector} → {status}  ({r.duration_s*1000:.0f} ms)")
            results.append(r)
            _pace()

    fleet_duration = time.monotonic() - t_fleet_start - _PACED_S
    single_drone_estimate = sum(r.duration_s for r in results)

    anomaly_count = sum(len(r.anomalies) for r in results)
    print("\n" + "=" * 60)
    print(f"  Survey complete in {fleet_duration*1000:.0f} ms wall-clock time")
    if single_drone_estimate > fleet_duration:
        # Only meaningful when per-sector work outweighs thread overhead
        # (i.e. real flights, not instant mocks).
        print(f"  Sequential equivalent: {single_drone_estimate*1000:.0f} ms  "
              f"(fleet is {single_drone_estimate/fleet_duration:.1f}× faster)")
    print(f"  Sectors covered: {len(results)}/{num_drones}  "
          f"({sum(r.coverage_pct for r in results)/len(results):.0f}% avg coverage)")
    print(f"  Anomalies detected: {anomaly_count}")
    if anomaly_count:
        for r in results:
            for a in r.anomalies:
                print(f"    ↳ [{r.drone_id}] {a}")
    print("=" * 60)
    print()
    print("  Next steps:")
    print("  • No drone adapter ships yet: `apyrobo init my-drone --adapter` scaffolds one")
    print("    (docs/adapter_authoring.md), then swap mock:// for my-drone://")
    print("  • Add `apyrobo policy add \"never exceed 0.5 m/s near humans\"`")
    print("  • Scale up: `python demo.py --fleet-size 100`")
    print()


if __name__ == "__main__":
    main()
