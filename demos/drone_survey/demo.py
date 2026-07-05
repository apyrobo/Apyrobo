"""
Demo 1: 10-Drone Coordinated Survey
====================================
10 drones survey a 1 km² grid in parallel using APYROBO's fleet coordination.
Each drone gets an assigned sector; results stream in as sectors complete.

Run:
    pip install apyrobo
    python demo.py
"""
from __future__ import annotations

import math
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
NUM_DRONES = 10
GRID_SIZE = 10          # 10×10 km² surveyed, 1 km² per sector
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
        anomalies.append(f"{anomaly_type} at ({cx + random.randint(-50, 50):.0f}, {cy + random.randint(-50, 50):.0f})")

    return SurveyResult(drone_id=drone_id, sector=(col, row), anomalies=anomalies, duration_s=duration)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(42)
    print("=" * 60)
    print("  APYROBO — 10-Drone Coordinated Survey")
    print("  Coverage: 10 km² | Drones: 10 | Backend: mock://")
    print("=" * 60)

    agent = Agent(provider="rule")

    # Assign one sector per drone (10 drones × 1 km² = 10 km² of a 10×10 grid)
    sectors = [(col, row) for row in range(2) for col in range(5)]
    assert len(sectors) == NUM_DRONES

    drone_ids = [f"drone_{i:02d}" for i in range(NUM_DRONES)]
    assignments = list(zip(drone_ids, sectors))

    print(f"\n  Launching {NUM_DRONES} drones simultaneously...\n")
    t_fleet_start = time.monotonic()
    results: list[SurveyResult] = []

    with ThreadPoolExecutor(max_workers=NUM_DRONES) as pool:
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
    print(f"  Sectors covered: {len(results)}/{NUM_DRONES}  "
          f"({sum(r.coverage_pct for r in results)/len(results):.0f}% avg coverage)")
    print(f"  Anomalies detected: {anomaly_count}")
    if anomaly_count:
        for r in results:
            for a in r.anomalies:
                print(f"    ↳ [{r.drone_id}] {a}")
    print("=" * 60)
    print()
    print("  Next steps:")
    print("  • Swap mock:// URIs for real drone URIs (DDS, MAVLink, etc.)")
    print("  • Add `apyrobo policy add \"never exceed 0.5 m/s near humans\"`")
    print("  • Scale to 100 drones with --fleet-size flag")
    print()


if __name__ == "__main__":
    main()
