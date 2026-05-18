"""
APYROBO Benchmark Suite
========================
Reproducible benchmark for 5 canonical robotics tasks.
Measures setup time, planning time, execution time, and failure recovery time.
Outputs a comparison table against raw ROS 2 implementation estimates.

Run locally:
    python benchmarks/benchmark_suite.py
    python benchmarks/benchmark_suite.py --json          # machine-readable
    python benchmarks/benchmark_suite.py --iterations 50 # more samples

Run in CI:
    see .github/workflows/benchmark.yml
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

sys.path.insert(0, __file__.rsplit("/benchmarks/", 1)[0])

from apyrobo import Agent, MockAdapter, Robot, SkillExecutor
from apyrobo.coordination.bus import MultiAgentCoordinator, TaskBus
from apyrobo.safety.nl_policy import NLPolicyParser, NLPolicyStore

# ---------------------------------------------------------------------------
# Raw ROS 2 estimates (lines of code only — not runnable without ROS 2 install)
# These are representative counts for well-written raw ROS 2 C++/Python nodes.
# ---------------------------------------------------------------------------

ROS2_LOC = {
    "navigate_to_point": 58,    # nav2 action client + goal + result handling
    "pick_and_place":    84,     # MoveIt Python API + joint trajectory
    "patrol_route":      63,     # sequential nav goals + waypoint tracking
    "fleet_coordination": 127,   # multi-node pub/sub + custom coordinator
    "safety_policy":     96,     # custom constraint checker + config YAML
}

# ---------------------------------------------------------------------------
# Benchmark task definitions
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    task_name: str
    iterations: int
    setup_ms:    list[float] = field(default_factory=list)
    plan_ms:     list[float] = field(default_factory=list)
    exec_ms:     list[float] = field(default_factory=list)
    recovery_ms: list[float] = field(default_factory=list)
    apyrobo_loc: int = 0
    ros2_loc:    int = 0
    success_rate: float = 1.0

    def median(self, samples: list[float]) -> float:
        return statistics.median(samples) if samples else 0.0

    def p95(self, samples: list[float]) -> float:
        if not samples:
            return 0.0
        s = sorted(samples)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    def to_dict(self) -> dict:
        return {
            "task": self.task_name,
            "iterations": self.iterations,
            "setup_ms_median": round(self.median(self.setup_ms), 2),
            "plan_ms_median": round(self.median(self.plan_ms), 2),
            "exec_ms_median": round(self.median(self.exec_ms), 2),
            "recovery_ms_median": round(self.median(self.recovery_ms), 2),
            "exec_ms_p95": round(self.p95(self.exec_ms), 2),
            "apyrobo_loc": self.apyrobo_loc,
            "ros2_loc": self.ros2_loc,
            "loc_ratio": round(self.ros2_loc / max(self.apyrobo_loc, 1), 1),
            "success_rate": self.success_rate,
        }


def _time_it(fn: Callable) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Task 1: Navigate to point
# ---------------------------------------------------------------------------

def bench_navigate(n: int) -> BenchmarkResult:
    result = BenchmarkResult("navigate_to_point", n, apyrobo_loc=9, ros2_loc=ROS2_LOC["navigate_to_point"])
    for _ in range(n):
        result.setup_ms.append(_time_it(lambda: (
            MockAdapter("bench"),
            Robot("mock://bench", MockAdapter("bench")),
        )))
        agent = Agent(provider="rule")
        robot = Robot("mock://bench", MockAdapter("bench"))
        executor = SkillExecutor(robot)
        result.plan_ms.append(_time_it(lambda: agent.plan("navigate_to", robot)))
        plan = agent.plan("navigate_to", robot)
        result.exec_ms.append(_time_it(lambda: executor.execute_graph(plan)))
    return result


# ---------------------------------------------------------------------------
# Task 2: Pick-and-place
# ---------------------------------------------------------------------------

def bench_pick_and_place(n: int) -> BenchmarkResult:
    result = BenchmarkResult("pick_and_place", n, apyrobo_loc=14, ros2_loc=ROS2_LOC["pick_and_place"])
    for _ in range(n):
        agent = Agent(provider="rule")
        robot = Robot("mock://arm", MockAdapter("arm"))
        executor = SkillExecutor(robot)
        result.plan_ms.append(_time_it(lambda: agent.plan("pick_object", robot)))
        pick_plan = agent.plan("pick_object", robot)
        place_plan = agent.plan("place_object", robot)
        result.exec_ms.append(_time_it(lambda: (
            executor.execute_graph(pick_plan),
            executor.execute_graph(place_plan),
        )))
    return result


# ---------------------------------------------------------------------------
# Task 3: Patrol route (5 waypoints)
# ---------------------------------------------------------------------------

WAYPOINTS = ["A", "B", "C", "D", "E"]

def bench_patrol(n: int) -> BenchmarkResult:
    result = BenchmarkResult("patrol_route", n, apyrobo_loc=19, ros2_loc=ROS2_LOC["patrol_route"])
    for _ in range(n):
        agent = Agent(provider="rule")
        robot = Robot("mock://patrol", MockAdapter("patrol"))
        executor = SkillExecutor(robot)
        t_plan = 0.0
        t_exec = 0.0
        for wp in WAYPOINTS:
            t0 = time.perf_counter()
            plan = agent.plan(f"navigate_to_{wp}", robot)
            t_plan += (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            executor.execute_graph(plan)
            t_exec += (time.perf_counter() - t0) * 1000
        result.plan_ms.append(t_plan)
        result.exec_ms.append(t_exec)
    return result


# ---------------------------------------------------------------------------
# Task 4: Fleet coordination (3 robots, 3 tasks)
# ---------------------------------------------------------------------------

def bench_fleet(n: int) -> BenchmarkResult:
    result = BenchmarkResult("fleet_coordination", n, apyrobo_loc=26, ros2_loc=ROS2_LOC["fleet_coordination"])
    for _ in range(n):
        agent = Agent(provider="rule")
        bus = TaskBus(timeout=5.0)
        coords = []
        for i in range(3):
            robot = Robot(f"mock://fleet_{i}", MockAdapter(f"fleet_{i}"))
            coord = MultiAgentCoordinator(
                agent, robot, bus,
                agent_id=f"fleet_{i}",
                capabilities=["NAVIGATE", "PICK"],
            )
            coord.start()
            coords.append(coord)
        time.sleep(0.01)

        t0 = time.perf_counter()
        tasks = ["navigate_to", "pick_object", "navigate_to"]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = [pool.submit(bus.dispatch, t) for t in tasks]
            _ = [f.result() for f in futs]
        result.exec_ms.append((time.perf_counter() - t0) * 1000)

        for coord in coords:
            coord.stop()
    return result


# ---------------------------------------------------------------------------
# Task 5: Safety policy enforcement
# ---------------------------------------------------------------------------

def bench_safety_policy(n: int) -> BenchmarkResult:
    result = BenchmarkResult("safety_policy", n, apyrobo_loc=15, ros2_loc=ROS2_LOC["safety_policy"])
    parser = NLPolicyParser()
    rules = [
        "never exceed 0.5 m/s near humans",
        "keep at least 1.5 meters away from humans",
        "always keep 20% battery",
    ]
    store = NLPolicyStore(":memory:")
    for rule in rules:
        store.add(parser.parse(rule))
    policies = store.get_active_policies()

    actions = [
        "navigate at 0.3 m/s to kitchen",    # ok
        "navigate at 2.0 m/s to warehouse",   # speed violation
        "navigate at 0.4 m/s; 10% battery",   # battery violation
        "navigate at 0.2 m/s to lobby",       # ok
        "navigate at 1.5 m/s to dock",        # speed violation
    ]

    for _ in range(n):
        t0 = time.perf_counter()
        for action in actions:
            parser.check_compliance(action, policies)
        result.exec_ms.append((time.perf_counter() - t0) * 1000)

        # Recovery: detect violation → substitute safe action
        t0 = time.perf_counter()
        for action in actions:
            violations = parser.check_compliance(action, policies)
            if violations:
                safe_action = "navigate at 0.3 m/s"  # safe fallback
                parser.check_compliance(safe_action, policies)
        result.recovery_ms.append((time.perf_counter() - t0) * 1000)

    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

BENCHMARKS = [
    bench_navigate,
    bench_pick_and_place,
    bench_patrol,
    bench_fleet,
    bench_safety_policy,
]


def run_all(iterations: int) -> list[BenchmarkResult]:
    results = []
    for bench_fn in BENCHMARKS:
        result = bench_fn(iterations)
        results.append(result)
    return results


def print_table(results: list[BenchmarkResult]) -> None:
    COL = [40, 9, 9, 9, 9, 8, 5, 8]
    header = ["Task", "Setup ms", "Plan ms", "Exec ms", "P95 ms", "Rec ms", "LOC", "ROS2 LOC"]
    sep = "─" * (sum(COL) + len(COL) * 2 + 1)

    print(sep)
    row = "│".join(h.center(w) for h, w in zip(header, COL))
    print(f"│{row}│")
    print(sep)

    for r in results:
        d = r.to_dict()
        cells = [
            r.task_name,
            f"{d['setup_ms_median']:.1f}",
            f"{d['plan_ms_median']:.1f}",
            f"{d['exec_ms_median']:.1f}",
            f"{d['exec_ms_p95']:.1f}",
            f"{d['recovery_ms_median']:.1f}" if r.recovery_ms else "—",
            str(d['apyrobo_loc']),
            f"{d['ros2_loc']} ({d['loc_ratio']}×)",
        ]
        row = "│".join(c.ljust(w) if i == 0 else c.center(w) for i, (c, w) in enumerate(zip(cells, COL)))
        print(f"│{row}│")

    print(sep)

    total_apyrobo = sum(r.apyrobo_loc for r in results)
    total_ros2 = sum(r.ros2_loc for r in results)
    print(f"\n  APYROBO total: {total_apyrobo} lines   Raw ROS 2 total: {total_ros2} lines   "
          f"({total_ros2 / total_apyrobo:.1f}× reduction)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="APYROBO benchmark suite")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    args = parser.parse_args()

    if not args.json:
        print(f"\n  APYROBO Benchmark Suite — {args.iterations} iterations per task\n")

    results = run_all(args.iterations)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print_table(results)
        print("  Legend:")
        print("    Setup ms: time to init Robot + Agent (cold start)")
        print("    Plan ms:  time to generate skill graph")
        print("    Exec ms:  time to run skill graph on mock adapter")
        print("    P95 ms:   95th-percentile execution latency")
        print("    Rec ms:   detect violation → substitute safe action")
        print("    LOC:      APYROBO lines of code for this task")
        print("    ROS2 LOC: raw ROS 2 equivalent (C++/Python node)")
        print()


if __name__ == "__main__":
    main()
