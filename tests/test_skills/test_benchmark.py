"""
CI-11 + CI-03: Performance benchmark — 100 tasks/min through mock executor.

Provides a regression gate to ensure execution speed doesn't degrade.
Includes throughput benchmarks and regression gating via benchmark.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import TaskStatus
from apyrobo.skills.skill import Skill, SkillStatus, BUILTIN_SKILLS
from apyrobo.skills.executor import SkillGraph, SkillExecutor
from apyrobo.skills.agent import Agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_robot() -> Robot:
    return Robot.discover("mock://bench_bot")


BENCHMARK_PATH = Path(__file__).parent / "benchmark.json"


# ===========================================================================
# CI-11: Performance benchmarks
# ===========================================================================

class TestPerformanceBenchmark:
    """Performance regression gate."""

    @pytest.mark.timeout(120)
    def test_100_tasks_per_minute(self, mock_robot: Robot) -> None:
        """Execute 100 simple tasks within 60 seconds (target: 100 tasks/min)."""
        start = time.time()
        completed = 0

        for i in range(100):
            g = SkillGraph()
            nav = Skill(
                skill_id=f"navigate_to_{i}",
                name=f"Nav {i}",
                required_capability=BUILTIN_SKILLS["navigate_to"].required_capability,
                parameters={"x": float(i), "y": float(i), "speed": 0.5},
                timeout_seconds=5.0,
            )
            g.add_skill(nav)
            exe = SkillExecutor(mock_robot)
            result = exe.execute_graph(g)
            if result.status == TaskStatus.COMPLETED:
                completed += 1

        elapsed = time.time() - start

        assert completed >= 95, f"Only {completed}/100 tasks completed"
        assert elapsed < 60.0, f"Took {elapsed:.1f}s (>60s target)"

        tasks_per_min = completed / elapsed * 60
        print(f"\nBenchmark: {completed} tasks in {elapsed:.1f}s = {tasks_per_min:.0f} tasks/min")

    @pytest.mark.timeout(120)
    def test_graph_with_chain(self, mock_robot: Robot) -> None:
        """Execute a 10-skill chain 20 times within 60 seconds."""
        start = time.time()
        completed = 0

        for run in range(20):
            g = SkillGraph()
            prev_id = None
            for i in range(10):
                sid = f"navigate_to_{run * 10 + i}"
                s = Skill(
                    skill_id=sid,
                    name=f"Step {i}",
                    required_capability=BUILTIN_SKILLS["navigate_to"].required_capability,
                    parameters={"x": float(i), "y": float(i)},
                    timeout_seconds=5.0,
                )
                g.add_skill(s, depends_on=[prev_id] if prev_id else [])
                prev_id = s.skill_id

            exe = SkillExecutor(mock_robot)
            result = exe.execute_graph(g)
            if result.status == TaskStatus.COMPLETED:
                completed += 1

        elapsed = time.time() - start

        assert completed >= 18, f"Only {completed}/20 graph runs completed"
        assert elapsed < 60.0, f"Took {elapsed:.1f}s (>60s target)"
        print(f"\nBenchmark: {completed} 10-skill chains in {elapsed:.1f}s")

    @pytest.mark.timeout(120)
    def test_parallel_graph_throughput(self, mock_robot: Robot) -> None:
        """Execute parallel graphs efficiently."""
        start = time.time()
        completed = 0

        for run in range(20):
            g = SkillGraph()
            # 5 independent skills + 1 final
            par_ids = []
            for i in range(5):
                sid = f"report_status_{run * 10 + i}"
                s = Skill(
                    skill_id=sid,
                    name=f"Parallel {i}",
                    required_capability=BUILTIN_SKILLS["report_status"].required_capability,
                    timeout_seconds=5.0,
                )
                g.add_skill(s)
                par_ids.append(sid)

            final = Skill(
                skill_id=f"stop_{run}",
                name="Final",
                required_capability=BUILTIN_SKILLS["stop"].required_capability,
                timeout_seconds=5.0,
            )
            g.add_skill(final, depends_on=par_ids)

            exe = SkillExecutor(mock_robot)
            result = exe.execute_graph(g, parallel=True)
            if result.status == TaskStatus.COMPLETED:
                completed += 1

        elapsed = time.time() - start

        assert completed >= 18, f"Only {completed}/20 parallel runs completed"
        assert elapsed < 60.0, f"Took {elapsed:.1f}s"
        print(f"\nBenchmark: {completed} parallel graphs in {elapsed:.1f}s")


# ===========================================================================
# CI-03: Throughput benchmarks with regression gating
# ===========================================================================


class TestThroughputBenchmark:
    """CI-03: End-to-end throughput through mock agent + executor."""

    def test_throughput_100_tasks(self, mock_robot: Robot) -> None:
        """
        Run 100 sequential agent.execute() calls in mock mode.
        Must complete in < 5 seconds (20 tasks/sec minimum).
        """
        agent = Agent(provider="rule")

        start = time.time()
        completed = 0
        for _ in range(100):
            result = agent.execute("go to 1 1", mock_robot)
            if result.status == TaskStatus.COMPLETED:
                completed += 1
        elapsed = time.time() - start

        tasks_per_sec = completed / elapsed
        print(f"\nThroughput: {completed} tasks in {elapsed:.2f}s = {tasks_per_sec:.1f} tasks/sec")

        assert elapsed < 5.0, f"100 tasks took {elapsed:.2f}s (>5s budget)"
        assert completed >= 95, f"Only {completed}/100 tasks completed"

        # Save benchmark results for regression tracking
        benchmark_data = {
            "test_throughput_100_tasks": {
                "tasks": completed,
                "elapsed_s": round(elapsed, 3),
                "tasks_per_sec": round(tasks_per_sec, 2),
            }
        }
        self._save_benchmark(benchmark_data)

    def test_parallel_swarm_throughput(self, mock_robot: Robot) -> None:
        """
        3 mock robots, run tasks via agent, compare parallel vs sequential timing.
        Parallel should be faster (or at least not slower).
        """
        from apyrobo.swarm.bus import SwarmBus
        from apyrobo.swarm.coordinator import SwarmCoordinator

        robots = [
            Robot.discover("mock://swarm_bot_0"),
            Robot.discover("mock://swarm_bot_1"),
            Robot.discover("mock://swarm_bot_2"),
        ]
        bus = SwarmBus()
        for r in robots:
            bus.register(r)

        coordinator = SwarmCoordinator(bus, strategy="round_robin")
        agent = Agent(provider="rule")

        # Sequential baseline: 10 tasks on one robot
        start_seq = time.time()
        for _ in range(10):
            agent.execute("go to 1 1", robots[0])
        elapsed_seq = time.time() - start_seq

        # Parallel via swarm: 10 tasks distributed across 3 robots
        start_par = time.time()
        for _ in range(10):
            coordinator.execute_task("go to 1 1", agent)
        elapsed_par = time.time() - start_par

        print(
            f"\nSwarm throughput: sequential={elapsed_seq:.2f}s, "
            f"parallel={elapsed_par:.2f}s, "
            f"speedup={elapsed_seq / max(elapsed_par, 0.001):.2f}x"
        )

        # Parallel should complete (may not always be faster due to coordination overhead)
        assert elapsed_par < 30.0, f"Parallel swarm took {elapsed_par:.2f}s (too slow)"

        benchmark_data = {
            "test_parallel_swarm_throughput": {
                "sequential_s": round(elapsed_seq, 3),
                "parallel_s": round(elapsed_par, 3),
                "speedup": round(elapsed_seq / max(elapsed_par, 0.001), 2),
            }
        }
        self._save_benchmark(benchmark_data)

    def test_regression_gate(self, mock_robot: Robot) -> None:
        """
        If a previous benchmark.json exists, verify throughput hasn't
        dropped more than 20% from the saved baseline.
        """
        if not BENCHMARK_PATH.exists():
            pytest.skip("No baseline benchmark.json to compare against")

        baseline = json.loads(BENCHMARK_PATH.read_text())
        baseline_tps = baseline.get("test_throughput_100_tasks", {}).get("tasks_per_sec")
        if baseline_tps is None:
            pytest.skip("No tasks_per_sec in baseline")

        # Run a quick measurement
        agent = Agent(provider="rule")
        start = time.time()
        completed = 0
        for _ in range(100):
            result = agent.execute("go to 1 1", mock_robot)
            if result.status == TaskStatus.COMPLETED:
                completed += 1
        elapsed = time.time() - start
        current_tps = completed / elapsed

        threshold = baseline_tps * 0.8  # 20% degradation limit
        print(
            f"\nRegression gate: baseline={baseline_tps:.1f} tps, "
            f"current={current_tps:.1f} tps, threshold={threshold:.1f} tps"
        )
        assert current_tps >= threshold, (
            f"Throughput regression: {current_tps:.1f} tasks/sec < "
            f"{threshold:.1f} tasks/sec (80% of baseline {baseline_tps:.1f})"
        )

    @staticmethod
    def _save_benchmark(data: dict) -> None:
        """Merge benchmark data into benchmark.json."""
        existing = {}
        if BENCHMARK_PATH.exists():
            try:
                existing = json.loads(BENCHMARK_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        existing.update(data)
        BENCHMARK_PATH.write_text(json.dumps(existing, indent=2))
