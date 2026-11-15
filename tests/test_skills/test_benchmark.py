"""
CI-11 + CI-03: Performance benchmark — 100 tasks/min through mock executor.

Provides a regression gate to ensure execution speed doesn't degrade.
Includes throughput benchmarks and regression gating via benchmark.json.
"""

from __future__ import annotations

import contextlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import pytest

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import TaskStatus
from apyrobo.skills.agent import Agent
from apyrobo.skills.executor import SkillExecutor, SkillGraph
from apyrobo.skills.skill import BUILTIN_SKILLS, Skill

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

    # Regression gate settings. The gate looks for throughput that decays
    # as tasks pile up (state that grows with every execute, a leak that
    # makes each task slower than the last). It scores PASSES timed passes
    # before and after SOAK_TASKS more executes, by the median pass of each.
    GATE_PASSES = 5
    GATE_TASKS_PER_PASS = 100
    GATE_SOAK_TASKS = 1000
    GATE_MIN_RATIO = 0.5

    @classmethod
    def _gate_ratio(cls, run_task: Any) -> tuple[float, list[float], list[float]]:
        """Time ``run_task`` before and after a soak; return late/early.

        ``run_task()`` runs one task and returns True when it completed.
        """

        def measure() -> float:
            start = time.perf_counter()
            completed = 0
            for _ in range(cls.GATE_TASKS_PER_PASS):
                if run_task():
                    completed += 1
            return completed / (time.perf_counter() - start)

        early = [measure() for _ in range(cls.GATE_PASSES)]
        for _ in range(cls.GATE_SOAK_TASKS):
            run_task()
        late = [measure() for _ in range(cls.GATE_PASSES)]
        return statistics.median(late) / statistics.median(early), early, late

    def test_regression_gate(self, mock_robot: Robot) -> None:
        """
        Throughput must not decay as tasks pile up: the median of five
        100-task passes after another 1000 tasks must stay above half the
        median of five passes before them.

        History. This first compared against benchmark.json's "baseline"
        (really just test_throughput_100_tasks moments earlier, with the
        swarm test in between, which dragged the second number to ~50% on
        Python 3.11 runners). Then it compared three passes against the next
        three, back to back, with a 20% limit; that still turned CI red on
        shared runners (1058 < 1078 tasks/sec on main at 33c5fa0), because
        three ~0.1 s passes in a row can all land in one slow stretch of a
        loaded runner. Run to run noise of 20-30% is normal there.

        Now the two halves are 1000 tasks apart and the limit is 50%. A
        slowdown that grows with every task (what this gate is for) gets
        much worse over those 1000 tasks, so it still fails:
        test_regression_gate_catches_decay checks that with a task whose
        cost grows. A one-off stall has to cover three of the five later
        passes and halve them to fail it. Absolute speed is
        test_throughput_100_tasks' job (100 tasks under 5 s).
        """
        agent = Agent(provider="rule")

        def run_task() -> bool:
            result = agent.execute("go to 1 1", mock_robot)
            return result.status == TaskStatus.COMPLETED

        ratio, early, late = self._gate_ratio(run_task)
        print(
            f"\nRegression gate: early={[round(p, 1) for p in early]}, "
            f"late={[round(p, 1) for p in late]}, ratio={ratio:.2f} "
            f"(limit {self.GATE_MIN_RATIO})"
        )
        assert ratio >= self.GATE_MIN_RATIO, (
            f"Throughput decayed as tasks piled up: late median is "
            f"{ratio:.0%} of early median (limit {self.GATE_MIN_RATIO:.0%}); "
            f"early={[round(p, 1) for p in early]}, "
            f"late={[round(p, 1) for p in late]}"
        )

    def test_regression_gate_catches_decay(self) -> None:
        """The gate fails a task whose cost grows with every run."""
        history: list[int] = []

        def leaky_task() -> bool:
            # Each run scans everything earlier runs left behind: the kind
            # of per-task growth the gate is there to catch.
            history.append(len(history))
            total = 0
            for item in history:
                total += item
            return total >= 0

        ratio, _, _ = self._gate_ratio(leaky_task)
        assert ratio < self.GATE_MIN_RATIO, (
            f"gate missed a growing per-task cost (ratio {ratio:.2f})"
        )

    @staticmethod
    def _save_benchmark(data: dict) -> None:
        """Merge benchmark data into benchmark.json."""
        existing = {}
        if BENCHMARK_PATH.exists():
            with contextlib.suppress(json.JSONDecodeError, OSError):
                existing = json.loads(BENCHMARK_PATH.read_text())
        existing.update(data)
        BENCHMARK_PATH.write_text(json.dumps(existing, indent=2))
