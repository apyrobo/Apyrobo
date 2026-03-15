"""
CI-11: Performance benchmark — 100 tasks/min through mock executor.

Provides a regression gate to ensure execution speed doesn't degrade.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import TaskStatus
from apyrobo.skills.skill import Skill, SkillStatus, BUILTIN_SKILLS
from apyrobo.skills.executor import SkillGraph, SkillExecutor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_robot() -> Robot:
    return Robot.discover("mock://bench_bot")


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
                skill_id=f"nav_{i}",
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
                s = Skill(
                    skill_id=f"step_{run}_{i}",
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
            for i in range(5):
                s = Skill(
                    skill_id=f"par_{run}_{i}",
                    name=f"Parallel {i}",
                    required_capability=BUILTIN_SKILLS["report_status"].required_capability,
                    timeout_seconds=5.0,
                )
                g.add_skill(s)

            final = Skill(
                skill_id=f"final_{run}",
                name="Final",
                required_capability=BUILTIN_SKILLS["stop"].required_capability,
                timeout_seconds=5.0,
            )
            g.add_skill(final, depends_on=[f"par_{run}_{i}" for i in range(5)])

            exe = SkillExecutor(mock_robot)
            result = exe.execute_graph(g, parallel=True)
            if result.status == TaskStatus.COMPLETED:
                completed += 1

        elapsed = time.time() - start

        assert completed >= 18, f"Only {completed}/20 parallel runs completed"
        assert elapsed < 60.0, f"Took {elapsed:.1f}s"
        print(f"\nBenchmark: {completed} parallel graphs in {elapsed:.1f}s")
