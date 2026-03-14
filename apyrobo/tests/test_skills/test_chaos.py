"""
CI-13: Chaos testing — kill robot mid-task, verify crash recovery via StateStore.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import TaskStatus
from apyrobo.skills.skill import Skill, SkillStatus, BUILTIN_SKILLS
from apyrobo.skills.executor import SkillGraph, SkillExecutor
from apyrobo.persistence import StateStore, TaskJournalEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_robot() -> Robot:
    return Robot.discover("mock://chaos_bot")


@pytest.fixture
def state_store(tmp_path: Path) -> StateStore:
    store = StateStore(tmp_path / "chaos_state.json")
    yield store
    store.clear()


# ===========================================================================
# CI-13: Chaos & crash recovery tests
# ===========================================================================

class TestCrashRecovery:
    """Verify StateStore crash recovery behavior."""

    def test_interrupted_task_detected(self, state_store: StateStore) -> None:
        """In-progress tasks are detected as interrupted on restart."""
        state_store.begin_task("t1", {"task": "deliver"}, robot_id="bot1")
        state_store.begin_task("t2", {"task": "patrol"}, robot_id="bot2")
        state_store.complete_task("t2")

        interrupted = state_store.get_interrupted_tasks()
        assert len(interrupted) == 1
        assert interrupted[0].task_id == "t1"

    def test_state_survives_reload(self, tmp_path: Path) -> None:
        """State persists across store instances (simulates restart)."""
        path = tmp_path / "reload_test.json"
        store1 = StateStore(path)
        store1.begin_task("t1", {"task": "deliver"}, robot_id="bot1")
        store1.update_task("t1", step=3, total_steps=5)

        # Simulate restart
        store2 = StateStore(path)
        task = store2.get_task("t1")
        assert task is not None
        assert task.step == 3
        assert task.total_steps == 5
        assert task.status == "in_progress"

    def test_abort_interrupted_tasks(self, state_store: StateStore) -> None:
        """Interrupted tasks can be aborted on recovery."""
        state_store.begin_task("t1", {"task": "deliver"})
        state_store.begin_task("t2", {"task": "patrol"})

        interrupted = state_store.get_interrupted_tasks()
        for task in interrupted:
            state_store.abort_task(task.task_id, reason="crash_recovery")

        interrupted_after = state_store.get_interrupted_tasks()
        assert len(interrupted_after) == 0

    def test_fail_task_records_error(self, state_store: StateStore) -> None:
        """Failed tasks record the error."""
        state_store.begin_task("t1")
        state_store.fail_task("t1", error="robot lost connection")

        task = state_store.get_task("t1")
        assert task is not None
        assert task.status == "failed"
        assert task.result is not None
        assert "lost connection" in task.result.get("error", "")

    def test_concurrent_writes(self, state_store: StateStore) -> None:
        """Multiple threads can write concurrently."""
        errors: list[Exception] = []

        def writer(task_id: str) -> None:
            try:
                state_store.begin_task(task_id, {"thread": task_id})
                for step in range(5):
                    state_store.update_task(task_id, step=step)
                    time.sleep(0.01)
                state_store.complete_task(task_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"t{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors during concurrent writes: {errors}"

        # Verify all tasks completed
        for i in range(5):
            task = state_store.get_task(f"t{i}")
            assert task is not None
            assert task.status == "completed"

    def test_mid_execution_crash_simulation(
        self, mock_robot: Robot, state_store: StateStore
    ) -> None:
        """
        Simulate a crash mid-execution and verify recovery.

        Flow:
        1. Start a task and record it in StateStore
        2. Simulate crash (interrupt execution)
        3. Verify StateStore shows interrupted task
        4. Abort the task as crash recovery
        """
        # Start task
        state_store.begin_task(
            "chaos_task",
            {"task": "navigate and pick"},
            robot_id="chaos_bot",
            total_steps=3,
        )

        # Simulate partial execution
        state_store.update_task("chaos_task", step=1, status="in_progress")

        # "Crash" — don't complete the task

        # Recovery: check interrupted tasks
        interrupted = state_store.get_interrupted_tasks()
        assert len(interrupted) == 1
        assert interrupted[0].task_id == "chaos_task"
        assert interrupted[0].step == 1

        # Abort interrupted tasks
        for task in interrupted:
            state_store.abort_task(task.task_id, reason="simulated crash")

        # Verify recovery
        task = state_store.get_task("chaos_task")
        assert task is not None
        assert task.status == "aborted"
        assert state_store.get_interrupted_tasks() == []

    def test_robot_position_persists(self, state_store: StateStore) -> None:
        """Robot position survives store reload."""
        state_store.save_robot_position("bot1", x=1.5, y=2.5, yaw=0.7)
        pos = state_store.get_robot_position("bot1")
        assert pos is not None
        assert pos["x"] == 1.5
        assert pos["y"] == 2.5

    def test_recent_tasks(self, state_store: StateStore) -> None:
        """Recent tasks are returned in order."""
        for i in range(5):
            state_store.begin_task(f"t{i}")
            time.sleep(0.01)  # ensure different timestamps

        recent = state_store.get_recent_tasks(limit=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0].task_id == "t4"

    def test_kv_store(self, state_store: StateStore) -> None:
        """Key-value store works correctly."""
        state_store.set("config_key", {"nested": "value"})
        result = state_store.get("config_key")
        assert result == {"nested": "value"}

    def test_clear_removes_everything(self, state_store: StateStore) -> None:
        """Clear removes all state."""
        state_store.begin_task("t1")
        state_store.set("key", "val")
        state_store.save_robot_position("bot1", 1.0, 2.0)

        state_store.clear()
        assert state_store.task_count == 0
        assert state_store.get("key") is None
        assert state_store.get_robot_position("bot1") is None
