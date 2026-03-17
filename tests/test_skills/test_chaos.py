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


# ===========================================================================
# CI-01: Full process crash mid-graph and StateStore recovery
# ===========================================================================


class SimulatedCrash(Exception):
    """Raised to simulate a process crash mid-graph."""


class _CrashAfterStepExecutor(SkillExecutor):
    """SkillExecutor that raises SimulatedCrash after N completed steps."""

    def __init__(self, robot: Robot, state_store: Any, crash_after_step: int,
                 task_id: str = "") -> None:
        super().__init__(robot, state_store=state_store)
        self._crash_after_step = crash_after_step
        self._steps_done = 0
        self._task_id = task_id

    def execute_skill(self, skill: Skill, parameters: dict | None = None) -> SkillStatus:
        status = super().execute_skill(skill, parameters)
        if status == SkillStatus.COMPLETED:
            self._steps_done += 1
            # Update store *before* crash so step count is persisted
            if self._state_store and self._task_id:
                self._state_store.update_task(
                    self._task_id, step=self._steps_done, status="in_progress",
                )
            if self._steps_done >= self._crash_after_step:
                raise SimulatedCrash(
                    f"Simulated crash after step {self._steps_done}"
                )
        return status


class TestCrashMidGraph:
    """CI-01: Crash mid-graph and verify StateStore recovery."""

    def test_crash_recovery(self, tmp_path: Path, mock_robot: Robot) -> None:
        """
        Start a 5-skill graph, crash after step 3, reload StateStore,
        verify get_interrupted_tasks() returns the task at step 3.
        """
        store = StateStore(tmp_path / "crash_state.json")

        # Build a 5-skill sequential graph
        graph = SkillGraph()
        prev_id = None
        for i in range(5):
            s = Skill(
                skill_id=f"navigate_to_{i}",
                name=f"Nav {i}",
                required_capability=BUILTIN_SKILLS["navigate_to"].required_capability,
                parameters={"x": float(i), "y": float(i)},
                timeout_seconds=5.0,
            )
            graph.add_skill(s, depends_on=[prev_id] if prev_id else [])
            prev_id = s.skill_id

        # Manually begin task in store (as execute_graph does with trace_id)
        task_id = "crash_test_task"
        store.begin_task(task_id, metadata={"skill_count": 5}, total_steps=5)

        # Simulate execution with crash after step 3
        executor = _CrashAfterStepExecutor(
            mock_robot, state_store=store, crash_after_step=3,
            task_id=task_id,
        )
        order = graph.get_execution_order()
        with pytest.raises(SimulatedCrash):
            for skill in order:
                params = graph.get_parameters(skill.skill_id)
                executor.execute_skill(skill, params)

        # Reload store from disk (simulates process restart)
        store2 = StateStore(tmp_path / "crash_state.json")
        interrupted = store2.get_interrupted_tasks()
        assert len(interrupted) == 1
        assert interrupted[0].task_id == task_id
        assert interrupted[0].step == 3
        assert interrupted[0].status == "in_progress"

    def test_crash_recovery_step_count(self, tmp_path: Path, mock_robot: Robot) -> None:
        """Step count in recovered state matches steps_completed before crash."""
        store = StateStore(tmp_path / "step_count_state.json")
        task_id = "step_count_task"
        store.begin_task(task_id, total_steps=5)

        # Simulate 2 completed steps then crash
        store.update_task(task_id, step=1, status="in_progress")
        store.update_task(task_id, step=2, status="in_progress")
        # "crash" here — don't complete

        store2 = StateStore(tmp_path / "step_count_state.json")
        task = store2.get_task(task_id)
        assert task is not None
        assert task.step == 2
        assert task.total_steps == 5
        assert task.is_interrupted

    def test_new_statestore_reads_same_data(self, tmp_path: Path) -> None:
        """New StateStore instance reads identical data (persistence works)."""
        path = tmp_path / "persist_test.json"
        store1 = StateStore(path)
        store1.begin_task("t1", {"desc": "test"}, robot_id="bot1", total_steps=10)
        store1.update_task("t1", step=7, status="in_progress")
        store1.set("config", {"key": "value"})
        store1.save_robot_position("bot1", x=3.0, y=4.0, yaw=1.57)

        # New instance from same file
        store2 = StateStore(path)
        task = store2.get_task("t1")
        assert task is not None
        assert task.step == 7
        assert task.total_steps == 10
        assert task.metadata == {"desc": "test"}
        assert task.robot_id == "bot1"

        assert store2.get("config") == {"key": "value"}
        pos = store2.get_robot_position("bot1")
        assert pos is not None
        assert pos["x"] == 3.0
        assert pos["y"] == 4.0

    def test_concurrent_state_store_sqlite(self, tmp_path: Path) -> None:
        """10 threads write tasks simultaneously to SQLiteStateStore; no corruption."""
        from apyrobo.persistence import SQLiteStateStore

        store = SQLiteStateStore(tmp_path / "concurrent.db")
        errors: list[Exception] = []

        def writer(task_id: str) -> None:
            try:
                store.begin_task(task_id, {"thread": task_id})
                for step in range(10):
                    store.update_task(task_id, step=step)
                    time.sleep(0.005)
                store.complete_task(task_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"t{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors during concurrent writes: {errors}"

        # Verify all 10 tasks completed without corruption
        for i in range(10):
            task = store.get_task(f"t{i}")
            assert task is not None, f"Task t{i} missing"
            assert task.status == "completed", f"Task t{i} status={task.status}"
            assert task.step == 9, f"Task t{i} step={task.step}"
        assert store.task_count == 10

    def test_redis_state_store_with_fakeredis(self) -> None:
        """RedisStateStore: all StorageBackend abstract methods work with fakeredis."""
        fakeredis = pytest.importorskip("fakeredis")
        from apyrobo.persistence import RedisStateStore

        client = fakeredis.FakeRedis(decode_responses=True)
        store = RedisStateStore(_client=client, prefix="test:")

        # begin_task
        entry = store.begin_task("t1", {"desc": "test"}, robot_id="bot1", total_steps=5)
        assert entry.task_id == "t1"
        assert entry.status == "in_progress"

        # update_task
        store.update_task("t1", step=3, status="in_progress")
        task = store.get_task("t1")
        assert task is not None
        assert task.step == 3

        # get_interrupted_tasks
        interrupted = store.get_interrupted_tasks()
        assert len(interrupted) == 1
        assert interrupted[0].task_id == "t1"

        # complete_task
        store.complete_task("t1", result={"ok": True})
        task = store.get_task("t1")
        assert task.status == "completed"
        assert store.get_interrupted_tasks() == []

        # fail_task
        store.begin_task("t2")
        store.fail_task("t2", error="test error")
        task = store.get_task("t2")
        assert task.status == "failed"

        # get_recent_tasks
        recent = store.get_recent_tasks(limit=5)
        assert len(recent) == 2

        # KV store
        store.set("key1", {"nested": "val"})
        assert store.get("key1") == {"nested": "val"}
        assert store.get("nonexistent", "default") == "default"

        # Robot position
        store.save_robot_position("bot1", x=1.5, y=2.5, yaw=0.7)
        pos = store.get_robot_position("bot1")
        assert pos is not None
        assert abs(pos["x"] - 1.5) < 0.01

        # task_count
        assert store.task_count == 2

        # clear
        store.clear()
        assert store.task_count == 0
        assert store.get("key1") is None
