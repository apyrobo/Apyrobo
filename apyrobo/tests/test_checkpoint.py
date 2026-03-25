"""Tests for apyrobo.skills.checkpoint"""
from __future__ import annotations

import time
import pytest

from apyrobo.skills.checkpoint import (
    CheckpointEntry,
    CheckpointStore,
    CheckpointedExecutor,
)


# ---------------------------------------------------------------------------
# CheckpointEntry
# ---------------------------------------------------------------------------

class TestCheckpointEntry:
    def test_checksum_auto_computed(self):
        entry = CheckpointEntry(
            task_id="t1",
            skill_name="navigate_to",
            step_index=1,
            total_steps=3,
            state={"x": 1.0},
            completed_steps=["navigate_to"],
        )
        assert entry.checksum != ""

    def test_verify_integrity_passes(self):
        entry = CheckpointEntry(
            task_id="t1",
            skill_name="s",
            step_index=1,
            total_steps=1,
            state={"key": "value"},
            completed_steps=["s"],
        )
        assert entry.verify_integrity() is True

    def test_verify_integrity_fails_on_tamper(self):
        entry = CheckpointEntry(
            task_id="t1",
            skill_name="s",
            step_index=1,
            total_steps=1,
            state={"key": "original"},
            completed_steps=["s"],
        )
        entry.state["key"] = "tampered"
        assert entry.verify_integrity() is False

    def test_to_dict_roundtrip(self):
        entry = CheckpointEntry(
            task_id="abc",
            skill_name="pick",
            step_index=2,
            total_steps=5,
            state={"status": "ok"},
            completed_steps=["load", "pick"],
        )
        d = entry.to_dict()
        assert d["task_id"] == "abc"
        assert d["step_index"] == 2
        assert d["completed_steps"] == ["load", "pick"]


# ---------------------------------------------------------------------------
# CheckpointStore
# ---------------------------------------------------------------------------

class TestCheckpointStore:
    def setup_method(self):
        self.store = CheckpointStore(":memory:")

    def _make_entry(self, task_id: str, step_index: int = 1) -> CheckpointEntry:
        return CheckpointEntry(
            task_id=task_id,
            skill_name="nav",
            step_index=step_index,
            total_steps=3,
            state={"step": step_index},
            completed_steps=[f"step_{i}" for i in range(step_index)],
        )

    def test_save_and_load(self):
        entry = self._make_entry("task-1")
        self.store.save(entry)
        loaded = self.store.load("task-1")
        assert loaded is not None
        assert loaded.task_id == "task-1"
        assert loaded.step_index == 1

    def test_load_nonexistent_returns_none(self):
        assert self.store.load("no-such-task") is None

    def test_delete_removes_entry(self):
        self.store.save(self._make_entry("task-del"))
        self.store.delete("task-del")
        assert self.store.load("task-del") is None

    def test_delete_nonexistent_is_noop(self):
        self.store.delete("ghost")  # should not raise

    def test_list_tasks(self):
        self.store.save(self._make_entry("t1"))
        self.store.save(self._make_entry("t2"))
        tasks = self.store.list_tasks()
        assert "t1" in tasks
        assert "t2" in tasks

    def test_upsert_updates_existing(self):
        self.store.save(self._make_entry("task-up", step_index=1))
        self.store.save(self._make_entry("task-up", step_index=2))
        loaded = self.store.load("task-up")
        assert loaded.step_index == 2

    def test_checksum_preserved_on_load(self):
        entry = self._make_entry("chk-task")
        self.store.save(entry)
        loaded = self.store.load("chk-task")
        assert loaded.checksum == entry.checksum

    def test_concurrent_task_isolation(self):
        for i in range(5):
            self.store.save(self._make_entry(f"task-{i}", step_index=i))
        for i in range(5):
            loaded = self.store.load(f"task-{i}")
            assert loaded.step_index == i


# ---------------------------------------------------------------------------
# CheckpointedExecutor
# ---------------------------------------------------------------------------

def _ok(name: str):
    """Return a step function that records its call."""
    return lambda: name


class TestCheckpointedExecutor:
    def setup_method(self):
        self.store = CheckpointStore(":memory:")
        self.executor = CheckpointedExecutor(self.store)

    def _make_steps(self, names):
        return [(n, _ok(n), {}) for n in names]

    def test_all_steps_complete_successfully(self):
        steps = self._make_steps(["a", "b", "c"])
        result = self.executor.execute_steps("task-1", steps)
        assert result["failed"] is None
        assert result["completed"] == ["a", "b", "c"]

    def test_checkpoint_deleted_after_full_success(self):
        steps = self._make_steps(["x", "y"])
        self.executor.execute_steps("task-clean", steps)
        assert self.store.load("task-clean") is None

    def test_failure_stops_execution_and_records_partial(self):
        def explode():
            raise RuntimeError("bang")

        steps = [("step1", _ok("step1"), {}), ("fail", explode, {}), ("step3", _ok("step3"), {})]
        result = self.executor.execute_steps("task-fail", steps)
        assert result["failed"] == "fail"
        assert result["completed"] == ["step1"]
        assert result["result"] is None

    def test_checkpoint_saved_after_each_step(self):
        checkpoint_after_first = {}

        def spy_step():
            # peek at checkpoint store mid-execution
            entry = self.store.load("task-spy")
            if entry:
                checkpoint_after_first["found"] = True
            return "done"

        steps = [("first", _ok("first"), {}), ("spy", spy_step, {})]
        self.executor.execute_steps("task-spy", steps)
        assert checkpoint_after_first.get("found") is True

    def test_resume_from_middle(self):
        calls = {"n": 0}

        def counted():
            calls["n"] += 1
            return "run"

        steps = [
            ("step1", counted, {}),
            ("step2", counted, {}),
            ("step3", counted, {}),
        ]

        # Simulate existing checkpoint after step1
        entry = CheckpointEntry(
            task_id="task-resume",
            skill_name="step1",
            step_index=1,
            total_steps=3,
            state={"step1": {"status": "ok", "result": "run"}},
            completed_steps=["step1"],
        )
        self.store.save(entry)

        result = self.executor.execute_steps("task-resume", steps, resume=True)
        # Only step2 and step3 should have been called
        assert calls["n"] == 2
        assert result["completed"] == ["step1", "step2", "step3"]
        assert result["failed"] is None

    def test_no_resume_ignores_checkpoint(self):
        calls = {"n": 0}

        def counted():
            calls["n"] += 1
            return "run"

        steps = [("a", counted, {}), ("b", counted, {})]
        # Pre-seed a checkpoint
        entry = CheckpointEntry(
            task_id="task-noresume",
            skill_name="a",
            step_index=1,
            total_steps=2,
            state={"a": {"status": "ok", "result": "run"}},
            completed_steps=["a"],
        )
        self.store.save(entry)

        result = self.executor.execute_steps("task-noresume", steps, resume=False)
        assert calls["n"] == 2  # both steps ran from scratch

    def test_tampered_checkpoint_restarts(self):
        calls = {"n": 0}

        def counted():
            calls["n"] += 1
            return "run"

        steps = [("x", counted, {}), ("y", counted, {})]
        entry = CheckpointEntry(
            task_id="task-tamper",
            skill_name="x",
            step_index=1,
            total_steps=2,
            state={"x": {"status": "ok"}},
            completed_steps=["x"],
        )
        # Tamper the state after computing checksum
        entry.state["x"]["status"] = "tampered"
        self.store.save(entry)

        result = self.executor.execute_steps("task-tamper", steps, resume=True)
        assert calls["n"] == 2  # full restart

    def test_kwargs_passed_to_step_fn(self):
        received = {}

        def capture(x, y):
            received["x"] = x
            received["y"] = y
            return x + y

        steps = [("compute", capture, {"x": 3, "y": 4})]
        result = self.executor.execute_steps("task-kwargs", steps)
        assert received == {"x": 3, "y": 4}
        assert result["result"] == 7
