"""
Targeted tests for SQLiteStateStore (persistence.py OB-06).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from apyrobo.persistence import SQLiteStateStore, TaskJournalEntry


class TestSQLiteStateStoreBasic:
    def test_create_in_tmp(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        assert store.task_count == 0

    def test_begin_task_returns_entry(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        entry = store.begin_task("t1", metadata={"action": "deliver"}, robot_id="tb4")
        assert isinstance(entry, TaskJournalEntry)
        assert entry.task_id == "t1"
        assert entry.status == "in_progress"

    def test_begin_task_stored(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        store.begin_task("t2")
        assert store.task_count == 1

    def test_get_task(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        store.begin_task("t3", metadata={"x": 1})
        entry = store.get_task("t3")
        assert entry is not None
        assert entry.task_id == "t3"

    def test_get_task_nonexistent(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        assert store.get_task("nope") is None

    def test_update_task_step(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        store.begin_task("t4", total_steps=5)
        store.update_task("t4", step=2)
        entry = store.get_task("t4")
        assert entry.step == 2

    def test_update_task_status(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        store.begin_task("t5")
        store.update_task("t5", status="paused")
        entry = store.get_task("t5")
        assert entry.status == "paused"

    def test_update_task_nonexistent_no_error(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        # Should not raise
        store.update_task("nonexistent", step=1)

    def test_complete_task(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        store.begin_task("t6")
        store.complete_task("t6", result={"delivered": True})
        entry = store.get_task("t6")
        assert entry.status == "completed"

    def test_fail_task(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        store.begin_task("t7")
        store.fail_task("t7", error="timeout")
        entry = store.get_task("t7")
        assert entry.status == "failed"

    def test_abort_task(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        store.begin_task("t8")
        store.abort_task("t8", reason="emergency stop")
        entry = store.get_task("t8")
        assert entry.status == "aborted"

    def test_get_interrupted_tasks(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        store.begin_task("ta")
        store.begin_task("tb")
        store.complete_task("tb")
        interrupted = store.get_interrupted_tasks()
        ids = [e.task_id for e in interrupted]
        assert "ta" in ids
        assert "tb" not in ids

    def test_get_recent_tasks(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        for i in range(5):
            store.begin_task(f"task_{i}")
        recent = store.get_recent_tasks(limit=3)
        assert len(recent) == 3

    def test_get_recent_tasks_all(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        store.begin_task("r1")
        store.begin_task("r2")
        recent = store.get_recent_tasks()
        assert len(recent) == 2


class TestSQLiteRobotPosition:
    def test_save_and_get(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "pos.db")
        store.save_robot_position("tb4", x=1.5, y=2.5, yaw=0.5)
        pos = store.get_robot_position("tb4")
        assert pos is not None
        assert pos["x"] == pytest.approx(1.5)
        assert pos["y"] == pytest.approx(2.5)
        assert pos["yaw"] == pytest.approx(0.5)

    def test_overwrite_position(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "pos.db")
        store.save_robot_position("tb4", x=0.0, y=0.0)
        store.save_robot_position("tb4", x=5.0, y=5.0)
        pos = store.get_robot_position("tb4")
        assert pos["x"] == pytest.approx(5.0)

    def test_get_position_nonexistent(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "pos.db")
        assert store.get_robot_position("unknown_bot") is None


class TestSQLiteKeyValue:
    def test_set_and_get(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "kv.db")
        store.set("config", {"max_speed": 1.5})
        val = store.get("config")
        assert val["max_speed"] == 1.5

    def test_get_default(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "kv.db")
        assert store.get("missing", "default_val") == "default_val"

    def test_overwrite_key(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "kv.db")
        store.set("k", 1)
        store.set("k", 2)
        assert store.get("k") == 2

    def test_set_various_types(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "kv.db")
        store.set("int_val", 42)
        store.set("list_val", [1, 2, 3])
        store.set("str_val", "hello")
        assert store.get("int_val") == 42
        assert store.get("list_val") == [1, 2, 3]
        assert store.get("str_val") == "hello"


class TestSQLiteClear:
    def test_clear_removes_all(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "clear.db")
        store.begin_task("tc1")
        store.begin_task("tc2")
        store.set("key1", "val1")
        store.save_robot_position("bot", x=1.0, y=2.0)
        store.clear()
        assert store.task_count == 0
        assert store.get("key1") is None
        assert store.get_robot_position("bot") is None

    def test_begin_task_with_total_steps(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "ts.db")
        entry = store.begin_task("t_steps", total_steps=10)
        assert entry.total_steps == 10
