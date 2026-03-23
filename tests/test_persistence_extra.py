"""
Extra coverage tests for apyrobo/persistence.py.

Targets missing lines not covered by test_persistence_sqlite.py:
  StateStore: repr, update_task on unknown task, abort_task,
  save/get robot position, clear with path cleanup,
  set/get KV, load from disk (persistence across restores),
  interrupted task detection on load.

  RedisStateStore: all methods via a mock Redis client.

  SQLiteStateStore: repr, update_task with extra kwargs,
  fail_task with explicit result, get kv with invalid json,
  abort_task.

  create_state_store factory.
  recover_interrupted_tasks helper.
  TaskJournalEntry: from_dict, is_interrupted, repr.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from apyrobo.persistence import (
    StateStore,
    SQLiteStateStore,
    RedisStateStore,
    TaskJournalEntry,
    create_state_store,
    recover_interrupted_tasks,
)


# ===========================================================================
# TaskJournalEntry helpers
# ===========================================================================

class TestTaskJournalEntry:
    def test_to_dict_roundtrip(self) -> None:
        entry = TaskJournalEntry(
            task_id="t1",
            status="in_progress",
            metadata={"action": "deliver"},
            step=2,
            total_steps=5,
            robot_id="tb4",
        )
        d = entry.to_dict()
        assert d["task_id"] == "t1"
        assert d["status"] == "in_progress"
        assert d["metadata"] == {"action": "deliver"}
        assert d["step"] == 2
        assert d["total_steps"] == 5
        assert d["robot_id"] == "tb4"

    def test_from_dict(self) -> None:
        now = time.time()
        data = {
            "task_id": "t2",
            "status": "completed",
            "metadata": {},
            "step": 3,
            "total_steps": 3,
            "robot_id": None,
            "created_at": now,
            "updated_at": now,
            "result": {"ok": True},
        }
        entry = TaskJournalEntry.from_dict(data)
        assert entry.task_id == "t2"
        assert entry.status == "completed"
        assert entry.result == {"ok": True}

    def test_is_interrupted_pending(self) -> None:
        entry = TaskJournalEntry(task_id="t", status="pending")
        assert entry.is_interrupted is True

    def test_is_interrupted_in_progress(self) -> None:
        entry = TaskJournalEntry(task_id="t", status="in_progress")
        assert entry.is_interrupted is True

    def test_is_interrupted_completed(self) -> None:
        entry = TaskJournalEntry(task_id="t", status="completed")
        assert entry.is_interrupted is False

    def test_is_interrupted_failed(self) -> None:
        entry = TaskJournalEntry(task_id="t", status="failed")
        assert entry.is_interrupted is False

    def test_is_interrupted_aborted(self) -> None:
        entry = TaskJournalEntry(task_id="t", status="aborted")
        assert entry.is_interrupted is False

    def test_repr(self) -> None:
        entry = TaskJournalEntry(task_id="myTask", status="in_progress", step=1, total_steps=4)
        r = repr(entry)
        assert "myTask" in r
        assert "in_progress" in r


# ===========================================================================
# StateStore (JSON backend)
# ===========================================================================

class TestStateStoreBasic:
    def test_begin_task(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.json")
        entry = store.begin_task("t1", metadata={"x": 1}, robot_id="tb4", total_steps=3)
        assert entry.task_id == "t1"
        assert entry.status == "in_progress"
        assert store.task_count == 1

    def test_update_task_step_and_status(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.json")
        store.begin_task("t2", total_steps=5)
        store.update_task("t2", step=2, status="custom_status")
        entry = store.get_task("t2")
        assert entry.step == 2
        assert entry.status == "custom_status"

    def test_update_task_total_steps(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.json")
        store.begin_task("t3", total_steps=3)
        store.update_task("t3", total_steps=10)
        entry = store.get_task("t3")
        assert entry.total_steps == 10

    def test_update_task_extra_metadata(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.json")
        store.begin_task("t4")
        store.update_task("t4", progress=0.5)
        entry = store.get_task("t4")
        assert entry.metadata.get("progress") == 0.5

    def test_update_task_unknown_is_noop(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.json")
        # Should not raise
        store.update_task("nonexistent", step=1)

    def test_complete_task(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.json")
        store.begin_task("t5")
        store.complete_task("t5", result={"delivered": True})
        entry = store.get_task("t5")
        assert entry.status == "completed"
        assert entry.result == {"delivered": True}

    def test_fail_task(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.json")
        store.begin_task("t6")
        store.fail_task("t6", error="timeout")
        entry = store.get_task("t6")
        assert entry.status == "failed"
        assert entry.result is not None

    def test_fail_task_with_result(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.json")
        store.begin_task("t7")
        store.fail_task("t7", result={"reason": "estop"})
        entry = store.get_task("t7")
        assert entry.result == {"reason": "estop"}

    def test_abort_task(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.json")
        store.begin_task("t8")
        store.abort_task("t8", reason="crash recovery")
        entry = store.get_task("t8")
        assert entry.status == "aborted"
        assert entry.result == {"abort_reason": "crash recovery"}

    def test_get_task_nonexistent(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.json")
        assert store.get_task("nope") is None

    def test_get_interrupted_tasks(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.json")
        store.begin_task("active")
        store.begin_task("done")
        store.complete_task("done")
        interrupted = store.get_interrupted_tasks()
        ids = [e.task_id for e in interrupted]
        assert "active" in ids
        assert "done" not in ids

    def test_get_recent_tasks_limit(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.json")
        for i in range(6):
            store.begin_task(f"task_{i}")
        recent = store.get_recent_tasks(limit=3)
        assert len(recent) == 3

    def test_get_recent_tasks_order(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.json")
        store.begin_task("first")
        time.sleep(0.01)
        store.begin_task("second")
        recent = store.get_recent_tasks()
        assert recent[0].task_id == "second"


class TestStateStoreRobotPosition:
    def test_save_and_get(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.json")
        store.save_robot_position("tb4", x=1.0, y=2.0, yaw=0.5)
        pos = store.get_robot_position("tb4")
        assert pos is not None
        assert pos["x"] == pytest.approx(1.0)
        assert pos["y"] == pytest.approx(2.0)
        assert pos["yaw"] == pytest.approx(0.5)

    def test_get_position_nonexistent(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.json")
        assert store.get_robot_position("unknown") is None

    def test_overwrite_position(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.json")
        store.save_robot_position("tb4", x=0.0, y=0.0)
        store.save_robot_position("tb4", x=5.0, y=6.0)
        pos = store.get_robot_position("tb4")
        assert pos["x"] == pytest.approx(5.0)


class TestStateStoreKeyValue:
    def test_set_and_get(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.json")
        store.set("config", {"speed": 1.5})
        assert store.get("config") == {"speed": 1.5}

    def test_get_default(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.json")
        assert store.get("missing", "fallback") == "fallback"

    def test_overwrite_key(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.json")
        store.set("k", 1)
        store.set("k", 99)
        assert store.get("k") == 99


class TestStateStorePersistence:
    def test_reload_from_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        store1 = StateStore(path)
        store1.begin_task("persistent_task", metadata={"type": "nav"})
        store1.set("swarm_key", "value")
        store1.save_robot_position("bot", x=3.0, y=4.0)

        # Load fresh from disk
        store2 = StateStore(path)
        entry = store2.get_task("persistent_task")
        assert entry is not None
        assert entry.task_id == "persistent_task"
        assert store2.get("swarm_key") == "value"
        pos = store2.get_robot_position("bot")
        assert pos is not None
        assert pos["x"] == pytest.approx(3.0)

    def test_interrupted_tasks_logged_on_load(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        store1 = StateStore(path)
        store1.begin_task("still_running")
        # Reload — should see interrupted task
        store2 = StateStore(path)
        interrupted = store2.get_interrupted_tasks()
        ids = [t.task_id for t in interrupted]
        assert "still_running" in ids

    def test_clear_removes_file(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        store = StateStore(path)
        store.begin_task("t")
        store.set("k", "v")
        store.clear()
        assert store.task_count == 0
        assert not path.exists()

    def test_repr(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "state.json")
        store.begin_task("t")
        r = repr(store)
        assert "StateStore" in r
        assert "tasks=1" in r

    def test_load_invalid_json_does_not_raise(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text("not valid json")
        # Should not raise; just warn
        store = StateStore(path)
        assert store.task_count == 0


# ===========================================================================
# SQLiteStateStore extra coverage
# ===========================================================================

class TestSQLiteStateStoreExtra:
    def test_repr(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        store.begin_task("t")
        r = repr(store)
        assert "SQLiteStateStore" in r
        assert "tasks=1" in r

    def test_update_task_with_extra_metadata(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        store.begin_task("t1")
        store.update_task("t1", custom_key="custom_value")
        entry = store.get_task("t1")
        assert entry.metadata.get("custom_key") == "custom_value"

    def test_fail_task_with_result(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        store.begin_task("t2")
        store.fail_task("t2", result={"reason": "timeout"})
        entry = store.get_task("t2")
        assert entry.status == "failed"
        assert entry.result == {"reason": "timeout"}

    def test_get_kv_invalid_json_returns_default(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        # Manually insert invalid JSON into kv table
        conn = store._get_conn()
        conn.execute("INSERT INTO kv (key, value) VALUES (?, ?)", ("bad", "not-json{{{"))
        conn.commit()
        result = store.get("bad", default="fallback")
        assert result == "fallback"

    def test_task_count_multiple(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        for i in range(5):
            store.begin_task(f"task_{i}")
        assert store.task_count == 5

    def test_begin_task_replace_existing(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        store.begin_task("dup_task")
        store.begin_task("dup_task", metadata={"updated": True})
        assert store.task_count == 1
        entry = store.get_task("dup_task")
        assert entry.metadata.get("updated") is True

    def test_update_task_all_fields(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        store.begin_task("t3", total_steps=10)
        store.update_task("t3", step=5, total_steps=10, status="in_progress")
        entry = store.get_task("t3")
        assert entry.step == 5
        assert entry.total_steps == 10
        assert entry.status == "in_progress"

    def test_complete_task_with_none_result(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        store.begin_task("t4")
        store.complete_task("t4", result=None)
        entry = store.get_task("t4")
        assert entry.status == "completed"

    def test_row_to_entry_with_result_json(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        store.begin_task("t5")
        store.complete_task("t5", result={"data": [1, 2, 3]})
        entry = store.get_task("t5")
        assert entry.result == {"data": [1, 2, 3]}

    def test_get_recent_tasks_ordering(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        store.begin_task("older")
        time.sleep(0.01)
        store.begin_task("newer")
        recent = store.get_recent_tasks(limit=2)
        assert recent[0].task_id == "newer"

    def test_multiple_robots_positions(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        store.save_robot_position("bot1", x=1.0, y=2.0)
        store.save_robot_position("bot2", x=3.0, y=4.0)
        assert store.get_robot_position("bot1")["x"] == pytest.approx(1.0)
        assert store.get_robot_position("bot2")["x"] == pytest.approx(3.0)


# ===========================================================================
# RedisStateStore via mock client
# ===========================================================================

def _make_redis_store() -> RedisStateStore:
    """Build a RedisStateStore backed by a MagicMock redis client."""
    client = MagicMock()
    # Default return values
    client.exists.return_value = True
    client.zrange.return_value = []
    client.zrevrange.return_value = []
    client.zcard.return_value = 0
    client.hgetall.return_value = {}
    client.hget.return_value = "{}"
    client.get.return_value = None
    client.keys.return_value = []
    return RedisStateStore(prefix="test:", _client=client)


class TestRedisStateStore:
    def test_begin_task(self) -> None:
        store = _make_redis_store()
        entry = store.begin_task("t1", metadata={"action": "nav"}, robot_id="tb4", total_steps=3)
        assert entry.task_id == "t1"
        assert entry.status == "in_progress"
        store._client.hset.assert_called()
        store._client.zadd.assert_called()

    def test_update_task_no_op_when_missing(self) -> None:
        store = _make_redis_store()
        store._client.exists.return_value = False
        # Should not raise
        store.update_task("missing", step=1)
        store._client.hset.assert_not_called()

    def test_update_task_with_step_and_status(self) -> None:
        store = _make_redis_store()
        store._client.exists.return_value = True
        store._client.hget.return_value = "{}"
        store.update_task("t1", step=2, status="in_progress")
        store._client.hset.assert_called()

    def test_update_task_with_extra_kwargs(self) -> None:
        store = _make_redis_store()
        store._client.exists.return_value = True
        store._client.hget.return_value = '{"existing": 1}'
        store.update_task("t1", extra_field="extra_val")
        store._client.hset.assert_called()

    def test_complete_task(self) -> None:
        store = _make_redis_store()
        store.complete_task("t1", result={"ok": True})
        store._client.hset.assert_called()
        store._client.zadd.assert_called()

    def test_fail_task(self) -> None:
        store = _make_redis_store()
        store.fail_task("t1", error="timeout")
        store._client.hset.assert_called()

    def test_fail_task_with_result(self) -> None:
        store = _make_redis_store()
        store.fail_task("t1", result={"reason": "estop"})
        store._client.hset.assert_called()

    def test_abort_task(self) -> None:
        store = _make_redis_store()
        store.abort_task("t1", reason="crash")
        store._client.hset.assert_called()

    def test_get_task_missing(self) -> None:
        store = _make_redis_store()
        store._client.hgetall.return_value = {}
        result = store.get_task("ghost")
        assert result is None

    def test_get_task_found(self) -> None:
        store = _make_redis_store()
        now = time.time()
        store._client.hgetall.return_value = {
            "task_id": "t1",
            "status": "completed",
            "metadata": "{}",
            "step": "3",
            "total_steps": "5",
            "robot_id": "tb4",
            "created_at": str(now),
            "updated_at": str(now),
            "result": '{"ok": true}',
        }
        entry = store.get_task("t1")
        assert entry is not None
        assert entry.task_id == "t1"
        assert entry.status == "completed"
        assert entry.result == {"ok": True}

    def test_get_interrupted_tasks(self) -> None:
        store = _make_redis_store()
        now = time.time()
        store._client.zrange.return_value = ["t1", "t2"]

        def hgetall_side(key: str) -> dict:
            task_id = key.split(":")[-1]
            return {
                "task_id": task_id,
                "status": "in_progress",
                "metadata": "{}",
                "step": "0",
                "total_steps": "0",
                "robot_id": "",
                "created_at": str(now),
                "updated_at": str(now),
                "result": "",
            }

        store._client.hgetall.side_effect = hgetall_side
        interrupted = store.get_interrupted_tasks()
        assert len(interrupted) == 2

    def test_get_recent_tasks(self) -> None:
        store = _make_redis_store()
        now = time.time()
        store._client.zrevrange.return_value = ["t1"]
        store._client.hgetall.return_value = {
            "task_id": "t1",
            "status": "completed",
            "metadata": "{}",
            "step": "1",
            "total_steps": "1",
            "robot_id": "",
            "created_at": str(now),
            "updated_at": str(now),
            "result": "",
        }
        tasks = store.get_recent_tasks(limit=5)
        assert len(tasks) == 1

    def test_save_and_get_robot_position(self) -> None:
        store = _make_redis_store()
        store.save_robot_position("tb4", x=1.0, y=2.0, yaw=0.5)
        store._client.hset.assert_called()
        store._client.hgetall.return_value = {"x": "1.0", "y": "2.0", "yaw": "0.5", "t": "0.0"}
        pos = store.get_robot_position("tb4")
        assert pos is not None
        assert pos["x"] == pytest.approx(1.0)

    def test_get_robot_position_missing(self) -> None:
        store = _make_redis_store()
        store._client.hgetall.return_value = {}
        assert store.get_robot_position("ghost") is None

    def test_set_and_get_kv(self) -> None:
        store = _make_redis_store()
        store.set("mykey", {"val": 42})
        store._client.set.assert_called()
        store._client.get.return_value = '{"val": 42}'
        result = store.get("mykey")
        assert result == {"val": 42}

    def test_get_kv_default(self) -> None:
        store = _make_redis_store()
        store._client.get.return_value = None
        assert store.get("nope", default=99) == 99

    def test_get_kv_invalid_json(self) -> None:
        store = _make_redis_store()
        store._client.get.return_value = "not-json"
        result = store.get("bad", default="fallback")
        assert result == "fallback"

    def test_clear(self) -> None:
        store = _make_redis_store()
        store._client.keys.return_value = ["test:task:t1", "test:kv:k"]
        store.clear()
        store._client.delete.assert_called_once()

    def test_clear_no_keys(self) -> None:
        store = _make_redis_store()
        store._client.keys.return_value = []
        store.clear()
        store._client.delete.assert_not_called()

    def test_task_count(self) -> None:
        store = _make_redis_store()
        store._client.zcard.return_value = 7
        assert store.task_count == 7

    def test_repr(self) -> None:
        store = _make_redis_store()
        store._client.zcard.return_value = 3
        r = repr(store)
        assert "RedisStateStore" in r
        assert "test:" in r

    def test_hash_to_entry_empty_robot_id(self) -> None:
        now = time.time()
        entry = RedisStateStore._hash_to_entry({
            "task_id": "t",
            "status": "pending",
            "metadata": "{}",
            "step": "0",
            "total_steps": "0",
            "robot_id": "",
            "created_at": str(now),
            "updated_at": str(now),
            "result": "",
        })
        assert entry.robot_id is None

    def test_no_redis_package_raises(self) -> None:
        with patch.dict("sys.modules", {"redis": None}):
            with pytest.raises(RuntimeError, match="redis"):
                RedisStateStore(host="localhost", port=6379)


# ===========================================================================
# create_state_store factory
# ===========================================================================

class TestCreateStateStore:
    def test_create_json(self, tmp_path: Path) -> None:
        store = create_state_store("json", path=tmp_path / "s.json")
        assert isinstance(store, StateStore)

    def test_create_sqlite(self, tmp_path: Path) -> None:
        store = create_state_store("sqlite", path=tmp_path / "s.db")
        assert isinstance(store, SQLiteStateStore)

    def test_create_redis(self) -> None:
        client = MagicMock()
        store = create_state_store("redis", _client=client)
        assert isinstance(store, RedisStateStore)

    def test_create_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            create_state_store("mongo")


# ===========================================================================
# recover_interrupted_tasks helper
# ===========================================================================

class TestRecoverInterruptedTasks:
    def test_no_interrupted(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "s.json")
        store.begin_task("done")
        store.complete_task("done")
        result = recover_interrupted_tasks(store)
        assert result == []

    def test_with_interrupted(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path / "s.json")
        store.begin_task("still_running")
        result = recover_interrupted_tasks(store)
        assert len(result) == 1
        assert result[0].task_id == "still_running"

    def test_sqlite_backend(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "s.db")
        store.begin_task("running_task")
        result = recover_interrupted_tasks(store)
        assert any(t.task_id == "running_task" for t in result)
