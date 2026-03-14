"""
State Persistence — survives crashes and restarts.

Maintains a task journal on disk so the system can recover after
power loss, Docker restart, or node crash. On startup, reads the
journal and either resumes pending tasks or safely aborts them.

Backends:
    - StateStore: JSON file (simple, inspectable, no dependencies)
    - SQLiteStateStore (OB-06): SQLite for concurrent access
    - RedisStateStore (OB-07): Redis for distributed deployments

Usage:
    store = StateStore("/workspace/data/state.json")

    # Record task start
    store.begin_task("task_001", {"task": "deliver", "robot": "tb4"})

    # Update progress
    store.update_task("task_001", step=2, total=4, status="in_progress")

    # On completion
    store.complete_task("task_001", result={"status": "completed"})

    # After restart — check for interrupted tasks
    interrupted = store.get_interrupted_tasks()

    # OB-06: SQLite backend
    store = SQLiteStateStore("/workspace/data/state.db")

    # OB-07: Redis backend
    store = RedisStateStore(host="redis", port=6379, prefix="apyrobo:")
"""

from __future__ import annotations

import abc
import json
import logging
import os
import time
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TaskJournalEntry:
    """A single entry in the task journal."""

    def __init__(
        self,
        task_id: str,
        status: str = "pending",
        metadata: dict[str, Any] | None = None,
        step: int = 0,
        total_steps: int = 0,
        robot_id: str | None = None,
        created_at: float | None = None,
        updated_at: float | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        self.task_id = task_id
        self.status = status  # pending, in_progress, completed, failed, aborted
        self.metadata = metadata or {}
        self.step = step
        self.total_steps = total_steps
        self.robot_id = robot_id
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or time.time()
        self.result = result

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "metadata": self.metadata,
            "step": self.step,
            "total_steps": self.total_steps,
            "robot_id": self.robot_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskJournalEntry:
        return cls(**data)

    @property
    def is_interrupted(self) -> bool:
        """Was this task running when the system went down?"""
        return self.status in ("pending", "in_progress")

    def __repr__(self) -> str:
        return f"<TaskEntry {self.task_id} status={self.status} step={self.step}/{self.total_steps}>"


# ---------------------------------------------------------------------------
# Abstract backend interface
# ---------------------------------------------------------------------------

class StorageBackend(abc.ABC):
    """Abstract interface for state storage backends."""

    @abc.abstractmethod
    def begin_task(self, task_id: str, metadata: dict[str, Any] | None = None,
                   robot_id: str | None = None, total_steps: int = 0) -> TaskJournalEntry: ...

    @abc.abstractmethod
    def update_task(self, task_id: str, step: int | None = None,
                    total_steps: int | None = None, status: str | None = None,
                    **extra: Any) -> None: ...

    @abc.abstractmethod
    def complete_task(self, task_id: str, result: dict[str, Any] | None = None) -> None: ...

    @abc.abstractmethod
    def fail_task(self, task_id: str, error: str = "", result: dict[str, Any] | None = None) -> None: ...

    @abc.abstractmethod
    def get_task(self, task_id: str) -> TaskJournalEntry | None: ...

    @abc.abstractmethod
    def get_interrupted_tasks(self) -> list[TaskJournalEntry]: ...

    @abc.abstractmethod
    def get_recent_tasks(self, limit: int = 20) -> list[TaskJournalEntry]: ...

    @abc.abstractmethod
    def set(self, key: str, value: Any) -> None: ...

    @abc.abstractmethod
    def get(self, key: str, default: Any = None) -> Any: ...

    @abc.abstractmethod
    def clear(self) -> None: ...

    @property
    @abc.abstractmethod
    def task_count(self) -> int: ...


# ---------------------------------------------------------------------------
# JSON file backend (original)
# ---------------------------------------------------------------------------

class StateStore(StorageBackend):
    """
    Persistent state store backed by a JSON file.

    Thread-safe. Writes to disk on every mutation (fsync for durability).
    """

    def __init__(self, path: str | Path = "data/apyrobo_state.json") -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskJournalEntry] = {}
        self._robot_positions: dict[str, dict[str, float]] = {}
        self._swarm_state: dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    # Task journal
    # ------------------------------------------------------------------

    def begin_task(self, task_id: str, metadata: dict[str, Any] | None = None,
                   robot_id: str | None = None, total_steps: int = 0) -> TaskJournalEntry:
        """Record that a task has started."""
        entry = TaskJournalEntry(
            task_id=task_id, status="in_progress", metadata=metadata,
            robot_id=robot_id, total_steps=total_steps,
        )
        with self._lock:
            self._tasks[task_id] = entry
            self._save()
        logger.info("State: task %s started (robot=%s)", task_id, robot_id)
        return entry

    def update_task(self, task_id: str, step: int | None = None,
                    total_steps: int | None = None, status: str | None = None,
                    **extra: Any) -> None:
        """Update task progress."""
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                logger.warning("State: unknown task %s", task_id)
                return
            if step is not None:
                entry.step = step
            if total_steps is not None:
                entry.total_steps = total_steps
            if status is not None:
                entry.status = status
            entry.metadata.update(extra)
            entry.updated_at = time.time()
            self._save()

    def complete_task(self, task_id: str, result: dict[str, Any] | None = None) -> None:
        """Mark a task as completed."""
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry:
                entry.status = "completed"
                entry.result = result
                entry.updated_at = time.time()
                self._save()
        logger.info("State: task %s completed", task_id)

    def fail_task(self, task_id: str, error: str = "", result: dict[str, Any] | None = None) -> None:
        """Mark a task as failed."""
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry:
                entry.status = "failed"
                entry.result = result or {"error": error}
                entry.updated_at = time.time()
                self._save()
        logger.info("State: task %s failed: %s", task_id, error)

    def abort_task(self, task_id: str, reason: str = "") -> None:
        """Mark a task as aborted (e.g. after crash recovery)."""
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry:
                entry.status = "aborted"
                entry.result = {"abort_reason": reason}
                entry.updated_at = time.time()
                self._save()

    def get_task(self, task_id: str) -> TaskJournalEntry | None:
        return self._tasks.get(task_id)

    def get_interrupted_tasks(self) -> list[TaskJournalEntry]:
        """Find tasks that were running when the system went down."""
        return [t for t in self._tasks.values() if t.is_interrupted]

    def get_recent_tasks(self, limit: int = 20) -> list[TaskJournalEntry]:
        """Most recent tasks, newest first."""
        sorted_tasks = sorted(self._tasks.values(), key=lambda t: t.updated_at, reverse=True)
        return sorted_tasks[:limit]

    # ------------------------------------------------------------------
    # Robot position persistence
    # ------------------------------------------------------------------

    def save_robot_position(self, robot_id: str, x: float, y: float, yaw: float = 0.0) -> None:
        """Persist robot position (for recovery after restart)."""
        with self._lock:
            self._robot_positions[robot_id] = {"x": x, "y": y, "yaw": yaw, "t": time.time()}
            self._save()

    def get_robot_position(self, robot_id: str) -> dict[str, float] | None:
        return self._robot_positions.get(robot_id)

    # ------------------------------------------------------------------
    # Generic key-value (for swarm state, config, etc.)
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._swarm_state[key] = value
            self._save()

    def get(self, key: str, default: Any = None) -> Any:
        return self._swarm_state.get(key, default)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load state from disk."""
        if not self._path.exists():
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            self._tasks = {
                k: TaskJournalEntry.from_dict(v) for k, v in data.get("tasks", {}).items()
            }
            self._robot_positions = data.get("robot_positions", {})
            self._swarm_state = data.get("swarm_state", {})
            interrupted = self.get_interrupted_tasks()
            if interrupted:
                logger.warning(
                    "State: found %d interrupted tasks from previous session: %s",
                    len(interrupted), [t.task_id for t in interrupted],
                )
        except Exception as e:
            logger.error("State: failed to load from %s: %s", self._path, e)

    def _save(self) -> None:
        """Write state to disk (atomic via tmp + rename)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        try:
            data = {
                "tasks": {k: v.to_dict() for k, v in self._tasks.items()},
                "robot_positions": self._robot_positions,
                "swarm_state": self._swarm_state,
                "saved_at": time.time(),
            }
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.rename(self._path)
        except Exception as e:
            logger.error("State: failed to save to %s: %s", self._path, e)

    def clear(self) -> None:
        """Clear all state (for testing)."""
        with self._lock:
            self._tasks.clear()
            self._robot_positions.clear()
            self._swarm_state.clear()
            if self._path.exists():
                self._path.unlink()

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    def __repr__(self) -> str:
        interrupted = len(self.get_interrupted_tasks())
        return f"<StateStore tasks={self.task_count} interrupted={interrupted} path={self._path}>"


# ---------------------------------------------------------------------------
# OB-06: SQLite backend
# ---------------------------------------------------------------------------

class SQLiteStateStore(StorageBackend):
    """
    OB-06: SQLite-backed state store for concurrent access.

    Handles multiple readers/writers safely via SQLite's WAL mode.
    Drop-in replacement for StateStore.

    Usage:
        store = SQLiteStateStore("/workspace/data/state.db")
        store.begin_task("task_001", {"task": "deliver"}, robot_id="tb4")
    """

    def __init__(self, path: str | Path = "data/apyrobo_state.db") -> None:
        import sqlite3
        self._path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # Initialize schema
        conn = self._get_conn()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                metadata TEXT DEFAULT '{}',
                step INTEGER DEFAULT 0,
                total_steps INTEGER DEFAULT 0,
                robot_id TEXT,
                created_at REAL,
                updated_at REAL,
                result TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS robot_positions (
                robot_id TEXT PRIMARY KEY,
                x REAL, y REAL, yaw REAL,
                t REAL
            )
        """)
        conn.commit()

    def _get_conn(self) -> Any:
        import sqlite3
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self._path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def begin_task(self, task_id: str, metadata: dict[str, Any] | None = None,
                   robot_id: str | None = None, total_steps: int = 0) -> TaskJournalEntry:
        now = time.time()
        entry = TaskJournalEntry(
            task_id=task_id, status="in_progress", metadata=metadata,
            robot_id=robot_id, total_steps=total_steps,
            created_at=now, updated_at=now,
        )
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO tasks (task_id, status, metadata, step, total_steps, robot_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, "in_progress", json.dumps(metadata or {}), 0, total_steps, robot_id, now, now),
        )
        conn.commit()
        return entry

    def update_task(self, task_id: str, step: int | None = None,
                    total_steps: int | None = None, status: str | None = None,
                    **extra: Any) -> None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            return
        now = time.time()
        new_step = step if step is not None else row["step"]
        new_total = total_steps if total_steps is not None else row["total_steps"]
        new_status = status if status is not None else row["status"]
        meta = json.loads(row["metadata"] or "{}")
        meta.update(extra)
        conn.execute(
            "UPDATE tasks SET step=?, total_steps=?, status=?, metadata=?, updated_at=? WHERE task_id=?",
            (new_step, new_total, new_status, json.dumps(meta), now, task_id),
        )
        conn.commit()

    def complete_task(self, task_id: str, result: dict[str, Any] | None = None) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE tasks SET status='completed', result=?, updated_at=? WHERE task_id=?",
            (json.dumps(result) if result else None, time.time(), task_id),
        )
        conn.commit()

    def fail_task(self, task_id: str, error: str = "", result: dict[str, Any] | None = None) -> None:
        r = result or {"error": error}
        conn = self._get_conn()
        conn.execute(
            "UPDATE tasks SET status='failed', result=?, updated_at=? WHERE task_id=?",
            (json.dumps(r), time.time(), task_id),
        )
        conn.commit()

    def abort_task(self, task_id: str, reason: str = "") -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE tasks SET status='aborted', result=?, updated_at=? WHERE task_id=?",
            (json.dumps({"abort_reason": reason}), time.time(), task_id),
        )
        conn.commit()

    def get_task(self, task_id: str) -> TaskJournalEntry | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def get_interrupted_tasks(self) -> list[TaskJournalEntry]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM tasks WHERE status IN ('pending', 'in_progress')"
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_recent_tasks(self, limit: int = 20) -> list[TaskJournalEntry]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM tasks ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def save_robot_position(self, robot_id: str, x: float, y: float, yaw: float = 0.0) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO robot_positions (robot_id, x, y, yaw, t) VALUES (?, ?, ?, ?, ?)",
            (robot_id, x, y, yaw, time.time()),
        )
        conn.commit()

    def get_robot_position(self, robot_id: str) -> dict[str, float] | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM robot_positions WHERE robot_id = ?", (robot_id,)).fetchone()
        if row is None:
            return None
        return {"x": row["x"], "y": row["y"], "yaw": row["yaw"], "t": row["t"]}

    def set(self, key: str, value: Any) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
            (key, json.dumps(value, default=str)),
        )
        conn.commit()

    def get(self, key: str, default: Any = None) -> Any:
        conn = self._get_conn()
        row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return default

    def clear(self) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM tasks")
        conn.execute("DELETE FROM kv")
        conn.execute("DELETE FROM robot_positions")
        conn.commit()

    @property
    def task_count(self) -> int:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM tasks").fetchone()
        return row["cnt"]

    @staticmethod
    def _row_to_entry(row: Any) -> TaskJournalEntry:
        return TaskJournalEntry(
            task_id=row["task_id"],
            status=row["status"],
            metadata=json.loads(row["metadata"] or "{}"),
            step=row["step"],
            total_steps=row["total_steps"],
            robot_id=row["robot_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            result=json.loads(row["result"]) if row["result"] else None,
        )

    def __repr__(self) -> str:
        return f"<SQLiteStateStore tasks={self.task_count} path={self._path}>"


# ---------------------------------------------------------------------------
# OB-07: Redis backend
# ---------------------------------------------------------------------------

class RedisStateStore(StorageBackend):
    """
    OB-07: Redis-backed state store for distributed deployments.

    Multiple APYROBO instances can share state through Redis.
    Tasks are stored as Redis hashes, KV as simple strings.

    Requires: pip install redis

    Usage:
        store = RedisStateStore(host="redis", port=6379, prefix="apyrobo:")
    """

    def __init__(self, host: str = "localhost", port: int = 6379,
                 db: int = 0, prefix: str = "apyrobo:",
                 password: str | None = None,
                 _client: Any = None) -> None:
        self._prefix = prefix
        self._client = _client

        if self._client is None:
            try:
                import redis
                self._client = redis.Redis(host=host, port=port, db=db,
                                            password=password, decode_responses=True)
            except ImportError:
                raise RuntimeError(
                    "redis package required for RedisStateStore. "
                    "Install with: pip install redis"
                )

    def _key(self, *parts: str) -> str:
        return self._prefix + ":".join(parts)

    def begin_task(self, task_id: str, metadata: dict[str, Any] | None = None,
                   robot_id: str | None = None, total_steps: int = 0) -> TaskJournalEntry:
        now = time.time()
        entry = TaskJournalEntry(
            task_id=task_id, status="in_progress", metadata=metadata,
            robot_id=robot_id, total_steps=total_steps,
            created_at=now, updated_at=now,
        )
        self._client.hset(self._key("task", task_id), mapping={
            "task_id": task_id,
            "status": "in_progress",
            "metadata": json.dumps(metadata or {}),
            "step": 0,
            "total_steps": total_steps,
            "robot_id": robot_id or "",
            "created_at": now,
            "updated_at": now,
            "result": "",
        })
        self._client.zadd(self._key("task_index"), {task_id: now})
        return entry

    def update_task(self, task_id: str, step: int | None = None,
                    total_steps: int | None = None, status: str | None = None,
                    **extra: Any) -> None:
        key = self._key("task", task_id)
        if not self._client.exists(key):
            return
        updates: dict[str, Any] = {"updated_at": time.time()}
        if step is not None:
            updates["step"] = step
        if total_steps is not None:
            updates["total_steps"] = total_steps
        if status is not None:
            updates["status"] = status
        if extra:
            existing_meta = json.loads(self._client.hget(key, "metadata") or "{}")
            existing_meta.update(extra)
            updates["metadata"] = json.dumps(existing_meta)
        self._client.hset(key, mapping=updates)
        self._client.zadd(self._key("task_index"), {task_id: updates["updated_at"]})

    def complete_task(self, task_id: str, result: dict[str, Any] | None = None) -> None:
        now = time.time()
        key = self._key("task", task_id)
        self._client.hset(key, mapping={
            "status": "completed",
            "result": json.dumps(result) if result else "",
            "updated_at": now,
        })
        self._client.zadd(self._key("task_index"), {task_id: now})

    def fail_task(self, task_id: str, error: str = "", result: dict[str, Any] | None = None) -> None:
        now = time.time()
        key = self._key("task", task_id)
        r = result or {"error": error}
        self._client.hset(key, mapping={
            "status": "failed",
            "result": json.dumps(r),
            "updated_at": now,
        })
        self._client.zadd(self._key("task_index"), {task_id: now})

    def abort_task(self, task_id: str, reason: str = "") -> None:
        now = time.time()
        key = self._key("task", task_id)
        self._client.hset(key, mapping={
            "status": "aborted",
            "result": json.dumps({"abort_reason": reason}),
            "updated_at": now,
        })

    def get_task(self, task_id: str) -> TaskJournalEntry | None:
        key = self._key("task", task_id)
        data = self._client.hgetall(key)
        if not data:
            return None
        return self._hash_to_entry(data)

    def get_interrupted_tasks(self) -> list[TaskJournalEntry]:
        task_ids = self._client.zrange(self._key("task_index"), 0, -1)
        result = []
        for tid in task_ids:
            entry = self.get_task(tid)
            if entry and entry.is_interrupted:
                result.append(entry)
        return result

    def get_recent_tasks(self, limit: int = 20) -> list[TaskJournalEntry]:
        task_ids = self._client.zrevrange(self._key("task_index"), 0, limit - 1)
        result = []
        for tid in task_ids:
            entry = self.get_task(tid)
            if entry:
                result.append(entry)
        return result

    def save_robot_position(self, robot_id: str, x: float, y: float, yaw: float = 0.0) -> None:
        self._client.hset(self._key("pos", robot_id), mapping={
            "x": x, "y": y, "yaw": yaw, "t": time.time(),
        })

    def get_robot_position(self, robot_id: str) -> dict[str, float] | None:
        data = self._client.hgetall(self._key("pos", robot_id))
        if not data:
            return None
        return {k: float(v) for k, v in data.items()}

    def set(self, key: str, value: Any) -> None:
        self._client.set(self._key("kv", key), json.dumps(value, default=str))

    def get(self, key: str, default: Any = None) -> Any:
        raw = self._client.get(self._key("kv", key))
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return default

    def clear(self) -> None:
        # Delete all keys with our prefix
        keys = self._client.keys(self._prefix + "*")
        if keys:
            self._client.delete(*keys)

    @property
    def task_count(self) -> int:
        return self._client.zcard(self._key("task_index"))

    @staticmethod
    def _hash_to_entry(data: dict[str, str]) -> TaskJournalEntry:
        return TaskJournalEntry(
            task_id=data.get("task_id", ""),
            status=data.get("status", "pending"),
            metadata=json.loads(data.get("metadata", "{}")),
            step=int(float(data.get("step", 0))),
            total_steps=int(float(data.get("total_steps", 0))),
            robot_id=data.get("robot_id") or None,
            created_at=float(data.get("created_at", 0)),
            updated_at=float(data.get("updated_at", 0)),
            result=json.loads(data["result"]) if data.get("result") else None,
        )

    def __repr__(self) -> str:
        return f"<RedisStateStore prefix={self._prefix} tasks={self.task_count}>"
