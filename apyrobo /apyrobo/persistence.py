"""
State Persistence — survives crashes and restarts.

Maintains a task journal on disk so the system can recover after
power loss, Docker restart, or node crash. On startup, reads the
journal and either resumes pending tasks or safely aborts them.

Storage: JSON file (simple, inspectable, no database dependency).
In production, swap to SQLite or Redis via the StorageBackend interface.

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
"""

from __future__ import annotations

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


class StateStore:
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
