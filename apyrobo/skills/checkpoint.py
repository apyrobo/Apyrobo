"""
Execution Checkpointing — resume skill graphs from last successful step.

Classes:
    CheckpointEntry      — serialisable snapshot after one completed step
    CheckpointStore      — SQLite-backed storage for checkpoint entries
    CheckpointedExecutor — wraps step-list execution with auto-checkpointing
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS checkpoints (
    task_id         TEXT    PRIMARY KEY,
    skill_name      TEXT    NOT NULL DEFAULT '',
    step_index      INTEGER NOT NULL DEFAULT 0,
    total_steps     INTEGER NOT NULL DEFAULT 0,
    state           TEXT    NOT NULL DEFAULT '{}',
    completed_steps TEXT    NOT NULL DEFAULT '[]',
    timestamp       REAL    NOT NULL DEFAULT 0.0,
    checksum        TEXT    NOT NULL DEFAULT ''
);
"""


# ---------------------------------------------------------------------------
# CheckpointEntry
# ---------------------------------------------------------------------------

@dataclass
class CheckpointEntry:
    """Serialisable snapshot of task progress after a completed step."""

    task_id: str
    skill_name: str
    step_index: int
    total_steps: int
    state: dict[str, Any]
    completed_steps: list[str]
    timestamp: float = field(default_factory=time.time)
    checksum: str = field(default="")

    def __post_init__(self) -> None:
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """SHA-256 of the serialised *state* dict."""
        payload = json.dumps(self.state, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    def verify_integrity(self) -> bool:
        """Return True if the stored checksum matches the state content."""
        return self.checksum == self._compute_checksum()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "skill_name": self.skill_name,
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "state": self.state,
            "completed_steps": self.completed_steps,
            "timestamp": self.timestamp,
            "checksum": self.checksum,
        }


# ---------------------------------------------------------------------------
# CheckpointStore
# ---------------------------------------------------------------------------

class CheckpointStore:
    """
    SQLite-backed checkpoint storage.

    One row per *task_id* — each save/update overwrites the previous entry
    for the same task (upsert semantics).

    Args:
        db_path: Path to SQLite file.  Use ``":memory:"`` for tests.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_CREATE_TABLE)
        logger.debug("CheckpointStore opened: %s", db_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, entry: CheckpointEntry) -> None:
        """Upsert a checkpoint entry for *entry.task_id*."""
        self._conn.execute(
            """
            INSERT INTO checkpoints
                (task_id, skill_name, step_index, total_steps,
                 state, completed_steps, timestamp, checksum)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                skill_name      = excluded.skill_name,
                step_index      = excluded.step_index,
                total_steps     = excluded.total_steps,
                state           = excluded.state,
                completed_steps = excluded.completed_steps,
                timestamp       = excluded.timestamp,
                checksum        = excluded.checksum
            """,
            (
                entry.task_id,
                entry.skill_name,
                entry.step_index,
                entry.total_steps,
                json.dumps(entry.state, sort_keys=True),
                json.dumps(entry.completed_steps),
                entry.timestamp,
                entry.checksum,
            ),
        )
        logger.debug(
            "CheckpointStore: saved task=%r step=%d/%d",
            entry.task_id, entry.step_index, entry.total_steps,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self, task_id: str) -> CheckpointEntry | None:
        """Load the latest checkpoint for *task_id*, or None if not found."""
        row = self._conn.execute(
            "SELECT * FROM checkpoints WHERE task_id = ?", (task_id,)
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def list_tasks(self) -> list[str]:
        """Return all task IDs with stored checkpoints."""
        rows = self._conn.execute(
            "SELECT task_id FROM checkpoints ORDER BY timestamp"
        ).fetchall()
        return [r["task_id"] for r in rows]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, task_id: str) -> None:
        """Remove the checkpoint for *task_id* (no-op if not present)."""
        self._conn.execute(
            "DELETE FROM checkpoints WHERE task_id = ?", (task_id,)
        )
        logger.debug("CheckpointStore: deleted task=%r", task_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> CheckpointEntry:
        entry = CheckpointEntry(
            task_id=row["task_id"],
            skill_name=row["skill_name"],
            step_index=row["step_index"],
            total_steps=row["total_steps"],
            state=json.loads(row["state"]),
            completed_steps=json.loads(row["completed_steps"]),
            timestamp=row["timestamp"],
            checksum=row["checksum"],
        )
        # Bypass __post_init__ recalculation — checksum already stored
        return entry

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# CheckpointedExecutor
# ---------------------------------------------------------------------------

class CheckpointedExecutor:
    """
    Wraps a list of named steps with automatic checkpointing.

    If *resume* is True and a checkpoint exists for *task_id*, execution
    skips already-completed steps and resumes from the next pending one.
    A checkpoint is saved after each successful step.

    Usage::

        store = CheckpointStore()
        executor = CheckpointedExecutor(store)

        steps = [
            ("load_map",   load_map_fn,   {}),
            ("navigate",   navigate_fn,   {"x": 1.0, "y": 2.0}),
            ("pick_object", pick_fn,      {}),
        ]
        result = executor.execute_steps("task-001", steps)
        # result = {"completed": [...], "failed": None, "result": <last output>}
    """

    def __init__(self, store: CheckpointStore) -> None:
        self._store = store

    def execute_steps(
        self,
        task_id: str,
        steps: list[tuple[str, Callable, dict]],
        resume: bool = True,
    ) -> dict[str, Any]:
        """
        Execute *steps* for *task_id* with checkpointing.

        Args:
            task_id:  Unique task identifier used as checkpoint key.
            steps:    List of ``(step_name, fn, kwargs)`` tuples.
            resume:   If True and a checkpoint exists, skip completed steps.

        Returns:
            ``{"completed": [str, ...], "failed": str | None, "result": Any}``
            where *failed* is the name of the step that raised an exception,
            or None if all steps completed successfully.
        """
        completed: list[str] = []
        state: dict[str, Any] = {}
        start_index = 0
        last_result: Any = None

        # Resume from checkpoint if available
        if resume:
            entry = self._store.load(task_id)
            if entry is not None:
                if not entry.verify_integrity():
                    logger.warning(
                        "CheckpointedExecutor: checksum mismatch for task=%r — starting fresh",
                        task_id,
                    )
                else:
                    completed = list(entry.completed_steps)
                    state = dict(entry.state)
                    start_index = entry.step_index
                    logger.info(
                        "CheckpointedExecutor: resuming task=%r from step %d/%d",
                        task_id, start_index, len(steps),
                    )

        # Execute remaining steps
        for idx in range(start_index, len(steps)):
            step_name, fn, kwargs = steps[idx]
            logger.debug(
                "CheckpointedExecutor: task=%r running step %d/%d %r",
                task_id, idx + 1, len(steps), step_name,
            )
            try:
                last_result = fn(**kwargs)
                completed.append(step_name)
                state[step_name] = {"status": "ok", "result": str(last_result)}
                # Save checkpoint after each successful step
                entry = CheckpointEntry(
                    task_id=task_id,
                    skill_name=step_name,
                    step_index=idx + 1,
                    total_steps=len(steps),
                    state=state,
                    completed_steps=completed,
                )
                self._store.save(entry)
            except Exception as exc:
                logger.error(
                    "CheckpointedExecutor: task=%r step %r failed: %s",
                    task_id, step_name, exc,
                )
                return {"completed": completed, "failed": step_name, "result": None}

        # All steps completed — clean up checkpoint
        self._store.delete(task_id)
        return {"completed": completed, "failed": None, "result": last_result}
