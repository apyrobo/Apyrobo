"""
Episodic Memory — SQLite-backed task execution history.

Stores structured records of skill executions: what ran, what the robot
state was, the outcome, timing, and any metadata.  Queryable by time
range, task type, robot ID, or outcome.

Integration:
    The EpisodicStore can be attached to a SkillExecutor to auto-record
    episodes after each skill graph run.

Usage::

    store = EpisodicStore()                   # in-memory (testing)
    store = EpisodicStore("~/.apyrobo/episodes.db")   # persisted

    episode = Episode(
        task="deliver package to room 3",
        robot_id="turtlebot4",
        skills_run=["navigate_to", "pick_object"],
        robot_state={"x": 1.0, "y": 2.0},
        outcome="success",
    )
    store.record(episode)

    recent = store.query(limit=10)
    successes = store.query(outcome="success", robot_id="turtlebot4")
    last_hour = store.query(time_from=time.time() - 3600)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# SQLite schema version — increment if schema changes
_SCHEMA_VERSION = 1

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   REAL    NOT NULL,
    robot_id    TEXT    NOT NULL DEFAULT '',
    task        TEXT    NOT NULL DEFAULT '',
    task_type   TEXT    NOT NULL DEFAULT '',
    skills_run  TEXT    NOT NULL DEFAULT '[]',
    robot_state TEXT    NOT NULL DEFAULT '{}',
    outcome     TEXT    NOT NULL DEFAULT 'unknown',
    duration_s  REAL    NOT NULL DEFAULT 0.0,
    metadata    TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp);
CREATE INDEX IF NOT EXISTS idx_robot_id  ON episodes(robot_id);
CREATE INDEX IF NOT EXISTS idx_outcome   ON episodes(outcome);
CREATE INDEX IF NOT EXISTS idx_task_type ON episodes(task_type);
"""


# ---------------------------------------------------------------------------
# Episode dataclass
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """
    A single task execution record.

    Attributes:
        task: Human-readable task description.
        robot_id: Identifier of the robot that executed the task.
        skills_run: Ordered list of skill IDs that were executed.
        robot_state: Snapshot of robot state at execution time.
        outcome: ``"success"``, ``"failure"``, ``"aborted"``, or ``"unknown"``.
        task_type: Optional category label (e.g. ``"navigation"``, ``"pick"``)
            for easier filtering.
        duration_s: Wall-clock duration of the task in seconds.
        metadata: Arbitrary extra data (step counts, error messages, etc.).
        timestamp: Unix timestamp of episode creation (auto-set if 0).
        id: Database row ID — set by EpisodicStore after recording.
    """

    task: str = ""
    robot_id: str = ""
    skills_run: list[str] = field(default_factory=list)
    robot_state: dict[str, Any] = field(default_factory=dict)
    outcome: str = "unknown"
    task_type: str = ""
    duration_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    id: int | None = field(default=None, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        d = asdict(self)
        d.pop("id", None)
        return d


# ---------------------------------------------------------------------------
# EpisodicStore
# ---------------------------------------------------------------------------

class EpisodicStore:
    """
    SQLite-backed store for task execution episodes.

    Thread-safe via ``check_same_thread=False``; callers that share an
    instance across threads should coordinate writes externally if strict
    ordering is needed.

    Args:
        db_path: Path to the SQLite database file.  Pass ``":memory:"`` or
            ``None`` for an in-memory database (useful in tests).
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None or str(db_path) == ":memory:":
            self._path = ":memory:"
        else:
            self._path = str(Path(db_path).expanduser())
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            self._path,
            check_same_thread=False,
            isolation_level=None,  # autocommit
        )
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("EpisodicStore opened: %s", self._path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self._conn.executescript(_CREATE_TABLE)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(self, episode: Episode) -> Episode:
        """
        Persist an episode and return it with its assigned database ID.

        If ``episode.timestamp`` is 0, it is set to the current time.
        """
        if episode.timestamp == 0:
            episode.timestamp = time.time()

        cur = self._conn.execute(
            """
            INSERT INTO episodes
                (timestamp, robot_id, task, task_type, skills_run,
                 robot_state, outcome, duration_s, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode.timestamp,
                episode.robot_id,
                episode.task,
                episode.task_type,
                json.dumps(episode.skills_run),
                json.dumps(episode.robot_state),
                episode.outcome,
                episode.duration_s,
                json.dumps(episode.metadata),
            ),
        )
        episode.id = cur.lastrowid
        logger.debug("Recorded episode %d: %r (%s)", episode.id, episode.task, episode.outcome)
        return episode

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        *,
        time_from: float | None = None,
        time_to: float | None = None,
        task_type: str | None = None,
        robot_id: str | None = None,
        outcome: str | None = None,
        limit: int = 100,
        offset: int = 0,
        order: str = "DESC",
    ) -> list[Episode]:
        """
        Query episodes with optional filters.

        Args:
            time_from: Include only episodes at or after this Unix timestamp.
            time_to: Include only episodes at or before this Unix timestamp.
            task_type: Filter by task type label (exact match).
            robot_id: Filter by robot identifier (exact match).
            outcome: Filter by outcome string (exact match).
            limit: Maximum number of results to return.
            offset: Number of rows to skip (for pagination).
            order: ``"DESC"`` (newest first) or ``"ASC"`` (oldest first).

        Returns a list of :class:`Episode` objects.
        """
        where_clauses: list[str] = []
        params: list[Any] = []

        if time_from is not None:
            where_clauses.append("timestamp >= ?")
            params.append(time_from)
        if time_to is not None:
            where_clauses.append("timestamp <= ?")
            params.append(time_to)
        if task_type is not None:
            where_clauses.append("task_type = ?")
            params.append(task_type)
        if robot_id is not None:
            where_clauses.append("robot_id = ?")
            params.append(robot_id)
        if outcome is not None:
            where_clauses.append("outcome = ?")
            params.append(outcome)

        order_clause = "ASC" if order.upper() == "ASC" else "DESC"
        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        sql = f"""
            SELECT * FROM episodes
            {where_sql}
            ORDER BY timestamp {order_clause}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_episode(row) for row in rows]

    def get(self, episode_id: int) -> Episode | None:
        """Fetch a single episode by its database ID."""
        row = self._conn.execute(
            "SELECT * FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        return self._row_to_episode(row) if row else None

    def count(
        self,
        *,
        robot_id: str | None = None,
        outcome: str | None = None,
    ) -> int:
        """Return the total number of stored episodes, optionally filtered."""
        where_clauses: list[str] = []
        params: list[Any] = []
        if robot_id is not None:
            where_clauses.append("robot_id = ?")
            params.append(robot_id)
        if outcome is not None:
            where_clauses.append("outcome = ?")
            params.append(outcome)
        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        row = self._conn.execute(
            f"SELECT COUNT(*) FROM episodes {where_sql}", params
        ).fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def delete_older_than(self, cutoff_timestamp: float) -> int:
        """
        Delete episodes older than *cutoff_timestamp*.

        Returns the number of rows deleted.
        """
        cur = self._conn.execute(
            "DELETE FROM episodes WHERE timestamp < ?", (cutoff_timestamp,)
        )
        deleted = cur.rowcount
        logger.info("EpisodicStore: deleted %d episodes older than %s", deleted, cutoff_timestamp)
        return deleted

    def clear(self) -> int:
        """Delete all episodes. Returns the count deleted."""
        cur = self._conn.execute("DELETE FROM episodes")
        return cur.rowcount

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_episode(row: sqlite3.Row) -> Episode:
        return Episode(
            id=row["id"],
            timestamp=row["timestamp"],
            robot_id=row["robot_id"],
            task=row["task"],
            task_type=row["task_type"],
            skills_run=json.loads(row["skills_run"]),
            robot_state=json.loads(row["robot_state"]),
            outcome=row["outcome"],
            duration_s=row["duration_s"],
            metadata=json.loads(row["metadata"]),
        )

    def __repr__(self) -> str:
        return f"<EpisodicStore path={self._path!r} count={self.count()}>"
