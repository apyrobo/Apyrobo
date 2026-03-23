"""
Audit Trail — immutable, cryptographically-chained log of all commands,
decisions, and violations.

Each event records a SHA-256 hash of the *previous* event, forming a chain
that detects any tampering or deletion.

Classes:
    AuditEvent  — one immutable record
    AuditTrail  — persists events to SQLite (default: :memory:)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_GENESIS_HASH = "0" * 64  # sentinel for the first event


@dataclass
class AuditEvent:
    """One immutable audit record."""

    event_id: str       # UUID
    timestamp: float
    actor: str          # user/system that triggered it
    action: str         # e.g. "skill_executed", "robot_registered"
    resource: str       # what was acted upon
    outcome: str        # "success" | "failure"
    metadata: dict[str, Any]
    prev_hash: str      # SHA-256 of previous event (or genesis sentinel)

    def compute_hash(self) -> str:
        """Return SHA-256 of the canonical representation of this event."""
        payload = json.dumps(
            {
                "event_id": self.event_id,
                "timestamp": self.timestamp,
                "actor": self.actor,
                "action": self.action,
                "resource": self.resource,
                "outcome": self.outcome,
                "metadata": self.metadata,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "outcome": self.outcome,
            "metadata": self.metadata,
            "prev_hash": self.prev_hash,
        }


class AuditTrail:
    """
    Tamper-evident audit trail backed by SQLite.

    Usage:
        trail = AuditTrail()          # in-memory
        trail = AuditTrail("audit.db") # persistent file

        event = trail.record(
            actor="operator_1",
            action="skill_executed",
            resource="navigate_to",
            outcome="success",
            metadata={"x": 1, "y": 2},
        )
        assert trail.verify_chain()
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        self._last_hash: str = _GENESIS_HASH

        # Recover last hash from existing DB
        row = self._conn.execute(
            "SELECT hash FROM audit_events ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if row:
            self._last_hash = row[0]

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id  TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                actor     TEXT NOT NULL,
                action    TEXT NOT NULL,
                resource  TEXT NOT NULL,
                outcome   TEXT NOT NULL,
                metadata  TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                hash      TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def record(
        self,
        actor: str,
        action: str,
        resource: str,
        outcome: str,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Create and persist a new audit event."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            actor=actor,
            action=action,
            resource=resource,
            outcome=outcome,
            metadata=metadata or {},
            prev_hash=self._last_hash,
        )
        event_hash = event.compute_hash()
        self._conn.execute(
            """
            INSERT INTO audit_events
              (event_id, timestamp, actor, action, resource, outcome, metadata, prev_hash, hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.timestamp,
                event.actor,
                event.action,
                event.resource,
                event.outcome,
                json.dumps(event.metadata),
                event.prev_hash,
                event_hash,
            ),
        )
        self._conn.commit()
        self._last_hash = event_hash
        logger.debug(
            "Audit: %s %s %s %s", actor, action, resource, outcome
        )
        return event

    def query(
        self,
        actor: str | None = None,
        action: str | None = None,
        since: float | None = None,
    ) -> list[AuditEvent]:
        """Return events matching the given filters (all optional)."""
        sql = "SELECT event_id, timestamp, actor, action, resource, outcome, metadata, prev_hash FROM audit_events WHERE 1=1"
        params: list[Any] = []
        if actor is not None:
            sql += " AND actor = ?"
            params.append(actor)
        if action is not None:
            sql += " AND action = ?"
            params.append(action)
        if since is not None:
            sql += " AND timestamp >= ?"
            params.append(since)
        sql += " ORDER BY timestamp ASC"

        rows = self._conn.execute(sql, params).fetchall()
        events = []
        for row in rows:
            event_id, ts, act, act_name, resource, outcome, meta_json, prev_hash = row
            events.append(
                AuditEvent(
                    event_id=event_id,
                    timestamp=ts,
                    actor=act,
                    action=act_name,
                    resource=resource,
                    outcome=outcome,
                    metadata=json.loads(meta_json),
                    prev_hash=prev_hash,
                )
            )
        return events

    def verify_chain(self) -> bool:
        """
        Verify hash-chain integrity.

        Re-computes each event's hash and checks that prev_hash matches the
        previous event's hash.  Returns True if the chain is intact.
        """
        rows = self._conn.execute(
            "SELECT event_id, timestamp, actor, action, resource, outcome, metadata, prev_hash, hash "
            "FROM audit_events ORDER BY rowid ASC"
        ).fetchall()

        expected_prev = _GENESIS_HASH
        for row in rows:
            event_id, ts, actor, action, resource, outcome, meta_json, prev_hash, stored_hash = row
            event = AuditEvent(
                event_id=event_id,
                timestamp=ts,
                actor=actor,
                action=action,
                resource=resource,
                outcome=outcome,
                metadata=json.loads(meta_json),
                prev_hash=prev_hash,
            )
            if prev_hash != expected_prev:
                logger.error("Chain broken at event %s: prev_hash mismatch", event_id)
                return False
            computed = event.compute_hash()
            if computed != stored_hash:
                logger.error("Chain broken at event %s: hash mismatch", event_id)
                return False
            expected_prev = stored_hash
        return True

    def close(self) -> None:
        self._conn.close()

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()
        return row[0] if row else 0
