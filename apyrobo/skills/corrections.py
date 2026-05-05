"""Correction learning — user overrides bias future LLM planning.

Records human corrections to skill steps and uses them to augment planning
prompts so the LLM avoids repeating past mistakes.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Correction:
    """A single human correction to a skill step."""

    correction_id: str
    task_description: str
    original_step: dict[str, Any]   # {skill_id, parameters}
    corrected_step: dict[str, Any]  # {skill_id, parameters}
    reason: str | None
    timestamp: str                  # ISO 8601
    applied_count: int = 0


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS corrections (
    correction_id   TEXT PRIMARY KEY,
    task_description TEXT NOT NULL,
    original_step   TEXT NOT NULL,
    corrected_step  TEXT NOT NULL,
    reason          TEXT,
    timestamp       TEXT NOT NULL,
    applied_count   INTEGER NOT NULL DEFAULT 0
)
"""


def _word_overlap(a: str, b: str) -> float:
    """Jaccard similarity on lowercased word sets."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa and not wb:
        return 1.0
    union = wa | wb
    return len(wa & wb) / len(union)


class CorrectionStore:
    """SQLite-backed store for human corrections.

    Args:
        db_path: Path to the SQLite database file.  Use ``":memory:"`` in tests.
    """

    def __init__(self, db_path: str = "corrections.db") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self,
        task_description: str,
        original_step: dict[str, Any],
        corrected_step: dict[str, Any],
        reason: str = "",
    ) -> Correction:
        """Persist a new correction and return it."""
        c = Correction(
            correction_id=str(uuid.uuid4()),
            task_description=task_description,
            original_step=original_step,
            corrected_step=corrected_step,
            reason=reason or None,
            timestamp=datetime.utcnow().isoformat(),
        )
        self._conn.execute(
            "INSERT INTO corrections VALUES (?,?,?,?,?,?,?)",
            (
                c.correction_id,
                c.task_description,
                json.dumps(c.original_step),
                json.dumps(c.corrected_step),
                c.reason,
                c.timestamp,
                c.applied_count,
            ),
        )
        self._conn.commit()
        return c

    def mark_applied(self, correction_id: str) -> None:
        """Increment applied_count for the given correction."""
        self._conn.execute(
            "UPDATE corrections SET applied_count = applied_count + 1 WHERE correction_id = ?",
            (correction_id,),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def list_all(self) -> list[Correction]:
        """Return all stored corrections."""
        cur = self._conn.execute(
            "SELECT correction_id, task_description, original_step, corrected_step, "
            "reason, timestamp, applied_count FROM corrections ORDER BY timestamp"
        )
        return [self._row_to_correction(r) for r in cur.fetchall()]

    def find_relevant(
        self,
        task: str,
        skill_id: str,
        top_k: int = 3,
    ) -> list[Correction]:
        """Return the top-k corrections most relevant to *task* and *skill_id*.

        Relevance = word-overlap similarity between *task* and the stored
        task_description, filtered to corrections whose original or corrected
        step involves *skill_id*.
        """
        all_corrections = self.list_all()

        scored: list[tuple[float, Correction]] = []
        for c in all_corrections:
            orig_sid = c.original_step.get("skill_id", "")
            corr_sid = c.corrected_step.get("skill_id", "")
            if skill_id and skill_id not in (orig_sid, corr_sid):
                continue
            score = _word_overlap(task, c.task_description)
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_correction(row: tuple) -> Correction:
        cid, task_desc, orig_json, corr_json, reason, ts, applied = row
        return Correction(
            correction_id=cid,
            task_description=task_desc,
            original_step=json.loads(orig_json),
            corrected_step=json.loads(corr_json),
            reason=reason,
            timestamp=ts,
            applied_count=applied,
        )


# ---------------------------------------------------------------------------
# CorrectionLearner
# ---------------------------------------------------------------------------

class CorrectionLearner:
    """Wraps a CorrectionStore and provides prompt-augmentation utilities."""

    def __init__(self, store: CorrectionStore) -> None:
        self._store = store

    @property
    def store(self) -> CorrectionStore:
        return self._store

    def record_correction(
        self,
        task_description: str,
        original_step: dict[str, Any],
        corrected_step: dict[str, Any],
        reason: str = "",
    ) -> Correction:
        """Record a user correction and return the persisted Correction."""
        return self._store.record(task_description, original_step, corrected_step, reason)

    def augment_prompt(
        self,
        task: str,
        base_prompt: str,
        skill_id: str = "",
    ) -> str:
        """Prepend relevant past corrections to *base_prompt*.

        If no relevant corrections are found, returns *base_prompt* unchanged.
        """
        relevant = self._store.find_relevant(task, skill_id)
        if not relevant:
            return base_prompt

        lines = ["[Past corrections — prefer these over default behaviour]"]
        for c in relevant:
            orig_sid = c.original_step.get("skill_id", "?")
            corr_sid = c.corrected_step.get("skill_id", "?")
            reason_note = f" (reason: {c.reason})" if c.reason else ""
            lines.append(
                f"  - For task similar to {c.task_description!r}: "
                f"use {corr_sid!r} instead of {orig_sid!r}{reason_note}"
            )
            self._store.mark_applied(c.correction_id)

        prefix = "\n".join(lines) + "\n\n"
        return prefix + base_prompt
