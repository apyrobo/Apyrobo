"""Natural language safety policies for APYROBO.

Allows operators to express safety constraints in plain English and have
them automatically enforced by the ``SafetyEnforcer``.

Quick start::

    from apyrobo.safety.nl_policy import NLPolicyStore, NLPolicyParser

    store = NLPolicyStore()
    parser = NLPolicyParser()

    # Translate a natural language rule into a structured SafetyPolicy
    policy = parser.parse("never exceed 0.5 m/s near humans")
    store.add(policy)

    # Enforcement happens in SafetyEnforcer automatically when store is wired:
    store.get_active_policies()  # → list[NLSafetyPolicy]

CLI usage::

    apyrobo policy add "never exceed 0.5 m/s near humans"
    apyrobo policy list
    apyrobo policy remove <id>
    apyrobo policy check "navigate at 2.0 m/s"

v7.0.0 — APYROBO Category Ownership
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Policy data model
# ---------------------------------------------------------------------------

# Supported constraint types derived from natural language
CONSTRAINT_TYPES = {
    "speed_limit",       # "never exceed X m/s"
    "proximity_limit",   # "stay X m away from humans"
    "no_go_zone",        # "never enter the warehouse"
    "battery_reserve",   # "always keep X% battery"
    "time_window",       # "only operate between Xam and Ypm"
    "load_limit",        # "never carry more than X kg"
    "custom",            # catch-all for unrecognised patterns
}

# Regex patterns for common constraint types (unit: SI base units)
_SPEED_RE = re.compile(
    r"(?:never\s+exceed|max(?:imum)?\s+speed|speed\s+limit)\s+"
    r"([0-9]+(?:\.[0-9]+)?)\s*m/s",
    re.IGNORECASE,
)
_PROXIMITY_RE = re.compile(
    r"(?:stay|keep|maintain)\s+(?:at\s+least\s+)?([0-9]+(?:\.[0-9]+)?)\s*m(?:eter|etre)?s?\s+"
    r"(?:away\s+from|from)\s+([a-z\s]+)",
    re.IGNORECASE,
)
_BATTERY_RE = re.compile(
    r"(?:always\s+keep|maintain|reserve)\s+(?:at\s+least\s+)?([0-9]+(?:\.[0-9]+)?)\s*%\s*battery",
    re.IGNORECASE,
)
_NOGO_RE = re.compile(
    r"(?:never\s+enter|do\s+not\s+enter|avoid)\s+(?:the\s+)?(.+)",
    re.IGNORECASE,
)


@dataclass
class NLSafetyPolicy:
    """A safety policy derived from a natural language description.

    Attributes
    ----------
    policy_id:
        Unique identifier (UUID-based).
    description:
        Original natural language description provided by the operator.
    constraint_type:
        Structured constraint category (e.g. ``"speed_limit"``).
    parameters:
        Constraint-specific parameters (e.g. ``{"max_speed_ms": 0.5}``).
    severity:
        ``"hard"`` (block the action) or ``"soft"`` (warn only).
    created_at:
        Unix timestamp when the policy was added.
    active:
        Whether the policy is currently enforced.
    source:
        How the policy was parsed: ``"regex"`` or ``"llm"``.
    """

    policy_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    description: str = ""
    constraint_type: str = "custom"
    parameters: dict[str, Any] = field(default_factory=dict)
    severity: str = "hard"   # "hard" | "soft"
    created_at: float = field(default_factory=time.time)
    active: bool = True
    source: str = "regex"

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "description": self.description,
            "constraint_type": self.constraint_type,
            "parameters": self.parameters,
            "severity": self.severity,
            "created_at": self.created_at,
            "active": self.active,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NLSafetyPolicy":
        return cls(
            policy_id=d.get("policy_id", str(uuid.uuid4())[:12]),
            description=d.get("description", ""),
            constraint_type=d.get("constraint_type", "custom"),
            parameters=d.get("parameters", {}),
            severity=d.get("severity", "hard"),
            created_at=d.get("created_at", time.time()),
            active=d.get("active", True),
            source=d.get("source", "regex"),
        )

    def summary(self) -> str:
        """One-line human-readable summary for CLI output."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return (
            f"[{self.policy_id}] {self.constraint_type}"
            + (f"({params_str})" if params_str else "")
            + f" [{self.severity}] — {self.description[:60]}"
        )


# ---------------------------------------------------------------------------
# NLPolicyParser
# ---------------------------------------------------------------------------

class NLPolicyParser:
    """Translates natural language safety rules into ``NLSafetyPolicy`` objects.

    Uses a regex-based rule set for common patterns (no API key required).
    When an LLM provider is available via *agent*, it falls back to LLM
    parsing for patterns the regex rules don't cover.

    Parameters
    ----------
    agent:
        Optional ``Agent`` instance used as LLM fallback.  When ``None``
        (default), unrecognised patterns produce a ``"custom"`` policy with
        the description stored verbatim.
    severity:
        Default severity for parsed policies (``"hard"`` or ``"soft"``).

    Examples
    --------
    ::

        parser = NLPolicyParser()
        p = parser.parse("never exceed 0.5 m/s near humans")
        # p.constraint_type == "speed_limit"
        # p.parameters == {"max_speed_ms": 0.5, "context": "near humans"}
    """

    def __init__(self, agent: Any = None, severity: str = "hard") -> None:
        self._agent = agent
        self._default_severity = severity

    def parse(self, description: str) -> NLSafetyPolicy:
        """Parse *description* into a structured ``NLSafetyPolicy``.

        Tries regex rules first.  Falls back to LLM if an agent is wired
        and regex rules produce no match.  Always succeeds — unknown
        patterns yield a ``"custom"`` policy.

        Parameters
        ----------
        description:
            Natural language safety rule (e.g. ``"never exceed 0.5 m/s"``).

        Returns
        -------
        NLSafetyPolicy
            Structured policy ready for storage and enforcement.
        """
        policy = self._try_regex(description)
        if policy is not None:
            return policy

        if self._agent is not None:
            policy = self._try_llm(description)
            if policy is not None:
                return policy

        # Fallback — store verbatim as custom policy
        return NLSafetyPolicy(
            description=description,
            constraint_type="custom",
            parameters={"raw": description},
            severity=self._default_severity,
            source="regex",
        )

    def _try_regex(self, description: str) -> NLSafetyPolicy | None:
        """Return a policy if *description* matches a known regex pattern."""
        # Speed limit
        m = _SPEED_RE.search(description)
        if m:
            context = re.sub(r".*m/s\s*", "", description, flags=re.IGNORECASE).strip()
            return NLSafetyPolicy(
                description=description,
                constraint_type="speed_limit",
                parameters={
                    "max_speed_ms": float(m.group(1)),
                    "context": context or "always",
                },
                severity=self._default_severity,
                source="regex",
            )

        # Proximity
        m = _PROXIMITY_RE.search(description)
        if m:
            return NLSafetyPolicy(
                description=description,
                constraint_type="proximity_limit",
                parameters={
                    "min_distance_m": float(m.group(1)),
                    "target": m.group(2).strip(),
                },
                severity=self._default_severity,
                source="regex",
            )

        # Battery reserve
        m = _BATTERY_RE.search(description)
        if m:
            return NLSafetyPolicy(
                description=description,
                constraint_type="battery_reserve",
                parameters={"min_battery_pct": float(m.group(1))},
                severity=self._default_severity,
                source="regex",
            )

        # No-go zone
        m = _NOGO_RE.search(description)
        if m:
            zone = m.group(1).strip().rstrip(".")
            return NLSafetyPolicy(
                description=description,
                constraint_type="no_go_zone",
                parameters={"zone_name": zone},
                severity=self._default_severity,
                source="regex",
            )

        return None

    def _try_llm(self, description: str) -> NLSafetyPolicy | None:
        """Ask the LLM to parse *description* into a structured policy."""
        prompt = (
            f"Parse this safety rule into a JSON object with keys: "
            f"constraint_type (one of: speed_limit, proximity_limit, no_go_zone, "
            f"battery_reserve, time_window, load_limit, custom), "
            f"parameters (dict of numeric/string values). "
            f"Safety rule: \"{description}\"\n"
            f"Reply with only valid JSON."
        )
        try:
            # Use the agent's LLM to parse
            result_text = self._agent.complete(prompt)
            # Extract JSON from the response
            json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
            if not json_match:
                return None
            data = json.loads(json_match.group())
            return NLSafetyPolicy(
                description=description,
                constraint_type=data.get("constraint_type", "custom"),
                parameters=data.get("parameters", {}),
                severity=self._default_severity,
                source="llm",
            )
        except Exception as exc:
            logger.debug("NLPolicyParser LLM parse failed: %s", exc)
            return None

    def check_compliance(self, action_description: str, policies: list[NLSafetyPolicy]) -> list[str]:
        """Check whether *action_description* would violate any active policy.

        Performs a lightweight textual check against each policy.  For
        hard enforcement, use ``NLPolicyEnforcer``.

        Parameters
        ----------
        action_description:
            Free-text description of the proposed action
            (e.g. ``"navigate at 2.0 m/s"``).
        policies:
            List of active ``NLSafetyPolicy`` objects to check against.

        Returns
        -------
        list[str]
            List of violation messages.  Empty list = compliant.
        """
        violations = []

        # Extract speed from action description for speed limit checks
        action_speed_m = self._extract_speed(action_description)

        for policy in policies:
            if not policy.active:
                continue

            if policy.constraint_type == "speed_limit":
                max_speed = policy.parameters.get("max_speed_ms")
                if max_speed is not None and action_speed_m is not None:
                    if action_speed_m > max_speed:
                        violations.append(
                            f"Policy [{policy.policy_id}] violated: "
                            f"speed {action_speed_m} m/s exceeds limit {max_speed} m/s "
                            f"({policy.description})"
                        )

            elif policy.constraint_type == "battery_reserve":
                min_battery = policy.parameters.get("min_battery_pct")
                battery_m = self._extract_battery(action_description)
                if min_battery is not None and battery_m is not None:
                    if battery_m < min_battery:
                        violations.append(
                            f"Policy [{policy.policy_id}] violated: "
                            f"battery {battery_m}% below reserve {min_battery}% "
                            f"({policy.description})"
                        )

            elif policy.constraint_type == "no_go_zone":
                zone = policy.parameters.get("zone_name", "").lower()
                if zone and zone in action_description.lower():
                    violations.append(
                        f"Policy [{policy.policy_id}] violated: "
                        f"action mentions no-go zone {zone!r} "
                        f"({policy.description})"
                    )

        return violations

    @staticmethod
    def _extract_speed(text: str) -> float | None:
        """Extract a speed value in m/s from *text*, or None."""
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*m/s", text, re.IGNORECASE)
        return float(m.group(1)) if m else None

    @staticmethod
    def _extract_battery(text: str) -> float | None:
        """Extract a battery percentage from *text*, or None."""
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%\s*battery", text, re.IGNORECASE)
        return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# NLPolicyStore
# ---------------------------------------------------------------------------

class NLPolicyStore:
    """SQLite-backed store for natural language safety policies.

    Provides a plain-English audit trail that non-engineers can read.
    All policies are persisted across process restarts.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Uses an in-memory database
        when ``":memory:"`` (default; ideal for testing).

    Examples
    --------
    ::

        store = NLPolicyStore("policies.db")
        policy = NLPolicyParser().parse("never exceed 0.5 m/s near humans")
        store.add(policy)

        for p in store.get_active_policies():
            print(p.summary())
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nl_policies (
                policy_id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                constraint_type TEXT NOT NULL,
                parameters TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'hard',
                created_at REAL NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                source TEXT NOT NULL DEFAULT 'regex'
            )
            """
        )
        self._conn.commit()

    def add(self, policy: NLSafetyPolicy) -> None:
        """Persist *policy* to the store.

        Parameters
        ----------
        policy:
            The policy to add.  If a policy with the same ID already
            exists, it is replaced.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO nl_policies
                (policy_id, description, constraint_type, parameters,
                 severity, created_at, active, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                policy.policy_id,
                policy.description,
                policy.constraint_type,
                json.dumps(policy.parameters),
                policy.severity,
                policy.created_at,
                int(policy.active),
                policy.source,
            ),
        )
        self._conn.commit()
        logger.info("NLPolicyStore: added policy %r (%s)", policy.policy_id, policy.description)

    def remove(self, policy_id: str) -> bool:
        """Delete a policy by ID.

        Returns
        -------
        bool
            True if a policy was found and removed.
        """
        cursor = self._conn.execute(
            "DELETE FROM nl_policies WHERE policy_id = ?", (policy_id,)
        )
        self._conn.commit()
        removed = cursor.rowcount > 0
        if removed:
            logger.info("NLPolicyStore: removed policy %r", policy_id)
        return removed

    def deactivate(self, policy_id: str) -> bool:
        """Deactivate a policy without deleting it.

        Returns
        -------
        bool
            True if the policy was found.
        """
        cursor = self._conn.execute(
            "UPDATE nl_policies SET active = 0 WHERE policy_id = ?", (policy_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def get_active_policies(self) -> list[NLSafetyPolicy]:
        """Return all active policies ordered by creation time."""
        cursor = self._conn.execute(
            "SELECT policy_id, description, constraint_type, parameters, "
            "severity, created_at, active, source "
            "FROM nl_policies WHERE active = 1 ORDER BY created_at ASC"
        )
        return [self._row_to_policy(row) for row in cursor.fetchall()]

    def get_all_policies(self) -> list[NLSafetyPolicy]:
        """Return all policies (active and inactive)."""
        cursor = self._conn.execute(
            "SELECT policy_id, description, constraint_type, parameters, "
            "severity, created_at, active, source "
            "FROM nl_policies ORDER BY created_at ASC"
        )
        return [self._row_to_policy(row) for row in cursor.fetchall()]

    def get(self, policy_id: str) -> NLSafetyPolicy | None:
        """Return a single policy by ID, or None if not found."""
        cursor = self._conn.execute(
            "SELECT policy_id, description, constraint_type, parameters, "
            "severity, created_at, active, source "
            "FROM nl_policies WHERE policy_id = ?",
            (policy_id,),
        )
        row = cursor.fetchone()
        return self._row_to_policy(row) if row else None

    def count(self) -> int:
        """Return the total number of policies (active and inactive)."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM nl_policies")
        return cursor.fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    @staticmethod
    def _row_to_policy(row: tuple) -> NLSafetyPolicy:
        policy_id, description, constraint_type, params_json, severity, \
            created_at, active, source = row
        return NLSafetyPolicy(
            policy_id=policy_id,
            description=description,
            constraint_type=constraint_type,
            parameters=json.loads(params_json),
            severity=severity,
            created_at=float(created_at),
            active=bool(active),
            source=source,
        )
