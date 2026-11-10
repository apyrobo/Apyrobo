"""Natural language safety policies for APYROBO.

Allows operators to express safety constraints in plain English, store
them, and check a written action against them before it runs
(``check_compliance`` / ``apyrobo policy check``). The policies are not
applied to motion by ``SafetyEnforcer``, ``apyrobo execute`` or
``apyrobo mcp``; use those components' own limits for that.

Quick start::

    from apyrobo.safety.nl_policy import NLPolicyStore, NLPolicyParser

    store = NLPolicyStore()
    parser = NLPolicyParser()

    # Translate a natural language rule into a structured SafetyPolicy
    policy = parser.parse("never exceed 0.5 m/s near humans")
    store.add(policy)

    # Check an action before running it (a pre-flight check on text):
    parser.check_compliance("navigate at 2.0 m/s", store.get_active_policies())

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
# Speeds in an action description. Only m/s is read; any other way of
# writing a speed is reported as unreadable rather than skipped.
_NUM = r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)"
_MPS = r"(?:m/s|m\s*s(?:\^?-1|⁻¹)|mps|m(?:eter|etre)s?\s*(?:/|per)\s*s(?:ec(?:ond)?)?)\b"
_ACTION_MPS_RE = re.compile(rf"(?<![0-9.])({_NUM})\s*{_MPS}", re.IGNORECASE)
_ACTION_OTHER_SPEED_RE = re.compile(
    rf"(?<![0-9.]){_NUM}\s*(?:km/h|kmh|kph|km\s*per\s*h(?:our)?|kilomet(?:er|re)s?\s*(?:/|per)\s*h(?:our)?"
    r"|mph|miles?\s*(?:/|per)\s*h(?:our)?|knots?|kn\b|ft/s|fps|feet\s*(?:/|per)\s*s(?:ec(?:ond)?)?"
    r"|in/s|cm/s|mm/s|cent(?:i)?met(?:er|re)s?\s*(?:/|per)\s*s(?:ec(?:ond)?)?|m/min|m/h\b"
    r"|m(?:eter|etre)s?\s*(?:/|per)\s*(?:min(?:ute)?|h(?:our)?))",
    re.IGNORECASE,
)
_ACTION_SPEED_WORD_RE = re.compile(
    rf"\b(?:speed|velocity|pace)\b(?:\s+(?:of|is|at|to|=|:|limit|up\s+to|about|around))*\s*[:=]?\s*({_NUM})(?!\s*[0-9.])",
    re.IGNORECASE,
)
_ACTION_GLUED_MPS_RE = re.compile(
    rf"([0-9][0-9.,eE+_]*|[.,][0-9][0-9.,eE+_]*)\s*{_MPS}", re.IGNORECASE
)
_ACTION_PER_SECOND_RE = re.compile(
    rf"(?<![0-9.]){_NUM}\s*\w*\s*(?:/|per)\s*s(?:ec(?:ond)?)?\b", re.IGNORECASE
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
    def from_dict(cls, d: dict[str, Any]) -> NLSafetyPolicy:
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
        Optional LLM fallback: any object with a ``complete(prompt) -> str``
        method, such as a small wrapper around ``litellm.completion``.
        ``apyrobo.Agent`` has no ``complete`` method and is refused with
        ``TypeError``.  When ``None`` (default), unrecognised patterns
        produce a ``"custom"`` policy with the description stored verbatim.
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
        if agent is not None and not callable(getattr(agent, "complete", None)):
            raise TypeError(
                "NLPolicyParser agent must have a complete(prompt) -> str method; "
                f"{type(agent).__name__} has none, so the LLM fallback would never run. "
                "Wrap your LLM call in a small class with complete(), "
                "see docs/concepts/nl_safety_policies.md"
            )
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

    def check_compliance(
        self, action_description: str, policies: list[NLSafetyPolicy]
    ) -> list[str]:
        """Check whether *action_description* would violate any active policy.

        Performs a textual pre-flight check against each policy; it does
        not stop a robot by itself. Speeds are read in m/s only, and every
        m/s value in the text is checked. A speed written some other way
        (``2 km/h``, ``speed 2``) counts as a violation of an active speed
        limit, since the check cannot tell whether it is over the limit.
        ``proximity_limit``, ``time_window``, ``load_limit`` and ``custom``
        policies are stored but not checked here.

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

        # Extract speeds from action description for speed limit checks
        action_speeds = self._extract_speeds(action_description)
        unread_speed = self._unread_speed(action_description)

        for policy in policies:
            if not policy.active:
                continue

            if policy.constraint_type == "speed_limit":
                max_speed = policy.parameters.get("max_speed_ms")
                if max_speed is None:
                    continue
                if unread_speed is not None:
                    violations.append(
                        f"Policy [{policy.policy_id}] cannot be checked: "
                        f"the speed {unread_speed!r} cannot be read as a number in m/s, "
                        f"so it may exceed limit {max_speed} m/s; write it like '0.3 m/s' "
                        f"({policy.description})"
                    )
                    continue
                if action_speeds:
                    action_speed_m = max(action_speeds)
                    if action_speed_m > max_speed:
                        violations.append(
                            f"Policy [{policy.policy_id}] violated: "
                            f"speed {action_speed_m} m/s exceeds limit {max_speed} m/s "
                            f"({policy.description})"
                        )

            elif policy.constraint_type == "battery_reserve":
                min_battery = policy.parameters.get("min_battery_pct")
                battery_m = self._extract_battery(action_description)
                if (
                    min_battery is not None
                    and battery_m is not None
                    and battery_m < min_battery
                ):
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
        """Extract the highest speed value in m/s from *text*, or None."""
        speeds = NLPolicyParser._extract_speeds(text)
        return max(speeds) if speeds else None

    @staticmethod
    def _extract_speeds(text: str) -> list[float]:
        """Every speed written in m/s in *text*."""
        return [float(m.group(1)) for m in _ACTION_MPS_RE.finditer(text)]

    @staticmethod
    def _unread_speed(text: str) -> str | None:
        """The first speed in *text* not written in m/s, or None."""
        read = [m.span() for m in _ACTION_MPS_RE.finditer(text)]

        def is_read(start: int, end: int) -> bool:
            return any(rs <= start < re_ or start <= rs < end for rs, re_ in read)

        m = _ACTION_OTHER_SPEED_RE.search(text)
        if m:
            return m.group(0)
        for m in _ACTION_GLUED_MPS_RE.finditer(text):
            if not re.fullmatch(_NUM, m.group(1)):
                return m.group(0)
        for m in _ACTION_PER_SECOND_RE.finditer(text):
            if not is_read(*m.span()):
                return m.group(0)
        for m in _ACTION_SPEED_WORD_RE.finditer(text):
            start, end = m.span(1)
            after = text[end:]
            if not re.match(rf"\s*{_MPS}", after, re.IGNORECASE):
                return m.group(0).strip()
        return None

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
        db = str(db_path)
        if db != ":memory:" and not db.startswith("file:"):
            path = Path(db).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            db = str(path)
        self._db_path = db
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
