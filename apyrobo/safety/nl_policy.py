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
import math
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

# The constraint types check_compliance reads; the others are stored and
# listed but never produce a violation.
CHECKED_CONSTRAINT_TYPES = frozenset({"speed_limit", "battery_reserve", "no_go_zone"})

# Regex patterns for common constraint types (unit: SI base units)
_PROXIMITY_RE = re.compile(
    r"(?:stay|keep|maintain)\s+(?:at\s+least\s+)?([0-9]+(?:\.[0-9]+)?)\s*m(?:eter|etre)?s?\s+"
    r"(?:away\s+from|from)\s+([a-z\s]+)",
    re.IGNORECASE,
)
# A battery reserve: a lower bound on the battery level, written as a
# percentage. Each pattern's one group is the percentage.
_PCT = r"([0-9]+(?:\.[0-9]+)?)\s*%"
_NEG = r"(?:never|do\s+not|don't|must\s+not|mustn't|should\s+not|shall\s+not)"
_FALL = r"(?:go|drop|fall|run|dip|get)\s+(?:below|under)"
_BATTERY_WORD = r"(?:the\s+)?battery(?:\s+level)?"
_BATTERY_RES = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # "always keep 20% battery", "keep at least 20% of the battery",
        # "maintain a 20% battery reserve", "reserve 20% battery"
        rf"\b(?:always\s+)?(?:keep|maintain|reserve|hold|retain)\s+"
        rf"(?:at\s+least\s+|a\s+minimum\s+of\s+|a\s+)?{_PCT}\s*(?:of\s+)?{_BATTERY_WORD}",
        # "never go below 20% battery", "do not drop under 20% of the battery"
        rf"\b{_NEG}\s+{_FALL}\s+{_PCT}\s*(?:of\s+)?{_BATTERY_WORD}",
        # "never let the battery fall below 20%"
        rf"\b{_NEG}\s+let\s+{_BATTERY_WORD}\s+{_FALL}\s+{_PCT}",
        # "battery must stay above 20%", "keep the battery above 20%"
        rf"\b(?:(?:keep|maintain)\s+)?{_BATTERY_WORD}\s+"
        r"(?:(?:must|should|shall)\s+)?(?:always\s+)?(?:(?:stay|remain|be)\s+)?"
        rf"(?:above|over|at\s+least|at\s+or\s+above)\s+{_PCT}",
        # "minimum battery 20%", "min battery level: 20%"
        rf"\b(?:minimum|min)\s+{_BATTERY_WORD}(?:\s+(?:of|is))?\s*[:=]?\s*{_PCT}",
    )
)
# A battery level in an action: "15% battery", "battery 15%", "battery at 15%".
_ACTION_BATTERY_RES = (
    re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%\s*battery", re.IGNORECASE),
    re.compile(
        r"\bbattery(?:\s+level)?(?:\s+(?:is|at|of))?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        re.IGNORECASE,
    ),
)
# Speeds in an action description. A speed in one of the units in
# ``_RULE_SPEED_UNITS`` is read (converted to m/s); any other way of writing a
# speed is reported as unreadable rather than skipped.
_NUM = r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)"
_MPS = r"(?:m/s|m\s*s(?:\^?-1|⁻¹)|mps|m(?:eter|etre)s?\s*(?:/|per)\s*s(?:ec(?:ond)?)?)\b"
# A rule's speed limit: an upper-bound phrase, then a speed in one of the
# units in ``_RULE_SPEED_UNITS`` (stored in m/s).
# Lower bounds ("never go below 0.2 m/s", "at least 0.2 m/s") are not matched,
# so they stay ``custom`` instead of becoming a maximum.
_SPEED_WORD = r"(?:speed|velocity)"
_NEVER = r"(?:never|do\s+not|don't|must\s+not|mustn't|should\s+not|shall\s+not)"
_SPEED_LEAD = (
    rf"(?:{_NEVER}\s+exceed"
    rf"|max(?:imum)?(?:\s+{_SPEED_WORD})?|{_SPEED_WORD}\s+limit|top\s+{_SPEED_WORD}"
    r"|(?:no|not|never)\s+(?:go\s+|drive\s+|move\s+|travel\s+)?faster\s+than"
    rf"|{_NEVER}\s+(?:go|drive|move|travel)\s+(?:above|over)"
    rf"|(?:stay|keep|remain|hold)\s+(?:(?:the\s+)?{_SPEED_WORD}\s+)?(?:below|under|at\s+or\s+below)"
    rf"|(?:limit|cap|restrict)\s+(?:the\s+)?(?:(?:robot(?:'s|s)?|its)\s+)?(?:{_SPEED_WORD}\s+)?(?:to|at)"
    r"|slow\s+down\s+to"
    rf"|{_SPEED_WORD}\s*(?:<=|<|≤)"
    r"|at\s+most|no\s+more\s+than)"
)
# A rule's speed may be written in any of these units; it is stored in m/s.
# (unit pattern, metres per second per unit)
_RULE_SPEED_UNITS = (
    (_MPS, 1.0),
    (
        r"(?:km\s*/\s*h(?:our|r)?|kmh|kph|km\s*per\s*h(?:our)?"
        r"|kilomet(?:er|re)s?\s*(?:/|per|an)\s*h(?:our)?)",
        1000 / 3600,
    ),
    (r"(?:mph|miles?\s*(?:/|per|an)\s*h(?:our)?)", 0.44704),
    (r"(?:cm/s|cent(?:i)?met(?:er|re)s?\s*(?:/|per)\s*s(?:ec(?:ond)?)?)", 0.01),
    (r"(?:mm/s|millimet(?:er|re)s?\s*(?:/|per)\s*s(?:ec(?:ond)?)?)", 0.001),
    (r"(?:ft/s|fps|f(?:ee|oo)t\s*(?:/|per)\s*s(?:ec(?:ond)?)?)", 0.3048),
    (r"(?:m/min|m(?:eter|etre)s?\s*(?:/|per)\s*min(?:ute)?)", 1 / 60),
)
_SPEED_RE = re.compile(
    rf"\b{_SPEED_LEAD}(?:\s+(?:of|is|at))?\s*[:=]?\s*({_NUM})\s*"
    rf"({'|'.join(u for u, _ in _RULE_SPEED_UNITS)})(?![a-z/])",
    re.IGNORECASE,
)


def _rule_speed_ms(number: str, unit: str) -> float:
    """Return a rule's speed in m/s from its number and the unit it was written in."""
    for pattern, factor in _RULE_SPEED_UNITS:
        if re.fullmatch(pattern, unit, re.IGNORECASE):
            value = float(number) * factor
            if factor == 1.0:
                return value
            # Six decimals, rounded down so the stored limit is never looser.
            return math.floor(value * 1e6 + 1e-6) / 1e6
    raise ValueError(f"unknown speed unit {unit!r}")


def _action_speed_ms(number: str, unit: str) -> float:
    """Return an action's speed in m/s; rounded up so it is never understated."""
    for pattern, factor in _RULE_SPEED_UNITS:
        if re.fullmatch(pattern, unit, re.IGNORECASE):
            value = float(number) * factor
            if factor == 1.0:
                return value
            return math.ceil(value * 1e6 - 1e-6) / 1e6
    raise ValueError(f"unknown speed unit {unit!r}")


_ACTION_SPEED_RE = re.compile(
    rf"(?<![0-9.])({_NUM})\s*({'|'.join(u for u, _ in _RULE_SPEED_UNITS)})(?![a-z/])",
    re.IGNORECASE,
)
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

# A no-go zone: "never enter", "do not enter", "don't go into", "keep out of",
# "stay out of" or "avoid", then the zone's name.
_NOGO_RE = re.compile(
    r"(?:(?:never|do\s+not|don't|must\s+not|mustn't|should\s+not|shall\s+not)"
    r"\s+(?:enter|go\s+into)"
    r"|(?:keep|stay)\s+out\s+of"
    r"|avoid)\s+(?:the\s+)?(.+)",
    re.IGNORECASE,
)
# What may follow a zone's name in a no-go rule: a condition ("during lunch",
# "without an escort"). The zone is refused at all times; the condition is
# kept as ``context`` and not checked.
_NOGO_CONDITION_RE = re.compile(
    r"\s*,?\s+(?:during|when|whenever|while|unless|if|without|except|before|after"
    r"|between|until|outside\s+of|at\s+night|at\s+any\s+time|at\s+all\s+times"
    r"|on\s+weekends?|on\s+weekdays?|for\s+any\s+reason)\b.*$",
    re.IGNORECASE,
)
# Several zones in one rule: "the kitchen or the lab", "the kitchen, the lab
# and the dock".
_NOGO_SPLIT_RE = re.compile(r"\s*,\s*(?:(?:and|or|nor)\s+)?|\s+(?:and|or|nor)\s+", re.IGNORECASE)
_LEADING_THE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


def _nogo_zones(text: str) -> tuple[list[str], str]:
    """Split the text after a no-go lead into zone names and a condition."""
    text = text.strip()
    context = ""
    m = _NOGO_CONDITION_RE.search(text)
    if m:
        context = text[m.start():].strip(" ,").rstrip(".!?;:")
        text = text[: m.start()]
    zones = []
    for part in _NOGO_SPLIT_RE.split(text):
        name = _LEADING_THE_RE.sub("", part.strip().rstrip(".!?;:,").strip())
        if name and name.lower() not in (z.lower() for z in zones):
            zones.append(name)
    return zones, context


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
            context = description[m.end():].strip()
            params = {
                "max_speed_ms": _rule_speed_ms(m.group(1), m.group(2)),
                "context": context or "always",
            }
            if not re.fullmatch(_MPS, m.group(2), re.IGNORECASE):
                params["as_written"] = f"{m.group(1)} {m.group(2)}"
            return NLSafetyPolicy(
                description=description,
                constraint_type="speed_limit",
                parameters=params,
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
        for battery_re in _BATTERY_RES:
            m = battery_re.search(description)
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
            zones, context = _nogo_zones(m.group(1))
            if not zones:
                return None
            params: dict[str, Any] = {"zone_name": zones[0]}
            if len(zones) > 1:
                params["zone_names"] = zones
            if context:
                params["context"] = context
            return NLSafetyPolicy(
                description=description,
                constraint_type="no_go_zone",
                parameters=params,
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
        not stop a robot by itself. Every speed in the text written in m/s,
        km/h, mph, cm/s, mm/s, ft/s or m/min is converted to m/s (rounded up)
        and checked. A speed written some other way (``2 knots``,
        ``speed 2``) counts as a violation of an active speed limit, since
        the check cannot tell whether it is over the limit.
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
        action_speeds = self._read_speeds(action_description)
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
                        f"the speed {unread_speed!r} cannot be read as a speed, "
                        f"so it may exceed limit {max_speed} m/s; write it like '0.3 m/s' "
                        f"or '1 km/h' "
                        f"({policy.description})"
                    )
                    continue
                if action_speeds:
                    action_speed_m, written = max(action_speeds, key=lambda sw: sw[0])
                    if action_speed_m > max_speed:
                        as_written = "" if written is None else f" ({written})"
                        violations.append(
                            f"Policy [{policy.policy_id}] violated: "
                            f"speed {action_speed_m} m/s{as_written} exceeds limit "
                            f"{max_speed} m/s ({policy.description})"
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
                zones = policy.parameters.get("zone_names") or [
                    policy.parameters.get("zone_name", "")
                ]
                action_lower = action_description.lower()
                for zone in (str(z).lower() for z in zones):
                    if zone and zone in action_lower:
                        violations.append(
                            f"Policy [{policy.policy_id}] violated: "
                            f"action mentions no-go zone {zone!r} "
                            f"({policy.description})"
                        )
                        break

        return violations

    @staticmethod
    def _extract_speed(text: str) -> float | None:
        """Extract the highest speed value in m/s from *text*, or None."""
        speeds = NLPolicyParser._extract_speeds(text)
        return max(speeds) if speeds else None

    @staticmethod
    def _extract_speeds(text: str) -> list[float]:
        """Every speed in *text* that can be read, in m/s."""
        return [speed for speed, _ in NLPolicyParser._read_speeds(text)]

    @staticmethod
    def _read_speeds(text: str) -> list[tuple[float, str | None]]:
        """Every readable speed in *text*: (m/s, as written or None if m/s)."""
        speeds: list[tuple[float, str | None]] = []
        for m in _ACTION_SPEED_RE.finditer(text):
            unit = m.group(2)
            if re.fullmatch(_MPS, unit, re.IGNORECASE):
                speeds.append((float(m.group(1)), None))
            else:
                speeds.append((_action_speed_ms(m.group(1), unit), m.group(0)))
        return speeds

    @staticmethod
    def _unread_speed(text: str) -> str | None:
        """The first speed in *text* that cannot be read, or None."""
        read = [m.span() for m in _ACTION_SPEED_RE.finditer(text)]

        def is_read(start: int, end: int) -> bool:
            return any(rs <= start < re_ or start <= rs < end for rs, re_ in read)

        for m in _ACTION_OTHER_SPEED_RE.finditer(text):
            if not is_read(*m.span()):
                return m.group(0)
        for m in _ACTION_GLUED_MPS_RE.finditer(text):
            if not re.fullmatch(_NUM, m.group(1)):
                return m.group(0)
        for m in _ACTION_PER_SECOND_RE.finditer(text):
            if not is_read(*m.span()):
                return m.group(0)
        for m in _ACTION_SPEED_WORD_RE.finditer(text):
            start, end = m.span(1)
            if not is_read(start, end):
                return m.group(0).strip()
        return None

    @staticmethod
    def _extract_battery(text: str) -> float | None:
        """The lowest battery percentage written in *text*, or None."""
        levels = [
            float(m.group(1))
            for battery_re in _ACTION_BATTERY_RES
            for m in battery_re.finditer(text)
        ]
        return min(levels) if levels else None


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
