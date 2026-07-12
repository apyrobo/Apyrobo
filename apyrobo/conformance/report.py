"""Conformance report model — check results and their machine-readable form.

A report is a list of :class:`CheckResult` plus target metadata. The JSON
form (``to_dict``/``to_json``) is the stable interface consumed by CI
pipelines and the badge program; its layout is versioned by
``REPORT_FORMAT_VERSION``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

#: The spec revision this suite tests against (spec/README.md).
SPEC_VERSION = "1.0"

#: Version of the JSON report layout itself.
REPORT_FORMAT_VERSION = "1"

#: Check statuses, in severity order.
STATUSES = ("pass", "warn", "fail", "skip")


@dataclass
class CheckResult:
    """Outcome of a single conformance check.

    ``level`` is the RFC-2119 requirement level of the spec clause being
    tested: a failed MUST check makes the target non-conformant; a failed
    SHOULD check is recorded as ``warn``.
    """

    check_id: str
    title: str
    level: str  # "MUST" | "SHOULD"
    spec_ref: str  # e.g. "adapter-contract.md §2"
    status: str  # "pass" | "warn" | "fail" | "skip"
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.check_id,
            "title": self.title,
            "level": self.level,
            "spec_ref": self.spec_ref,
            "status": self.status,
            "details": self.details,
        }


@dataclass
class ConformanceReport:
    """Full result of one conformance run against one target."""

    target: str
    kind: str  # "adapter" | "wire-protocol"
    checks: list[CheckResult] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def add(
        self,
        check_id: str,
        title: str,
        level: str,
        spec_ref: str,
        status: str,
        details: str = "",
    ) -> CheckResult:
        result = CheckResult(check_id, title, level, spec_ref, status, details)
        self.checks.append(result)
        return result

    @property
    def summary(self) -> dict[str, int]:
        counts = {status: 0 for status in STATUSES}
        for check in self.checks:
            counts[check.status] += 1
        return counts

    @property
    def conformant(self) -> bool:
        """True when no MUST-level check failed (SHOULD warnings allowed)."""
        return not any(c.status == "fail" for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "apyrobo_conformance_report": REPORT_FORMAT_VERSION,
            "spec_version": SPEC_VERSION,
            "apyrobo_version": _apyrobo_version(),
            "target": self.target,
            "kind": self.kind,
            "timestamp": self.timestamp,
            "checks": [c.to_dict() for c in self.checks],
            "summary": self.summary,
            "conformant": self.conformant,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def render_text(self) -> str:
        """Human-readable summary for terminal output."""
        icons = {"pass": "✓", "warn": "!", "fail": "✗", "skip": "-"}
        lines = [
            f"APYROBO conformance — spec {SPEC_VERSION} — {self.kind}",
            f"Target: {self.target}",
            "",
        ]
        for c in self.checks:
            line = f"  {icons[c.status]} [{c.check_id}] {c.title}"
            if c.status != "pass" and c.details:
                line += f"\n      {c.details}"
            lines.append(line)
        s = self.summary
        lines.append("")
        lines.append(
            f"{s['pass']} passed, {s['fail']} failed, "
            f"{s['warn']} warnings, {s['skip']} skipped"
        )
        verdict = "CONFORMANT" if self.conformant else "NOT CONFORMANT"
        lines.append(f"Result: {verdict} (spec {SPEC_VERSION})")
        return "\n".join(lines)


def _apyrobo_version() -> str:
    try:
        from apyrobo.__version__ import __version__
        return __version__
    except Exception:
        return "unknown"
