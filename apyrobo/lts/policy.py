"""LTS release policy definitions for apyrobo."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class LTSRelease:
    """Metadata for a Long-Term Support release.

    Attributes:
        version: The release version string (e.g. ``"1.0.0"``).
        release_date: ISO 8601 date of the release (``"YYYY-MM-DD"``).
        eol_date: End-of-life date — no further patches after this date.
        security_only: When ``True``, only security fixes are backported.
    """

    version: str
    release_date: str
    eol_date: str
    security_only: bool = False

    def is_eol(self, as_of: str | None = None) -> bool:
        """Return ``True`` if this release has passed its end-of-life date.

        Args:
            as_of: ISO date to compare against.  Defaults to today.
        """
        check = date.fromisoformat(as_of) if as_of else date.today()
        return check > date.fromisoformat(self.eol_date)


class LTSPolicy:
    """apyrobo LTS release policy.

    The policy follows a cadence of one LTS every major release, supported
    for 24 months from the release date (security fixes only after 12 months).

    Attributes:
        LTS_RELEASES: Ordered list of :class:`LTSRelease` objects.
    """

    LTS_RELEASES: list[LTSRelease] = [
        LTSRelease(
            version="1.0.0",
            release_date="2026-04-01",  # Planned
            eol_date="2028-04-01",
            security_only=False,
        ),
    ]

    def is_lts(self, version: str) -> bool:
        """Return ``True`` if *version* is a designated LTS release."""
        return any(r.version == version for r in self.LTS_RELEASES)

    def is_eol(self, version: str, as_of: str | None = None) -> bool:
        """Return ``True`` if the LTS release for *version* is past its EOL.

        Args:
            version: Release version to check.
            as_of: ISO date string; defaults to today.
        """
        for r in self.LTS_RELEASES:
            if r.version == version:
                return r.is_eol(as_of)
        return False

    def supported_versions(self, as_of: str | None = None) -> list[LTSRelease]:
        """Return all LTS releases that are not yet past EOL.

        Args:
            as_of: ISO date string; defaults to today.
        """
        return [r for r in self.LTS_RELEASES if not r.is_eol(as_of)]

    def next_lts(self) -> str | None:
        """Return the next planned LTS version string, or ``None`` if unknown."""
        today = date.today()
        for r in self.LTS_RELEASES:
            if date.fromisoformat(r.release_date) > today:
                return r.version
        return None

    def latest_lts(self) -> LTSRelease | None:
        """Return the most recently released LTS, or ``None``."""
        today = date.today()
        released = [
            r for r in self.LTS_RELEASES
            if date.fromisoformat(r.release_date) <= today
        ]
        return released[-1] if released else None
