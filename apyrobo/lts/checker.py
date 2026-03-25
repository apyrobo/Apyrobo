"""Version update and security advisory checker."""

from __future__ import annotations

from typing import Optional

from .policy import LTSPolicy, LTSRelease


class VersionChecker:
    """Check the current apyrobo installation against the LTS policy.

    This class is intentionally offline-first: it reads from the local
    :class:`~apyrobo.lts.policy.LTSPolicy` by default, and can optionally
    reach out to a registry URL to fetch up-to-date release metadata.

    Example::

        checker = VersionChecker()
        info = checker.check_for_updates("0.9.0")
        print(info)
        # {'current': '0.9.0', 'latest_lts': '1.0.0', 'is_lts': False, 'is_eol': False}
    """

    def __init__(self, policy: Optional[LTSPolicy] = None) -> None:
        self._policy = policy or LTSPolicy()

    def check_for_updates(
        self,
        current: str,
        registry_url: Optional[str] = None,
    ) -> dict:
        """Return version status information for *current*.

        Args:
            current: The installed version string.
            registry_url: Optional URL to fetch live release data from.
                          Currently informational; live fetch not implemented.

        Returns:
            Dictionary with keys:

            - ``current`` — the queried version
            - ``is_lts`` — whether the current version is an LTS release
            - ``is_eol`` — whether the current LTS release is past EOL
            - ``latest_lts`` — version string of the most recent LTS, or ``None``
            - ``next_lts`` — planned next LTS version, or ``None``
            - ``supported`` — whether the current release receives updates
        """
        latest: Optional[LTSRelease] = self._policy.latest_lts()
        return {
            "current": current,
            "is_lts": self._policy.is_lts(current),
            "is_eol": self._policy.is_eol(current),
            "latest_lts": latest.version if latest else None,
            "next_lts": self._policy.next_lts(),
            "supported": not self._policy.is_eol(current),
        }

    def security_advisory(self, version: str) -> list[str]:
        """Return any security advisories applicable to *version*.

        In a production deployment this would query a CVE database or the
        apyrobo security advisory feed.  Currently returns an empty list
        for versions that are supported, or a generic warning for EOL versions.

        Args:
            version: The version to check.

        Returns:
            List of advisory strings (empty if version is clean).
        """
        advisories: list[str] = []
        if self._policy.is_eol(version):
            advisories.append(
                f"apyrobo {version} has reached end-of-life and no longer "
                "receives security patches.  Please upgrade to a supported release."
            )
        return advisories
