"""Migration guide generator for apyrobo version upgrades."""

from __future__ import annotations

from dataclasses import dataclass

from .changelog import ChangelogEntry


@dataclass
class MigrationStep:
    """A single step in a migration guide."""

    description: str
    from_code: str = ""   # Example of old code
    to_code: str = ""     # Example of new code
    automated: bool = False  # Can be auto-fixed by a codemod


class MigrationGuide:
    """Generate migration guides from changelog entries.

    Example::

        guide = MigrationGuide()
        entries = parser.parse_file("CHANGELOG.md")
        steps = guide.generate("0.9.0", "1.0.0", entries)
        print(guide.to_markdown(steps))
    """

    def generate(
        self,
        from_version: str,
        to_version: str,
        changelog: list[ChangelogEntry],
    ) -> list[MigrationStep]:
        """Generate migration steps between two versions.

        Args:
            from_version: The version the user is migrating *from*.
            to_version: The version the user is migrating *to*.
            changelog: Parsed changelog entries.

        Returns:
            List of :class:`MigrationStep` in migration order.
        """
        steps: list[MigrationStep] = []

        # Collect entries in [from_version, to_version] range
        collecting = False
        for entry in reversed(changelog):  # oldest first
            if entry.version == from_version:
                collecting = True
                continue
            if not collecting:
                continue
            for change in entry.breaking_changes:
                steps.append(
                    MigrationStep(
                        description=f"[{entry.version}] {change}",
                        automated=self._is_automatable(change),
                    )
                )
            for dep in entry.deprecations:
                steps.append(
                    MigrationStep(
                        description=f"[{entry.version}] Deprecation — {dep}",
                        automated=False,
                    )
                )
            if entry.version == to_version:
                break

        return steps

    def to_markdown(self, steps: list[MigrationStep]) -> str:
        """Render migration steps as a Markdown document.

        Args:
            steps: Steps produced by :meth:`generate`.

        Returns:
            Markdown string.
        """
        if not steps:
            return "No breaking changes — upgrade should be smooth.\n"

        lines = ["# Migration Guide\n"]
        for i, step in enumerate(steps, 1):
            lines.append(f"## Step {i}: {step.description}\n")
            if step.automated:
                lines.append("*This change can be applied automatically.*\n")
            if step.from_code:
                lines.append("**Before:**\n```python\n" + step.from_code + "\n```\n")
            if step.to_code:
                lines.append("**After:**\n```python\n" + step.to_code + "\n```\n")

        return "\n".join(lines)

    # ------------------------------------------------------------------

    _AUTOMATABLE_KEYWORDS = ("rename", "moved to", "import path")

    def _is_automatable(self, description: str) -> bool:
        lower = description.lower()
        return any(kw in lower for kw in self._AUTOMATABLE_KEYWORDS)
