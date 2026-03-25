"""CHANGELOG parser for apyrobo versioning support."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ChangelogEntry:
    """A single version entry from CHANGELOG.md."""

    version: str
    date: str
    breaking_changes: list[str] = field(default_factory=list)
    new_features: list[str] = field(default_factory=list)
    deprecations: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)

    def has_breaking_changes(self) -> bool:
        return bool(self.breaking_changes)


class ChangelogParser:
    """Parse Keep-a-Changelog formatted CHANGELOG.md files.

    Example::

        parser = ChangelogParser()
        entries = parser.parse_file("CHANGELOG.md")
        breaking = parser.get_breaking_changes("0.9.0", "1.0.0")
    """

    # Matches: ## [1.2.3] - 2024-01-15  or  ## [Unreleased]
    VERSION_HEADER = re.compile(
        r"^##\s+\[(?P<version>[^\]]+)\](?:\s+-\s+(?P<date>\d{4}-\d{2}-\d{2}))?",
        re.MULTILINE,
    )
    SECTION_HEADER = re.compile(r"^###\s+(.+)$", re.MULTILINE)
    LIST_ITEM = re.compile(r"^[-*]\s+(.+)$", re.MULTILINE)

    def parse_file(self, path: str) -> list[ChangelogEntry]:
        """Parse a CHANGELOG.md file.

        Args:
            path: Path to the changelog file.

        Returns:
            List of :class:`ChangelogEntry` objects, newest first.
        """
        content = Path(path).read_text(encoding="utf-8")
        return self._parse(content)

    def parse_text(self, text: str) -> list[ChangelogEntry]:
        """Parse changelog from a string."""
        return self._parse(text)

    def get_breaking_changes(
        self, from_version: str, to_version: str
    ) -> list[str]:
        """Collect all breaking changes between two versions.

        Args:
            from_version: Starting version (exclusive).
            to_version: Ending version (inclusive).

        Returns:
            Flattened list of breaking change descriptions.
        """
        entries = getattr(self, "_entries", [])
        breaking: list[str] = []
        in_range = False
        for entry in reversed(entries):  # oldest first
            if entry.version == from_version:
                in_range = True
                continue
            if in_range:
                breaking.extend(entry.breaking_changes)
            if entry.version == to_version:
                break
        return breaking

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse(self, content: str) -> list[ChangelogEntry]:
        entries: list[ChangelogEntry] = []

        # Split content at each version header
        splits = list(self.VERSION_HEADER.finditer(content))
        for i, match in enumerate(splits):
            version = match.group("version")
            date = match.group("date") or ""
            # Extract the block for this version
            start = match.end()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(content)
            block = content[start:end]

            entry = ChangelogEntry(version=version, date=date)
            self._populate_sections(block, entry)
            entries.append(entry)

        self._entries = entries
        return entries

    def _populate_sections(self, block: str, entry: ChangelogEntry) -> None:
        """Fill entry fields from section headers in *block*."""
        section_map = {
            "breaking changes": "breaking_changes",
            "breaking": "breaking_changes",
            "added": "new_features",
            "new features": "new_features",
            "deprecated": "deprecations",
            "fixed": "fixes",
            "bug fixes": "fixes",
            "changed": "new_features",  # treat as feature for simplicity
        }

        current_section: Optional[str] = None
        for line in block.splitlines():
            sec_match = self.SECTION_HEADER.match(line)
            if sec_match:
                label = sec_match.group(1).lower().strip()
                current_section = section_map.get(label)
                continue
            item_match = self.LIST_ITEM.match(line)
            if item_match and current_section:
                target = getattr(entry, current_section)
                target.append(item_match.group(1).strip())
