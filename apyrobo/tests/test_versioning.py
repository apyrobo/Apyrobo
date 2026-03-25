"""Tests for versioning: changelog, migration, and compatibility."""

import textwrap
import pytest

from apyrobo.versioning.changelog import ChangelogEntry, ChangelogParser
from apyrobo.versioning.migration import MigrationStep, MigrationGuide
from apyrobo.versioning.compatibility import APICompatibilityChecker


SAMPLE_CHANGELOG = textwrap.dedent("""\
    # Changelog

    ## [1.0.0] - 2026-04-01

    ### Breaking Changes
    - Removed `MemoryStore.get_all()` — use `MemoryStore.query()` instead
    - Renamed `run_skill` to `execute_skill`

    ### Added
    - Plugin system
    - Skill registry

    ### Deprecated
    - `apyrobo.legacy.OldAdapter` will be removed in 2.0

    ## [0.9.0] - 2025-12-01

    ### Fixed
    - Bug in skill executor retry logic

    ## [0.8.0] - 2025-09-01

    ### Added
    - Initial voice adapter
""")


class TestChangelogParser:
    def setup_method(self):
        self.parser = ChangelogParser()
        self.entries = self.parser.parse_text(SAMPLE_CHANGELOG)

    def test_parses_three_versions(self):
        assert len(self.entries) == 3

    def test_versions_in_order(self):
        assert self.entries[0].version == "1.0.0"
        assert self.entries[1].version == "0.9.0"
        assert self.entries[2].version == "0.8.0"

    def test_date_parsed(self):
        assert self.entries[0].date == "2026-04-01"

    def test_breaking_changes(self):
        entry = self.entries[0]
        assert len(entry.breaking_changes) == 2
        assert any("MemoryStore" in b for b in entry.breaking_changes)

    def test_new_features(self):
        entry = self.entries[0]
        assert any("Plugin" in f for f in entry.new_features)

    def test_deprecations(self):
        entry = self.entries[0]
        assert len(entry.deprecations) == 1
        assert "OldAdapter" in entry.deprecations[0]

    def test_fixes(self):
        entry = self.entries[1]
        assert len(entry.fixes) == 1

    def test_has_breaking_changes(self):
        assert self.entries[0].has_breaking_changes()
        assert not self.entries[1].has_breaking_changes()

    def test_get_breaking_changes_range(self):
        changes = self.parser.get_breaking_changes("0.9.0", "1.0.0")
        assert len(changes) == 2

    def test_get_breaking_changes_no_range(self):
        changes = self.parser.get_breaking_changes("0.8.0", "0.9.0")
        assert changes == []


class TestMigrationGuide:
    def setup_method(self):
        parser = ChangelogParser()
        self.entries = parser.parse_text(SAMPLE_CHANGELOG)
        self.guide = MigrationGuide()

    def test_generate_steps(self):
        steps = self.guide.generate("0.9.0", "1.0.0", self.entries)
        assert len(steps) >= 2  # 2 breaking + 1 deprecation

    def test_step_has_description(self):
        steps = self.guide.generate("0.9.0", "1.0.0", self.entries)
        assert all(s.description for s in steps)

    def test_to_markdown_has_content(self):
        steps = self.guide.generate("0.9.0", "1.0.0", self.entries)
        md = self.guide.to_markdown(steps)
        assert "Step 1" in md
        assert "Migration Guide" in md

    def test_to_markdown_empty(self):
        steps = self.guide.generate("0.8.0", "0.9.0", self.entries)
        md = self.guide.to_markdown(steps)
        assert "smooth" in md.lower()

    def test_automatable_rename(self):
        step = MigrationStep(description="rename run_skill to execute_skill", automated=False)
        assert self.guide._is_automatable(step.description)

    def test_not_automatable(self):
        step = MigrationStep(description="Rewrote core scheduler", automated=False)
        assert not self.guide._is_automatable(step.description)


class TestAPICompatibilityChecker:
    def setup_method(self):
        self.checker = APICompatibilityChecker()

    def test_no_deprecated_usage(self, tmp_path):
        f = tmp_path / "clean.py"
        f.write_text("x = 1\n")
        usages = self.checker.check(str(tmp_path), ["MemoryStore"])
        assert usages == []

    def test_detects_usage(self, tmp_path):
        f = tmp_path / "old.py"
        f.write_text("from apyrobo import MemoryStore\nstore = MemoryStore()\n")
        usages = self.checker.check(str(tmp_path), ["MemoryStore"])
        assert len(usages) >= 1
        assert "MemoryStore" in usages[0]

    def test_report_clean(self):
        report = self.checker.report([])
        assert "No deprecated" in report

    def test_report_with_usages(self):
        usages = ["src/foo.py:12 — usage of deprecated 'MemoryStore'"]
        report = self.checker.report(usages)
        assert "1 deprecated" in report
        assert "MemoryStore" in report

    def test_scan_missing_directory(self):
        # Should not raise, just return empty
        usages = self.checker.check("/nonexistent/path", ["foo"])
        assert usages == []
