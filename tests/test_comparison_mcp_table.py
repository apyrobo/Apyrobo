"""The "Which robot MCP server fits you" table in docs/comparison.md.

Each row other than APYROBO's says it was read from the project's own
README or code at a named commit, so every such row must link the
project's repository and name that commit.
"""

from __future__ import annotations

import re
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "comparison.md"

ROW_LINK = re.compile(r"^\| \[[^\]]+\]\((https://github\.com/[\w.-]+/[\w.-]+)\)")
COMMIT = re.compile(r"\(([0-9a-f]{7})[;)]")


def _table_rows() -> list[str]:
    text = DOC.read_text(encoding="utf-8")
    section = text.split("## Which robot MCP server fits you", 1)[1]
    section = section.split("\n---", 1)[0]
    lines = [line for line in section.splitlines() if line.startswith("| ")]
    # Header row, then the separator ("|---|") is not "| "-prefixed.
    return lines[1:]


def test_table_has_apyrobo_and_other_projects() -> None:
    rows = _table_rows()
    assert rows[0].startswith("| APYROBO")
    assert len(rows) >= 2


def test_every_other_row_links_a_repo_and_names_a_commit() -> None:
    for row in _table_rows()[1:]:
        first_cell = row.split(" | ", 1)[0]
        assert ROW_LINK.match(row), f"no repository link: {first_cell}"
        assert COMMIT.search(first_cell), f"no commit named: {first_cell}"


def test_wisevision_row_uses_the_current_repository_name() -> None:
    rows = [r for r in _table_rows() if "WiseVision" in r.split(" | ", 1)[0]]
    assert len(rows) == 1
    assert "https://github.com/wise-vision/ros2_mcp)" in rows[0]
    assert "ROS2_MCP_READONLY=1" in rows[0]
