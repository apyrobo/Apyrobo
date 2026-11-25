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


def test_rde_mcp_ros_2_row_links_repo_and_pypi() -> None:
    rows = [r for r in _table_rows() if "rde-mcp-ros-2" in r.split(" | ", 1)[0]]
    assert len(rows) == 1
    assert "https://github.com/ranch-hand-robotics/rde-mcp-ros-2)" in rows[0]
    assert "https://pypi.org/project/rde-mcp-ros-2/)" in rows[0]
    assert rows[0].rstrip().endswith("| MIT |")


RAW_CALLS = "Raw topic, service or action calls"


def _raw_calls_cells() -> list[tuple[str, str]]:
    text = DOC.read_text(encoding="utf-8")
    section = text.split("## Which robot MCP server fits you", 1)[1]
    section = section.split("\n---", 1)[0]
    lines = [line for line in section.splitlines() if line.startswith("| ")]
    header = [c.strip() for c in lines[0].strip("|").split(" | ")]
    col = header.index(RAW_CALLS)
    cells = []
    for row in lines[1:]:
        parts = [c.strip() for c in row.strip().strip("|").split(" | ")]
        assert len(parts) == len(header), f"wrong column count: {parts[0]}"
        cells.append((parts[0], parts[col]))
    return cells


def test_every_row_says_whether_it_exposes_raw_calls() -> None:
    cells = _raw_calls_cells()
    assert len(cells) >= 2
    for project, cell in cells:
        assert cell.startswith(("Yes", "No", "not checked")), (
            f"{project}: raw-calls cell must start Yes, No or 'not checked'"
        )


def test_apyrobo_raw_calls_cell_is_no_and_links_the_tool_list() -> None:
    project, cell = _raw_calls_cells()[0]
    assert project.startswith("APYROBO")
    assert cell.startswith("No")
    assert "(mcp.md#tools)" in cell


def test_raw_calls_yes_cells_name_the_tools_and_the_file_read() -> None:
    for project, cell in _raw_calls_cells()[1:]:
        if cell.startswith(("Yes", "No")):
            assert "`" in cell and "(`" in cell, (
                f"{project}: name the tools and the file they were read from"
            )


def test_raw_calls_note_under_the_table() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "check\nwhether yours has an allowlist" in text
