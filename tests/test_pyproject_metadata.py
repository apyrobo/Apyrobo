"""The package metadata PyPI shows says what apyrobo is: an MCP safety layer."""

from __future__ import annotations

import re
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _project_table() -> str:
    # tomllib is 3.11+ and CI also runs 3.10: read the [project] table as text.
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r"^\[project\]\n(.*?)(?=^\[)", text, re.M | re.S)
    assert match, "pyproject.toml has no [project] table"
    return match.group(1)


def _description() -> str:
    match = re.search(r'^description = "([^"]*)"$', _project_table(), re.M)
    assert match, "pyproject.toml [project] has no one-line description"
    return match.group(1)


def _keywords() -> list[str]:
    match = re.search(r"^keywords = \[(.*?)\]", _project_table(), re.M | re.S)
    assert match, "pyproject.toml [project] has no keywords list"
    return re.findall(r'"([^"]+)"', match.group(1))


def test_description_leads_with_the_mcp_safety_layer():
    description = _description()
    assert description.startswith("Safety layer between AI agents and robots")
    assert "MCP" in description
    assert "mock://" in description
    assert "orchestration layer" not in description


def test_keywords_add_mcp_terms_and_keep_the_old_ones():
    keywords = _keywords()
    for word in (
        "robotics", "ros2", "ai", "llm", "swarm", "orchestration",
        "mcp", "mcp-server", "model-context-protocol", "robot-safety",
        "ai-agents", "nav2",
    ):
        assert word in keywords, word
    assert len(keywords) == len(set(keywords))
