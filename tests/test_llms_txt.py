"""Keep llms.txt honest: an agent reading it should find the MCP server."""

from __future__ import annotations

import re
from pathlib import Path

from apyrobo import mcp_server

ROOT = Path(__file__).resolve().parent.parent


def _text() -> str:
    return (ROOT / "llms.txt").read_text(encoding="utf-8")


def test_the_mcp_server_comes_before_everything_else():
    text = _text()
    assert text.index("## MCP server") < text.index("## Python framework")
    assert text.index("apyrobo mcp") < text.index("## Python framework")


def test_it_gives_the_try_out_the_config_command_and_the_guide():
    text = _text()
    assert "apyrobo mcp --max-speed 0.5 --no-go 2,2,4,4" in text
    assert "apyrobo mcp config claude-desktop" in text
    assert "(docs/mcp.md)" in text


def test_it_names_every_tool_the_server_has():
    server = mcp_server.build_server(robot_uri="mock://llms-tools")
    text = _text()
    missing = [t["name"] for t in server.toolbox.list_tools() if f"`{t['name']}`" not in text]
    assert not missing, f"llms.txt does not name tools {missing}"


def test_it_says_what_the_server_does_not_guarantee():
    text = _text()
    assert "not a certified safety system" in text
    assert "policy, not live sensors" in text
    assert "e-stops" in text


def test_it_makes_no_claim_that_safety_cannot_be_bypassed():
    assert "bypassed" not in _text().lower()


def test_every_relative_link_resolves():
    links = re.findall(r"\]\((?!https?:)([^)]+)\)", _text())
    assert links
    missing = [link for link in links if not (ROOT / link).exists()]
    assert not missing, f"llms.txt links to missing files {missing}"


def test_readme_and_llms_txt_lead_with_the_pypi_command():
    """From 4.2.0 apyrobo mcp is on PyPI: the first command is the plain uvx form."""
    for name in ("README.md", "llms.txt"):
        text = (ROOT / name).read_text(encoding="utf-8")
        first = text.index("apyrobo mcp --max-speed")
        assert text[:first].rstrip().endswith("uvx"), f"{name} does not lead with uvx apyrobo mcp"
        assert "not in a PyPI release yet" not in text, name
