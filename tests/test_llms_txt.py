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


def _section(title: str) -> str:
    text = _text()
    start = text.index(title)
    end = text.find("\n## ", start + len(title))
    return text[start:] if end == -1 else text[start:end]


def test_the_quick_start_runs_and_moves_the_robot():
    section = _section("## Quick start")
    block = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    scope: dict = {}
    exec(compile(block, "llms.txt", "exec"), scope)
    assert scope["result"].status.value == "completed"
    assert len(scope["plan"].skills) >= 1
    assert scope["robot"].get_position() != (0.0, 0.0)


def test_every_adapter_scheme_it_names_is_registered():
    import importlib

    from apyrobo.core.adapters import _ADAPTER_REGISTRY

    section = _section("### Hardware adapter pattern")
    for module in re.findall(r"`import (apyrobo[\w.]+)`", section):
        importlib.import_module(module)
    schemes = re.findall(r"`(\w+)://", section)
    assert schemes
    missing = [s for s in schemes if s not in _ADAPTER_REGISTRY]
    assert not missing, f"llms.txt names adapter schemes with no adapter: {missing}"


def test_it_names_every_client_mcp_config_supports():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", "mcp", "config", "--help"],
        capture_output=True, text=True, timeout=60, check=True,
    ).stdout
    choices = re.search(r"\{(claude-desktop,[a-z,-]+)\}", out).group(1).split(",")
    assert "claude-desktop" in choices
    text = _text()
    missing = [c for c in choices if f"`{c}`" not in text]
    assert not missing, f"llms.txt does not name mcp config clients {missing}"
