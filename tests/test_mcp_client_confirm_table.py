"""docs/mcp.md: the 'Where --confirm-moves works' table stays in step with the code."""

from __future__ import annotations

import re
from pathlib import Path

from apyrobo import mcp_clients
from apyrobo.mcp_server import RobotToolbox

DOC = Path(__file__).resolve().parent.parent / "docs" / "mcp.md"

CLIENT_NAMES = {
    "claude-desktop": "Claude Desktop",
    "claude-code": "Claude Code",
    "cursor": "Cursor",
    "vscode": "VS Code",
    "codex": "Codex CLI",
    "gemini": "Gemini CLI",
    "zed": "Zed",
}


def _table() -> dict[str, list[str]]:
    text = DOC.read_text(encoding="utf-8")
    start = text.index("#### Where `--confirm-moves` works")
    rows: dict[str, list[str]] = {}
    for line in text[start:].splitlines()[1:]:
        if line.startswith("#"):
            break
        if line.startswith("| ") and not line.startswith(("| ---", "| Client")):
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows[cells[0]] = cells[1:]
    return rows


def test_every_config_client_has_a_row():
    rows = _table()
    assert set(CLIENT_NAMES) == set(mcp_clients.CLIENTS)
    missing = [name for name in CLIENT_NAMES.values() if name not in rows]
    assert not missing, f"no 'Where --confirm-moves works' row for {missing}"


def test_clients_config_refuses_are_marked_no():
    rows = _table()
    for client in mcp_clients.NO_ELICITATION:
        assert rows[CLIENT_NAMES[client]][0].startswith("No"), client


def test_yes_only_with_a_linked_source():
    for name, (elicitation, _asks, source) in _table().items():
        if elicitation.startswith("Yes"):
            assert re.search(r"\]\(https://", source), f"{name}: 'Yes' without a linked source"


def _toolbox() -> RobotToolbox:
    from apyrobo import Agent, Robot
    from apyrobo.safety.enforcer import SafetyEnforcer

    robot = Robot.discover("mock://turtlebot4")
    return RobotToolbox(robot, SafetyEnforcer(robot), Agent(provider="rule"), "mock://doc-check")


def test_vscode_line_names_exactly_the_tools_that_are_not_read_only():
    specs = [spec for spec, _ in _toolbox()._tools.values()]
    prompted = {s["name"] for s in specs if not s["annotations"]["readOnlyHint"]}
    unprompted = {s["name"] for s in specs if s["annotations"]["readOnlyHint"]}
    text = DOC.read_text(encoding="utf-8")
    start = text.index("VS Code asks you before every tool call")
    para = text[start:text.index("\n\n", start)]
    marked, _, rest = para.partition("so VS Code shows")
    assert set(re.findall(r"`(\w+)`", marked)) - {"readOnlyHint"} == prompted
    assert set(re.findall(r"`(\w+)`", rest)) - {"confirm"} == unprompted
