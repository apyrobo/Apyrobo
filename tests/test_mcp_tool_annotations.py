"""apyrobo mcp tools carry a human title and MCP tool annotations.

Clients (Claude Desktop, VS Code and others) show the title and use the
hints when deciding what to ask the user before a call. The field names
are checked against the installed ``mcp`` package's ``Tool`` and
``ToolAnnotations`` models, so a misspelled hint fails here.
"""

from __future__ import annotations

import pytest

from apyrobo.mcp_server import build_server


@pytest.fixture
def tools():
    server = build_server(max_speed=0.5)
    resp = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    return {t["name"]: t for t in resp["result"]["tools"]}


READ_ONLY = ("robot_status", "list_capabilities", "plan_task", "safety_report")
MOVES = ("move_to", "rotate", "execute_task")


def test_every_tool_has_a_title_on_the_tool_and_in_its_annotations(tools):
    for name, tool in tools.items():
        title = tool.get("title")
        assert isinstance(title, str) and title.strip(), name
        assert tool["annotations"]["title"] == title, name
    titles = [t["title"] for t in tools.values()]
    assert len(set(titles)) == len(titles)


@pytest.mark.parametrize("name", READ_ONLY)
def test_read_only_tools_say_so_and_leave_out_write_hints(tools, name):
    ann = tools[name]["annotations"]
    assert ann["readOnlyHint"] is True
    # Only meaningful when the tool is not read-only.
    assert "destructiveHint" not in ann
    assert "idempotentHint" not in ann


@pytest.mark.parametrize("name", MOVES)
def test_moves_are_not_read_only_and_reach_the_physical_world(tools, name):
    ann = tools[name]["annotations"]
    assert ann["readOnlyHint"] is False
    assert ann["openWorldHint"] is True
    # A move cannot be undone: keep the cautious default for approval prompts.
    assert ann["destructiveHint"] is True
    assert ann["idempotentHint"] is False


def test_stop_is_idempotent_and_not_destructive(tools):
    ann = tools["stop"]["annotations"]
    assert ann["readOnlyHint"] is False
    assert ann["idempotentHint"] is True
    assert ann["destructiveHint"] is False
    assert ann["openWorldHint"] is True


def test_server_side_tools_are_closed_world(tools):
    assert tools["plan_task"]["annotations"]["openWorldHint"] is False
    assert tools["safety_report"]["annotations"]["openWorldHint"] is False


def test_tools_parse_with_the_official_mcp_types(tools):
    types = pytest.importorskip("mcp.types")
    for name, spec in tools.items():
        tool = types.Tool.model_validate(spec)
        assert tool.title == spec["title"], name
        ann = tool.annotations
        assert ann is not None, name
        # Every hint we send maps onto a declared field (no silent drops).
        sent = {k for k in spec["annotations"]}
        parsed = set(ann.model_dump(by_alias=True, exclude_none=True))
        assert sent == parsed, (name, sent ^ parsed)
        assert ann.read_only_hint is spec["annotations"]["readOnlyHint"]
