"""docs/mcp.md must not say a move_to with speed 0 drives.

The /cmd_vel paragraph under "Speed on ros2://" used to say a ``move_to`` with
``speed`` 0 drives at the adapter's default speed. ``apyrobo mcp`` refuses
speed 0 (and below) before the robot moves, on any robot, so a user reading
the doc expected a move that never came. These tests tie the paragraph to
what the server does.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from apyrobo.mcp_server import build_server

DOC = Path(__file__).resolve().parent.parent / "docs" / "mcp.md"


def _cmd_vel_paragraph() -> str:
    text = DOC.read_text(encoding="utf-8")
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    found = [p for p in paragraphs if p.startswith("Without Nav2, on the `/cmd_vel` fallback")]
    assert len(found) == 1, found
    return found[0]


def test_the_doc_no_longer_says_speed_0_drives():
    paragraph = _cmd_vel_paragraph()
    assert "or with `speed` 0" not in paragraph
    assert "A `move_to` with no `speed` drives at the adapter's default of 0.5 m/s" in paragraph


def test_the_doc_says_speed_0_is_refused():
    assert "A `speed` of 0 or less is refused before anything moves, on every robot." in (
        _cmd_vel_paragraph()
    )


def _call(server, msg_id, name, **arguments):
    return server.handle({
        "jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    })


@pytest.mark.parametrize("speed", [0, -0.2])
def test_the_server_refuses_what_the_doc_says_it_refuses(tmp_path, speed):
    server = build_server(audit_log=tmp_path / "audit.jsonl", max_speed=0.5)
    server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "t", "version": "0"}},
    })
    before = server.toolbox._position()
    reply = _call(server, 2, "move_to", x=1.0, y=1.0, speed=speed)
    assert reply["result"]["isError"] is True
    assert "must be more than 0 m/s" in reply["result"]["content"][0]["text"]
    assert server.toolbox._position() == before
