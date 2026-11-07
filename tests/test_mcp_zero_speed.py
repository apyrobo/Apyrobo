"""move_to and rotate refuse speed 0 instead of reporting a move.

speed 0 on move_to used to drive at the default speed and return "moved";
rotate with speed 0 returned "rotated". Neither is what a caller asking for 0
meant, so both are refused before anything moves.
"""
from __future__ import annotations

import json

import pytest

from apyrobo.mcp_server import build_server


def call(name, msg_id=2, **arguments):
    return {
        "jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def initialize():
    return {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "t", "version": "0"}},
    }


def text(reply):
    return reply["result"]["content"][0]["text"]


@pytest.fixture
def server(tmp_path):
    server = build_server(audit_log=tmp_path / "audit.jsonl", max_speed=0.5)
    server.handle(initialize())
    return server


@pytest.mark.parametrize("speed", [0, 0.0, -0.0])
def test_move_to_speed_0_is_refused(server, speed):
    before = server.toolbox._position()
    reply = server.handle(call("move_to", x=1.0, y=1.0, speed=speed))
    assert reply["result"]["isError"] is True
    assert "must be more than 0 m/s" in text(reply)
    assert "Leave it out for the default speed" in text(reply)
    assert server.toolbox._position() == before


@pytest.mark.parametrize("speed", [0, 0.0, -0.0])
def test_rotate_speed_0_is_refused(server, speed):
    heading = server.toolbox.enforcer.get_orientation()
    reply = server.handle(call("rotate", angle_rad=1.0, speed=speed))
    assert reply["result"]["isError"] is True
    assert "must be a turn rate other than 0 rad/s" in text(reply)
    assert server.toolbox.enforcer.get_orientation() == heading


def test_small_and_negative_turn_speeds_still_run(server):
    reply = server.handle(call("move_to", x=1.0, y=1.0, speed=0.01))
    assert reply["result"]["isError"] is False
    reply = server.handle(call("rotate", angle_rad=0.5, speed=0.1, msg_id=3))
    assert reply["result"]["isError"] is False
    reply = server.handle(call("rotate", angle_rad=-0.5, speed=-0.1, msg_id=4))
    assert reply["result"]["isError"] is False


def test_the_refusal_is_in_the_audit_log(server, tmp_path):
    server.handle(call("move_to", x=1.0, y=1.0, speed=0))
    line = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[-1])
    assert line["tool"] == "move_to"
    assert line["decision"] == "failed"
    assert "must be more than 0 m/s" in line["detail"]


def test_confirm_moves_does_not_ask_about_speed_0():
    server = build_server(confirm_moves=True)
    meta = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {"elicitation": {"form": {}}},
    }
    before = server.toolbox._position()
    reply = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "move_to", "arguments": {"x": 1.0, "y": 1.0, "speed": 0},
                   "_meta": meta},
    })
    assert reply["result"].get("resultType") != "input_required"
    assert reply["result"]["isError"] is True
    assert server.toolbox._position() == before
