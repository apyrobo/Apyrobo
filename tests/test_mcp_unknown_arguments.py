"""Motion tools refuse argument names their schema does not list."""
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


def position(server):
    return server.toolbox._position()


def test_move_to_with_velocity_instead_of_speed_is_refused(server):
    before = position(server)
    reply = server.handle(call("move_to", x=1.0, y=1.0, velocity=0.1))
    assert reply["result"]["isError"] is True
    assert "Unknown argument 'velocity' for move_to" in text(reply)
    assert "'x', 'y', 'speed'" in text(reply)
    assert "did not move" in text(reply)
    assert position(server) == before


def test_rotate_with_an_extra_speed_name_is_refused(server):
    reply = server.handle(call("rotate", angle_rad=1.0, angular_speed=100))
    assert reply["result"]["isError"] is True
    assert "Unknown argument 'angular_speed' for rotate" in text(reply)
    assert "rotated" not in text(reply)


def test_several_unknown_arguments_are_all_named(server):
    reply = server.handle(call("move_to", x=1.0, y=1.0, max_speed=0.1, velocity=0.1))
    assert reply["result"]["isError"] is True
    assert "Unknown arguments 'max_speed', 'velocity' for move_to" in text(reply)


def test_execute_task_with_an_extra_argument_is_refused(server):
    before = position(server)
    reply = server.handle(call("execute_task", task="go to (1, 1)", speed=0.1))
    assert reply["result"]["isError"] is True
    assert "Unknown argument 'speed' for execute_task" in text(reply)
    assert position(server) == before


def test_known_arguments_still_run(server):
    reply = server.handle(call("move_to", x=1.0, y=1.0, speed=0.2))
    assert reply["result"]["isError"] is False
    reply = server.handle(call("rotate", angle_rad=0.5, speed=0.2, msg_id=3))
    assert reply["result"]["isError"] is False


def test_read_only_tools_still_ignore_extra_arguments(server):
    reply = server.handle(call("robot_status", host_meta="x"))
    assert reply["result"]["isError"] is False
    reply = server.handle(call("plan_task", task="go to (1, 1)", host_meta="x", msg_id=3))
    assert reply["result"]["isError"] is False


def test_the_refusal_is_in_the_audit_log(server, tmp_path):
    server.handle(call("move_to", x=1.0, y=1.0, velocity=0.1))
    line = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[-1])
    assert line["tool"] == "move_to"
    assert line["decision"] == "failed"
    assert "Unknown argument 'velocity'" in line["detail"]


def test_confirm_moves_does_not_ask_about_an_unknown_argument():
    server = build_server(confirm_moves=True)
    meta = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {"elicitation": {"form": {}}},
    }
    before = position(server)
    reply = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "move_to", "arguments": {"x": 1.0, "y": 1.0, "velocity": 0.1},
                   "_meta": meta},
    })
    assert reply["result"].get("resultType") != "input_required"
    assert reply["result"]["isError"] is True
    assert "Unknown argument 'velocity'" in text(reply)
    assert position(server) == before
