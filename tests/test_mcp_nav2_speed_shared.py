"""Another node publishing Nav2 speed limits: the reply and audit log say so."""
from __future__ import annotations

import json

import pytest

from apyrobo.mcp_server import (
    SPEED_NOT_APPLIED,
    SPEED_NOT_CHECKED,
    SPEED_SHARED,
    build_server,
)
from apyrobo.skills.agent import Agent


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


def reply_payload(reply):
    assert not reply["result"]["isError"], reply
    return json.loads(reply["result"]["content"][0]["text"])


def make(tmp_path, *, shared=True, agent=None):
    """A mock:// server whose adapter reports a second speed_limit publisher."""
    log = tmp_path / "audit.jsonl"
    server = build_server(audit_log=log, max_speed=0.2, agent=agent)
    server.handle(initialize())
    server.toolbox.robot._adapter.speed_limit_shared = shared
    return server, log


def last_line(log):
    return json.loads(log.read_text().splitlines()[-1])


def test_a_shared_limit_is_said_in_the_reply_and_audit(tmp_path):
    server, log = make(tmp_path)
    moved = reply_payload(server.handle(call("move_to", x=1.0, y=0.0, speed=0.1)))
    assert moved["status"] == "moved"
    assert moved["speed_limit_shared"] is True
    assert moved["speed_note"] == SPEED_SHARED
    line = last_line(log)
    assert line["decision"] == "ran"
    assert line["enforced_speed"] == pytest.approx(0.1)
    assert line["detail"] == SPEED_SHARED


def test_a_clamped_move_keeps_its_decision(tmp_path):
    server, log = make(tmp_path)
    reply_payload(server.handle(call("move_to", x=1.0, y=0.0, speed=5.0)))
    line = last_line(log)
    assert line["decision"] == "clamped"
    assert line["detail"] == SPEED_SHARED


def test_execute_task_with_a_shared_limit_says_so(tmp_path):
    server, _ = make(tmp_path, agent=Agent(provider="rule"))
    done = reply_payload(server.handle(call("execute_task", task="go to (1, 1)")))
    assert done["status"] == "completed"
    assert done["speed_limit_shared"] is True
    assert done["speed_note"] == SPEED_SHARED


def test_shared_and_unchecked_both_said(tmp_path):
    server, log = make(tmp_path)
    server.toolbox.robot._adapter.speed_limit_checked = False
    moved = reply_payload(server.handle(call("move_to", x=1.0, y=0.0)))
    assert moved["speed_limit_checked"] is False
    assert moved["speed_limit_shared"] is True
    assert moved["speed_note"] == f"{SPEED_NOT_CHECKED} {SPEED_SHARED}"
    detail = last_line(log)["detail"]  # the audit log caps detail's length
    assert "not checked" in detail
    assert moved["speed_note"].startswith(detail)


def test_not_applied_wins_over_shared(tmp_path):
    server, log = make(tmp_path)
    server.toolbox.robot._adapter.speed_limit_applies = False
    moved = reply_payload(server.handle(call("move_to", x=1.0, y=0.0)))
    assert moved["speed_note"] == SPEED_NOT_APPLIED
    assert "speed_limit_shared" not in moved
    assert last_line(log)["decision"] == "not_limited"


def test_a_rotate_is_not_marked(tmp_path):
    server, _ = make(tmp_path)
    turned = reply_payload(server.handle(call("rotate", angle_rad=0.5)))
    assert "speed_limit_shared" not in turned
    assert "speed_note" not in turned


def test_not_shared_adds_nothing(tmp_path):
    server, log = make(tmp_path, shared=False)
    moved = reply_payload(server.handle(call("move_to", x=1.0, y=0.0, speed=0.1)))
    assert "speed_limit_shared" not in moved
    assert "speed_note" not in moved
    assert "detail" not in last_line(log)


def test_an_adapter_without_the_check_adds_nothing(tmp_path):
    server, _ = make(tmp_path)
    del server.toolbox.robot._adapter.speed_limit_shared
    moved = reply_payload(server.handle(call("move_to", x=1.0, y=0.0, speed=0.1)))
    assert "speed_limit_shared" not in moved
