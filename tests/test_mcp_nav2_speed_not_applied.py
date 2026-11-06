"""Through Nav2 no speed reaches the robot: the reply and audit log say so."""
from __future__ import annotations

import json

import pytest

from apyrobo.mcp_server import SPEED_NOT_APPLIED, build_server
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


def make(tmp_path, *, nav2, agent=None):
    """A mock:// server whose adapter reports what ros2:// on Nav2 reports."""
    log = tmp_path / "audit.jsonl"
    server = build_server(audit_log=log, max_speed=0.2, agent=agent)
    server.handle(initialize())
    if nav2:
        server.toolbox.robot._adapter.speed_limit_applies = False
    return server, log


def last_line(log):
    return json.loads(log.read_text().splitlines()[-1])


def test_a_nav2_move_says_the_limit_was_not_applied(tmp_path):
    server, log = make(tmp_path, nav2=True)
    moved = reply_payload(server.handle(call("move_to", x=1.0, y=0.0, speed=1.0)))
    assert moved["status"] == "moved"
    assert moved["speed_limit_applied"] is False
    assert moved["speed_note"] == SPEED_NOT_APPLIED
    line = last_line(log)
    assert line["decision"] == "not_limited"
    assert line["enforced_speed"] is None
    assert "not applied" in line["detail"]


def test_a_nav2_move_without_a_speed_is_not_called_ran(tmp_path):
    server, log = make(tmp_path, nav2=True)
    moved = reply_payload(server.handle(call("move_to", x=1.0, y=0.0)))
    assert moved["speed_limit_applied"] is False
    assert last_line(log)["decision"] == "not_limited"


@pytest.mark.parametrize("speed, decision", [(1.0, "clamped"), (0.1, "ran")])
def test_mock_keeps_clamped_and_ran(tmp_path, speed, decision):
    server, log = make(tmp_path, nav2=False)
    moved = reply_payload(server.handle(call("move_to", x=1.0, y=0.0, speed=speed)))
    assert "speed_limit_applied" not in moved
    assert "speed_note" not in moved
    assert last_line(log)["decision"] == decision


def test_a_refused_nav2_move_is_still_refused(tmp_path):
    log = tmp_path / "audit.jsonl"
    server = build_server(audit_log=log, max_speed=0.2, agent=None,
                          no_go_zones=[{"x_min": 2, "y_min": 2, "x_max": 4, "y_max": 4}])
    server.handle(initialize())
    server.toolbox.robot._adapter.speed_limit_applies = False
    reply = server.handle(call("move_to", x=3.0, y=3.0))
    assert reply["result"]["isError"]
    assert "speed_limit_applied" not in reply["result"]["content"][0]["text"]
    assert last_line(log)["decision"] == "refused"


def test_a_nav2_rotate_is_not_marked(tmp_path):
    server, log = make(tmp_path, nav2=True)
    rotated = reply_payload(server.handle(call("rotate", angle_rad=0.5)))
    assert "speed_limit_applied" not in rotated
    assert last_line(log)["decision"] in {"ran", "clamped"}


def test_execute_task_through_nav2_says_the_limit_was_not_applied(tmp_path):
    server, log = make(tmp_path, nav2=True, agent=Agent(provider="rule"))
    done = reply_payload(server.handle(call("execute_task", task="go to (1, 1)")))
    assert done["status"] == "completed"
    assert done["speed_limit_applied"] is False
    assert last_line(log)["decision"] == "not_limited"


def test_execute_task_with_no_drive_is_not_marked(tmp_path):
    server, log = make(tmp_path, nav2=True, agent=Agent(provider="rule"))
    done = reply_payload(server.handle(call("execute_task", task="report status")))
    assert "speed_limit_applied" not in done
    assert last_line(log)["decision"] == "ran"


def test_without_an_audit_log_the_reply_still_says_so():
    server = build_server(max_speed=0.2, agent=None)
    server.handle(initialize())
    server.toolbox.robot._adapter.speed_limit_applies = False
    moved = reply_payload(server.handle(call("move_to", x=1.0, y=0.0, speed=1.0)))
    assert moved["speed_limit_applied"] is False
