"""Tests for `apyrobo mcp` — the MCP server over the safety enforcer."""

from __future__ import annotations

import io
import json
import subprocess
import sys

import pytest

from apyrobo.mcp_server import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    SUPPORTED_PROTOCOL_VERSIONS,
    build_server,
    parse_zone,
)

ZONE = {"x_min": 2.0, "y_min": 2.0, "x_max": 4.0, "y_max": 4.0}


@pytest.fixture
def server():
    return build_server(max_speed=0.5, no_go_zones=[ZONE])


def rpc(server, method, params=None, msg_id=1):
    msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        msg["params"] = params
    return server.handle(msg)


def call(server, name, arguments=None):
    resp = rpc(server, "tools/call", {"name": name, "arguments": arguments or {}})
    result = resp["result"]
    return result["isError"], result["content"][0]["text"]


# -- protocol ---------------------------------------------------------------


def test_initialize_echoes_supported_version(server):
    resp = rpc(server, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {}})
    result = resp["result"]
    assert result["protocolVersion"] == "2025-03-26"
    assert result["serverInfo"]["name"] == "apyrobo"
    assert "tools" in result["capabilities"]
    assert "safety" in result["instructions"]


def test_initialize_unknown_version_falls_back_to_newest(server):
    resp = rpc(server, "initialize", {"protocolVersion": "1999-01-01"})
    assert resp["result"]["protocolVersion"] == SUPPORTED_PROTOCOL_VERSIONS[0]


def test_notifications_get_no_reply(server):
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_ping_and_unknown_method(server):
    assert rpc(server, "ping")["result"] == {}
    assert rpc(server, "resources/list")["error"]["code"] == METHOD_NOT_FOUND


def test_tools_list_has_schemas(server):
    tools = rpc(server, "tools/list")["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {
        "robot_status",
        "move_to",
        "stop",
        "plan_task",
        "execute_task",
        "safety_report",
    } <= names
    move = next(t for t in tools if t["name"] == "move_to")
    assert move["inputSchema"]["required"] == ["x", "y"]
    assert move["annotations"]["readOnlyHint"] is False


def test_unknown_tool_is_invalid_params(server):
    resp = rpc(server, "tools/call", {"name": "self_destruct", "arguments": {}})
    assert resp["error"]["code"] == INVALID_PARAMS


def test_serve_loop_parse_error_and_stdout_isolation(server, capsys):
    stdin = io.StringIO(
        "not json\n"
        + json.dumps({"jsonrpc": "2.0", "id": 7, "method": "ping"})
        + "\n"
        + json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
        + "\n"
    )
    out = io.StringIO()
    server.serve(stdin=stdin, stdout=out)
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert lines[0]["error"]["code"] == PARSE_ERROR
    assert lines[1] == {"jsonrpc": "2.0", "id": 7, "result": {}}
    assert len(lines) == 2


# -- safety behaviour -------------------------------------------------------


def test_status_reports_policy(server):
    is_error, text = call(server, "robot_status")
    assert not is_error
    status = json.loads(text)
    assert status["safety_policy"]["max_speed_m_s"] == 0.5
    assert status["safety_policy"]["no_go_zones"] == [ZONE]
    assert status["position"] == {"x": 0.0, "y": 0.0}


def test_move_speed_is_clamped(server):
    is_error, text = call(server, "move_to", {"x": 1, "y": 1, "speed": 2.5})
    assert not is_error
    result = json.loads(text)
    assert result["position"] == {"x": 1.0, "y": 1.0}
    assert result["safety_interventions"][0]["type"] == "speed_clamped"
    assert result["safety_interventions"][0]["enforced"] == 0.5


def test_move_into_no_go_zone_is_refused_and_robot_stays(server):
    is_error, text = call(server, "move_to", {"x": 3, "y": 3})
    assert is_error
    assert text.startswith("REFUSED by safety policy")
    _, status = call(server, "robot_status")
    assert json.loads(status)["position"] == {"x": 0.0, "y": 0.0}
    _, report = call(server, "safety_report")
    assert json.loads(report)["violations"][0]["type"] == "collision_zone"


def test_bad_arguments_are_tool_errors(server):
    is_error, text = call(server, "move_to", {"x": "far", "y": 1})
    assert is_error and "must be a number" in text
    is_error, text = call(server, "move_to", {"x": 1})
    assert is_error and "Missing required argument 'y'" in text


def test_stop_always_works(server):
    is_error, text = call(server, "stop")
    assert not is_error
    assert json.loads(text)["status"] == "stopped"


def test_plan_task_does_not_move(server):
    is_error, text = call(server, "plan_task", {"task": "go to (5, 1)"})
    assert not is_error
    plan = json.loads(text)
    assert plan["steps"][0]["name"] == "Navigate To"
    _, status = call(server, "robot_status")
    assert json.loads(status)["position"] == {"x": 0.0, "y": 0.0}


def test_execute_task_completes(server):
    is_error, text = call(server, "execute_task", {"task": "deliver the package to (5, 1)"})
    assert not is_error, text
    result = json.loads(text)
    assert result["status"] == "completed"
    assert result["position"] == {"x": 5.0, "y": 1.0}


def test_execute_task_into_no_go_zone_reports_refusal(server):
    is_error, text = call(server, "execute_task", {"task": "go to (3, 3)"})
    assert is_error
    assert text.startswith("REFUSED by safety policy")
    assert "at step 1" in text


# -- configuration ----------------------------------------------------------


def test_parse_zone():
    assert parse_zone("2,2,4,4") == ZONE
    with pytest.raises(ValueError):
        parse_zone("1,2,3")
    with pytest.raises(ValueError):
        parse_zone("4,4,2,2")


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        build_server(policy="yolo")


def test_strict_policy_without_overrides_keeps_name():
    srv = build_server(policy="strict")
    assert srv.toolbox.enforcer.policy.name == "strict"
    assert srv.toolbox.enforcer.policy.max_speed == 0.5


# -- end to end through the CLI --------------------------------------------


def test_cli_stdio_round_trip():
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "move_to", "arguments": {"x": 3, "y": 3}},
        },
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "apyrobo", "mcp", "--no-go", "2,2,4,4"],
        input="".join(json.dumps(r) + "\n" for r in requests),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    replies = [json.loads(line) for line in proc.stdout.splitlines()]
    assert [r["id"] for r in replies] == [1, 2]  # nothing else on stdout
    assert replies[1]["result"]["isError"] is True
    assert "apyrobo mcp" in proc.stderr
