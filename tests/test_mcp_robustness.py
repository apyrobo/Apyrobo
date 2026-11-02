"""`apyrobo mcp` on garbage input: answer with an error, never crash, never move.

Every case feeds raw lines to the stdio loop, the way a buggy or hostile
client would, and checks three things: each request gets a JSON-RPC error
or an ``isError`` tool result, every line written to stdout is strict JSON,
and the mock robot is still where it started.
"""

from __future__ import annotations

import io
import json
import math
import subprocess
import sys

import pytest

from apyrobo.mcp_server import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    MODERN_PROTOCOL_VERSIONS,
    PARSE_ERROR,
    build_server,
)
from apyrobo.safety.enforcer import SafetyEnforcer, SafetyPolicy, SafetyViolation

ZONE = {"x_min": 2.0, "y_min": 2.0, "x_max": 4.0, "y_max": 4.0}
INIT = {
    "jsonrpc": "2.0",
    "id": 0,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
}


def _strict(text: str):
    """json.loads that refuses NaN/Infinity, like strict JSON parsers do."""

    def bad(token: str):
        raise ValueError(f"non-standard JSON constant {token}")

    return json.loads(text, parse_constant=bad)


def serve_lines(lines: list[str], **kw):
    """Run the stdio loop on raw *lines*; return (replies, server)."""
    server = build_server(max_speed=0.5, no_go_zones=[ZONE], **kw)
    out = io.StringIO()
    server.serve(io.StringIO("".join(line + "\n" for line in lines)), out)
    replies = [_strict(line) for line in out.getvalue().splitlines() if line.strip()]
    return replies, server


def assert_not_moved(server) -> None:
    assert server.toolbox._position() == {"x": 0.0, "y": 0.0}
    assert server.toolbox.enforcer.get_orientation() == 0.0


def tool_call(arguments: str, name: str = "move_to", msg_id: int = 1) -> str:
    # Raw text on purpose: json.dumps cannot write NaN the way a client might.
    return (
        f'{{"jsonrpc":"2.0","id":{msg_id},"method":"tools/call",'
        f'"params":{{"name":"{name}","arguments":{arguments}}}}}'
    )


# -- framing and JSON-RPC shape ---------------------------------------------


@pytest.mark.parametrize(
    "line, code",
    [
        ("not json", PARSE_ERROR),
        ("{", PARSE_ERROR),
        ('{"jsonrpc":"2.0","id":1,"method":"tools/list"', PARSE_ERROR),
        ("null", INVALID_REQUEST),
        ("42", INVALID_REQUEST),
        ('"tools/list"', INVALID_REQUEST),
        ("[]", INVALID_REQUEST),
        ('{"id":1,"method":"tools/list"}', INVALID_REQUEST),
        ('{"jsonrpc":"1.0","id":1,"method":"tools/list"}', INVALID_REQUEST),
        ('{"jsonrpc":"2.0","id":1}', INVALID_REQUEST),
        ('{"jsonrpc":"2.0","id":1,"method":42}', INVALID_REQUEST),
        ('{"jsonrpc":"2.0","id":1,"method":"tools/list","params":[1]}', INVALID_PARAMS),
        ('{"jsonrpc":"2.0","id":1,"method":"no/such/method"}', METHOD_NOT_FOUND),
    ],
)
def test_malformed_message_gets_one_error(line, code):
    replies, server = serve_lines([line])
    assert len(replies) == 1
    assert replies[0]["error"]["code"] == code
    assert_not_moved(server)


def test_empty_and_blank_lines_are_skipped():
    replies, _ = serve_lines(["", "   ", "\t", json.dumps(INIT)])
    assert [r["id"] for r in replies] == [0]


def test_batch_mixes_errors_and_results():
    batch = [
        INIT,
        1,
        {"jsonrpc": "2.0", "id": 2, "method": "nope"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ]
    replies, server = serve_lines([json.dumps(batch)])
    assert len(replies) == 1 and isinstance(replies[0], list)
    by_id = {r["id"]: r for r in replies[0]}
    assert "result" in by_id[0]
    assert by_id[None]["error"]["code"] == INVALID_REQUEST
    assert by_id[2]["error"]["code"] == METHOD_NOT_FOUND
    assert_not_moved(server)


def test_batch_of_only_notifications_gets_no_reply():
    note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    replies, _ = serve_lines([json.dumps([note, note])])
    assert replies == []


def test_request_without_id_is_a_notification_and_never_moves():
    move = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": "move_to", "arguments": {"x": 1, "y": 1}},
    }
    replies, server = serve_lines([json.dumps(INIT), json.dumps(move)])
    assert [r["id"] for r in replies] == [0]
    assert_not_moved(server)


def test_very_long_line_is_a_parse_error_and_the_loop_goes_on():
    replies, _ = serve_lines(["x" * 2_000_000, json.dumps(INIT)])
    assert replies[0]["error"]["code"] == PARSE_ERROR
    assert replies[1]["id"] == 0


def test_long_string_argument_is_a_tool_error():
    huge = "9" * 100_000
    replies, server = serve_lines([json.dumps(INIT), tool_call(f'{{"x":"{huge}","y":0}}')])
    assert replies[-1]["result"]["isError"] is True
    assert_not_moved(server)


# -- tools/call arguments -----------------------------------------------------


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("move_to", "{}"),
        ("move_to", '{"x":1}'),
        ("move_to", '{"x":"1","y":1}'),
        ("move_to", '{"x":true,"y":1}'),
        ("move_to", '{"x":null,"y":1}'),
        ("move_to", '{"x":[1],"y":1}'),
        ("move_to", '{"x":{"v":1},"y":1}'),
        ("move_to", '{"x":1,"y":1,"speed":"fast"}'),
        ("rotate", '{"angle_rad":"half"}'),
        ("rotate", "{}"),
        ("execute_task", '{"task":42}'),
        ("execute_task", '{"task":"   "}'),
    ],
)
def test_wrong_argument_types_are_tool_errors(name, arguments):
    replies, server = serve_lines([json.dumps(INIT), tool_call(arguments, name)])
    assert replies[-1]["result"]["isError"] is True
    assert_not_moved(server)


@pytest.mark.parametrize("arguments", ['"x=1"', "[1,2]", "7"])
def test_non_object_arguments_are_invalid_params(arguments):
    replies, server = serve_lines([json.dumps(INIT), tool_call(arguments)])
    assert replies[-1]["error"]["code"] == INVALID_PARAMS
    assert_not_moved(server)


@pytest.mark.parametrize("name", ['"no_such_tool"', "42", "null"])
def test_unknown_tool_is_invalid_params(name):
    line = f'{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":{name}}}}}'
    replies, server = serve_lines([json.dumps(INIT), line])
    assert replies[-1]["error"]["code"] == INVALID_PARAMS
    assert_not_moved(server)


NON_FINITE = [
    ("move_to", '{"x":NaN,"y":0}'),
    ("move_to", '{"x":0,"y":NaN}'),
    ("move_to", '{"x":Infinity,"y":0}'),
    ("move_to", '{"x":-Infinity,"y":0}'),
    ("move_to", '{"x":1e400,"y":0}'),
    ("move_to", '{"x":1' + "0" * 400 + ',"y":0}'),
    ("move_to", '{"x":1,"y":1,"speed":NaN}'),
    ("move_to", '{"x":1,"y":1,"speed":Infinity}'),
    ("move_to", '{"x":1,"y":1,"speed":-Infinity}'),
    ("move_to", '{"x":3,"y":3,"speed":NaN}'),
    ("rotate", '{"angle_rad":NaN}'),
    ("rotate", '{"angle_rad":Infinity}'),
    ("rotate", '{"angle_rad":1,"speed":NaN}'),
    ("rotate", '{"angle_rad":1,"speed":-Infinity}'),
]


@pytest.mark.parametrize("name, arguments", NON_FINITE)
def test_nan_and_infinity_are_refused_before_any_motion(name, arguments):
    replies, server = serve_lines([json.dumps(INIT), tool_call(arguments, name)])
    result = replies[-1]["result"]
    assert result["isError"] is True
    assert "finite" in result["content"][0]["text"]
    assert_not_moved(server)


@pytest.mark.parametrize("name, arguments", NON_FINITE[:4] + NON_FINITE[10:12])
def test_nan_is_refused_in_the_modern_era_too(name, arguments):
    line = (
        f'{{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{{"name":"{name}",'
        f'"arguments":{arguments},"_meta":{{"io.modelcontextprotocol/protocolVersion":'
        f'"{MODERN_PROTOCOL_VERSIONS[0]}","io.modelcontextprotocol/clientCapabilities":{{}}}}}}}}'
    )
    replies, server = serve_lines([line])
    assert replies[-1]["result"]["isError"] is True
    assert_not_moved(server)


@pytest.mark.parametrize("name, arguments", NON_FINITE[:2] + NON_FINITE[6:7] + NON_FINITE[10:11])
def test_nan_is_refused_under_confirm_moves(name, arguments):
    replies, server = serve_lines(
        [json.dumps(INIT), tool_call(arguments, name)], confirm_moves=True
    )
    reply = replies[-1]
    assert reply.get("error") or reply["result"]["isError"] is True
    assert_not_moved(server)


@pytest.mark.parametrize("speed", ["-1", "-0.5", "-1000"])
def test_negative_linear_speed_is_refused(speed):
    move = tool_call(f'{{"x":1,"y":1,"speed":{speed}}}')
    replies, server = serve_lines([json.dumps(INIT), move])
    result = replies[-1]["result"]
    assert result["isError"] is True
    assert "0 or more" in result["content"][0]["text"]
    assert_not_moved(server)


def test_server_keeps_serving_after_garbage():
    lines = [
        "garbage",
        json.dumps(INIT),
        tool_call('{"x":NaN,"y":NaN}', msg_id=1),
        "[]",
        tool_call('{"x":1,"y":0,"speed":5}', msg_id=2),
    ]
    replies, server = serve_lines(lines)
    last = replies[-1]
    assert last["id"] == 2 and last["result"]["isError"] is False
    moved = json.loads(last["result"]["content"][0]["text"])
    assert moved["safety_interventions"][0]["enforced"] == 0.5
    assert server.toolbox._position() == {"x": 1.0, "y": 0.0}


def test_real_stdio_survives_invalid_utf8_and_nan():
    payload = (
        b"\xff\xfe\x00 not utf-8\n"
        + json.dumps(INIT).encode()
        + b"\n"
        + tool_call('{"x":NaN,"y":0}').encode()
        + b"\n"
        + b'{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"robot_status"}}\n'
    )
    proc = subprocess.run(
        [sys.executable, "-m", "apyrobo", "mcp"],
        input=payload,
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    replies = [_strict(line) for line in proc.stdout.decode().splitlines() if line.strip()]
    assert replies[0]["error"]["code"] == PARSE_ERROR
    assert replies[2]["result"]["isError"] is True
    status = json.loads(replies[3]["result"]["content"][0]["text"])
    assert status["position"] == {"x": 0.0, "y": 0.0}


# -- the enforcer itself --------------------------------------------------------


def _enforcer(**policy):
    from apyrobo.core.robot import Robot

    robot = Robot.discover("mock://robustness")
    return robot, SafetyEnforcer(robot, policy=SafetyPolicy(name="t", **policy))


@pytest.mark.parametrize(
    "kw",
    [
        {"x": math.nan, "y": 0.0},
        {"x": 0.0, "y": math.inf},
        {"x": 1.0, "y": 1.0, "speed": math.nan},
        {"x": 1.0, "y": 1.0, "speed": math.inf},
        {"x": 1.0, "y": 1.0, "speed": -5.0},
    ],
)
def test_enforcer_refuses_bad_move_numbers(kw):
    robot, enforcer = _enforcer(max_speed=0.5, collision_zones=[ZONE])
    with pytest.raises(SafetyViolation):
        enforcer.move(**kw)
    assert tuple(robot.get_position()) == (0.0, 0.0)
    assert enforcer.violations[-1]["type"] == "invalid_command"


@pytest.mark.parametrize(
    "kw",
    [{"angle_rad": math.nan}, {"angle_rad": -math.inf}, {"angle_rad": 1.0, "speed": math.nan}],
)
def test_enforcer_refuses_bad_rotate_numbers(kw):
    robot, enforcer = _enforcer()
    with pytest.raises(SafetyViolation):
        enforcer.rotate(**kw)
    assert enforcer.get_orientation() == 0.0
    assert enforcer.violations[-1]["type"] == "invalid_command"
