"""docs/mcp.md Troubleshooting: "The server writes its start-up line and any
errors to stderr; stdout carries only MCP messages."

A stray print or log line on stdout breaks a stdio MCP client (the official
SDK client skips or errors on a line that is not JSON-RPC), so this checks
the promise byte for byte: a session that lists, moves, refuses, plans, runs
a task and calls an unknown tool writes nothing to stdout but one JSON-RPC
2.0 message per line; a server that cannot start writes nothing to stdout
and says why on stderr.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[1] / "docs" / "mcp.md"
PROMISE = "stdout carries only MCP messages"


def _server(*flags: str) -> list[str]:
    return [sys.executable, "-m", "apyrobo", "mcp", *flags]


def _request(proc: subprocess.Popen, msg_id: int, method: str, params: dict) -> dict:
    proc.stdin.write(json.dumps(
        {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}) + "\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        assert line, f"stdout closed before the answer to {method}"
        message = _parse(line)
        if message.get("id") == msg_id:
            return message


def _parse(line: str) -> dict:
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        pytest.fail(f"stdout line is not JSON: {line!r}")
    assert isinstance(message, dict) and message.get("jsonrpc") == "2.0", line
    return message


def test_the_doc_still_makes_the_promise():
    assert PROMISE in " ".join(DOC.read_text(encoding="utf-8").split())


@pytest.mark.parametrize("robot", ["mock://turtlebot4", "gazebo://warehouse"])
def test_a_full_session_writes_only_json_rpc_to_stdout(tmp_path, robot):
    # gazebo:// logs a warning at start: it must land on stderr.
    audit = tmp_path / "audit.jsonl"
    stderr_path = tmp_path / "stderr.txt"
    with stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(
            _server("--robot", robot, "--max-speed", "0.5", "--no-go", "2,2,4,4",
                    "--audit-log", str(audit)),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
            text=True, encoding="utf-8",
        )
        try:
            init = _request(proc, 1, "initialize", {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "stdout-check", "version": "0"},
            })
            assert "result" in init, init
            proc.stdin.write(json.dumps(
                {"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
            calls = [
                ("tools/list", {}),
                ("tools/call", {"name": "move_to", "arguments": {"x": 1, "y": 1, "speed": 2}}),
                ("tools/call", {"name": "move_to", "arguments": {"x": 3, "y": 3}}),
                ("tools/call", {"name": "rotate", "arguments": {"angle_rad": 0.5}}),
                ("tools/call", {"name": "plan_task", "arguments": {"task": "pick up the cup"}}),
                ("tools/call", {"name": "execute_task",
                                "arguments": {"task": "go to (1, 0) then go to (0, 1)"}}),
                ("tools/call", {"name": "safety_report", "arguments": {}}),
                ("tools/call", {"name": "stop", "arguments": {}}),
                ("tools/call", {"name": "no_such_tool", "arguments": {}}),
            ]
            answers = [_request(proc, i, method, params)
                       for i, (method, params) in enumerate(calls, start=2)]
        finally:
            proc.stdin.close()
            rest = proc.stdout.read()
            proc.wait(timeout=30)
    for line in rest.splitlines():
        _parse(line)
    moved = answers[1]["result"]
    assert not moved.get("isError"), moved  # the session really drove
    assert answers[2]["result"].get("isError"), answers[2]  # and really refused
    stderr_text = stderr_path.read_text(encoding="utf-8")
    assert any(line.startswith("apyrobo mcp — robot=")
               for line in stderr_text.splitlines()), stderr_text
    assert "Error:" not in stderr_text
    assert audit.read_text(encoding="utf-8").strip(), "audit log stayed empty"


def _start_errors() -> list[list[str]]:
    cases = [["--no-go", "1,x,2,2"], ["--max-speed", "-1"],
             ["--robot", "nosuchscheme://bot"]]
    if importlib.util.find_spec("litellm") is None:
        cases.append(["--provider", "llm"])
    if importlib.util.find_spec("rclpy") is None:
        cases.append(["--robot", "ros2://burger"])
    return cases


@pytest.mark.parametrize("flags", _start_errors(), ids=" ".join)
def test_a_server_that_cannot_start_writes_nothing_to_stdout(flags):
    proc = subprocess.run(_server(*flags), input="", capture_output=True,
                          text=True, encoding="utf-8", timeout=60)
    assert proc.returncode != 0, proc
    assert proc.stdout == "", proc.stdout
    assert "Error" in proc.stderr, proc.stderr
