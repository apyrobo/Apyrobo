"""--no-go and --allow-area take rectangles whose corners may be negative."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from apyrobo.cli import _bind_rectangle_values
from tests.mcp_stdio import talk


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["--no-go", "-1.7,-50,-1.5,50"], ["--no-go=-1.7,-50,-1.5,50"]),
        (["--allow-area", "-.5,0,1e1,2"], ["--allow-area=-.5,0,1e1,2"]),
        (["--no-go", "2,2,4,4"], ["--no-go", "2,2,4,4"]),
        (["--no-go=-1,-1,0,0"], ["--no-go=-1,-1,0,0"]),
        (["--no-go", "--max-speed", "1"], ["--no-go", "--max-speed", "1"]),
        (["-v", "-1,2"], ["-v", "-1,2"]),
    ],
)
def test_bind_rectangle_values(argv, expected):
    assert _bind_rectangle_values(argv) == expected


def test_server_starts_and_enforces_a_zone_with_negative_corners():
    calls = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "move_to", "arguments": {"x": -2.0, "y": -2.0}}},
    ]
    done = talk(
        [sys.executable, "-m", "apyrobo", "mcp", "--no-go", "-3,-3,-1,-1"],
        "".join(json.dumps(c) + "\n" for c in calls),
        last_id=2,
    )
    reply = [json.loads(line) for line in done.stdout.splitlines() if line.strip()][-1]
    assert reply["result"]["isError"] is True, done.stderr
    assert "zone" in json.dumps(reply)


def test_selftest_accepts_negative_corners():
    done = subprocess.run(
        [sys.executable, "-m", "apyrobo", "mcp", "selftest",
         "--no-go", "-3,-3,-1,-1", "--allow-area", "-5,-5,5,5"],
        capture_output=True, text=True, timeout=60,
    )
    assert "expected one argument" not in done.stderr
    assert done.returncode == 0, done.stdout + done.stderr


def test_config_accepts_negative_corners():
    done = subprocess.run(
        [sys.executable, "-m", "apyrobo", "mcp", "config", "cursor", "--no-go", "-3,-3,-1,-1"],
        capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert "-3,-3,-1,-1" in done.stdout
