"""docker/docker-compose-demo.yml keeps its orchestration server running.

`apyrobo serve` on its default stdio transport exits as soon as stdin closes,
and a compose service has no stdin: the old `apyrobo-demo` service exited 0
right after start and restart-looped forever, reinstalling the package each
time. These tests pin the fixed setup: the server speaks WebSocket on a
published port, the command it runs stays up with stdin closed and answers a
task, and the dashboard healthcheck uses a tool the slim image has.
"""
from __future__ import annotations

import asyncio
import json
import re
import shlex
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "docker" / "docker-compose-demo.yml"
README = ROOT / "docker" / "README.md"


def _services() -> dict:
    return yaml.safe_load(COMPOSE.read_text())["services"]


def _serve_args() -> list[str]:
    """The `apyrobo serve ...` part of the apyrobo-demo command, as argv."""
    command = _services()["apyrobo-demo"]["command"]
    inner = shlex.split(command)[-1]  # the sh -c "..." string
    serve = inner.split("&&")[-1].strip()
    # sh runs each line as its own command: a flag on a line of its own
    # would fail with "not found" (exit 127) instead of reaching serve.
    assert "\n" not in serve, f"serve command spans lines: {serve!r}"
    args = shlex.split(serve)
    assert args[:2] == ["apyrobo", "serve"], serve
    return args


def test_orchestration_server_uses_websocket_on_published_port() -> None:
    args = _serve_args()
    assert args[args.index("--transport") + 1] == "websocket"
    port = args[args.index("--ws-port") + 1]
    assert f"{port}:{port}" in _services()["apyrobo-demo"].get("ports", [])
    # the image installs the extra that the WebSocket transport needs
    assert "[websocket]" in _services()["apyrobo-demo"]["command"]


def test_dashboard_healthcheck_does_not_need_curl() -> None:
    # python:3.12-slim ships no curl, so a curl healthcheck never passes.
    services = _services()
    for name, svc in services.items():
        if "healthcheck" in svc:
            assert svc["image"].startswith("python:"), name
            assert "curl" not in json.dumps(svc["healthcheck"]["test"]), name


def test_header_names_no_release_version() -> None:
    assert not re.search(r"\bv\d+\.\d+\.\d+\b", COMPOSE.read_text())


def test_readme_names_the_server_port() -> None:
    port = _serve_args()[_serve_args().index("--ws-port") + 1]
    assert f"ws://localhost:{port}" in README.read_text()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, proc: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            pytest.fail(f"server exited with {proc.returncode} before listening:\n{out}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    pytest.fail(f"server did not listen on {port} within {timeout}s")


def test_compose_server_command_stays_up_and_runs_a_task() -> None:
    pytest.importorskip("websockets")
    exe = shutil.which("apyrobo")
    if exe is None:
        pytest.skip("apyrobo console script not on PATH")

    args = _serve_args()
    port = _free_port()
    args[args.index("--ws-port") + 1] = str(port)
    proc = subprocess.Popen(
        [exe, *args[1:]],
        stdin=subprocess.DEVNULL,  # as in a compose service
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_port(port, proc)

        async def _submit() -> dict:
            from websockets.asyncio.client import connect

            async with connect(f"ws://127.0.0.1:{port}") as ws:
                await ws.send(json.dumps({
                    "task": "navigate to (2, 1)",
                    "robot_uri": "mock://compose_probe",
                }))
                while True:
                    reply = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
                    if reply.get("task") == "navigate to (2, 1)":
                        return reply

        reply = asyncio.run(_submit())
        assert proc.poll() is None, "server exited after one task"
        meta = reply["metadata"]
        assert meta["status"] == "planned", meta
        assert meta["execution"]["status"] == "completed", meta["execution"]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
