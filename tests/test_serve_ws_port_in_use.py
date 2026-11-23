"""apyrobo serve --transport websocket on a port that is already taken.

Before the fix the bind error was logged with a traceback inside the server
thread, startup() then waited the full 10 s and raised a RuntimeError that
said only "did not start within 10 s", and the CLI printed a traceback. Now
startup() raises ServerStartError at once, naming the port, and the CLI
prints one line and exits 1.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import pytest

pytest.importorskip("websockets")

from apyrobo.orchestration.adapter import (  # noqa: E402
    ServerStartError,
    WebSocketOrchestrationAdapter,
)


@pytest.fixture()
def taken_port():
    """A port with a listening socket already bound on all interfaces."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", 0))
    sock.listen(1)
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def test_startup_on_taken_port_fails_fast_and_names_it(taken_port):
    adapter = WebSocketOrchestrationAdapter(port=taken_port)
    start = time.monotonic()
    with pytest.raises(ServerStartError) as info:
        adapter.startup()
    elapsed = time.monotonic() - start
    assert elapsed < 5, f"startup took {elapsed:.1f}s; it should not wait the 10 s timeout"
    message = str(info.value)
    assert f"port {taken_port} is already in use" in message
    assert "--ws-port" in message
    assert isinstance(info.value.__cause__, OSError)
    adapter.shutdown()  # safe after a failed start
    assert adapter._thread is not None and not adapter._thread.is_alive()


def test_startup_on_taken_port_logs_no_traceback(taken_port, caplog):
    adapter = WebSocketOrchestrationAdapter(port=taken_port)
    with (
        caplog.at_level("DEBUG", logger="apyrobo.orchestration.adapter"),
        pytest.raises(ServerStartError),
    ):
        adapter.startup()
    adapter.shutdown()
    assert not any(r.exc_info for r in caplog.records)
    assert "server loop failed" not in caplog.text


def test_free_port_still_starts():
    adapter = WebSocketOrchestrationAdapter(port=0)
    adapter.startup()
    try:
        assert adapter._start_error is None
    finally:
        adapter.shutdown()


def test_cli_serve_on_taken_port_exits_1_with_one_line(taken_port):
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    start = time.monotonic()
    proc = subprocess.run(
        [
            sys.executable, "-m", "apyrobo.cli", "serve",
            "--transport", "websocket", "--ws-port", str(taken_port),
        ],
        capture_output=True, text=True, timeout=60, env=env,
        stdin=subprocess.DEVNULL,
    )
    elapsed = time.monotonic() - start
    assert proc.returncode == 1, proc.stderr
    assert "Traceback" not in proc.stderr
    errors = [line for line in proc.stderr.splitlines() if line.startswith("Error:")]
    assert errors == [
        f"Error: cannot listen on ws://0.0.0.0:{taken_port}: port {taken_port} "
        f"is already in use; stop the other server or pick another port with --ws-port"
    ]
    assert elapsed < 30
    assert proc.stdout == ""


def test_server_start_error_is_a_runtime_error():
    # Code that caught the old RuntimeError keeps working.
    assert issubclass(ServerStartError, RuntimeError)
