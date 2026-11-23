"""docs/conformance.md 'Quick start' runs as written.

The quick start installs `apyrobo[conformance]` and then runs
`apyrobo conformance ws://localhost:8765`. Run that way, the ws:// line
exits 2 ("The 'websockets' package is required for ws:// targets"), because
websockets ships in the `websocket` extra, and nothing in the block said a
server must already be listening. These tests pin the fixed block: the
install line it gives for ws:// pulls in websockets, the server command it
names is a real one, and that server plus the conformance line pass.
"""
from __future__ import annotations

import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "conformance.md"
CLI = [sys.executable, "-m", "apyrobo"]


def _quickstart() -> str:
    text = DOC.read_text(encoding="utf-8")
    section = text.split("## Quick start", 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"```bash\n(.*?)```", section, re.S)
    assert blocks, "no bash block under '## Quick start'"
    return blocks[0]


def _lines(block: str) -> list[str]:
    """Every command in the block, commented-out example commands included."""
    out = []
    for raw in block.splitlines():
        line = raw.strip().lstrip("#").strip()
        line = line.split("  #", 1)[0].strip()
        if line.startswith(("pip install", "apyrobo ")):
            out.append(line)
    return out


def _extras_in(line: str) -> set[str]:
    match = re.search(r"apyrobo\[([^\]]+)\]", line)
    return {e.strip() for e in match.group(1).split(",")} if match else set()


def _extra_deps(extras: set[str]) -> list[str]:
    # pyproject.toml read with a regex: tomllib is 3.11+ and CI runs 3.10 too.
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = text.split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]
    table = {
        name: re.findall(r'"([^"]+)"', body)
        for name, body in re.findall(r"(?m)^([\w-]+)\s*=\s*\[(.*?)\]", section, re.S)
    }
    deps: list[str] = []
    for extra in extras:
        assert extra in table, f"docs name an extra pyproject.toml lacks: {extra}"
        deps.extend(table[extra])
    return deps


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_ws_target_has_an_install_line_that_brings_websockets():
    lines = _lines(_quickstart())
    assert any("ws://" in line for line in lines)
    installs = [line for line in lines if line.startswith("pip install")]
    covered = [
        line for line in installs
        if any(dep.startswith("websockets") for dep in _extra_deps(_extras_in(line)))
    ]
    assert covered, f"no quick-start install line brings websockets for ws://: {installs}"


def test_ws_server_command_in_doc_is_accepted_by_the_cli():
    serves = [line for line in _lines(_quickstart()) if line.startswith("apyrobo serve")]
    ws = [line for line in serves if "--transport websocket" in line]
    assert ws, "the quick start names no command that starts a WebSocket server"
    args = shlex.split(ws[0].rstrip("&").strip())[1:]
    result = subprocess.run(
        [*CLI, *args, "--help"], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--transport" in result.stdout


def test_ws_quickstart_passes_against_the_named_server():
    pytest.importorskip("websockets")
    serve = next(
        line for line in _lines(_quickstart())
        if line.startswith("apyrobo serve") and "--transport websocket" in line
    )
    check = next(line for line in _lines(_quickstart()) if "ws://localhost:8765" in line)
    port = _free_port()
    server = subprocess.Popen(
        [*CLI, *shlex.split(serve.rstrip("&").strip())[1:], "--ws-port", str(port)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            assert server.poll() is None, "the WebSocket server exited on start"
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail(f"nothing listening on port {port} after 30 s")
        args = shlex.split(check.replace("8765", str(port)))[1:]
        result = subprocess.run(
            [*CLI, *args], capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "CONFORMANT" in result.stdout
    finally:
        server.terminate()
        server.wait(timeout=10)


def test_ws_target_without_websockets_names_the_extra():
    """Without websockets the run says which extra to install and exits 2."""
    code = (
        "import sys; sys.modules['websockets'] = None\n"
        "sys.argv = ['apyrobo', 'conformance', 'ws://localhost:1']\n"
        "from apyrobo.cli import main\n"
        "main()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "apyrobo[websocket]" in result.stdout + result.stderr
