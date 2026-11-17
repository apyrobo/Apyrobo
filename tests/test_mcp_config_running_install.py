"""apyrobo mcp config points a client at the apyrobo install that printed it.

A user who runs ``.venv/bin/apyrobo mcp config claude-desktop --launcher
apyrobo`` without activating the venv used to get whichever ``apyrobo`` came
first on PATH (another version, maybe one without ``apyrobo mcp``), or a bare
``apyrobo`` that GUI clients can't start.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import pytest

from apyrobo import mcp_clients

EXE = "apyrobo.exe" if sys.platform == "win32" else "apyrobo"


def _script(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / EXE
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    return script


@pytest.fixture
def scripts(tmp_path, monkeypatch):
    """The running Python's scripts folder is tmp/venv/bin; PATH has another apyrobo."""
    venv_bin = tmp_path / "venv" / "bin"
    monkeypatch.setattr(mcp_clients.sysconfig, "get_path",
                        lambda name, *a, **k: str(venv_bin) if name == "scripts" else None)
    monkeypatch.setattr(mcp_clients.sys, "executable", str(tmp_path / "elsewhere" / "python"))
    other = _script(tmp_path / "usr" / "bin")
    found = {"apyrobo": str(other), "ssh": "/usr/bin/ssh"}
    monkeypatch.setattr(mcp_clients.shutil, "which", lambda name: found.get(name))
    return venv_bin, other


def test_the_running_install_wins_over_path(scripts):
    venv_bin, _ = scripts
    own = _script(venv_bin)
    assert mcp_clients.find_executable("apyrobo") == str(own)
    launch = mcp_clients.resolve_launch(["--max-speed", "0.5"], "apyrobo")
    assert launch.command == str(own.resolve())
    assert launch.args == ["mcp", "--max-speed", "0.5"]
    assert launch.note is None


def test_auto_without_uvx_uses_the_running_install(scripts):
    venv_bin, _ = scripts
    own = _script(venv_bin)
    assert mcp_clients.resolve_launch([], "auto").command == str(own.resolve())


def test_falls_back_to_path_when_the_running_python_has_no_script(scripts):
    _, other = scripts
    assert mcp_clients.find_executable("apyrobo") == str(other)
    assert mcp_clients.resolve_launch([], "apyrobo").command == str(other.resolve())


def test_script_beside_the_interpreter_counts(scripts, tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_clients.sysconfig, "get_path", lambda name, *a, **k: None)
    own = _script(tmp_path / "conda" / "bin")
    monkeypatch.setattr(mcp_clients.sys, "executable", str(tmp_path / "conda" / "bin" / "python"))
    assert mcp_clients.find_executable("apyrobo") == str(own)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX execute bit")
def test_a_non_executable_file_is_not_used(scripts):
    venv_bin, other = scripts
    stub = _script(venv_bin)
    stub.chmod(0o644)
    assert not os.access(stub, os.X_OK)
    assert mcp_clients.find_executable("apyrobo") == str(other)


def test_other_tools_still_come_from_path(scripts):
    venv_bin, _ = scripts
    _script(venv_bin)
    assert mcp_clients.find_executable("uvx") is None
    assert mcp_clients.find_executable("ssh") == "/usr/bin/ssh"


def test_printed_config_names_the_running_install(scripts):
    venv_bin, _ = scripts
    own = _script(venv_bin)
    args = argparse.Namespace(client="claude-desktop", launcher="apyrobo", name=None,
                              robot=None, policy=None, policy_file=None, max_speed=0.5,
                              no_go=None, allow_area=None, confirm_moves=False,
                              audit_log=None)
    out, err = io.StringIO(), io.StringIO()
    assert mcp_clients.main(args, stdout=out, stderr=err) == 0, err.getvalue()
    text = out.getvalue()
    block = json.loads(text[text.index("{"):text.rindex("}") + 1])
    entry = block["mcpServers"]["apyrobo"]
    assert entry["command"] == str(own.resolve())
    assert entry["args"][:1] == ["mcp"]
