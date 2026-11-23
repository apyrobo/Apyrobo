"""``apyrobo mcp config zed``: the ``context_servers`` entry for Zed's settings.

Zed's MCP docs (zed.dev/docs/ai/mcp, read 2026-11-23) give the entry as
``{"context_servers": {name: {"command", "args", "env"}}}`` in the settings
file opened with ``zed: open settings file``, and say Zed does not support
elicitation, which ``--confirm-moves`` needs.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from apyrobo import mcp_clients
from apyrobo.mcp_clients import GIT_SOURCE


def _args(**kw):
    base = dict(client="zed", launcher="auto", name="apyrobo", cfg_robot=None,
                cfg_policy=None, cfg_policy_file=None, cfg_max_speed=None, cfg_no_go=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _run(args, found=None):
    out, err = io.StringIO(), io.StringIO()
    code = mcp_clients.main(args, stdout=out, stderr=err,
                            which=lambda name: (found or {}).get(name))
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def before_pypi(monkeypatch):
    monkeypatch.setattr(mcp_clients, "__version__", "4.1.0")


def test_zed_prints_a_context_servers_entry(tmp_path, before_pypi):
    uvx = tmp_path / "uvx"
    uvx.write_text("")
    code, out, err = _run(_args(cfg_max_speed=0.5, cfg_no_go=["2,2,4,4"]), {"uvx": str(uvx)})
    assert code == 0, err
    block = json.loads(out)
    assert list(block) == ["context_servers"]
    assert block["context_servers"]["apyrobo"] == {
        "command": str(uvx.resolve()),
        "args": ["--from", GIT_SOURCE, "apyrobo", "mcp", "--max-speed", "0.5",
                 "--no-go", "2,2,4,4"],
        "env": {},
    }


def test_zed_hints_name_the_settings_file_and_the_approval_setting():
    code, _, err = _run(_args(launcher="uvx"))
    assert code == 0
    assert "zed: open settings file" in err
    assert 'Keep any other servers already under "context_servers".' in err
    assert "mcpServers" not in err
    assert 'agent.tool_permissions.default is "confirm"' in err
    assert '"mcp:apyrobo:move_to"' in err


def test_zed_hint_uses_the_chosen_name():
    code, out, err = _run(_args(launcher="uvx", name="tb4"))
    assert code == 0
    assert list(json.loads(out)["context_servers"]) == ["tb4"]
    assert '"mcp:tb4:move_to"' in err


@pytest.mark.parametrize("flag", ["confirm_moves", "cfg_confirm_moves"])
def test_zed_refuses_confirm_moves_on_either_side_of_config(flag):
    code, out, err = _run(_args(launcher="uvx", **{flag: True}))
    assert code == 1 and out == ""
    lines = err.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("Error: ")
    assert "Zed does not support MCP elicitation" in err
    assert "every move would be refused" in err


def test_other_clients_keep_confirm_moves():
    code, out, err = _run(_args(client="cursor", launcher="uvx", cfg_confirm_moves=True))
    assert code == 0, err
    assert "--confirm-moves" in json.loads(out)["mcpServers"]["apyrobo"]["args"]


def test_zed_checks_flags_like_every_client():
    code, out, err = _run(_args(cfg_max_speed=-1.0))
    assert code == 1 and out == ""
    assert "--max-speed must be positive" in err


def test_zed_over_ssh_runs_the_server_on_the_robot():
    code, out, err = _run(_args(cfg_ssh="robot", cfg_robot="mock://tb"), {"ssh": "/usr/bin/ssh"})
    assert code == 0, err
    entry = json.loads(out)["context_servers"]["apyrobo"]
    assert entry["command"] == str(Path("/usr/bin/ssh").resolve())
    assert entry["args"][-1] == "apyrobo mcp --robot mock://tb"
    assert entry["env"] == {}


def test_cli_accepts_zed(tmp_path):
    exe = tmp_path / "apyrobo"
    exe.write_text("")
    exe.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}"}
    result = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", "mcp", "--max-speed", "0.5", "config", "zed",
         "--launcher", "apyrobo"],
        capture_output=True, text=True, cwd=tmp_path, timeout=60, env=env,
    )
    assert result.returncode == 0, result.stderr
    expected = Path(mcp_clients._own_script() or exe).resolve()
    assert json.loads(result.stdout)["context_servers"]["apyrobo"] == {
        "command": str(expected), "args": ["mcp", "--max-speed", "0.5"], "env": {}}
    refused = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", "mcp", "config", "zed", "--confirm-moves",
         "--launcher", "apyrobo"],
        capture_output=True, text=True, cwd=tmp_path, timeout=60, env=env,
    )
    assert refused.returncode == 1 and refused.stdout == ""
    assert "Traceback" not in refused.stderr
    assert "Zed does not support MCP elicitation" in refused.stderr
