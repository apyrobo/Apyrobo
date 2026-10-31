"""Tests for ``apyrobo mcp config`` (apyrobo/mcp_clients.py)."""

from __future__ import annotations

import io
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from apyrobo import mcp_clients
from apyrobo.mcp_clients import GIT_SOURCE, Launch


def _which(found: dict[str, str]):
    return lambda name: found.get(name)


def _args(client, **kw):
    base = dict(client=client, launcher="auto", name="apyrobo", cfg_robot=None,
                cfg_policy=None, cfg_policy_file=None, cfg_max_speed=None, cfg_no_go=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _run(args, which):
    out, err = io.StringIO(), io.StringIO()
    code = mcp_clients.main(args, stdout=out, stderr=err, which=which)
    return code, out.getvalue(), err.getvalue()


def test_claude_desktop_uses_absolute_uvx_and_passes_flags(tmp_path):
    uvx = tmp_path / "uvx"
    uvx.write_text("")
    code, out, err = _run(
        _args("claude-desktop", cfg_max_speed=0.5, cfg_no_go=["2,2,4,4"], cfg_robot="mock://tb"),
        _which({"uvx": str(uvx)}),
    )
    assert code == 0
    server = json.loads(out)["mcpServers"]["apyrobo"]
    assert server["command"] == str(uvx.resolve())
    assert Path(server["command"]).is_absolute()
    assert server["args"] == ["--from", GIT_SOURCE, "apyrobo", "mcp", "--robot", "mock://tb",
                              "--max-speed", "0.5", "--no-go", "2,2,4,4"]
    assert "fully quit Claude Desktop" in err


def test_falls_back_to_installed_apyrobo(tmp_path):
    exe = tmp_path / "apyrobo"
    exe.write_text("")
    code, out, _ = _run(_args("cursor"), _which({"apyrobo": str(exe)}))
    assert code == 0
    server = json.loads(out)["mcpServers"]["apyrobo"]
    assert server == {"command": str(exe.resolve()), "args": ["mcp"]}


def test_launcher_apyrobo_beats_uvx(tmp_path):
    flags: list[str] = []
    launch = mcp_clients.resolve_launch(
        flags, "apyrobo", _which({"uvx": "/x/uvx", "apyrobo": str(tmp_path / "apyrobo")}))
    assert launch.args == ["mcp"]
    assert launch.command.endswith("apyrobo")


def test_nothing_on_path_uses_bare_uvx_with_a_note():
    code, out, err = _run(_args("claude-desktop"), _which({}))
    assert code == 0
    assert json.loads(out)["mcpServers"]["apyrobo"]["command"] == "uvx"
    assert "Note:" in err and "which uvx" in err


def test_claude_code_prints_a_shell_command(tmp_path):
    launch = Launch("/opt/my tools/uvx",
                    ["--from", GIT_SOURCE, "apyrobo", "mcp", "--max-speed", "0.5"])
    text = mcp_clients.render("claude-code", launch, platform="linux")
    assert shlex.split(text) == ["claude", "mcp", "add", "apyrobo", "--", "/opt/my tools/uvx",
                                 "--from", GIT_SOURCE, "apyrobo", "mcp", "--max-speed", "0.5"]


def test_policy_file_is_made_absolute(tmp_path, monkeypatch):
    policy = tmp_path / "lab.json"
    policy.write_text("{}")
    monkeypatch.chdir(tmp_path)
    flags = mcp_clients.server_args(policy_file="lab.json")
    assert flags == ["--policy-file", str(policy.resolve())]


@pytest.mark.parametrize("kw, message", [
    ({"cfg_policy_file": "/no/such/policy.json"}, "policy file not found"),
    ({"cfg_no_go": ["1,2"]}, "no-go zone"),
    ({"cfg_max_speed": 0.0}, "--max-speed must be positive"),
])
def test_bad_flags_are_readable_errors(kw, message):
    code, out, err = _run(_args("cursor", **kw), _which({}))
    assert code == 1
    assert out == ""
    assert err.startswith("Error:") and message in err


@pytest.mark.parametrize("platform, expected", [
    ("darwin", "~/Library/Application Support/Claude/claude_desktop_config.json"),
    ("win32", r"%APPDATA%\Claude\claude_desktop_config.json"),
])
def test_claude_desktop_location_per_os(platform, expected):
    assert mcp_clients.config_location("claude-desktop", platform) == expected


def test_cli_prints_config_and_writes_nothing(tmp_path):
    env_home = tmp_path / "home"
    env_home.mkdir()
    proc = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", "mcp", "config", "cursor", "--launcher", "apyrobo",
         "--max-speed", "0.3"],
        capture_output=True, text=True, timeout=60,
        env={"HOME": str(env_home), "PATH": "",
             "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )
    assert proc.returncode == 0, proc.stderr
    server = json.loads(proc.stdout)["mcpServers"]["apyrobo"]
    assert server["args"] == ["mcp", "--max-speed", "0.3"]
    assert list(env_home.iterdir()) == []


def test_plain_mcp_command_still_serves(monkeypatch):
    import apyrobo.cli as cli

    seen = {}
    monkeypatch.setattr(cli, "cmd_mcp", lambda args: seen.update(vars(args)))
    monkeypatch.setattr(sys, "argv", ["apyrobo", "mcp", "--max-speed", "0.5", "--no-go", "1,1,2,2"])
    cli.main()
    assert seen["mcp_command"] is None
    assert seen["max_speed"] == 0.5
    assert seen["no_go"] == ["1,1,2,2"]
