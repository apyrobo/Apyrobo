"""apyrobo mcp --provider ... config prints a config whose server starts.

``apyrobo mcp --provider llm config cursor`` used to print ``uvx --from
git+... apyrobo mcp --provider llm``. uvx installs plain apyrobo, without
litellm, so the server refused to start inside the client. A typo such as
``--provider lmm`` was printed as given and failed the same way, where a GUI
client shows the user nothing.
"""
from __future__ import annotations

import io
import json
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from apyrobo import mcp_clients
from apyrobo.mcp_clients import GIT_SOURCE


def _args(client, **kw):
    base = dict(client=client, launcher="auto", name="apyrobo", cfg_robot=None,
                cfg_policy=None, cfg_policy_file=None, cfg_max_speed=None, cfg_no_go=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _run(args, found):
    out, err = io.StringIO(), io.StringIO()
    code = mcp_clients.main(args, stdout=out, stderr=err, which=lambda n: found.get(n))
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def before_pypi(monkeypatch):
    monkeypatch.setattr(mcp_clients, "__version__", "4.1.0")


@pytest.fixture
def on_pypi(monkeypatch):
    monkeypatch.setattr(mcp_clients, "__version__", "4.2.0")


@pytest.mark.parametrize("provider", ["llm", "tool_calling"])
def test_uvx_config_installs_the_llm_extra_from_github(tmp_path, before_pypi, provider):
    uvx = tmp_path / "uvx"
    uvx.write_text("")
    code, out, _ = _run(_args("cursor", provider=provider), {"uvx": str(uvx)})
    assert code == 0
    args = json.loads(out)["mcpServers"]["apyrobo"]["args"]
    assert args == ["--from", f"apyrobo[llm] @ {GIT_SOURCE}", "apyrobo", "mcp",
                    "--provider", provider]


def test_uvx_config_installs_the_llm_extra_from_pypi(tmp_path, on_pypi):
    uvx = tmp_path / "uvx"
    uvx.write_text("")
    code, out, _ = _run(_args("claude-code", provider="llm"), {"uvx": str(uvx)})
    assert code == 0
    argv = shlex.split(out.strip())
    assert argv[argv.index("--") + 2:] == ["--from", "apyrobo[llm]", "apyrobo", "mcp",
                                           "--provider", "llm"]


@pytest.mark.parametrize("provider", [None, "rule", "multi_turn"])
def test_other_providers_keep_the_plain_source(before_pypi, provider):
    flags = mcp_clients.server_args(provider=provider)
    assert mcp_clients._uvx_source(flags=flags) == ["--from", GIT_SOURCE]


def test_ssh_uvx_adds_the_extra_on_the_robot(before_pypi):
    flags = mcp_clients.server_args(provider="llm")
    argv = shlex.split(mcp_clients.remote_command(flags, "uvx"))
    assert argv == ["uvx", "--from", f"apyrobo[llm] @ {GIT_SOURCE}", "apyrobo", "mcp",
                    "--provider", "llm"]


def test_unknown_provider_is_refused_with_the_choices(tmp_path):
    uvx = tmp_path / "uvx"
    uvx.write_text("")
    code, out, err = _run(_args("cursor", provider="lmm"), {"uvx": str(uvx)})
    assert code == 1
    assert out == ""
    assert "unknown --provider 'lmm'" in err
    for name in ("rule", "auto", "llm", "tool_calling", "multi_turn"):
        assert name in err


def test_own_install_without_litellm_is_refused(monkeypatch, tmp_path):
    own = tmp_path / "venv" / "bin" / "apyrobo"
    own.parent.mkdir(parents=True)
    own.write_text("#!/bin/sh\n")
    own.chmod(0o755)
    monkeypatch.setattr(mcp_clients, "_own_script", lambda name="apyrobo": str(own))
    real_find_spec = mcp_clients.importlib.util.find_spec
    monkeypatch.setattr(mcp_clients.importlib.util, "find_spec",
                        lambda name, *a: None if name == "litellm" else real_find_spec(name, *a))
    code, out, err = _run(_args("cursor", provider="llm", launcher="apyrobo"),
                          {"apyrobo": str(own)})
    assert code == 1
    assert out == ""
    assert "needs litellm" in err and "pip install 'apyrobo[llm]'" in err


def test_own_install_with_litellm_is_accepted(monkeypatch, tmp_path):
    own = tmp_path / "venv" / "bin" / "apyrobo"
    own.parent.mkdir(parents=True)
    own.write_text("#!/bin/sh\n")
    own.chmod(0o755)
    monkeypatch.setattr(mcp_clients, "_own_script", lambda name="apyrobo": str(own))
    monkeypatch.setattr(mcp_clients.importlib.util, "find_spec", lambda name, *a: object())
    code, out, _ = _run(_args("cursor", provider="llm", launcher="apyrobo"),
                        {"apyrobo": str(own)})
    assert code == 0
    assert json.loads(out)["mcpServers"]["apyrobo"]["args"] == ["mcp", "--provider", "llm"]


def test_cli_refuses_a_provider_typo_before_config(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", "mcp", "--provider", "lmm", "config", "cursor"],
        capture_output=True, text=True, cwd=tmp_path, timeout=60,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert "unknown --provider 'lmm'" in result.stderr
