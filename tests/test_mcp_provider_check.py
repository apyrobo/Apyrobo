"""apyrobo mcp refuses to start with an LLM planner it cannot import.

Without litellm, --provider llm used to start, and every plan_task and
execute_task quietly used the rule planner; only a stderr line said so.
"""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
from argparse import Namespace

import pytest

from apyrobo import mcp_selftest
from apyrobo.mcp_server import build_server


def _no_litellm(monkeypatch):
    real = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "litellm":
            return None
        return real(name, *args, **kwargs)

    monkeypatch.setattr("apyrobo.mcp_server.importlib.util.find_spec", find_spec)


@pytest.mark.parametrize("provider", ["llm", "tool_calling"])
def test_build_server_refuses_an_llm_provider_without_litellm(monkeypatch, provider):
    _no_litellm(monkeypatch)
    with pytest.raises(ImportError) as info:
        build_server(provider=provider)
    message = str(info.value)
    assert f"--provider {provider} needs litellm" in message
    assert "pip install 'apyrobo[llm]'" in message
    assert "--provider rule" in message


def test_rule_provider_still_starts_without_litellm(monkeypatch):
    _no_litellm(monkeypatch)
    server = build_server(provider="rule")
    assert server.toolbox.enforcer is not None


def test_a_given_agent_is_not_second_guessed(monkeypatch):
    _no_litellm(monkeypatch)
    from apyrobo.skills.agent import Agent

    server = build_server(provider="llm", agent=Agent(provider="rule"))
    assert server.toolbox.agent is not None


def test_unknown_provider_is_still_its_own_error():
    with pytest.raises(ValueError, match="Unknown provider"):
        build_server(provider="nope")


def _selftest_args(**kw):
    base = dict(robot="mock://turtlebot4", policy=None, policy_file=None, max_speed=None,
                no_go=None, allow_area=None, confirm_moves=False, st_robot=None, st_policy=None,
                st_policy_file=None, st_max_speed=None, st_no_go=None, st_allow_area=None,
                provider="rule")
    base.update(kw)
    return Namespace(**base)


def test_selftest_fails_when_the_server_would_not_start(monkeypatch):
    _no_litellm(monkeypatch)
    out, err = io.StringIO(), io.StringIO()
    code = mcp_selftest.main(_selftest_args(provider="llm"), stdout=out, stderr=err)
    assert code == 1
    assert err.getvalue().startswith("Error: --provider llm needs litellm")
    assert "checks passed" not in out.getvalue()


def test_selftest_with_the_rule_provider_still_passes(monkeypatch):
    _no_litellm(monkeypatch)
    out, err = io.StringIO(), io.StringIO()
    code = mcp_selftest.main(_selftest_args(), stdout=out, stderr=err)
    assert code == 0, err.getvalue()


@pytest.mark.skipif(importlib.util.find_spec("litellm") is not None,
                    reason="litellm is installed, --provider llm would start")
def test_cli_exits_with_the_install_line():
    proc = subprocess.run(
        [sys.executable, "-m", "apyrobo", "mcp", "--provider", "llm"],
        input="", capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 1
    assert proc.stderr.startswith("Error: --provider llm needs litellm")
    assert "Traceback" not in proc.stderr
