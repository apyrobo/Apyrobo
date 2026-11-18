"""``--provider`` works after ``config`` and ``selftest``, like every other server flag.

``apyrobo mcp config claude-code --provider llm`` and ``apyrobo mcp selftest
--provider llm`` used to stop with "unrecognized arguments: --provider" (exit
2), although --max-speed, --no-go and the rest work on either side.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", *argv],
        capture_output=True, text=True, timeout=120,
    )


def test_config_accepts_provider_after_the_client():
    proc = _run("mcp", "config", "claude-code", "--launcher", "apyrobo",
                "--provider", "multi_turn")
    assert proc.returncode == 0, proc.stderr
    assert "unrecognized arguments" not in proc.stderr
    assert "mcp --provider multi_turn" in proc.stdout


def test_config_provider_after_wins_over_before():
    proc = _run("mcp", "--provider", "multi_turn", "config", "claude-code",
                "--launcher", "apyrobo", "--provider", "auto")
    assert proc.returncode == 0, proc.stderr
    assert "mcp --provider auto" in proc.stdout
    assert "multi_turn" not in proc.stdout


def test_config_provider_before_is_kept_when_none_after():
    proc = _run("mcp", "--provider", "multi_turn", "config", "claude-code",
                "--launcher", "apyrobo")
    assert proc.returncode == 0, proc.stderr
    assert "mcp --provider multi_turn" in proc.stdout


def test_config_refuses_an_unknown_provider_after_the_client():
    proc = _run("mcp", "config", "claude-code", "--launcher", "apyrobo",
                "--provider", "bogus")
    assert proc.returncode == 1
    assert "unknown --provider 'bogus'" in proc.stderr


@pytest.mark.parametrize("provider", ["rule", "auto", "multi_turn"])
def test_selftest_accepts_provider_after_selftest(provider):
    proc = _run("mcp", "selftest", "--provider", provider)
    assert proc.returncode == 0, proc.stderr
    assert "unrecognized arguments" not in proc.stderr
    assert "3/3 checks passed" in proc.stdout


def test_selftest_refuses_an_unknown_provider_after_selftest():
    proc = _run("mcp", "selftest", "--provider", "bogus")
    assert proc.returncode == 1
    assert "unknown --provider 'bogus'" in proc.stderr


def test_selftest_provider_after_wins_over_before():
    proc = _run("mcp", "--provider", "bogus", "selftest", "--provider", "rule")
    assert proc.returncode == 0, proc.stderr
    assert "3/3 checks passed" in proc.stdout


@pytest.mark.parametrize("sub", ["config", "selftest"])
def test_help_names_the_provider_value_plainly(sub):
    # The flag is stored under cfg_/st_ so the one after the subcommand wins;
    # --help must still read "--provider PROVIDER", like `apyrobo mcp --help`.
    proc = _run("mcp", sub, "--help")
    assert proc.returncode == 0, proc.stderr
    assert "--provider PROVIDER" in proc.stdout
    assert "CFG_PROVIDER" not in proc.stdout
    assert "ST_PROVIDER" not in proc.stdout
