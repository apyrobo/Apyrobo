"""A bad --robot URI gets a one-line error and exit 1, never a traceback.

discover, plan, execute, listen, voice, serve and skill compose used to let
Robot.discover's ValueError escape, so a typo like 'mock://' or 'ros://bot'
printed a Python stack. connect, test-skill and dashboard already printed a
plain error; these now match them.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from apyrobo import cli

BAD_URIS = [
    ("notascheme", "Invalid robot URI"),
    ("mock://", "Invalid robot URI"),
    ("nosuch://bot", "No adapter registered for scheme 'nosuch'"),
]

COMMANDS = [
    ["discover", "{uri}"],
    ["plan", "go to (1, 2)", "--robot", "{uri}"],
    ["execute", "go to (1, 2)", "--robot", "{uri}"],
    ["listen", "--robot", "{uri}", "--adapter", "mock"],
    ["voice", "--robot", "{uri}"],
    ["serve", "--robot", "{uri}"],
    ["skill", "compose", "--robot", "{uri}"],
]


def _argv(cmd: list[str], uri: str) -> list[str]:
    return ["apyrobo"] + [a.format(uri=uri) for a in cmd]


@pytest.mark.parametrize("uri,reason", BAD_URIS)
@pytest.mark.parametrize("cmd", COMMANDS, ids=lambda c: " ".join(c[:2]))
def test_bad_robot_uri_exits_1_with_one_line_error(monkeypatch, capsys, cmd, uri, reason):
    monkeypatch.setattr(sys, "argv", _argv(cmd, uri))
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert f"Error: could not connect to {uri!r}" in err
    assert reason in err
    assert "Traceback" not in err


def test_discover_or_exit_returns_the_robot_for_a_good_uri():
    robot = cli._discover_or_exit("mock://bot")
    assert robot.capabilities().name


@pytest.mark.parametrize("cmd", [COMMANDS[0], COMMANDS[2]], ids=["discover", "execute"])
def test_installed_cli_prints_no_traceback(cmd):
    proc = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli"] + _argv(cmd, "mock://")[1:],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert "Error: could not connect to 'mock://'" in proc.stderr
