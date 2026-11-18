"""``apyrobo mcp config`` refuses an empty ``--ssh``; config and selftest flags have help.

``apyrobo mcp config claude-code --ssh ""`` used to print a config for this
machine, as if ``--ssh`` had been left out, and an empty ``--ssh-setup`` was
dropped silently. ``apyrobo mcp config --help`` and ``apyrobo mcp selftest
--help`` listed the server flags with no text beside them.
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys

import pytest

from apyrobo import mcp_clients


def _run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", *argv],
        capture_output=True, text=True, timeout=120,
    )


def _config(**kw) -> tuple[int, str, str]:
    ns = argparse.Namespace(client="claude-code", launcher="apyrobo", name="apyrobo",
                            cfg_ssh=None, cfg_ssh_setup=None)
    for key, value in kw.items():
        setattr(ns, key, value)
    out, err = io.StringIO(), io.StringIO()
    code = mcp_clients.main(ns, stdout=out, stderr=err,
                            which=lambda name: f"/usr/bin/{name}")
    return code, out.getvalue(), err.getvalue()


@pytest.mark.parametrize("value", ["", "  "])
def test_empty_ssh_is_refused(value):
    code, out, err = _config(cfg_ssh=value)
    assert code == 1
    assert out == ""
    assert "--ssh is empty" in err
    assert "user@host" in err


@pytest.mark.parametrize("ssh", [None, "robot@turtlebot"])
@pytest.mark.parametrize("value", ["", " "])
def test_empty_ssh_setup_is_refused(ssh, value):
    code, out, err = _config(cfg_ssh=ssh, cfg_ssh_setup=value)
    assert code == 1
    assert out == ""
    assert "--ssh-setup is empty" in err


def test_ssh_and_setup_still_work():
    code, out, err = _config(cfg_ssh="robot@turtlebot",
                             cfg_ssh_setup="/opt/ros/jazzy/setup.bash")
    assert code == 0, err
    assert "robot@turtlebot" in out
    assert "/opt/ros/jazzy/setup.bash" in out


def test_no_ssh_still_prints_a_local_config():
    code, out, err = _config()
    assert code == 0, err
    assert "claude mcp add apyrobo" in out
    assert "ssh" not in out


def test_cli_refuses_empty_ssh():
    proc = _run("mcp", "config", "claude-code", "--ssh", "")
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert "--ssh is empty" in proc.stderr


def _flag_help(text: str, flag: str) -> str:
    """The help text argparse prints for ``flag`` (empty when it has none)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(rf"^\s+{re.escape(flag)}(?: \S+)?(\s{{2,}}(.*))?$", line)
        if not m:
            continue
        if m.group(2):
            return m.group(2)
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        return nxt.strip() if nxt.startswith(" " * 20) and not nxt.strip().startswith("-") else ""
    raise AssertionError(f"{flag} not in help")


CONFIG_FLAGS = ["--robot", "--policy", "--policy-file", "--max-speed", "--no-go",
                "--allow-area", "--confirm-moves", "--audit-log", "--provider",
                "--ssh", "--ssh-setup", "--launcher", "--name"]
SELFTEST_FLAGS = ["--robot", "--policy", "--policy-file", "--max-speed", "--no-go",
                  "--allow-area", "--confirm-moves", "--audit-log", "--provider"]


@pytest.mark.parametrize("sub,flags", [("config", CONFIG_FLAGS),
                                       ("selftest", SELFTEST_FLAGS)])
def test_every_flag_has_help(sub, flags):
    proc = _run("mcp", sub, "--help")
    assert proc.returncode == 0, proc.stderr
    missing = [flag for flag in flags if not _flag_help(proc.stdout, flag)]
    assert missing == []
