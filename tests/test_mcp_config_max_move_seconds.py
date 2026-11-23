"""``apyrobo mcp config`` carries ``--max-move-seconds`` into the printed config.

The flag was parsed on ``apyrobo mcp`` and then left out of the config, so a
user who asked for a move time limit got a server with none, and exit 0.
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


def _args(**kw):
    base = dict(
        client="cursor", launcher="apyrobo", name="apyrobo",
        robot="mock://turtlebot4", policy=None, policy_file=None, max_speed=None,
        no_go=None, allow_area=None, confirm_moves=False, audit_log=None, provider="rule",
        max_move_seconds=None,
        cfg_robot=None, cfg_policy=None, cfg_policy_file=None, cfg_max_speed=None,
        cfg_no_go=None, cfg_allow_area=None, cfg_confirm_moves=False, cfg_audit_log=None,
        cfg_max_move_seconds=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _which(name):
    return {"apyrobo": "/usr/bin/apyrobo", "ssh": "/usr/bin/ssh"}.get(name)


def _run(args):
    out, err = io.StringIO(), io.StringIO()
    code = mcp_clients.main(args, stdout=out, stderr=err, which=_which)
    return code, out.getvalue(), err.getvalue()


def _server_args(args):
    code, out, err = _run(args)
    assert code == 0, err
    return json.loads(out)["mcpServers"]["apyrobo"]["args"]


def test_flag_before_config_reaches_the_config():
    assert _server_args(_args(max_move_seconds=30.0)) == ["mcp", "--max-move-seconds", "30"]


def test_flag_after_config_reaches_the_config_and_wins():
    args = _server_args(_args(max_move_seconds=30.0, cfg_max_move_seconds=12.5))
    assert args == ["mcp", "--max-move-seconds", "12.5"]


def test_value_reads_back_as_the_same_number():
    args = _server_args(_args(cfg_max_move_seconds=1234567.0))
    assert float(args[-1]) == 1234567.0


@pytest.mark.parametrize("value", [0.0, -5.0, float("nan"), float("inf")])
def test_value_the_server_would_refuse_is_refused(value):
    code, out, err = _run(_args(cfg_max_move_seconds=value))
    assert code == 1
    assert out == ""
    assert err.startswith("Error:") and "--max-move-seconds must be positive" in err


def test_ssh_config_passes_the_flag_to_the_remote_server():
    code, out, err = _run(_args(cfg_ssh="ubuntu@robot.local", cfg_ssh_setup=None,
                                cfg_robot="ros2://myrobot", max_move_seconds=20.0))
    assert code == 0, err
    entry = json.loads(out)["mcpServers"]["apyrobo"]
    remote = shlex.split(entry["args"][-1])
    assert remote[-2:] == ["--max-move-seconds", "20"]


@pytest.mark.parametrize("argv", [
    ["mcp", "--max-move-seconds", "30", "config", "cursor", "--launcher", "apyrobo"],
    ["mcp", "config", "cursor", "--launcher", "apyrobo", "--max-move-seconds", "30"],
])
def test_cli_prints_the_flag_on_either_side(argv):
    proc = subprocess.run([sys.executable, "-m", "apyrobo.cli", *argv],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    args = json.loads(proc.stdout)["mcpServers"]["apyrobo"]["args"]
    assert args[-2:] == ["--max-move-seconds", "30"]
