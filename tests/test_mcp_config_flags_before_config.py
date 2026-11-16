"""``apyrobo mcp <flags> config <client>`` keeps the flags given before "config".

Users build the server line first (``apyrobo mcp --max-speed 0.5 --no-go
2,2,4,4``) and then add ``config cursor``; the printed config must carry
those limits, as selftest and keepout already do.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from types import SimpleNamespace

from apyrobo import mcp_clients


def _args(**kw):
    base = dict(
        client="cursor", launcher="apyrobo", name="apyrobo",
        robot="mock://turtlebot4", policy=None, policy_file=None, max_speed=None,
        no_go=None, allow_area=None, confirm_moves=False, audit_log=None, provider="rule",
        cfg_robot=None, cfg_policy=None, cfg_policy_file=None, cfg_max_speed=None,
        cfg_no_go=None, cfg_allow_area=None, cfg_confirm_moves=False, cfg_audit_log=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _server_args(args):
    out, err = io.StringIO(), io.StringIO()
    code = mcp_clients.main(args, stdout=out, stderr=err,
                            which=lambda name: "/usr/bin/apyrobo" if name == "apyrobo" else None)
    assert code == 0, err.getvalue()
    return json.loads(out.getvalue())["mcpServers"]["apyrobo"]["args"]


def test_flags_before_config_reach_the_config():
    args = _server_args(_args(max_speed=0.5, no_go=["2,2,4,4"], allow_area=["0,0,10,6"],
                              confirm_moves=True, robot="ros2://burger", policy="strict"))
    assert args == ["mcp", "--robot", "ros2://burger", "--policy", "strict",
                    "--max-speed", "0.5", "--no-go", "2,2,4,4", "--allow-area", "0,0,10,6",
                    "--confirm-moves"]


def test_flag_after_config_wins_and_zones_from_both_sides_are_kept():
    args = _server_args(_args(max_speed=0.5, cfg_max_speed=0.3,
                              no_go=["0,0,1,1"], cfg_no_go=["5,5,6,6"]))
    assert args == ["mcp", "--max-speed", "0.3", "--no-go", "0,0,1,1", "--no-go", "5,5,6,6"]


def test_default_robot_and_rule_planner_add_nothing():
    assert _server_args(_args()) == ["mcp"]


def test_planner_before_config_is_passed_on():
    assert _server_args(_args(provider="openai")) == ["mcp", "--provider", "openai"]


def test_bad_flag_before_config_is_refused_not_dropped(tmp_path):
    out, err = io.StringIO(), io.StringIO()
    code = mcp_clients.main(_args(policy_file=str(tmp_path / "missing.yaml")),
                            stdout=out, stderr=err, which=lambda name: None)
    assert code == 1
    assert "policy file not found" in err.getvalue()
    assert out.getvalue() == ""


def test_cli_keeps_a_zone_given_before_config():
    out = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", "mcp", "--max-speed", "0.5", "--no-go", "2,2,4,4",
         "config", "cursor", "--launcher", "apyrobo"],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    args = json.loads(out.stdout)["mcpServers"]["apyrobo"]["args"]
    assert args[-4:] == ["--max-speed", "0.5", "--no-go", "2,2,4,4"]


def test_cli_refuses_a_typo_given_before_config():
    out = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", "mcp", "--no-go", "1,x,2,2", "config", "cursor"],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 1
    assert "y_min 'x' is not a number" in out.stderr
