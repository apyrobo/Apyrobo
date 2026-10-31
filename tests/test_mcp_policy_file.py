"""Tests for `apyrobo mcp --policy-file` — a safety policy loaded from JSON or YAML."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from apyrobo.mcp_server import build_server, load_policy_file

YAML_POLICY = """\
name: warehouse
base: default
max_speed: 0.8
max_angular_speed: 1.2
no_go_zones:
  - name: loading-dock
    x_min: 2
    y_min: 2
    x_max: 4
    y_max: 4
  - "10,10,12,12"
"""


def call(server, name, arguments=None):
    resp = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    })
    return resp["result"]


@pytest.fixture
def yaml_file(tmp_path):
    p = tmp_path / "warehouse.yaml"
    p.write_text(YAML_POLICY)
    return p


def test_yaml_file_loads(yaml_file):
    cfg = load_policy_file(yaml_file)
    assert cfg["name"] == "warehouse"
    assert cfg["max_speed"] == 0.8
    assert cfg["max_angular_speed"] == 1.2
    assert cfg["no_go_zones"] == [
        {"name": "loading-dock", "x_min": 2.0, "y_min": 2.0, "x_max": 4.0, "y_max": 4.0},
        {"x_min": 10.0, "y_min": 10.0, "x_max": 12.0, "y_max": 12.0},
    ]


def test_json_file_loads(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"max_speed": 0.3, "no_go_zones": ["1,1,2,2"]}))
    cfg = load_policy_file(p)
    assert cfg == {
        "max_speed": 0.3,
        "no_go_zones": [{"x_min": 1.0, "y_min": 1.0, "x_max": 2.0, "y_max": 2.0}],
    }


def test_server_uses_file_policy(yaml_file):
    srv = build_server(policy_file=yaml_file)
    pol = srv.toolbox.enforcer.policy
    assert pol.name == "warehouse"
    assert pol.max_speed == 0.8
    assert pol.max_angular_speed == 1.2
    assert len(pol.collision_zones) == 2


def test_file_speed_limit_clamps_moves(yaml_file):
    srv = build_server(policy_file=yaml_file)
    res = call(srv, "move_to", {"x": 1.0, "y": 0.0, "speed": 3.0})
    assert not res.get("isError")
    report = call(srv, "safety_report")
    text = json.dumps(report)
    assert "0.8" in text


def test_named_zone_refuses_moves_and_names_the_zone(yaml_file):
    srv = build_server(policy_file=yaml_file)
    res = call(srv, "move_to", {"x": 3.0, "y": 3.0})
    assert res["isError"] is True
    assert "loading-dock" in res["content"][0]["text"]


def test_cli_flags_override_the_file(yaml_file):
    zone = {"x_min": 20.0, "y_min": 20.0, "x_max": 21.0, "y_max": 21.0}
    srv = build_server(policy_file=yaml_file, max_speed=0.2, no_go_zones=[zone])
    pol = srv.toolbox.enforcer.policy
    assert pol.max_speed == 0.2
    assert pol.name == "warehouse+cli"
    assert len(pol.collision_zones) == 3  # file zones kept, flag zone added


def test_policy_flag_overrides_file_base(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("base: default\n")
    strict = build_server(policy="strict").toolbox.enforcer.policy
    srv = build_server(policy="strict", policy_file=p)
    assert srv.toolbox.enforcer.policy.max_speed == strict.max_speed


def test_file_base_is_used_when_no_flag(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("base: strict\n")
    strict = build_server(policy="strict").toolbox.enforcer.policy
    assert build_server(policy_file=p).toolbox.enforcer.policy.max_speed == strict.max_speed


@pytest.mark.parametrize(
    ("content", "suffix", "message"),
    [
        ("max_sped: 1\n", ".yaml", "unknown keys ['max_sped']"),
        ("max_speed: -1\n", ".yaml", "max_speed must be a positive number"),
        ("max_speed: fast\n", ".yaml", "max_speed must be a positive number"),
        ("- 1\n- 2\n", ".yaml", "mapping of settings"),
        ("no_go_zones: {x_min: 1}\n", ".yaml", "no_go_zones must be a list"),
        ("no_go_zones:\n  - {x_min: 1, y_min: 1, x_max: 2}\n", ".yaml", "missing ['y_max']"),
        ("no_go_zones:\n  - {x_min: 3, y_min: 1, x_max: 2, y_max: 2}\n", ".yaml", "min > max"),
        ("no_go_zones:\n  - '1,2,3'\n", ".yaml", "no_go_zones[0]"),
        ("no_go_zones:\n  - {x_min: 1, y_min: 1, x_max: 2, y_max: 2, z: 0}\n", ".yaml",
         "unknown keys ['z']"),
        ("max_speed: [1\n", ".yaml", "not valid YAML"),
        ("{\"max_speed\": 1,}", ".json", "not valid JSON"),
        ("base: yolo\n", ".yaml", "unknown policy 'yolo'"),
    ],
)
def test_bad_files_are_readable_errors(tmp_path, content, suffix, message):
    p = tmp_path / f"bad{suffix}"
    p.write_text(content)
    with pytest.raises(ValueError) as exc:
        build_server(policy_file=p)
    assert message in str(exc.value)


def test_missing_file_is_a_readable_error(tmp_path):
    with pytest.raises(ValueError, match="cannot read policy file"):
        load_policy_file(tmp_path / "nope.yaml")


def test_cli_bad_policy_file_exits_with_error(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("max_sped: 1\n")
    proc = subprocess.run(
        [sys.executable, "-m", "apyrobo", "mcp", "--policy-file", str(p)],
        input="", capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 1
    assert "unknown keys ['max_sped']" in proc.stderr


def test_cli_policy_file_reaches_the_server(yaml_file):
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "t", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "safety_report", "arguments": {}}},
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "apyrobo", "mcp", "--policy-file", str(yaml_file)],
        input="\n".join(json.dumps(m) for m in msgs) + "\n",
        capture_output=True, text=True, timeout=60,
    )
    assert "policy=warehouse" in proc.stderr
    assert "no_go_zones=2" in proc.stderr
    assert "loading-dock" in proc.stdout
