"""The example policies in examples/policies/ load and do what their comments say."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apyrobo.mcp_server import build_server, load_policy_file

POLICY_DIR = Path(__file__).resolve().parent.parent / "examples" / "policies"
POLICIES = sorted(POLICY_DIR.glob("*.yaml"))


def call(server, name, arguments=None):
    resp = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    })
    return resp["result"]


def zone_center(zone):
    return (zone["x_min"] + zone["x_max"]) / 2, (zone["y_min"] + zone["y_max"]) / 2


def test_three_example_policies_ship():
    assert [p.name for p in POLICIES] == [
        "home_stairs.yaml", "lab_bench.yaml", "warehouse_aisle.yaml",
    ]


@pytest.mark.parametrize("path", POLICIES, ids=lambda p: p.stem)
def test_example_loads_with_named_zones_and_limits(path):
    cfg = load_policy_file(path)
    assert cfg["name"]
    assert 0 < cfg["max_speed"] < 1.0
    assert cfg["max_angular_speed"] > 0
    assert any("name" in z for z in cfg["no_go_zones"])
    # Each file explains the rectangle-only, target-only behaviour.
    text = path.read_text()
    assert "axis-aligned rectangle" in text
    assert "TARGET point" in text


@pytest.mark.parametrize("path", POLICIES, ids=lambda p: p.stem)
def test_example_refuses_a_move_into_each_zone(path):
    cfg = load_policy_file(path)
    srv = build_server(policy_file=path)
    for zone in cfg["no_go_zones"]:
        x, y = zone_center(zone)
        res = call(srv, "move_to", {"x": x, "y": y})
        assert res["isError"] is True, zone
        if "name" in zone:
            assert zone["name"] in res["content"][0]["text"]


@pytest.mark.parametrize("path", POLICIES, ids=lambda p: p.stem)
def test_example_clamps_an_over_limit_move(path):
    cfg = load_policy_file(path)
    srv = build_server(policy_file=path)
    res = call(srv, "move_to", {"x": 0.5, "y": -0.5, "speed": 5.0})
    assert not res.get("isError"), res
    moved = json.loads(res["content"][0]["text"])
    assert moved["status"] == "moved"
    assert moved["safety_interventions"] == [{
        "type": "speed_clamped", "requested": 5.0,
        "enforced": cfg["max_speed"], "max_allowed": cfg["max_speed"],
    }]


def test_warehouse_example_refuses_a_target_off_the_allowed_floor():
    path = POLICY_DIR / "warehouse_aisle.yaml"
    cfg = load_policy_file(path)
    assert [a["name"] for a in cfg["allowed_areas"]] == ["aisle-floor"]
    srv = build_server(policy_file=path)
    res = call(srv, "move_to", {"x": 100.0, "y": 0.0})
    assert res["isError"] is True
    assert "outside every allowed area" in res["content"][0]["text"]
