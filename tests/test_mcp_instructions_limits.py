"""apyrobo mcp's instructions name the limits it enforces.

Most clients hand a server's instructions to the model before any tool call,
so the model can plan inside the limits instead of learning them one refusal
at a time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from apyrobo.mcp_server import INSTRUCTIONS, build_server

ZONE = {"x_min": 2.0, "y_min": 2.0, "x_max": 4.0, "y_max": 4.0}
MODERN = "2026-07-28"


def initialize(server) -> str:
    resp = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
    })
    return resp["result"]["instructions"]


def discover(server) -> str:
    meta = {
        "io.modelcontextprotocol/protocolVersion": MODERN,
        "io.modelcontextprotocol/clientInfo": {"name": "t", "version": "1"},
        "io.modelcontextprotocol/clientCapabilities": {},
    }
    resp = server.handle({
        "jsonrpc": "2.0", "id": 2, "method": "server/discover", "params": {"_meta": meta},
    })
    return resp["result"]["instructions"]


def test_guidance_sentences_stay_first():
    text = initialize(build_server())
    assert text.startswith(INSTRUCTIONS)
    assert "stop is always allowed" in text


def test_default_policy_names_speeds_and_no_zones():
    server = build_server()
    policy = server.toolbox.enforcer.policy
    text = initialize(server)
    assert f"driving at most {policy.max_speed:g} m/s" in text
    assert f"turning at most {policy.max_angular_speed:g} rad/s" in text
    assert "no no-go zones" in text
    assert "allowed areas" not in text
    assert "confirm" not in text
    assert "robot_status" in text


def test_cli_speed_and_zone_are_named():
    text = initialize(build_server(max_speed=0.5, no_go_zones=[ZONE]))
    assert "driving at most 0.5 m/s" in text
    assert "(2, 2)-(4, 4)" in text
    assert "no no-go zones" not in text


def test_discover_carries_the_same_instructions():
    server = build_server(max_speed=0.5, no_go_zones=[ZONE])
    assert discover(server) == initialize(server)


def test_policy_file_zone_names_are_shown(tmp_path: Path):
    path = tmp_path / "home.yaml"
    path.write_text(
        "name: home\n"
        "max_speed: 0.3\n"
        "no_go_zones:\n"
        "  - {name: stairs, x_min: 0, y_min: 0, x_max: 1, y_max: 2}\n"
        "allowed_areas:\n"
        "  - {name: kitchen, x_min: -3, y_min: -3, x_max: 3, y_max: 3}\n",
        encoding="utf-8",
    )
    text = initialize(build_server(policy_file=path))
    assert "(home policy)" in text
    assert "driving at most 0.3 m/s" in text
    assert "stairs (0, 0)-(1, 2)" in text
    assert "allowed areas kitchen (-3, -3)-(3, 3)" in text


def test_allow_area_flag_is_named():
    area = {"x_min": -5.0, "y_min": -5.0, "x_max": 5.0, "y_max": 5.0}
    text = initialize(build_server(allowed_areas=[area]))
    assert "targets only inside the allowed areas (-5, -5)-(5, 5)" in text


def test_confirm_moves_is_said():
    text = initialize(build_server(confirm_moves=True))
    assert "asks the user to confirm before it runs" in text


@pytest.mark.parametrize("count", [4, 50])
def test_many_zones_are_cut_short(count):
    zones = [
        {"x_min": float(i), "y_min": 0.0, "x_max": i + 0.5, "y_max": 1.0}
        for i in range(count)
    ]
    server = build_server(no_go_zones=zones)
    limits = server.toolbox.limits_text(True)
    assert f"and {count - 3} more" in limits
    assert "(3, 0)" not in limits
    assert len(limits) < 600


def test_no_battery_floor_without_a_battery_monitor():
    # apyrobo mcp starts its enforcer without a battery monitor, so the
    # floor does not apply; the instructions must not claim it does.
    assert "battery" not in initialize(build_server())


def test_battery_floor_named_when_a_monitor_is_set():
    server = build_server()
    server.toolbox.enforcer._battery_monitor = object()
    floor = server.toolbox.enforcer.policy.min_battery_pct
    assert f"no new move below {floor:g}% battery" in initialize(server)


def test_speed_cap_not_claimed_when_it_does_not_reach_the_drive():
    server = build_server(max_speed=0.5)

    class Adapter:
        speed_limit_applies = False

    server.toolbox.robot._adapter = Adapter()
    text = initialize(server)
    assert "driving at most" not in text
    assert "the 0.5 m/s policy limit does not reach this robot's drive" in text


def test_limits_change_nothing_that_is_refused():
    server = build_server(max_speed=0.5, no_go_zones=[ZONE])
    initialize(server)
    resp = server.handle({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "move_to", "arguments": {"x": 3, "y": 3}},
    })
    assert resp["result"]["isError"] is True
