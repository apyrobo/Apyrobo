"""MCP safety refusals carry a stable code beside their text (structuredContent)."""

from __future__ import annotations

import json

import pytest

from apyrobo.mcp_server import build_server

ZONE = {"x_min": 2.0, "y_min": 2.0, "x_max": 4.0, "y_max": 4.0}
AREA = {"x_min": -1.0, "y_min": -1.0, "x_max": 6.0, "y_max": 6.0}


def _server(version="2025-11-25", **kwargs):
    server = build_server(max_speed=0.5, **kwargs)
    server.handle({
        "jsonrpc": "2.0", "id": 0, "method": "initialize",
        "params": {"protocolVersion": version, "capabilities": {}},
    })
    return server


def _call(server, name, arguments):
    resp = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    })
    return resp["result"]


def test_collision_zone_refusal_has_code_and_zone():
    result = _call(_server(no_go_zones=[ZONE]), "move_to", {"x": 3, "y": 3})
    assert result["isError"] is True
    structured = result["structuredContent"]
    assert structured["status"] == "refused"
    assert structured["code"] == "collision_zone"
    assert structured["policy"] == "default+cli"
    assert structured["violation"]["zone"] == ZONE
    assert structured["violation"]["requested_position"] == [3, 3]
    # The text is unchanged, so readers of the sentence see no difference.
    text = result["content"][0]["text"]
    assert text.startswith(f"REFUSED by safety policy '{structured['policy']}': ")
    assert text.endswith("The robot did not move. Choose a different target.")
    json.dumps(result)  # plain JSON all the way down


def test_outside_allowed_area_refusal_has_code_and_areas():
    result = _call(_server(allowed_areas=[AREA]), "move_to", {"x": 9, "y": 9})
    assert result["isError"] is True
    structured = result["structuredContent"]
    assert structured["code"] == "outside_allowed_area"
    assert structured["violation"]["allowed_areas"] == [AREA]


def test_refusal_with_nothing_recorded_gets_the_generic_code(monkeypatch):
    from apyrobo.safety.enforcer import SafetyViolation

    server = _server()

    def refuse(**kwargs):
        raise SafetyViolation("no route")

    monkeypatch.setattr(server.toolbox.enforcer, "move", refuse)
    result = _call(server, "move_to", {"x": 1, "y": 1})
    assert result["isError"] is True
    assert result["structuredContent"]["code"] == "safety_violation"
    assert result["structuredContent"]["violation"] == {}


@pytest.mark.parametrize("version", ["2025-03-26", "2024-11-05"])
def test_older_protocols_get_text_only(version):
    result = _call(_server(version, no_go_zones=[ZONE]), "move_to", {"x": 3, "y": 3})
    assert result["isError"] is True
    assert "structuredContent" not in result
    assert result["content"][0]["text"].startswith("REFUSED by safety policy")


def test_text_is_identical_with_and_without_structured_content():
    new = _call(_server("2025-11-25", no_go_zones=[ZONE]), "move_to", {"x": 3, "y": 3})
    old = _call(_server("2025-03-26", no_go_zones=[ZONE]), "move_to", {"x": 3, "y": 3})
    assert new["content"] == old["content"]


def test_successful_and_argument_errors_carry_no_structured_content():
    server = _server(no_go_zones=[ZONE])
    assert "structuredContent" not in _call(server, "move_to", {"x": 1, "y": 1})
    bad = _call(server, "move_to", {"x": "far", "y": 1})
    assert bad["isError"] is True
    assert "structuredContent" not in bad
