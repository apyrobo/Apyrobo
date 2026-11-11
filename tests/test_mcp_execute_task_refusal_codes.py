"""execute_task safety refusals carry the same stable code as move_to (structuredContent)."""

from __future__ import annotations

import json

from apyrobo.mcp_server import build_server

ZONE = {"x_min": 2.0, "y_min": 2.0, "x_max": 4.0, "y_max": 4.0}


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


def test_first_step_refusal_has_code_zone_and_step():
    result = _call(_server(no_go_zones=[ZONE]), "execute_task", {"task": "go to (3, 3)"})
    assert result["isError"] is True
    structured = result["structuredContent"]
    assert structured["status"] == "refused"
    assert structured["code"] == "collision_zone"
    assert structured["policy"] == "default+cli"
    assert structured["violation"]["zone"] == ZONE
    assert structured["step"] == 1
    assert structured["steps_completed"] == 0
    assert structured["steps_total"] >= 1
    text = result["content"][0]["text"]
    assert text.startswith(f"REFUSED by safety policy '{structured['policy']}' at step 1: ")
    assert "The robot did not move. Choose a different target." in text
    json.dumps(result)  # plain JSON all the way down


def test_later_step_refusal_names_that_step():
    server = _server(no_go_zones=[ZONE])
    result = _call(server, "execute_task", {"task": "deliver from (1, 0) to (3, 3)"})
    assert result["isError"] is True
    structured = result["structuredContent"]
    assert structured["code"] == "collision_zone"
    assert structured["steps_completed"] >= 1
    assert structured["step"] == structured["steps_completed"] + 1
    text = result["content"][0]["text"]
    assert f"at step {structured['step']}: " in text
    assert "The robot did not move." not in text


def test_text_is_identical_with_and_without_structured_content():
    task = {"task": "go to (3, 3)"}
    new = _call(_server("2025-11-25", no_go_zones=[ZONE]), "execute_task", task)
    old = _call(_server("2025-03-26", no_go_zones=[ZONE]), "execute_task", task)
    assert "structuredContent" in new
    assert "structuredContent" not in old
    assert new["content"] == old["content"]


def test_completed_and_unplannable_tasks_carry_no_structured_content():
    server = _server(no_go_zones=[ZONE])
    done = _call(server, "execute_task", {"task": "go to (1, 1)"})
    assert done["isError"] is False
    assert "structuredContent" not in done
    unread = _call(server, "execute_task", {"task": "dance a little"})
    assert unread["isError"] is True
    assert "structuredContent" not in unread
