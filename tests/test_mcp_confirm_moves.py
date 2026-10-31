"""Tests for ``apyrobo mcp --confirm-moves``.

With the flag, every motion command (move_to, rotate, execute_task) asks the
user to confirm it through MCP elicitation before the robot moves. For
2026-07-28 clients this is the multi-round-trip pattern: the first
tools/call returns ``resultType: input_required`` with an
``elicitation/create`` request and a sealed ``requestState``; the client asks
the user and calls again with ``inputResponses``. Clients that cannot
confirm get motion refused (fail closed). Approval never lifts policy limits.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from apyrobo.mcp_server import CONFIRM_INPUT_KEY, INVALID_PARAMS, build_server

ZONE = {"x_min": 2.0, "y_min": 2.0, "x_max": 4.0, "y_max": 4.0}
MODERN = "2026-07-28"
ELICIT = {"elicitation": {"form": {}}}


@pytest.fixture
def server():
    return build_server(max_speed=0.5, no_go_zones=[ZONE], confirm_moves=True)


def rpc(server, method, params, msg_id=1):
    return server.handle({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})


def modern_call(server, name, arguments, capabilities=ELICIT, **extra):
    meta = {
        "io.modelcontextprotocol/protocolVersion": MODERN,
        "io.modelcontextprotocol/clientCapabilities": capabilities,
    }
    params = {"name": name, "arguments": arguments, "_meta": meta, **extra}
    return rpc(server, "tools/call", params)


def position(server):
    return server.toolbox.enforcer.get_position()


def confirm(server, name, arguments, action="accept", state=None):
    first = modern_call(server, name, arguments)["result"]
    return modern_call(
        server,
        name,
        arguments,
        inputResponses={CONFIRM_INPUT_KEY: {"action": action}},
        requestState=state if state is not None else first["requestState"],
    )


# -- the confirmation request ----------------------------------------------


def test_motion_call_asks_for_confirmation_and_does_not_move(server):
    start = position(server)
    result = modern_call(server, "move_to", {"x": 1.0, "y": 0.5, "speed": 3.0})["result"]
    assert result["resultType"] == "input_required"
    request = result["inputRequests"][CONFIRM_INPUT_KEY]
    assert request["method"] == "elicitation/create"
    assert request["params"]["mode"] == "form"
    assert request["params"]["requestedSchema"] == {"type": "object", "properties": {}}
    message = request["params"]["message"]
    assert "x=1.0, y=0.5" in message
    assert "0.5 m/s (requested 3.0, clamped" in message
    assert isinstance(result["requestState"], str)
    assert position(server) == start


def test_rotate_and_execute_task_also_ask(server):
    rot = modern_call(server, "rotate", {"angle_rad": 1.0})["result"]
    assert rot["resultType"] == "input_required"
    assert "rotate by 1.0 rad" in rot["inputRequests"][CONFIRM_INPUT_KEY]["params"]["message"]
    task = modern_call(server, "execute_task", {"task": "go to (1, 1)"})["result"]
    assert task["resultType"] == "input_required"
    assert "go to (1, 1)" in task["inputRequests"][CONFIRM_INPUT_KEY]["params"]["message"]


@pytest.mark.parametrize("name", ["robot_status", "stop", "safety_report"])
def test_non_motion_tools_never_ask(server, name):
    result = modern_call(server, name, {})["result"]
    assert result["resultType"] == "complete"
    assert result["isError"] is False


# -- the answer ------------------------------------------------------------


def test_accepted_move_runs_with_the_clamp(server):
    result = confirm(server, "move_to", {"x": 1.0, "y": 0.5, "speed": 3.0})["result"]
    assert result["resultType"] == "complete"
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert payload["status"] == "moved"
    assert payload["safety_interventions"][0]["enforced"] == 0.5
    assert position(server) == pytest.approx((1.0, 0.5))


@pytest.mark.parametrize("action", ["decline", "cancel"])
def test_declined_move_does_not_move(server, action):
    start = position(server)
    result = confirm(server, "move_to", {"x": 1.0, "y": 0.5}, action=action)["result"]
    assert result["resultType"] == "complete"
    payload = json.loads(result["content"][0]["text"])
    assert payload["status"] == "declined"
    assert payload["user_action"] == action
    assert position(server) == start


def test_confirmation_cannot_be_reused_for_other_arguments(server):
    start = position(server)
    first = modern_call(server, "move_to", {"x": 1.0, "y": 0.5})["result"]
    swapped = modern_call(
        server,
        "move_to",
        {"x": 1.5, "y": 1.5},
        inputResponses={CONFIRM_INPUT_KEY: {"action": "accept"}},
        requestState=first["requestState"],
    )
    assert swapped["error"]["code"] == INVALID_PARAMS
    assert position(server) == start


def test_forged_or_expired_state_is_rejected(server, monkeypatch):
    forged = confirm(server, "move_to", {"x": 1.0, "y": 0.5}, state="1.deadbeef")
    assert forged["error"]["code"] == INVALID_PARAMS
    first = modern_call(server, "move_to", {"x": 1.0, "y": 0.5})["result"]
    import apyrobo.mcp_server as mod

    real = mod.time.time()
    monkeypatch.setattr(mod.time, "time", lambda: real + mod.CONFIRM_TTL_S + 5)
    expired = modern_call(
        server,
        "move_to",
        {"x": 1.0, "y": 0.5},
        inputResponses={CONFIRM_INPUT_KEY: {"action": "accept"}},
        requestState=first["requestState"],
    )
    assert expired["error"]["code"] == INVALID_PARAMS


# -- policy still decides --------------------------------------------------


def test_no_go_target_is_refused_without_asking(server):
    result = modern_call(server, "move_to", {"x": 3.0, "y": 3.0})["result"]
    assert result["resultType"] == "complete"
    assert result["isError"] is True
    assert "REFUSED by safety policy" in result["content"][0]["text"]


def test_invalid_arguments_are_reported_without_asking(server):
    result = modern_call(server, "move_to", {"x": 1.0})["result"]
    assert result["isError"] is True
    assert "resultType" in result and result["resultType"] == "complete"


# -- fail closed -----------------------------------------------------------


@pytest.mark.parametrize("caps", [{}, {"elicitation": {"url": {}}}])
def test_client_without_form_elicitation_is_refused(server, caps):
    start = position(server)
    result = modern_call(server, "move_to", {"x": 1.0, "y": 0.5}, capabilities=caps)["result"]
    assert result["isError"] is True
    assert "does not support form elicitation" in result["content"][0]["text"]
    assert position(server) == start


def test_empty_elicitation_capability_means_form(server):
    result = modern_call(server, "move_to", {"x": 1.0, "y": 0.5}, capabilities={"elicitation": {}})
    assert result["result"]["resultType"] == "input_required"


def test_legacy_session_motion_is_refused(server):
    rpc(server, "initialize", {"protocolVersion": "2025-11-25", "capabilities": {}})
    start = position(server)
    resp = rpc(server, "tools/call", {"name": "move_to", "arguments": {"x": 1.0, "y": 0.5}})
    assert resp["result"]["isError"] is True
    assert "--confirm-moves" in resp["result"]["content"][0]["text"]
    assert position(server) == start
    status = rpc(server, "tools/call", {"name": "robot_status", "arguments": {}})
    assert status["result"]["isError"] is False


def test_default_is_unchanged():
    plain = build_server(max_speed=0.5)
    result = modern_call(plain, "move_to", {"x": 1.0, "y": 0.5})["result"]
    assert result["resultType"] == "complete"
    assert result["isError"] is False


# -- CLI -------------------------------------------------------------------


def test_cli_flag_reaches_the_server():
    msgs = [
        {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "move_to", "arguments": {"x": 1.0, "y": 0.0},
                "_meta": {
                    "io.modelcontextprotocol/protocolVersion": MODERN,
                    "io.modelcontextprotocol/clientCapabilities": ELICIT,
                },
            },
        }
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "apyrobo", "mcp", "--confirm-moves"],
        input="".join(json.dumps(m) + "\n" for m in msgs),
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "confirm_moves=on" in proc.stderr
    reply = json.loads(proc.stdout.strip().splitlines()[0])
    assert reply["result"]["resultType"] == "input_required"


# -- official SDK client ---------------------------------------------------


def test_official_client_round_trip():
    pytest.importorskip("mcp")
    from importlib.metadata import version

    if int(version("mcp").split(".")[0]) < 2:
        pytest.skip("needs mcp>=2.2 (2026-07-28 client)")
    import anyio
    from mcp import ClientSession, types
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "apyrobo", "mcp", "--confirm-moves", "--max-speed", "0.5"],
    )

    async def on_elicit(context, request_params):  # pragma: no cover - not called here
        return types.ElicitResult(action="accept")

    async def main():
        with anyio.fail_after(60):
            async with (
                stdio_client(params) as (read, write),
                ClientSession(read, write, elicitation_callback=on_elicit) as session,
            ):
                await session.discover()
                args = {"x": 1.0, "y": 0.0, "speed": 3.0}
                asked = await session.call_tool("move_to", args, allow_input_required=True)
                assert isinstance(asked, types.InputRequiredResult)
                request = asked.input_requests[CONFIRM_INPUT_KEY]
                assert "clamped" in request.params.message
                done = await session.call_tool(
                    "move_to",
                    args,
                    input_responses={CONFIRM_INPUT_KEY: types.ElicitResult(action="accept")},
                    request_state=asked.request_state,
                    allow_input_required=True,
                )
                return done

    done = anyio.run(main)
    assert done.is_error is False
    assert json.loads(done.content[0].text)["status"] == "moved"
