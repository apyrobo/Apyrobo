"""
``apyrobo mcp`` against the official MCP Python SDK client.

tests/test_mcp_server.py checks the server against our own reading of the
spec. This module checks it against the reference client (the ``mcp``
package, installed by the ``dev`` extra), the same library many MCP hosts
are built on, over a real stdio subprocess, in both protocol eras:

* the ``initialize`` handshake (the client's newest handshake version), and
* the stateless 2026-07-28 era (``server/discover``, no handshake).

Each era lists the tools and calls them: a normal move, an over-limit move
that is clamped and a no-go move that is refused with a readable error.
The module is skipped when the ``mcp`` package is not installed.
"""
from __future__ import annotations

import json
import sys
from importlib.metadata import version

import pytest

pytest.importorskip("mcp")
if int(version("mcp").split(".")[0]) < 2:
    pytest.skip("needs mcp>=2.2 (2026-07-28 client)", allow_module_level=True)

import anyio  # noqa: E402  (installed with mcp)
from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402
from mcp_types.version import LATEST_HANDSHAKE_VERSION  # noqa: E402

SERVER = StdioServerParameters(
    command=sys.executable,
    args=["-m", "apyrobo", "mcp", "--max-speed", "0.5", "--no-go", "2,2,4,4"],
)
TIMEOUT_S = 60


def _payload(result) -> dict:
    assert result.content, result
    return json.loads(result.content[0].text)


async def _exercise(session: ClientSession) -> dict:
    tools = await session.list_tools()
    moved = await session.call_tool("move_to", {"x": 1.0, "y": 0.0})
    clamped = await session.call_tool("move_to", {"x": 1.5, "y": 0.0, "speed": 3.0})
    refused = await session.call_tool("move_to", {"x": 3.0, "y": 3.0})
    report = await session.call_tool("safety_report", {})
    return {
        "tools": {t.name for t in tools.tools},
        "moved": moved,
        "clamped": clamped,
        "refused": refused,
        "report": report,
    }


def _run(era: str) -> dict:
    async def main() -> dict:
        with anyio.fail_after(TIMEOUT_S):
            async with (
                stdio_client(SERVER) as (read, write),
                ClientSession(read, write) as session,
            ):
                if era == "handshake":
                    init = await session.initialize()
                    info = init.server_info
                else:
                    await session.discover()
                    info = session.server_info
                out = await _exercise(session)
                out["version"] = session.protocol_version
                out["server_name"] = info.name if info else None
                return out

    return anyio.run(main)


@pytest.fixture(scope="module", params=["handshake", "stateless"])
def era(request):
    return request.param, _run(request.param)


def test_negotiates_a_version_the_server_supports(era):
    name, out = era
    from apyrobo.mcp_server import SUPPORTED_PROTOCOL_VERSIONS

    if name == "stateless":
        assert out["version"] == "2026-07-28"
    else:
        # The client proposes its newest handshake version; we must accept it
        # rather than fall back to an older one.
        assert out["version"] == LATEST_HANDSHAKE_VERSION
        assert out["version"] in SUPPORTED_PROTOCOL_VERSIONS
    assert out["server_name"] == "apyrobo"


def test_lists_the_robot_tools(era):
    _, out = era
    for tool in ("robot_status", "move_to", "rotate", "stop", "safety_report"):
        assert tool in out["tools"]


def test_normal_move_runs(era):
    _, out = era
    assert not out["moved"].is_error
    result = _payload(out["moved"])
    assert result["position"] == {"x": 1.0, "y": 0.0}
    assert result["safety_interventions"] == []


def test_over_limit_move_is_clamped(era):
    _, out = era
    assert not out["clamped"].is_error
    result = _payload(out["clamped"])
    assert result["position"] == {"x": 1.5, "y": 0.0}
    [clamp] = result["safety_interventions"]
    assert clamp["type"] == "speed_clamped"
    assert clamp["requested"] == 3.0
    assert clamp["enforced"] == 0.5


def test_no_go_move_is_refused_with_a_readable_error(era):
    _, out = era
    refused = out["refused"]
    assert refused.is_error
    text = refused.content[0].text
    assert text.startswith("REFUSED by safety policy")
    assert "collision zone" in text
    assert "did not move" in text


def test_safety_report_records_the_clamp_and_the_refusal(era):
    _, out = era
    assert not out["report"].is_error
    report = _payload(out["report"])
    assert report["safety_policy"]["no_go_zones"] == [
        {"x_min": 2.0, "y_min": 2.0, "x_max": 4.0, "y_max": 4.0}
    ]
    assert [i["type"] for i in report["interventions"]] == ["speed_clamped"]
    assert len(report["violations"]) == 1
