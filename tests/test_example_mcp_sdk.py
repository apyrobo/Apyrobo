"""
examples/06_mcp_client_sdk.py drives the real ``apyrobo mcp`` server with the
official MCP Python SDK: the over-limit move is clamped, the no-go move is
refused with its code read from structuredContent, the agent falls back to
another target, and the robot never enters the zone.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")
anyio = pytest.importorskip("anyio")

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "06_mcp_client_sdk.py"


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("mcp_sdk_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def log(example):
    async def run_with_timeout():
        with anyio.fail_after(60):
            return await example.run()

    return anyio.run(run_with_timeout)


def test_lists_the_server_tools(log):
    tools = log[0].removeprefix("tools: ").split(", ")
    for name in ("move_to", "rotate", "stop", "robot_status", "safety_report"):
        assert name in tools


def test_over_limit_move_is_clamped(log):
    assert log[1] == "moved to (1.0, 1.0)"
    assert log[2] == "  clamped: asked 2.0 m/s, ran at 0.5 m/s"


def test_no_go_move_is_refused_with_its_code(log):
    assert log[3].startswith("refused [collision_zone]: REFUSED by safety policy")
    assert "did not move" in log[3]


def test_agent_falls_back_and_never_enters_the_zone(log):
    assert log[4].startswith("  agent: target is in a no-go zone")
    assert log[5] == "moved to (1.0, 4.0)"
    assert log[-1] == "robot ends at (1.0, 4.0)"


def test_refusal_code_falls_back_to_the_text(example):
    def result(is_error, text, structured=None):
        content = [SimpleNamespace(text=text)]
        return SimpleNamespace(is_error=is_error, content=content,
                               structured_content=structured)

    assert example.refusal_code(result(False, "{}")) is None
    assert example.refusal_code(
        result(True, "x", {"status": "refused", "code": "outside_allowed_area"})
    ) == "outside_allowed_area"
    # Pre-2025-06-18 clients get text only.
    assert example.refusal_code(
        result(True, "REFUSED ... inside collision zone ...")
    ) == "collision_zone"
    assert example.refusal_code(result(True, "REFUSED: battery low")) == "safety_violation"
