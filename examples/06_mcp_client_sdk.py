#!/usr/bin/env python3
"""
Drive ``apyrobo mcp`` from your own agent code with the official MCP Python SDK.

05_mcp_safety_demo.py shows the raw JSON-RPC lines. This one is the loop an
agent framework runs: start the server over stdio, list its tools, call
``move_to``, and decide what to do next from the result. A refused move comes
back as an error result whose ``structuredContent`` carries a machine-readable
``code`` (``collision_zone``, ``outside_allowed_area``, ...); older protocol
versions send text only, so the code falls back to reading the text.

Runs on mock:// (no ROS 2, no hardware) with a 0.5 m/s limit and one no-go
zone. Needs the SDK, version 2.2 or later: ``pip install "mcp>=2.2"``.
From the repo root:

    python examples/06_mcp_client_sdk.py
"""
from __future__ import annotations

import json
import sys

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# The same server Claude, Cursor or VS Code would start, with its limits.
SERVER = StdioServerParameters(
    command=sys.executable,
    args=["-m", "apyrobo", "mcp", "--robot", "mock://turtlebot4",
          "--max-speed", "0.5", "--no-go", "2,2,4,4"],
)

# What the "agent" wants to do: the second target sits inside the no-go zone.
GOALS = [
    {"x": 1.0, "y": 1.0, "speed": 2.0},  # over the limit: runs, clamped
    {"x": 3.0, "y": 3.0},                # in the zone: refused, robot stays put
]
FALLBACK = {"x": 1.0, "y": 4.0}          # where the agent goes instead


def refusal_code(result) -> str | None:
    """The refusal code of a tool result, or None if the call went through."""
    if not result.is_error:
        return None
    if result.structured_content:  # MCP 2025-06-18 and later
        return result.structured_content.get("code", "safety_violation")
    text = result.content[0].text if result.content else ""
    return "collision_zone" if "collision zone" in text else "safety_violation"


async def move(session: ClientSession, args: dict, log: list[str]) -> str | None:
    result = await session.call_tool("move_to", args)
    code = refusal_code(result)
    if code is None:
        body = json.loads(result.content[0].text)
        pos = body["position"]
        log.append(f"moved to ({pos['x']}, {pos['y']})")
        for step in body.get("safety_interventions", []):
            if step["type"] == "speed_clamped":
                log.append(f"  clamped: asked {step['requested']} m/s, "
                           f"ran at {step['enforced']} m/s")
    else:
        log.append(f"refused [{code}]: {result.content[0].text.splitlines()[0]}")
    return code


async def run() -> list[str]:
    """Run the agent loop once and return what it printed."""
    log: list[str] = []
    async with stdio_client(SERVER) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        log.append("tools: " + ", ".join(sorted(t.name for t in tools.tools)))

        for goal in GOALS:
            code = await move(session, goal, log)
            # Branch on the code, not the wording: a zone refusal means pick
            # another target; anything else (battery, area) needs a human.
            if code == "collision_zone":
                log.append(f"  agent: target is in a no-go zone, trying {FALLBACK}")
                await move(session, FALLBACK, log)
            elif code is not None:
                log.append("  agent: stopping and asking the operator")
                await session.call_tool("stop", {})
                break

        status = json.loads((await session.call_tool("robot_status", {})).content[0].text)
        pos = status["position"]
        log.append(f"robot ends at ({pos['x']}, {pos['y']})")
    return log


def main() -> None:
    for line in anyio.run(run):
        print(line)


if __name__ == "__main__":
    main()
