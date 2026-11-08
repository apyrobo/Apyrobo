"""
Progress notifications during a move, seen through the official MCP client.

tests/test_mcp_progress.py checks notifications/progress with our own stdio
reader. This module checks the same behaviour with the reference client
(the ``mcp`` package from the ``dev`` extra), the library many MCP hosts are
built on, against a real ``apyrobo mcp``-style stdio subprocess:

* a move called with a progress callback gets progress while it runs,
* the progress values increase,
* nothing arrives after the move's result, and
* a move called without a callback (so no progressToken) gets none.

The server runs on mock:// with its move slowed to about a second and the
progress interval cut to 0.05 s, so the module runs in a few seconds.
It is skipped when the ``mcp`` package is not installed.
"""
from __future__ import annotations

import sys
import textwrap
from importlib.metadata import version

import pytest

pytest.importorskip("mcp")
if int(version("mcp").split(".")[0]) < 2:
    pytest.skip("needs mcp>=2.2 (2026-07-28 client)", allow_module_level=True)

import anyio  # noqa: E402  (installed with mcp)
from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402

MOVE_S = 1.0
INTERVAL_S = 0.05
TIMEOUT_S = 60

# The same server cmd_mcp builds, on mock://, with a move that takes MOVE_S
# and a short progress interval.
LAUNCH = textwrap.dedent(
    f"""
    import time
    from apyrobo.mcp_server import build_server

    server = build_server(robot_uri="mock://official-progress")
    server.progress_interval = {INTERVAL_S}
    robot = server.toolbox.enforcer._robot
    _move = robot.move

    def slow_move(x, y, speed=None):
        time.sleep({MOVE_S})
        return _move(x, y, speed=speed) if speed is not None else _move(x, y)

    robot.move = slow_move
    server.serve(watch_client=True)
    """
)
SERVER = StdioServerParameters(command=sys.executable, args=["-c", LAUNCH])


def _is_progress(message) -> bool:
    root = getattr(message, "root", message)
    return getattr(root, "method", None) == "notifications/progress"


def _run(era: str) -> dict:
    progress: list[float] = []
    # Every progress the client sees, by its callback or as a raw notification.
    events: list[tuple[str, object]] = []

    async def on_progress(value, total, message):
        progress.append(value)
        events.append(("callback", value))

    async def on_message(message):
        if _is_progress(message):
            events.append(("notification", message))

    async def main() -> dict:
        with anyio.fail_after(TIMEOUT_S):
            async with (
                stdio_client(SERVER) as (read, write),
                ClientSession(read, write, message_handler=on_message) as session,
            ):
                if era == "handshake":
                    await session.initialize()
                else:
                    await session.discover()
                with_token = await session.call_tool(
                    "move_to", {"x": 1.0, "y": 0.0}, progress_callback=on_progress,
                )
                at_result = len(events)
                # Anything still in flight after the result would land here.
                await anyio.sleep(10 * INTERVAL_S)
                after_result = events[at_result:]
                at_second = len(events)
                without_token = await session.call_tool("move_to", {"x": 2.0, "y": 0.0})
                await anyio.sleep(10 * INTERVAL_S)
                return {
                    "with_token": with_token,
                    "progress": list(progress),
                    "after_result": after_result,
                    "without_token": without_token,
                    "without_token_events": events[at_second:],
                }

    return anyio.run(main)


@pytest.fixture(scope="module", params=["handshake", "stateless"])
def era(request):
    return request.param, _run(request.param)


def test_the_move_with_a_callback_completes(era):
    _, out = era
    assert not out["with_token"].is_error, out["with_token"]


def test_progress_arrives_while_the_move_runs(era):
    _, out = era
    # A 1 s move at a 0.05 s interval: expect many, ask for a few.
    assert len(out["progress"]) >= 3, out["progress"]


def test_progress_values_increase(era):
    _, out = era
    values = out["progress"]
    assert values == sorted(values), values
    assert len(set(values)) == len(values), f"repeated progress values: {values}"


def test_nothing_arrives_after_the_result(era):
    _, out = era
    assert out["after_result"] == [], out["after_result"]


def test_no_progress_without_a_token(era):
    _, out = era
    assert not out["without_token"].is_error, out["without_token"]
    assert out["without_token_events"] == [], out["without_token_events"]
