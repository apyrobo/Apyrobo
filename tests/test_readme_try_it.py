"""The README's "Try it in 60 seconds" block must do what it says.

It is the first minute of every new user, so this module reads the block
itself rather than restating it:

* the ``uvx ... apyrobo mcp <flags>`` line gives the server flags; the same
  server is started locally (``python -m apyrobo mcp <flags>``, not uvx) and
  driven with the official MCP Python SDK client;
* the prompt and the sentence under it give the moves and the promised
  outcome: the first move is clamped to the stated speed, the second is
  refused and the robot stays where the first move left it;
* the ``claude mcp add`` line must match what ``apyrobo mcp config
  claude-code --launcher uvx`` prints for the same flags.

The server part is skipped when the ``mcp`` package is not installed.
"""
from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

import pytest

from apyrobo import mcp_clients

README = Path(__file__).resolve().parent.parent / "README.md"
HEADING = "### Try it in 60 seconds"
TIMEOUT_S = 60
_POINT = r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)"


def _block() -> str:
    text = README.read_text(encoding="utf-8")
    assert HEADING in text, f"README.md has no {HEADING!r} section"
    lines, in_fence = [], False
    for line in text.split(HEADING, 1)[1].splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
        elif line.startswith("#") and not in_fence:
            break  # the next heading ends the section
        lines.append(line)
    return "\n".join(lines)


def _uvx_argv() -> list[str]:
    fence = re.search(r"```bash\n(.*?)```", _block(), re.S)
    assert fence, "the try-it section has no bash block"
    lines = [ln for ln in fence.group(1).splitlines() if ln.startswith("uvx ")]
    assert len(lines) == 1, f"expected one uvx line in the try-it block, got {lines}"
    return shlex.split(lines[0])


def _flags(argv: list[str]) -> list[str]:
    i = argv.index("mcp")
    assert argv[i - 1] == "apyrobo", argv
    return argv[i + 1:]


def _claude_add_line() -> str:
    m = re.search(r"`(claude mcp add [^`]+)`", _block())
    assert m, "the try-it section has no `claude mcp add ...` line"
    return m.group(1)


def _claims() -> dict:
    """The moves and outcomes the prose promises."""
    block = _block()
    prompt = re.search(
        rf"drive to {_POINT} at (\d+(?:\.\d+)?) m/s, then to {_POINT}", block
    )
    assert prompt, "the try-it prompt no longer reads 'drive to (x, y) at N m/s, then to (x, y)'"
    clamp = re.search(r"first move is clamped to (\d+(?:\.\d+)?) m/s", block)
    assert clamp, "the try-it text no longer says what the first move is clamped to"
    assert re.search(r"second is refused", block), (
        "the try-it text no longer says the second move is refused"
    )
    x1, y1, speed, x2, y2 = map(float, prompt.groups())
    return {
        "first": (x1, y1),
        "speed": speed,
        "second": (x2, y2),
        "clamped_to": float(clamp.group(1)),
    }


# --- the text agrees with itself and with the code ---------------------------


def test_uvx_line_runs_apyrobo_mcp_from_github():
    argv = _uvx_argv()
    assert argv[:3] == ["uvx", "--from", mcp_clients.GIT_SOURCE]
    assert _flags(argv), "the uvx line passes no safety flags"


def test_claude_code_line_starts_the_same_server():
    readme_argv = shlex.split(_claude_add_line())
    assert readme_argv[:5] == ["claude", "mcp", "add", "apyrobo", "--"]
    assert readme_argv[5:] == _uvx_argv()


def test_claude_code_line_matches_mcp_config_output():
    flags = _flags(_uvx_argv())
    # Parse the README flags the way `apyrobo mcp config` does, so a flag the
    # config command would rewrite (spacing, number format) shows up here.
    opts = {"max_speed": None, "no_go": []}
    it = iter(flags)
    for flag in it:
        value = next(it)
        if flag == "--max-speed":
            opts["max_speed"] = float(value)
        elif flag == "--no-go":
            opts["no_go"].append(value)
        else:
            pytest.fail(f"try-it block uses {flag}; extend this test to cover it")
    launch = mcp_clients.resolve_launch(
        mcp_clients.server_args(**opts), launcher="uvx", which=lambda _name: None
    )
    printed = mcp_clients.render("claude-code", launch, platform="linux")
    assert printed == _claude_add_line()


def test_claims_name_the_flag_values():
    claims = _claims()
    flags = _flags(_uvx_argv())
    assert claims["clamped_to"] == float(flags[flags.index("--max-speed") + 1])
    assert claims["speed"] > claims["clamped_to"], "the first move must ask for more than the limit"
    zone = [float(v) for v in flags[flags.index("--no-go") + 1].split(",")]

    def in_zone(point: tuple[float, float]) -> bool:
        x, y = point
        return zone[0] <= x <= zone[2] and zone[1] <= y <= zone[3]

    assert in_zone(claims["second"]), "the second target is not in the no-go zone"
    assert not in_zone(claims["first"]), "the first target is in the no-go zone"


# --- the server does what the text says --------------------------------------


@pytest.fixture(scope="module")
def session_result() -> dict:
    pytest.importorskip("mcp")
    import anyio
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    claims = _claims()
    server = StdioServerParameters(
        command=sys.executable, args=["-m", "apyrobo", "mcp", *_flags(_uvx_argv())]
    )

    async def main() -> dict:
        with anyio.fail_after(TIMEOUT_S):
            async with (
                stdio_client(server) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                x1, y1 = claims["first"]
                x2, y2 = claims["second"]
                first = await session.call_tool(
                    "move_to", {"x": x1, "y": y1, "speed": claims["speed"]}
                )
                second = await session.call_tool("move_to", {"x": x2, "y": y2})
                status = await session.call_tool("robot_status", {})
                return {"claims": claims, "first": first, "second": second, "status": status}

    return anyio.run(main)


def _json(result) -> dict:
    assert result.content, result
    return json.loads(result.content[0].text)


def test_first_move_is_clamped_to_the_stated_speed(session_result):
    claims, first = session_result["claims"], session_result["first"]
    assert not first.is_error, first
    body = _json(first)
    assert body["status"] == "moved"
    assert (body["position"]["x"], body["position"]["y"]) == claims["first"]
    [clamp] = body["safety_interventions"]
    assert clamp["type"] == "speed_clamped"
    assert clamp["requested"] == claims["speed"]
    assert clamp["enforced"] == claims["clamped_to"]


def test_second_move_is_refused_and_the_robot_stays_put(session_result):
    claims, second = session_result["claims"], session_result["second"]
    assert second.is_error, second
    text = second.content[0].text
    assert text.startswith("REFUSED by safety policy")
    assert "no-go" in text or "collision zone" in text
    assert "did not move" in text
    status = _json(session_result["status"])
    assert (status["position"]["x"], status["position"]["y"]) == claims["first"]
