"""Every ``apyrobo`` command docs/mcp.md tells a new user to run must work as written.

This module reads the shell lines in the "Try it in a minute", "Check your
limits first", "Options" and "Troubleshooting" sections of docs/mcp.md and
runs the ones that work on ``mock://`` with no ROS:

* ``apyrobo mcp …`` server lines are started locally (``python -m apyrobo mcp``)
  and driven with the official MCP Python SDK client, and the outcome the
  text around them promises is checked (the clamp, the refusal, the four-step
  delivery, the allowed area, the audit-log lines);
* ``apyrobo mcp selftest …`` lines are run and must exit 0;
* the start-up line Troubleshooting quotes must match what the server prints.

Lines that need the network, ROS 2 or a real client are listed in ``SKIPPED``
with the reason. A new shell line in those sections fails
``test_every_shell_line_is_run_or_listed`` until it is added to one of the two
lists, so the doc cannot grow a command nobody ran. The client config
snippets (row 166, tests/test_mcp_docs.py) and the README try-it block
(tests/test_readme_try_it.py) are tested elsewhere.

The server tests are skipped when the ``mcp`` package is not installed.
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parent.parent / "docs" / "mcp.md"
TIMEOUT_S = 60

# Heading -> whether the section runs on through its subsections (to the next
# "## " heading) or stops at the next heading of any level.
SECTIONS = {
    "Try it in a minute": False,  # its ### client subsections are row 166's
    "Check your limits first": False,
    "Options": True,
    "Troubleshooting": True,
}

TRY_IT = "apyrobo mcp --max-speed 0.5 --no-go 2,2,4,4"
SELFTEST = "apyrobo mcp selftest --max-speed 0.5 --no-go 2,2,4,4"
ALLOW_AREA = "apyrobo mcp --max-speed 0.5 --allow-area 0,0,10,6 --no-go 2,2,4,4"
AUDIT_LOG = "apyrobo mcp --max-speed 0.5 --no-go 2,2,4,4 --audit-log ~/apyrobo-audit.jsonl"

RUN = {TRY_IT, SELFTEST, ALLOW_AREA, AUDIT_LOG}

_GIT = "git+https://github.com/apyrobo/Apyrobo"
SKIPPED = {
    f"uvx --from {_GIT} apyrobo mcp --max-speed 0.5 --no-go 2,2,4,4":
        "fetches from GitHub; the same server runs locally as TRY_IT",
    f"pipx run --spec {_GIT} apyrobo mcp --max-speed 0.5 --no-go 2,2,4,4":
        "fetches from GitHub; the same server runs locally as TRY_IT",
    f'pip install "{_GIT}"':
        "installs from GitHub; the test runs against this checkout",
    f"uvx --no-managed-python --python 3.12 --with 'numpy<2' --with transforms3d "
    f"--from {_GIT} apyrobo mcp --robot ros2://burger":
        "needs a sourced ROS 2 Jazzy and the network",
}


# --- reading the doc -----------------------------------------------------------


def _section(heading: str, with_subsections: bool) -> str:
    lines = DOC.read_text(encoding="utf-8").splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if re.fullmatch(rf"#+ {re.escape(heading)}", ln)),
        None,
    )
    assert start is not None, f"docs/mcp.md has no {heading!r} heading"
    body, in_fence = [], False
    for line in lines[start + 1:]:
        if line.startswith("```"):
            in_fence = not in_fence
        elif (
            not in_fence
            and line.startswith("#")
            and (not with_subsections or line.startswith("## "))
        ):
            break
        body.append(line)
    return "\n".join(body)


def _shell_lines(text: str) -> list[str]:
    """Commands in the section's ```bash blocks that run apyrobo."""
    found = []
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        joined = re.sub(r"\\\n\s*", "", block)  # backslash continuations
        found += [ln.strip() for ln in joined.splitlines() if "apyrobo" in ln]
    return found


def _all_shell_lines() -> list[str]:
    return [ln for h, sub in SECTIONS.items() for ln in _shell_lines(_section(h, sub))]


def _server_argv(line: str, audit_path: Path | None = None) -> list[str]:
    argv = shlex.split(line)
    assert argv[:2] == ["apyrobo", "mcp"], line
    args = argv[2:]
    if audit_path is not None:
        i = args.index("--audit-log")
        args[i + 1] = str(audit_path)
    return [sys.executable, "-m", "apyrobo", "mcp", *args]


def _prose(heading: str, with_subsections: bool = False) -> str:
    return " ".join(_section(heading, with_subsections).split())


# --- the lists stay in step with the doc ---------------------------------------


def test_every_shell_line_is_run_or_listed():
    lines = _all_shell_lines()
    assert lines, "found no apyrobo shell lines in docs/mcp.md"
    unknown = [ln for ln in lines if ln not in RUN and ln not in SKIPPED]
    assert not unknown, (
        "docs/mcp.md has shell lines this test neither runs nor skips; add each "
        f"to RUN (with a check) or SKIPPED (with a reason): {unknown}"
    )


def test_listed_lines_are_still_in_the_doc():
    lines = set(_all_shell_lines())
    stale = [ln for ln in (*RUN, *SKIPPED) if ln not in lines]
    assert not stale, f"these lines are no longer in docs/mcp.md: {stale}"


# --- apyrobo mcp selftest --------------------------------------------------------


def test_selftest_line_exits_zero():
    argv = shlex.split(SELFTEST)
    proc = subprocess.run(
        [sys.executable, "-m", *argv], capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # The doc says it shows an over-limit move clamped to your limit and tries
    # a move into each no-go zone.
    assert "clamped to 0.5 m/s" in proc.stdout, proc.stdout
    assert re.search(r"no-go zone #1: move to \([^)]+\) refused", proc.stdout), proc.stdout
    assert "checks passed" in proc.stdout


# --- server lines through the official MCP client ---------------------------------


def _session(argv: list[str], calls: list[tuple[str, dict]], errlog_path: Path):
    pytest.importorskip("mcp")
    import anyio
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    server = StdioServerParameters(command=argv[0], args=argv[1:])

    async def main() -> list:
        with anyio.fail_after(TIMEOUT_S), open(errlog_path, "w", encoding="utf-8") as err:
            async with (
                stdio_client(server, errlog=err) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                return [await session.call_tool(name, args) for name, args in calls]

    return anyio.run(main)


def _json(result) -> dict:
    assert result.content, result
    return json.loads(result.content[0].text)


def _pos(body: dict) -> tuple[float, float]:
    return body["position"]["x"], body["position"]["y"]


@pytest.fixture(scope="module")
def try_it(tmp_path_factory) -> dict:
    """The try-it server, asked what the doc's sample prompt asks."""
    prose = _prose("Try it in a minute")
    prompt = re.search(
        r"drive to \((\d+), (\d+)\) at (\d+(?:\.\d+)?) m/s, then drive to \((\d+), (\d+)\)",
        prose,
    )
    assert prompt, (
        "the try-it prompt no longer reads 'drive to (x, y) at N m/s, then drive to (x, y)'"
    )
    deliver = re.search(r'"(deliver the package to \(\d+, \d+\))"', prose)
    assert deliver, "the try-it text no longer suggests a 'deliver the package to (x, y)' task"
    x1, y1, speed, x2, y2 = map(float, prompt.groups())
    errlog = tmp_path_factory.mktemp("try_it") / "stderr.txt"
    status, first, second, after, task = _session(
        _server_argv(TRY_IT),
        [
            ("robot_status", {}),
            ("move_to", {"x": x1, "y": y1, "speed": speed}),
            ("move_to", {"x": x2, "y": y2}),
            ("robot_status", {}),
            ("execute_task", {"task": deliver.group(1)}),
        ],
        errlog,
    )
    return {
        "prose": prose, "first_target": (x1, y1), "speed": speed,
        "status": status, "first": first, "second": second, "after": after,
        "task": task, "stderr": errlog.read_text(encoding="utf-8"),
    }


def test_try_it_first_move_is_clamped(try_it):
    clamp = re.search(r"first move's speed clamped to (\d+(?:\.\d+)?) m/s", try_it["prose"])
    assert clamp, "the try-it text no longer says what the first move is clamped to"
    first = try_it["first"]
    assert not try_it["status"].is_error and not first.is_error, first
    body = _json(first)
    assert body["status"] == "moved"
    assert _pos(body) == try_it["first_target"]
    [intervention] = body["safety_interventions"]
    assert intervention["type"] == "speed_clamped"
    assert intervention["requested"] == try_it["speed"]
    assert intervention["enforced"] == float(clamp.group(1))


def test_try_it_move_into_the_zone_is_refused(try_it):
    assert "refused because that point is inside the no-go zone" in try_it["prose"]
    second = try_it["second"]
    assert second.is_error, second
    assert second.content[0].text.startswith("REFUSED by safety policy")
    assert _pos(_json(try_it["after"])) == try_it["first_target"], "the robot moved"


def test_try_it_delivery_plans_and_runs_four_steps(try_it):
    assert "four-step task" in try_it["prose"]
    task = try_it["task"]
    assert not task.is_error, task
    body = _json(task)
    assert body["status"] == "completed", body
    assert body["steps_total"] == 4 and body["steps_completed"] == 4, body


def test_start_up_line_is_the_one_troubleshooting_quotes(try_it):
    quoted = re.search(r"`(apyrobo mcp — robot=… \(waiting for an MCP client on stdio\))`",
                       _prose("Troubleshooting", True))
    assert quoted, "Troubleshooting no longer quotes the server's start-up line"
    pattern = re.escape(quoted.group(1)).replace("…", ".+")
    first_line = try_it["stderr"].splitlines()[0]
    assert re.fullmatch(pattern, first_line), first_line
    assert "Error" not in try_it["stderr"]


def test_allowed_area_line_refuses_targets_outside_it(tmp_path):
    flags = shlex.split(ALLOW_AREA)
    area = [float(v) for v in flags[flags.index("--allow-area") + 1].split(",")]
    outside = {"x": area[2] + 2, "y": (area[1] + area[3]) / 2}
    inside = {"x": 1.0, "y": 1.0}
    in_zone = {"x": 3.0, "y": 3.0}  # no-go 2,2,4,4 still wins inside the area
    out, ok, zone = _session(
        _server_argv(ALLOW_AREA),
        [("move_to", outside), ("move_to", inside), ("move_to", in_zone)],
        tmp_path / "stderr.txt",
    )
    assert out.is_error and out.content[0].text.startswith("REFUSED by safety policy"), out
    assert "allowed area" in out.content[0].text
    assert not ok.is_error and _pos(_json(ok)) == (1.0, 1.0), ok
    assert zone.is_error and zone.content[0].text.startswith("REFUSED by safety policy"), zone


def test_audit_log_line_writes_the_lines_the_doc_shows(tmp_path):
    section = _section("Keep an audit log", False)
    sample = re.search(r"```json\n(.*?)```", section, re.S)
    assert sample, "the audit-log section has no sample JSON lines"
    shown = [json.loads(ln) for ln in sample.group(1).splitlines() if ln.strip()]
    assert [ln["decision"] for ln in shown] == ["started", "clamped"], shown
    log = tmp_path / "apyrobo-audit.jsonl"
    [result] = _session(
        _server_argv(AUDIT_LOG, audit_path=log),
        [(shown[0]["tool"], shown[0]["arguments"])],
        tmp_path / "stderr.txt",
    )
    assert not result.is_error, result
    written = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines()]
    for want, got in zip(shown, written[:2], strict=True):
        assert set(got) == set(want), (want, got)
        assert {k: v for k, v in got.items() if k != "time"} == {
            k: v for k, v in want.items() if k != "time"
        }
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}\+00:00", got["time"])
