#!/usr/bin/env python3
"""Release-day smoke test: does the published package do what we promise?

Run on release day, after the upload, from any machine with Python 3.10+:

    python scripts/release_smoke.py 4.2.0             # installs apyrobo==4.2.0 from PyPI
    python scripts/release_smoke.py --wheel dist/apyrobo-4.2.0-py3-none-any.whl

It makes a fresh virtual environment in a temp folder, installs apyrobo
there (from PyPI, or the given wheel) together with the official MCP Python
SDK, starts ``apyrobo mcp --max-speed 0.5 --no-go 2,2,4,4`` on ``mock://``
over stdio, and talks to it with the SDK's client. It prints PASS/FAIL for:

  1. the package installs in a fresh venv
  2. the installed version is the one asked for, imported from that venv
  3. apyrobo mcp answers the client's handshake and lists move_to
  4. a 2 m/s move is clamped to 0.5 m/s
  5. a move to (3, 3), inside the no-go zone, is refused
  6. the robot did not move on that refusal

It never tags, bumps or uploads anything. It needs network access to
install from PyPI. ``--system-site-packages`` lets the venv reuse this
Python's packages (dependencies, the MCP SDK) so only apyrobo itself is
installed; the tests use it to run offline.

Exit code: 0 when every check passes, 1 otherwise, 2 on bad arguments.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_SPEED = 0.5
REQUESTED_SPEED = 2.0
NO_GO = "2,2,4,4"
REFUSED_TARGET = (3.0, 3.0)
CLAMP_TARGET = (1.0, 0.0)
MCP_SDK = "mcp>=2.2,<3"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""

    def line(self) -> str:
        tail = f" ({self.detail})" if self.detail else ""
        return f"{'PASS' if self.ok else 'FAIL'}  {self.name}{tail}"


# The client runs inside the new venv, so it uses the SDK installed there.
# It prints one JSON object on its last stdout line.
CLIENT = r'''
import json, sys
import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

server = StdioServerParameters(command=sys.argv[1], args=json.loads(sys.argv[2]))
clamp_xy = json.loads(sys.argv[3])
refused_xy = json.loads(sys.argv[4])
speed = float(sys.argv[5])


def payload(result):
    try:
        return json.loads(result.content[0].text)
    except Exception:
        return None


def text(result):
    try:
        return result.content[0].text
    except Exception:
        return ""


async def main():
    out = {}
    with anyio.fail_after(60):
        async with stdio_client(server) as (read, write), ClientSession(read, write) as s:
            init = await s.initialize()
            out["server_name"] = init.server_info.name if init.server_info else None
            out["server_version"] = init.server_info.version if init.server_info else None
            out["tools"] = sorted(t.name for t in (await s.list_tools()).tools)
            clamped = await s.call_tool(
                "move_to", {"x": clamp_xy[0], "y": clamp_xy[1], "speed": speed})
            out["clamped"] = {"is_error": bool(clamped.is_error),
                              "payload": payload(clamped), "text": text(clamped)}
            before = payload(await s.call_tool("robot_status", {}))
            refused = await s.call_tool("move_to", {"x": refused_xy[0], "y": refused_xy[1]})
            after = payload(await s.call_tool("robot_status", {}))
            out["refused"] = {"is_error": bool(refused.is_error), "text": text(refused),
                              "structured": refused.structured_content}
            out["before"] = (before or {}).get("position")
            out["after"] = (after or {}).get("position")
    print(json.dumps(out))


anyio.run(main)
'''


def _run(cmd: list[str], timeout: float = 600) -> tuple[bool, str, str]:
    # Run outside any checkout: ``python -c`` puts the working directory on
    # sys.path, so a source tree there would shadow the installed package.
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              cwd=tempfile.gettempdir())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, "", str(exc)
    text = re.sub(r"\x1b\[[0-9;]*m", "", proc.stderr).strip()
    return proc.returncode == 0, proc.stdout, text


def _last_line(text: str) -> str:
    """Last meaningful line, without the box drawing of exception groups."""
    lines = [ln.strip(" |+-\t") for ln in text.splitlines()]
    lines = [ln for ln in lines if any(ch.isalnum() for ch in ln)]
    return lines[-1] if lines else ""


def wheel_version(wheel: Path) -> str | None:
    """Version from a wheel file name: apyrobo-4.2.0-py3-none-any.whl -> 4.2.0."""
    parts = wheel.name.split("-")
    return parts[1] if wheel.suffix == ".whl" and len(parts) >= 5 else None


def bin_dir(venv: Path) -> Path:
    return venv / ("Scripts" if sys.platform == "win32" else "bin")


def exe(venv: Path, name: str) -> Path:
    return bin_dir(venv) / (f"{name}.exe" if sys.platform == "win32" else name)


def server_args() -> list[str]:
    return ["mcp", "--robot", "mock://turtlebot4",
            "--max-speed", str(MAX_SPEED), "--no-go", NO_GO]


def install(venv: Path, target: str, *, system_site: bool) -> Check:
    """Create *venv* and install *target* (a requirement or a wheel path)."""
    cmd = [sys.executable, "-m", "venv", str(venv)]
    if system_site:
        cmd.insert(3, "--system-site-packages")
    ok, _, err = _run(cmd)
    if not ok:
        return Check("installs in a fresh venv", False, _last_line(err) or "venv failed")
    pip = [str(exe(venv, "python")), "-m", "pip", "install", "-q",
           "--disable-pip-version-check"]
    if system_site:
        # Dependencies come from this Python; put apyrobo itself in the venv.
        pip += ["--no-deps", "--force-reinstall", target]
    else:
        pip += [target, MCP_SDK]
    ok, _, err = _run(pip)
    return Check("installs in a fresh venv", ok, target if ok else _last_line(err))


def check_version(venv: Path, expected: str | None) -> Check:
    ok, out, err = _run([str(exe(venv, "python")), "-c",
                         "import apyrobo, importlib.metadata as m; "
                         "print(m.version('apyrobo')); print(apyrobo.__file__)"])
    if not ok:
        return Check("installed version", False, _last_line(err) or "import failed")
    lines = out.strip().splitlines()
    got, path = (lines + ["", ""])[:2]
    inside = Path(path).resolve().is_relative_to(venv.resolve())
    good = inside and (expected is None or got == expected)
    detail = f"{got}" + (f", expected {expected}" if expected and got != expected else "")
    if not inside:
        detail += f", imported from {path}, not the venv"
    return Check("installed version", good, detail)


def drive(venv: Path) -> dict[str, Any] | str:
    """Run the SDK client against the venv's apyrobo mcp. A str is an error."""
    ok, out, err = _run([
        str(exe(venv, "python")), "-c", CLIENT,
        str(exe(venv, "apyrobo")), json.dumps(server_args()),
        json.dumps(list(CLAMP_TARGET)), json.dumps(list(REFUSED_TARGET)),
        str(REQUESTED_SPEED),
    ], timeout=120)
    if not ok:
        return _last_line(err) or "client failed"
    try:
        result = json.loads(_last_line(out))
    except json.JSONDecodeError:
        return "no result from the client"
    return result if isinstance(result, dict) else "no result from the client"


def evaluate(result: dict[str, Any] | str) -> list[Check]:
    """Turn the client's report into the four MCP checks."""
    names = ["apyrobo mcp answers the MCP SDK client",
             f"a {REQUESTED_SPEED:g} m/s move is clamped to {MAX_SPEED} m/s",
             f"a move to {REFUSED_TARGET} in the no-go zone is refused",
             "the robot did not move on the refusal"]
    if isinstance(result, str):
        return [Check(names[0], False, result)] + [Check(n, False, "not run") for n in names[1:]]

    tools = result.get("tools") or []
    up = result.get("server_name") == "apyrobo" and "move_to" in tools
    checks = [Check(names[0], up, f"version {result.get('server_version')}, {len(tools)} tools"
                    if up else f"server {result.get('server_name')!r}, tools {tools}")]

    clamped = result.get("clamped") or {}
    enforced = None
    for item in ((clamped.get("payload") or {}).get("safety_interventions") or []):
        if isinstance(item, dict) and item.get("type") == "speed_clamped":
            enforced = item.get("enforced")
    checks.append(Check(
        names[1], not clamped.get("is_error") and enforced == MAX_SPEED,
        f"enforced {enforced} m/s" if enforced is not None
        else (clamped.get("text") or "no speed_clamped intervention")[:120]))

    refused = result.get("refused") or {}
    text = refused.get("text") or ""
    code = (refused.get("structured") or {}).get("code")
    is_refusal = bool(refused.get("is_error")) and text.startswith("REFUSED")
    checks.append(Check(names[2], is_refusal,
                        f"code {code}" if is_refusal else (text[:120] or "no reply")))

    before, after = result.get("before"), result.get("after")
    checks.append(Check(names[3], before is not None and before == after,
                        f"at {before}" if before == after else f"{before} -> {after}"))
    return checks


def run(target: str, expected: str | None, *, system_site: bool, keep: bool) -> list[Check]:
    work = Path(tempfile.mkdtemp(prefix="apyrobo-release-smoke-"))
    try:
        venv = work / "venv"
        checks = [install(venv, target, system_site=system_site)]
        if not checks[0].ok:
            return checks
        checks.append(check_version(venv, expected))
        checks.extend(evaluate(drive(venv)))
        return checks
    finally:
        if keep:
            print(f"kept {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("version", nargs="?",
                        help="version to install from PyPI, e.g. 4.2.0")
    parser.add_argument("--wheel", type=Path, help="install this wheel instead of PyPI")
    parser.add_argument("--system-site-packages", action="store_true",
                        help="reuse this Python's packages; install only apyrobo itself")
    parser.add_argument("--keep", action="store_true", help="keep the temp venv")
    args = parser.parse_args(argv)

    if args.wheel is not None:
        if not args.wheel.is_file():
            parser.error(f"no such wheel: {args.wheel}")
        target = str(args.wheel.resolve())
        expected = args.version or wheel_version(args.wheel)
    elif args.version:
        target, expected = f"apyrobo=={args.version}", args.version
    else:
        parser.error("give a VERSION to install from PyPI, or --wheel PATH")

    checks = run(target, expected, system_site=args.system_site_packages, keep=args.keep)
    for check in checks:
        print(check.line())
    failed = sum(not c.ok for c in checks)
    print(f"\n{len(checks) - failed}/{len(checks)} passed" if failed else
          f"\nall {len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
