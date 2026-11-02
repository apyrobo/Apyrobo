#!/usr/bin/env python3
"""Pre-release check: is this checkout ready to upload to PyPI?

Run from the repo root before tagging a release:

    python scripts/release_check.py            # full check (builds, installs, runs)
    python scripts/release_check.py --static   # metadata checks only, no build

It checks, and prints PASS/FAIL for each:

  1. pyproject.toml, server.json and server.json's PyPI package agree on the version
  2. CHANGELOG.md has a "## [<version>]" section with real entries in it
     (the release notes come from it, so an empty one ships blank notes)
  3. README.md carries the "mcp-name: <server.json name>" marker the MCP
     Registry uses to verify PyPI ownership
  4. sdist and wheel build (python -m build)
  5. twine check passes on both
  6. the wheel's METADATA contains the mcp-name marker
  7. the wheel installs into a fresh venv and its ``apyrobo mcp`` answers
     initialize and tools/list over stdio, and clamps an over-limit move

It never bumps the version, tags, or uploads anything. Steps 4-7 need the
``build`` and ``twine`` packages (``pip install build twine``); step 7
installs the wheel's dependencies from PyPI, so it needs network access.

Exit code: 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

CLAMP_LIMIT = 0.5
CLAMP_REQUEST = 5.0


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""

    def line(self) -> str:
        return f"  {'PASS' if self.ok else 'FAIL'}  {self.name}" + (
            f" — {self.detail}" if self.detail else ""
        )


# -- metadata checks (pure, unit-tested) ------------------------------


def pyproject_version(root: Path) -> str | None:
    """The [project] version from pyproject.toml, or None if missing."""
    path = root / "pyproject.toml"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if "[project]" not in text:
        return None
    project = text.split("[project]", 1)[1].split("\n[", 1)[0]
    match = re.search(r'^version\s*=\s*"([^"]+)"', project, re.MULTILINE)
    return match.group(1) if match else None


def _server_json(root: Path) -> dict[str, Any] | None:
    path = root / "server.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def check_versions(root: Path) -> Check:
    name = "pyproject.toml and server.json versions match"
    version = pyproject_version(root)
    if version is None:
        return Check(name, False, "no [project] version in pyproject.toml")
    server = _server_json(root)
    if server is None:
        return Check(name, False, "server.json missing or not valid JSON")
    found = {"server.json version": server.get("version")}
    for i, pkg in enumerate(server.get("packages") or []):
        if isinstance(pkg, dict) and pkg.get("registryType") == "pypi":
            found[f"server.json packages[{i}].version"] = pkg.get("version")
    wrong = [f"{k}={v!r}" for k, v in found.items() if v != version]
    if wrong:
        return Check(name, False, f"pyproject has {version}; " + ", ".join(wrong))
    return Check(name, True, version)


def changelog_section(text: str, version: str) -> str | None:
    """The body of the ``## [version]`` section, or None if there is no heading."""
    heading = re.search(rf"^## \[{re.escape(version)}\][^\n]*$", text, re.MULTILINE)
    if heading is None:
        return None
    rest = text[heading.end():]
    following = re.search(r"^## \[", rest, re.MULTILINE)
    return rest[:following.start()] if following else rest


def check_changelog(root: Path, version: str | None) -> Check:
    name = "CHANGELOG.md has a section for this version, with entries"
    if version is None:
        return Check(name, False, "unknown version")
    path = root / "CHANGELOG.md"
    if not path.is_file():
        return Check(name, False, "CHANGELOG.md missing")
    section = changelog_section(path.read_text(encoding="utf-8"), version)
    if section is None:
        return Check(
            name, False,
            f"no '## [{version}]' heading (move the [Unreleased] entries under it)",
        )
    if "<!--" in section:
        return Check(name, False, f"'## [{version}]' still holds placeholder comments")
    entries = [
        line for line in section.splitlines()
        if line.strip() and line.strip() != "---" and not line.startswith("#")
    ]
    if not entries:
        return Check(
            name, False,
            f"'## [{version}]' is empty (are the entries still under [Unreleased]?)",
        )
    return Check(name, True, f"## [{version}], {len(entries)} lines")


def mcp_marker(root: Path) -> str | None:
    """The marker line the registry looks for, built from server.json's name."""
    server = _server_json(root)
    if not server or not isinstance(server.get("name"), str):
        return None
    return f"mcp-name: {server['name']}"


def check_marker_in(text: str, marker: str | None, where: str) -> Check:
    name = f"mcp-name marker in {where}"
    if marker is None:
        return Check(name, False, "server.json has no name")
    if marker in text:
        return Check(name, True, marker)
    return Check(name, False, f"'{marker}' not found")


def wheel_metadata(wheel: Path) -> str:
    """The METADATA file of a built wheel, as text."""
    with zipfile.ZipFile(wheel) as zf:
        for member in zf.namelist():
            if member.endswith(".dist-info/METADATA"):
                return zf.read(member).decode("utf-8")
    raise FileNotFoundError(f"no METADATA in {wheel.name}")


# -- stdio smoke test --------------------------------------------------


def drive_mcp(cmd: list[str], timeout: float = 60.0,
              expected_version: str | None = None) -> list[Check]:
    """Start ``apyrobo mcp`` with *cmd* and talk to it like an MCP client.

    *cmd* must start the server with ``--max-speed`` set to CLAMP_LIMIT.
    With *expected_version*, the version the server reports must equal it
    (catches an ``apyrobo/__version__.py`` that missed the bump).
    """
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "release-check", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "move_to",
            "arguments": {"x": 1.0, "y": 0.0, "speed": CLAMP_REQUEST}}},
    ]
    stdin = "".join(json.dumps(m) + "\n" for m in messages)
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        return [Check("apyrobo mcp starts", False, str(exc))]
    # The server refuses moves once stdin closes, so hold it open like a real
    # client until the last reply is in.
    watchdog = threading.Timer(timeout, proc.kill)
    watchdog.start()
    replies: dict[Any, dict[str, Any]] = {}
    try:
        with contextlib.suppress(OSError):
            proc.stdin.write(stdin)
            proc.stdin.flush()
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and "id" in msg:
                replies[msg["id"]] = msg
            if 3 in replies:
                break
    finally:
        with contextlib.suppress(OSError):
            proc.stdin.close()
        stderr = proc.stderr.read()
        proc.wait()
        watchdog.cancel()

    checks: list[Check] = []
    init = replies.get(1, {}).get("result") or {}
    server_info = init.get("serverInfo") or {}
    reported = server_info.get("version")
    checks.append(Check(
        "apyrobo mcp answers initialize",
        server_info.get("name") == "apyrobo"
        and (expected_version is None or reported == expected_version),
        (f"version {reported}" + (
            f", expected {expected_version}"
            if expected_version and reported != expected_version else ""))
        if server_info else
        (stderr.strip().splitlines() or ["no reply"])[-1],
    ))

    tools = [t.get("name") for t in (replies.get(2, {}).get("result") or {}).get("tools", [])]
    checks.append(Check(
        "apyrobo mcp lists its tools",
        "move_to" in tools,
        f"{len(tools)} tools" if tools else "no tools/list reply",
    ))

    call = replies.get(3, {}).get("result") or {}
    enforced = None
    try:
        payload = json.loads(call["content"][0]["text"])
        for item in payload.get("safety_interventions", []):
            if item.get("type") == "speed_clamped":
                enforced = item.get("enforced")
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        pass
    checks.append(Check(
        "an over-limit move is clamped",
        not call.get("isError") and enforced == CLAMP_LIMIT,
        f"{CLAMP_REQUEST} m/s requested, {enforced} m/s enforced"
        if enforced is not None else "no speed_clamped intervention in the reply",
    ))
    return checks


def mcp_command(apyrobo_exe: str) -> list[str]:
    return [apyrobo_exe, "mcp", "--robot", "mock://turtlebot4",
            "--max-speed", str(CLAMP_LIMIT)]


# -- build + install ---------------------------------------------------


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[bool, str]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    text = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout + proc.stderr)
    out = text.strip().splitlines()
    return proc.returncode == 0, (out[-1] if out else "")


def _has_module(name: str) -> bool:
    ok, _ = _run([sys.executable, "-c", f"import {name}"])
    return ok


def build_and_install(root: Path, work: Path, marker: str | None) -> list[Check]:
    checks: list[Check] = []
    missing = [m for m in ("build", "twine") if not _has_module(m)]
    if missing:
        return [Check("build tools installed", False,
                      f"pip install {' '.join(missing)}")]

    dist = work / "dist"
    ok, detail = _run([sys.executable, "-m", "build", "--sdist", "--wheel",
                       "--outdir", str(dist), str(root)])
    wheels = sorted(dist.glob("*.whl")) if ok else []
    sdists = sorted(dist.glob("*.tar.gz")) if ok else []
    checks.append(Check("sdist and wheel build", ok and bool(wheels) and bool(sdists),
                        ", ".join(p.name for p in wheels + sdists) or detail))
    if not (wheels and sdists):
        return checks
    wheel = wheels[0]

    ok, detail = _run([sys.executable, "-m", "twine", "check", "--strict",
                       *map(str, wheels + sdists)])
    checks.append(Check("twine check", ok, detail))

    try:
        checks.append(check_marker_in(wheel_metadata(wheel), marker, "wheel METADATA"))
    except (OSError, zipfile.BadZipFile, FileNotFoundError) as exc:
        checks.append(Check("mcp-name marker in wheel METADATA", False, str(exc)))

    venv = work / "venv"
    ok, detail = _run([sys.executable, "-m", "venv", str(venv)])
    bindir = venv / ("Scripts" if sys.platform == "win32" else "bin")
    python = bindir / ("python.exe" if sys.platform == "win32" else "python")
    if ok:
        ok, detail = _run([str(python), "-m", "pip", "install", "-q", str(wheel)])
    checks.append(Check("wheel installs in a fresh venv", ok, detail or wheel.name))
    if not ok:
        return checks

    exe = bindir / ("apyrobo.exe" if sys.platform == "win32" else "apyrobo")
    checks.extend(drive_mcp(mcp_command(str(exe)),
                            expected_version=pyproject_version(root)))
    return checks


# -- main --------------------------------------------------------------


def run(root: Path, *, static: bool, keep: bool) -> list[Check]:
    version = pyproject_version(root)
    marker = mcp_marker(root)
    readme = root / "README.md"
    checks = [
        check_versions(root),
        check_changelog(root, version),
        check_marker_in(readme.read_text(encoding="utf-8") if readme.is_file() else "",
                        marker, "README.md"),
    ]
    if static:
        return checks
    work = Path(tempfile.mkdtemp(prefix="apyrobo-release-"))
    try:
        checks.extend(build_and_install(root, work, marker))
    finally:
        if keep:
            print(f"  (kept build output in {work})")
        else:
            shutil.rmtree(work, ignore_errors=True)
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="project root (default: this checkout)")
    parser.add_argument("--static", action="store_true",
                        help="metadata checks only; no build, venv or server start")
    parser.add_argument("--keep", action="store_true",
                        help="keep the dist/ and venv in the temp folder")
    args = parser.parse_args(argv)

    version = pyproject_version(args.root) or "?"
    print(f"Release check for apyrobo {version} in {args.root}")
    checks = run(args.root, static=args.static, keep=args.keep)
    for c in checks:
        print(c.line())
    failed = [c for c in checks if not c.ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed"
          + ("" if not failed else " — not ready to release"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
