"""Ready-to-paste MCP client configs for ``apyrobo mcp``.

``apyrobo mcp config <client>`` prints the snippet a client needs to start
``apyrobo mcp``, with the executable resolved to an absolute path on this
machine (GUI clients such as Claude Desktop and Cursor do not read your
shell's PATH) and the config file location for this operating system.

It only prints. It never reads or writes the user's config files.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

#: Source used with ``uvx --from`` until a PyPI release ships ``apyrobo mcp``.
GIT_SOURCE = "git+https://github.com/apyrobo/Apyrobo"

CLIENTS = ("claude-desktop", "cursor", "claude-code")
LAUNCHERS = ("auto", "uvx", "apyrobo")

Which = Callable[[str], str | None]


@dataclass(frozen=True)
class Launch:
    """How the client should start the server."""

    command: str
    args: list[str]
    note: str | None = None


def server_args(
    robot: str | None = None,
    policy: str | None = None,
    policy_file: str | None = None,
    max_speed: float | None = None,
    no_go: Sequence[str] | None = None,
) -> list[str]:
    """The ``apyrobo mcp`` flags to pass through, in a stable order.

    ``policy_file`` is made absolute because a GUI client starts the server
    from its own working directory, not yours.
    """
    from apyrobo.mcp_server import parse_zone

    out: list[str] = []
    if robot:
        out += ["--robot", robot]
    if policy:
        out += ["--policy", policy]
    if policy_file:
        path = Path(policy_file).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"policy file not found: {policy_file}")
        out += ["--policy-file", str(path)]
    if max_speed is not None:
        if max_speed <= 0:
            raise ValueError(f"--max-speed must be positive, got {max_speed}")
        out += ["--max-speed", f"{max_speed:g}"]
    for zone in no_go or []:
        parse_zone(zone)  # fail now, not when the client starts the server
        out += ["--no-go", zone.replace(" ", "")]
    return out


def resolve_launch(
    flags: list[str], launcher: str = "auto", which: Which = shutil.which
) -> Launch:
    """Pick the executable and arguments that start ``apyrobo mcp``.

    ``auto`` prefers ``uvx`` (an isolated, always-fresh environment, as in
    docs/mcp.md), then the installed ``apyrobo`` executable.
    """
    if launcher not in LAUNCHERS:
        raise ValueError(f"unknown launcher {launcher!r}; choose one of {', '.join(LAUNCHERS)}")
    uvx_args = ["--from", GIT_SOURCE, "apyrobo", "mcp", *flags]
    apyrobo_args = ["mcp", *flags]

    if launcher in ("auto", "uvx"):
        uvx = which("uvx")
        if uvx:
            return Launch(str(Path(uvx).resolve()), uvx_args)
        if launcher == "uvx":
            return Launch("uvx", uvx_args, note=(
                "uvx was not found on PATH, so the config uses the bare name "
                "'uvx'. Install uv, then replace it with the output of "
                "'which uvx' (or 'where uvx' on Windows)."
            ))
    apyrobo = which("apyrobo")
    if apyrobo:
        return Launch(str(Path(apyrobo).resolve()), apyrobo_args)
    if launcher == "apyrobo":
        return Launch("apyrobo", apyrobo_args, note=(
            "the apyrobo executable was not found on PATH, so the config uses "
            "the bare name 'apyrobo'. Replace it with its absolute path."
        ))
    return Launch("uvx", uvx_args, note=(
        "neither uvx nor apyrobo was found on PATH, so the config uses the "
        "bare name 'uvx'. Install uv (https://docs.astral.sh/uv/), then "
        "replace 'uvx' with the output of 'which uvx' (or 'where uvx' on "
        "Windows)."
    ))


def config_location(client: str, platform: str | None = None) -> str:
    """Where this client keeps its MCP config, in words, for this OS."""
    platform = platform or sys.platform
    if client == "claude-desktop":
        if platform == "darwin":
            return "~/Library/Application Support/Claude/claude_desktop_config.json"
        if platform.startswith("win"):
            return r"%APPDATA%\Claude\claude_desktop_config.json"
        return ("Claude Desktop ships for macOS and Windows only; there the file is "
                "~/Library/Application Support/Claude/claude_desktop_config.json "
                r"(macOS) or %APPDATA%\Claude\claude_desktop_config.json (Windows)")
    if client == "cursor":
        if platform.startswith("win"):
            return r"%USERPROFILE%\.cursor\mcp.json (or .cursor\mcp.json in a project)"
        return "~/.cursor/mcp.json (or .cursor/mcp.json in a project)"
    if client == "claude-code":
        return "run the command in a terminal; Claude Code stores it for you"
    raise ValueError(f"unknown client {client!r}; choose one of {', '.join(CLIENTS)}")


def render(client: str, launch: Launch, name: str = "apyrobo",
           platform: str | None = None) -> str:
    """The text to paste: JSON for desktop clients, a command for Claude Code."""
    platform = platform or sys.platform
    if client in ("claude-desktop", "cursor"):
        block = {"mcpServers": {name: {"command": launch.command, "args": launch.args}}}
        return json.dumps(block, indent=2)
    if client == "claude-code":
        argv = ["claude", "mcp", "add", name, "--", launch.command, *launch.args]
        if platform.startswith("win"):
            return subprocess.list2cmdline(argv)
        return shlex.join(argv)
    raise ValueError(f"unknown client {client!r}; choose one of {', '.join(CLIENTS)}")


def hints(client: str, launch: Launch, platform: str | None = None) -> list[str]:
    """Short lines for stderr: where it goes and what to do next."""
    lines = []
    if launch.note:
        lines.append(f"Note: {launch.note}")
    if client == "claude-code":
        lines.append("Run the command above in a terminal.")
    else:
        lines.append(f"Merge this into: {config_location(client, platform)}")
        lines.append("Keep any other servers already under \"mcpServers\".")
    if client == "claude-desktop":
        lines.append("Then fully quit Claude Desktop (menu bar or system tray) "
                     "and open it again; it reads the config only at start-up.")
    return lines


def main(args: argparse.Namespace, stdout: TextIO | None = None,
         stderr: TextIO | None = None, which: Which = shutil.which) -> int:
    """Entry point for ``apyrobo mcp config``. Returns the exit code."""
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    try:
        flags = server_args(
            robot=getattr(args, "cfg_robot", None),
            policy=getattr(args, "cfg_policy", None),
            policy_file=getattr(args, "cfg_policy_file", None),
            max_speed=getattr(args, "cfg_max_speed", None),
            no_go=getattr(args, "cfg_no_go", None),
        )
        launch = resolve_launch(flags, getattr(args, "launcher", "auto") or "auto", which)
        client = args.client
        text = render(client, launch, name=getattr(args, "name", None) or "apyrobo")
        notes = hints(client, launch)
    except ValueError as exc:
        print(f"Error: {exc}", file=stderr)
        return 1
    print(text, file=stdout)
    for line in notes:
        print(line, file=stderr)
    return 0


__all__ = [
    "CLIENTS", "GIT_SOURCE", "LAUNCHERS", "Launch", "config_location", "hints",
    "main", "render", "resolve_launch", "server_args",
]
