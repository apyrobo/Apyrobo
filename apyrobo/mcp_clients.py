"""Ready-to-paste MCP client configs for ``apyrobo mcp``.

``apyrobo mcp config <client>`` prints the snippet a client needs to start
``apyrobo mcp``, with the executable resolved to an absolute path on this
machine (GUI clients such as Claude Desktop and Cursor do not read your
shell's PATH) and the config file location for this operating system.

With ``--ssh user@host`` the client on your laptop starts ``apyrobo mcp`` on
the robot over ssh instead, so MCP's stdio transport needs no new port or
transport. The server, and so the safety policy, runs on the robot.

It only prints. It never reads or writes the user's config files.
"""

from __future__ import annotations

import argparse
import json
import re
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

#: ssh options for an unattended stdio pipe. BatchMode makes a missing key fail
#: at once: over stdio there is nobody to answer a password prompt.
SSH_OPTIONS = (
    "-T",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=5",
    "-o", "ServerAliveCountMax=3",
)

_SSH_TARGET = re.compile(r"[A-Za-z0-9_.\[\]:%-]+(@[A-Za-z0-9_.\[\]:%-]+)?(?<![:@])\Z")
_HOST_WITH_PORT = re.compile(r"(.*@)?[^:]+:\d+\Z")

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
    confirm_moves: bool = False,
    remote: bool = False,
) -> list[str]:
    """The ``apyrobo mcp`` flags to pass through, in a stable order.

    ``policy_file`` is made absolute because a GUI client starts the server
    from its own working directory, not yours. With ``remote`` the server runs
    on another machine, so the path is the robot's: it cannot be checked here
    and must already be absolute.
    """
    from apyrobo.mcp_server import parse_zone

    out: list[str] = []
    if robot:
        out += ["--robot", robot]
    if policy:
        out += ["--policy", policy]
    if policy_file and remote:
        out += ["--policy-file", _remote_path(policy_file, "--policy-file")]
    elif policy_file:
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
    if confirm_moves:
        out.append("--confirm-moves")
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


def _remote_path(path: str, flag: str) -> str:
    if not path.startswith("/"):
        raise ValueError(f"with --ssh, {flag} is a path on the robot and must be "
                         f"absolute, got {path!r}")
    return path


def check_ssh_target(target: str) -> str:
    """Validate ``user@host``, so it can't be read as an ssh option."""
    if target.startswith("-") or not _SSH_TARGET.match(target):
        raise ValueError(f"--ssh expects user@host or a host name, got {target!r}")
    if _HOST_WITH_PORT.match(target):
        raise ValueError(f"--ssh takes no port ({target!r}): give the host a Port "
                         "in ~/.ssh/config and use its name")
    return target


def remote_command(
    flags: list[str], launcher: str = "auto", setup: str | None = None
) -> str:
    """The one string ssh hands to the robot's shell.

    ssh joins its arguments with spaces and the remote shell parses the result,
    so this is quoted here, once. ``setup`` is sourced in bash first, for what a
    non-interactive ssh shell lacks: ROS 2 most of all.
    """
    if launcher == "uvx":
        argv = ["uvx", "--from", GIT_SOURCE, "apyrobo", "mcp", *flags]
    else:
        argv = ["apyrobo", "mcp", *flags]
    command = shlex.join(argv)
    if setup is None:
        return command
    script = f". {shlex.quote(_remote_path(setup, '--ssh-setup'))} && exec {command}"
    return shlex.join(["bash", "-c", script])


def resolve_ssh_launch(
    target: str,
    flags: list[str],
    launcher: str = "auto",
    setup: str | None = None,
    which: Which = shutil.which,
) -> Launch:
    """Start ``apyrobo mcp`` on another machine through this machine's ssh."""
    if launcher not in LAUNCHERS:
        raise ValueError(f"unknown launcher {launcher!r}; choose one of {', '.join(LAUNCHERS)}")
    check_ssh_target(target)
    args = [*SSH_OPTIONS, target, remote_command(flags, launcher, setup)]
    ssh = which("ssh")
    if ssh:
        return Launch(str(Path(ssh).resolve()), args)
    return Launch("ssh", args, note=(
        "ssh was not found on PATH, so the config uses the bare name 'ssh'. "
        "Install an OpenSSH client, then replace it with its absolute path."
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


def ssh_hints(target: str, robot: str | None) -> list[str]:
    """What an ssh config needs that a local one does not."""
    lines = [
        f"Needs key-based login, with no passphrase prompt: 'ssh -o BatchMode=yes "
        f"{target} true' must work from this machine.",
        "The server runs on the robot, so apyrobo must be installed there and any "
        "--policy-file path is the robot's.",
    ]
    if not robot:
        lines.append("No --robot given, so the server on the robot uses its default, "
                     "mock://turtlebot4.")
    return lines


def main(args: argparse.Namespace, stdout: TextIO | None = None,
         stderr: TextIO | None = None, which: Which = shutil.which) -> int:
    """Entry point for ``apyrobo mcp config``. Returns the exit code."""
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    ssh = getattr(args, "cfg_ssh", None)
    setup = getattr(args, "cfg_ssh_setup", None)
    robot = getattr(args, "cfg_robot", None)
    try:
        if setup and not ssh:
            raise ValueError("--ssh-setup needs --ssh")
        flags = server_args(
            robot=robot,
            policy=getattr(args, "cfg_policy", None),
            policy_file=getattr(args, "cfg_policy_file", None),
            max_speed=getattr(args, "cfg_max_speed", None),
            no_go=getattr(args, "cfg_no_go", None),
            confirm_moves=bool(getattr(args, "cfg_confirm_moves", False)),
            remote=bool(ssh),
        )
        launcher = getattr(args, "launcher", "auto") or "auto"
        if ssh:
            launch = resolve_ssh_launch(ssh, flags, launcher, setup, which)
        else:
            launch = resolve_launch(flags, launcher, which)
        client = args.client
        text = render(client, launch, name=getattr(args, "name", None) or "apyrobo")
        notes = hints(client, launch)
        if ssh:
            notes += ssh_hints(ssh, robot)
    except ValueError as exc:
        print(f"Error: {exc}", file=stderr)
        return 1
    print(text, file=stdout)
    for line in notes:
        print(line, file=stderr)
    return 0


__all__ = [
    "CLIENTS", "GIT_SOURCE", "LAUNCHERS", "Launch", "SSH_OPTIONS", "check_ssh_target",
    "config_location", "hints", "main", "remote_command", "render", "resolve_launch",
    "resolve_ssh_launch", "server_args", "ssh_hints",
]
