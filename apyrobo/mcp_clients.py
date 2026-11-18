"""Ready-to-paste MCP client configs for ``apyrobo mcp``.

``apyrobo mcp config <client>`` prints the snippet a client needs to start
``apyrobo mcp``, with the executable resolved to an absolute path on this
machine (GUI clients such as Claude Desktop, Cursor and VS Code do not read your
shell's PATH) and the config file location for this operating system. For
Claude Code and the OpenAI Codex CLI it prints the ``mcp add`` command
instead, and the client writes its own config.

With ``--ssh user@host`` the client on your laptop starts ``apyrobo mcp`` on
the robot over ssh instead, so MCP's stdio transport needs no new port or
transport. The server, and so the safety policy, runs on the robot.

It only prints. It never reads or writes the user's config files.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from apyrobo.__version__ import __version__

#: Source used with ``uvx --from`` for versions PyPI has no ``apyrobo mcp`` in.
GIT_SOURCE = "git+https://github.com/apyrobo/Apyrobo"
#: First release that ships ``apyrobo mcp`` on PyPI.
FIRST_PYPI_MCP_RELEASE = (4, 2, 0)
#: ``--provider`` values that plan through litellm, which only ``apyrobo[llm]``
#: installs; ``apyrobo mcp`` refuses to start with one of them without it.
LLM_PROVIDERS = ("llm", "tool_calling")

CLIENTS = ("claude-desktop", "cursor", "claude-code", "vscode", "codex")
#: Clients set up with a ``<cli> mcp add <name> -- <command>`` shell line.
ADD_COMMANDS = {"claude-code": "claude", "codex": "codex"}
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


def _own_script(name: str = "apyrobo") -> str | None:
    """The ``name`` script installed beside the Python running this process.

    That is the install the user just ran (a venv, a conda env, the system
    Python), whether or not its bin directory is on ``PATH``, so the config
    starts the same version that checked the flags.
    """
    exe = name + (".exe" if sys.platform == "win32" else "")
    for folder in (sysconfig.get_path("scripts"), str(Path(sys.executable).parent)):
        if folder:
            script = Path(folder) / exe
            if script.is_file() and os.access(script, os.X_OK):
                return str(script)
    return None


def find_executable(name: str) -> str | None:
    """``shutil.which``, except ``apyrobo`` is the running install's script first."""
    if name == "apyrobo":
        own = _own_script(name)
        if own:
            return own
    return shutil.which(name)


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
    allow_area: Sequence[str] | None = None,
    audit_log: str | None = None,
    provider: str | None = None,
) -> list[str]:
    """The ``apyrobo mcp`` flags to pass through, in a stable order.

    ``policy_file`` and ``audit_log`` are made absolute because a GUI client
    starts the server from its own working directory, not yours. With
    ``remote`` the server runs on another machine, so the paths are the
    robot's: they cannot be checked here and must already be absolute.
    """
    from apyrobo.mcp_server import load_policy_file, parse_zone

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
        load_policy_file(path)  # the server reads it the same way at start
        out += ["--policy-file", str(path)]
    if max_speed is not None:
        # NaN and inf pass a plain "> 0" test, and the server refuses both.
        if not math.isfinite(max_speed) or max_speed <= 0:
            raise ValueError(f"--max-speed must be positive, got {max_speed}")
        out += ["--max-speed", f"{max_speed:g}"]
    for zone in no_go or []:
        parse_zone(zone)  # fail now, not when the client starts the server
        out += ["--no-go", zone.replace(" ", "")]
    for area in allow_area or []:
        parse_zone(area, "allowed area")
        out += ["--allow-area", area.replace(" ", "")]
    if confirm_moves:
        out.append("--confirm-moves")
    if audit_log and remote:
        out += ["--audit-log", _remote_path(audit_log, "--audit-log")]
    elif audit_log:
        path = Path(audit_log).expanduser().resolve()
        if path.is_dir() or not path.parent.is_dir():
            raise ValueError(f"cannot write an audit log at {audit_log}: "
                             "give a file in a folder that exists")
        out += ["--audit-log", str(path)]
    if provider and provider != "rule":
        known = _known_providers()
        if provider not in known:
            raise ValueError(f"unknown --provider {provider!r}; choose one of "
                             f"{', '.join(known)}")
        out += ["--provider", provider]
    return out


def _known_providers() -> list[str]:
    """The ``--provider`` names the server can start with, as ``Agent`` reads them."""
    from apyrobo.skills.agent import _PROVIDERS

    return ["rule", "auto", *(name for name in _PROVIDERS if name != "rule")]


def _provider(flags: Sequence[str]) -> str | None:
    """The ``--provider`` value in ``flags``, if any."""
    for i, flag in enumerate(flags[:-1]):
        if flag == "--provider":
            return flags[i + 1]
    return None


def _uvx_source(version: str | None = None, flags: Sequence[str] = ()) -> list[str]:
    """``uvx`` arguments naming where the package comes from.

    A version that contains ``apyrobo mcp`` on PyPI needs none. Older ones
    install from GitHub, and so does anything unparsable: that form always works.
    An LLM ``--provider`` in ``flags`` adds the ``llm`` extra, since uvx
    installs only what it is told to and the server will not start without it.
    """
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", __version__ if version is None else version)
    on_pypi = bool(match) and tuple(map(int, match.groups())) >= FIRST_PYPI_MCP_RELEASE
    if _provider(flags) in LLM_PROVIDERS:
        return ["--from", "apyrobo[llm]" if on_pypi else f"apyrobo[llm] @ {GIT_SOURCE}"]
    if on_pypi:
        return []
    return ["--from", GIT_SOURCE]


def resolve_launch(
    flags: list[str], launcher: str = "auto", which: Which = find_executable
) -> Launch:
    """Pick the executable and arguments that start ``apyrobo mcp``.

    ``auto`` prefers ``uvx`` (an isolated, always-fresh environment, as in
    docs/mcp.md), then the installed ``apyrobo`` executable: by default the
    one beside the running Python, else the first on ``PATH``.
    """
    if launcher not in LAUNCHERS:
        raise ValueError(f"unknown launcher {launcher!r}; choose one of {', '.join(LAUNCHERS)}")
    uvx_args = [*_uvx_source(flags=flags), "apyrobo", "mcp", *flags]
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
        argv = ["uvx", *_uvx_source(flags=flags), "apyrobo", "mcp", *flags]
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
    if client == "codex":
        return "run the command in a terminal; Codex writes it to its config.toml for you"
    if client == "vscode":
        # VS Code's MCP configuration reference names this user file, under
        # $COPILOT_HOME when that is set, as the place for new setups.
        return ("~/.copilot/mcp-config.json "
                "(or $COPILOT_HOME/mcp-config.json if COPILOT_HOME is set)")
    raise ValueError(f"unknown client {client!r}; choose one of {', '.join(CLIENTS)}")


def render(client: str, launch: Launch, name: str = "apyrobo",
           platform: str | None = None) -> str:
    """The text to paste: JSON for desktop clients, a command for Claude Code and Codex."""
    platform = platform or sys.platform
    if client in ("claude-desktop", "cursor"):
        block = {"mcpServers": {name: {"command": launch.command, "args": launch.args}}}
        return json.dumps(block, indent=2)
    if client == "vscode":
        entry = {"type": "stdio", "command": launch.command, "args": launch.args,
                 "tools": ["*"]}
        return json.dumps({"mcpServers": {name: entry}}, indent=2)
    if client in ADD_COMMANDS:
        # Both take ``<cli> mcp add <name> -- <command> [args...]``; Codex's
        # usage is ``codex mcp add [OPTIONS] <NAME> (--url <URL> | -- <COMMAND>...)``.
        argv = [ADD_COMMANDS[client], "mcp", "add", name, "--", launch.command, *launch.args]
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
    elif client == "codex":
        lines.append("Run the command above in a terminal. Codex writes the entry to "
                     "its config.toml; 'codex mcp list' shows it.")
    else:
        lines.append(f"Merge this into: {config_location(client, platform)}")
        lines.append("Keep any other servers already under \"mcpServers\".")
    if client == "vscode":
        lines.append("An older VS Code setup with .vscode/mcp.json: paste the same "
                     "entry under its top-level \"servers\" key instead, without "
                     "\"tools\".")
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


def launches_uvx(launch: Launch) -> bool:
    """Whether this launch starts the server through ``uvx``.

    Local configs name ``uvx`` as the command; an ssh config carries the remote
    command as its last argument, with ``uvx`` at the front of it.
    """
    if Path(launch.command).stem == "uvx":
        return True
    return bool(launch.args) and shlex.split(launch.args[-1])[:1] == ["uvx"]


def ros2_hints(robot: str | None, launch: Launch) -> list[str]:
    """What a ``ros2://`` config needs beyond the flags, for stderr."""
    if not robot or not robot.startswith("ros2://"):
        return []
    lines = [
        "ros2:// needs a sourced ROS 2 environment, and a client starts the "
        "server with its own, not your shell's: wrap the command as "
        "\"command\": \"bash\" with \"args\": [\"-c\", \"source "
        "/opt/ros/<distro>/setup.bash && exec <the command above>\"].",
    ]
    if launches_uvx(launch):
        lines.append(
            "uvx runs in an isolated environment, which rclpy reaches through "
            "PYTHONPATH but the apt-installed numpy and transforms3d do not, "
            "and uv may pick a Python rclpy was not built for. Add "
            "--no-managed-python --python 3.12 (3.10 on Humble) "
            "--with 'numpy<2' --with transforms3d."
        )
    lines.append("The ros2:// section and troubleshooting in docs/mcp.md have the detail.")
    return lines


# ``apyrobo mcp``'s own --robot default; the server falls back to it anyway.
_DEFAULT_ROBOT = "mock://turtlebot4"


def _merged_flags(args: argparse.Namespace) -> dict:
    """Server flags given on either side of "config".

    ``apyrobo mcp --max-speed 0.5 config cursor`` must not print a config
    without the limit. A flag after "config" wins over the same flag before
    it; zones from both sides are kept.
    """
    def pick(name: str):
        value = getattr(args, f"cfg_{name}", None)
        return value if value is not None else getattr(args, name, None)

    robot = getattr(args, "cfg_robot", None)
    if robot is None:
        parent = getattr(args, "robot", None)
        robot = parent if parent and parent != _DEFAULT_ROBOT else None
    return {
        "robot": robot,
        "policy": pick("policy"),
        "policy_file": pick("policy_file"),
        "max_speed": pick("max_speed"),
        "no_go": list(getattr(args, "no_go", None) or [])
        + list(getattr(args, "cfg_no_go", None) or []),
        "allow_area": list(getattr(args, "allow_area", None) or [])
        + list(getattr(args, "cfg_allow_area", None) or []),
        "confirm_moves": bool(
            getattr(args, "confirm_moves", False) or getattr(args, "cfg_confirm_moves", False)
        ),
        "audit_log": pick("audit_log"),
        "provider": pick("provider"),
    }


def _check_llm_installed(flags: Sequence[str], launch: Launch) -> None:
    """Refuse a config whose server would stop at start for want of litellm.

    Only for this install's own ``apyrobo``: uvx adds the extra itself, and
    another executable may sit in a Python this process cannot look into.
    """
    provider = _provider(flags)
    if provider not in LLM_PROVIDERS or launch.command != _own_script():
        return
    if importlib.util.find_spec("litellm") is None:
        raise ValueError(
            f"--provider {provider} needs litellm, which this install does not "
            "have, so the server would not start. Install it with: "
            "pip install 'apyrobo[llm]', or use --launcher uvx"
        )


def main(args: argparse.Namespace, stdout: TextIO | None = None,
         stderr: TextIO | None = None, which: Which = find_executable) -> int:
    """Entry point for ``apyrobo mcp config``. Returns the exit code."""
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    ssh = getattr(args, "cfg_ssh", None)
    setup = getattr(args, "cfg_ssh_setup", None)
    flags_in = _merged_flags(args)
    robot = flags_in["robot"]
    try:
        if setup and not ssh:
            raise ValueError("--ssh-setup needs --ssh")
        flags = server_args(**flags_in, remote=bool(ssh))
        launcher = getattr(args, "launcher", "auto") or "auto"
        if ssh:
            launch = resolve_ssh_launch(ssh, flags, launcher, setup, which)
        else:
            launch = resolve_launch(flags, launcher, which)
            _check_llm_installed(flags, launch)
        client = args.client
        text = render(client, launch, name=getattr(args, "name", None) or "apyrobo")
        notes = hints(client, launch)
        if ssh:
            notes += ssh_hints(ssh, robot)
        notes += ros2_hints(robot, launch)
    except ValueError as exc:
        print(f"Error: {exc}", file=stderr)
        return 1
    print(text, file=stdout)
    for line in notes:
        print(line, file=stderr)
    return 0


__all__ = [
    "ADD_COMMANDS", "CLIENTS", "GIT_SOURCE", "LAUNCHERS", "Launch", "SSH_OPTIONS",
    "check_ssh_target", "config_location", "hints", "launches_uvx", "main",
    "remote_command", "render",
    "resolve_launch", "resolve_ssh_launch", "ros2_hints", "server_args", "ssh_hints",
]
