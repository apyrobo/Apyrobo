"""Tests for ``apyrobo mcp config`` (apyrobo/mcp_clients.py)."""

from __future__ import annotations

import io
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from apyrobo import mcp_clients
from apyrobo.mcp_clients import GIT_SOURCE, Launch


def _which(found: dict[str, str]):
    return lambda name: found.get(name)


def _args(client, **kw):
    base = dict(client=client, launcher="auto", name="apyrobo", cfg_robot=None,
                cfg_policy=None, cfg_policy_file=None, cfg_max_speed=None, cfg_no_go=None)
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def before_pypi(monkeypatch):
    """A version whose PyPI release has no ``apyrobo mcp``."""
    monkeypatch.setattr(mcp_clients, "__version__", "4.1.0")


@pytest.fixture
def on_pypi(monkeypatch):
    """A version whose PyPI release has ``apyrobo mcp``."""
    monkeypatch.setattr(mcp_clients, "__version__", "4.2.0")


def _run(args, which):
    out, err = io.StringIO(), io.StringIO()
    code = mcp_clients.main(args, stdout=out, stderr=err, which=which)
    return code, out.getvalue(), err.getvalue()


def test_claude_desktop_uses_absolute_uvx_and_passes_flags(tmp_path, before_pypi):
    uvx = tmp_path / "uvx"
    uvx.write_text("")
    code, out, err = _run(
        _args("claude-desktop", cfg_max_speed=0.5, cfg_no_go=["2,2,4,4"], cfg_robot="mock://tb"),
        _which({"uvx": str(uvx)}),
    )
    assert code == 0
    server = json.loads(out)["mcpServers"]["apyrobo"]
    assert server["command"] == str(uvx.resolve())
    assert Path(server["command"]).is_absolute()
    assert server["args"] == ["--from", GIT_SOURCE, "apyrobo", "mcp", "--robot", "mock://tb",
                              "--max-speed", "0.5", "--no-go", "2,2,4,4"]
    assert "fully quit Claude Desktop" in err


def test_falls_back_to_installed_apyrobo(tmp_path):
    exe = tmp_path / "apyrobo"
    exe.write_text("")
    code, out, _ = _run(_args("cursor"), _which({"apyrobo": str(exe)}))
    assert code == 0
    server = json.loads(out)["mcpServers"]["apyrobo"]
    assert server == {"command": str(exe.resolve()), "args": ["mcp"]}


@pytest.mark.parametrize("version, expected", [
    ("4.2.0", []),
    ("4.2.1", []),
    ("4.10.0", []),
    ("5.0.0", []),
    ("4.2.0.dev1", []),
    ("4.1.0", ["--from", GIT_SOURCE]),
    ("4.1.9", ["--from", GIT_SOURCE]),
    ("4.0.10", ["--from", GIT_SOURCE]),
    ("3.9.0", ["--from", GIT_SOURCE]),
    ("unknown", ["--from", GIT_SOURCE]),
])
def test_uvx_source_follows_the_version(version, expected):
    assert mcp_clients._uvx_source(version) == expected


def test_config_prints_the_pypi_form_once_released(on_pypi, tmp_path):
    uvx = tmp_path / "uvx"
    uvx.write_text("")
    launch = mcp_clients.resolve_launch(["--max-speed", "0.5"], "auto",
                                        _which({"uvx": str(uvx)}))
    assert launch.args == ["apyrobo", "mcp", "--max-speed", "0.5"]


def test_bare_uvx_note_uses_the_pypi_form_once_released(on_pypi):
    assert mcp_clients.resolve_launch([], "auto", _which({})).args == ["apyrobo", "mcp"]


def test_ssh_prints_the_pypi_form_once_released(on_pypi):
    argv = shlex.split(mcp_clients.remote_command(["--max-speed", "0.5"], "uvx"))
    assert argv == ["uvx", "apyrobo", "mcp", "--max-speed", "0.5"]


def test_launcher_apyrobo_beats_uvx(tmp_path):
    flags: list[str] = []
    launch = mcp_clients.resolve_launch(
        flags, "apyrobo", _which({"uvx": "/x/uvx", "apyrobo": str(tmp_path / "apyrobo")}))
    assert launch.args == ["mcp"]
    assert launch.command.endswith("apyrobo")


def test_nothing_on_path_uses_bare_uvx_with_a_note():
    code, out, err = _run(_args("claude-desktop"), _which({}))
    assert code == 0
    assert json.loads(out)["mcpServers"]["apyrobo"]["command"] == "uvx"
    assert "Note:" in err and "which uvx" in err


def test_claude_code_prints_a_shell_command(tmp_path):
    launch = Launch("/opt/my tools/uvx",
                    ["--from", GIT_SOURCE, "apyrobo", "mcp", "--max-speed", "0.5"])
    text = mcp_clients.render("claude-code", launch, platform="linux")
    assert shlex.split(text) == ["claude", "mcp", "add", "apyrobo", "--", "/opt/my tools/uvx",
                                 "--from", GIT_SOURCE, "apyrobo", "mcp", "--max-speed", "0.5"]


def test_confirm_moves_passes_through(tmp_path):
    uvx = tmp_path / "uvx"
    uvx.write_text("")
    code, out, _ = _run(_args("cursor", cfg_max_speed=0.5, cfg_confirm_moves=True),
                        _which({"uvx": str(uvx)}))
    assert code == 0
    args = json.loads(out)["mcpServers"]["apyrobo"]["args"]
    assert args[-3:] == ["--max-speed", "0.5", "--confirm-moves"]
    assert mcp_clients.server_args() == []


def test_cli_parses_config_confirm_moves(monkeypatch):
    import apyrobo.cli as cli

    seen = {}
    monkeypatch.setattr(cli, "cmd_mcp", lambda args: seen.update(vars(args)))
    monkeypatch.setattr(sys, "argv", ["apyrobo", "mcp", "config", "cursor", "--confirm-moves"])
    cli.main()
    assert seen["cfg_confirm_moves"] is True


def test_policy_file_is_made_absolute(tmp_path, monkeypatch):
    policy = tmp_path / "lab.json"
    policy.write_text("{}")
    monkeypatch.chdir(tmp_path)
    flags = mcp_clients.server_args(policy_file="lab.json")
    assert flags == ["--policy-file", str(policy.resolve())]


@pytest.mark.parametrize("kw, message", [
    ({"cfg_policy_file": "/no/such/policy.json"}, "policy file not found"),
    ({"cfg_no_go": ["1,2"]}, "no-go zone"),
    ({"cfg_max_speed": 0.0}, "--max-speed must be positive"),
])
def test_bad_flags_are_readable_errors(kw, message):
    code, out, err = _run(_args("cursor", **kw), _which({}))
    assert code == 1
    assert out == ""
    assert err.startswith("Error:") and message in err


@pytest.mark.parametrize("platform, expected", [
    ("darwin", "~/Library/Application Support/Claude/claude_desktop_config.json"),
    ("win32", r"%APPDATA%\Claude\claude_desktop_config.json"),
])
def test_claude_desktop_location_per_os(platform, expected):
    assert mcp_clients.config_location("claude-desktop", platform) == expected


def test_cli_prints_config_and_writes_nothing(tmp_path):
    env_home = tmp_path / "home"
    env_home.mkdir()
    proc = subprocess.run(
        [sys.executable, "-m", "apyrobo.cli", "mcp", "config", "cursor", "--launcher", "apyrobo",
         "--max-speed", "0.3"],
        capture_output=True, text=True, timeout=60,
        env={"HOME": str(env_home), "PATH": "",
             "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )
    assert proc.returncode == 0, proc.stderr
    server = json.loads(proc.stdout)["mcpServers"]["apyrobo"]
    assert server["args"] == ["mcp", "--max-speed", "0.3"]
    assert list(env_home.iterdir()) == []


def test_plain_mcp_command_still_serves(monkeypatch):
    import apyrobo.cli as cli

    seen = {}
    monkeypatch.setattr(cli, "cmd_mcp", lambda args: seen.update(vars(args)))
    monkeypatch.setattr(sys, "argv", ["apyrobo", "mcp", "--max-speed", "0.5", "--no-go", "1,1,2,2"])
    cli.main()
    assert seen["mcp_command"] is None
    assert seen["max_speed"] == 0.5
    assert seen["no_go"] == ["1,1,2,2"]


# --ssh: the client on a laptop, the server on the robot


def _ssh_argv(out):
    entry = json.loads(out)["mcpServers"]["apyrobo"]
    return [entry["command"], *entry["args"]]


def test_ssh_config_starts_the_server_remotely(tmp_path):
    ssh = tmp_path / "ssh"
    ssh.write_text("")
    code, out, err = _run(
        _args("claude-desktop", cfg_ssh="ubuntu@robot.local", cfg_robot="ros2://",
              cfg_max_speed=0.5, cfg_no_go=["2,2,4,4"]),
        _which({"ssh": str(ssh), "uvx": "/local/uvx"}),
    )
    assert code == 0
    argv = _ssh_argv(out)
    assert argv[0] == str(ssh.resolve())
    assert argv[1:-2] == list(mcp_clients.SSH_OPTIONS)
    assert "-T" in argv and "BatchMode=yes" in argv
    assert argv[-2] == "ubuntu@robot.local"
    assert shlex.split(argv[-1]) == [
        "apyrobo", "mcp", "--robot", "ros2://", "--max-speed", "0.5", "--no-go", "2,2,4,4",
    ]
    assert "/local/uvx" not in out  # this machine's paths mean nothing on the robot
    assert "key-based login" in err


def test_ssh_setup_is_sourced_before_the_server():
    command = mcp_clients.remote_command(["--robot", "ros2://"], setup="/opt/ros/jazzy/setup.bash")
    argv = shlex.split(command)
    assert argv[:2] == ["bash", "-c"]
    assert argv[2] == ". /opt/ros/jazzy/setup.bash && exec apyrobo mcp --robot ros2://"


def test_ssh_uvx_launcher_runs_uvx_on_the_robot(before_pypi):
    argv = shlex.split(mcp_clients.remote_command([], "uvx"))
    assert argv == ["uvx", "--from", GIT_SOURCE, "apyrobo", "mcp"]


def test_ssh_policy_file_is_the_robots_path_and_not_checked_here():
    code, out, _ = _run(
        _args("cursor", cfg_ssh="robot", cfg_policy_file="/etc/apyrobo/policy.yaml"),
        _which({"ssh": "/usr/bin/ssh"}),
    )
    assert code == 0
    assert "--policy-file /etc/apyrobo/policy.yaml" in _ssh_argv(out)[-1]


@pytest.mark.parametrize("kwargs", [
    {"cfg_ssh": "robot", "cfg_policy_file": "policy.yaml"},
    {"cfg_ssh": "robot", "cfg_ssh_setup": "~/env.sh"},
    {"cfg_ssh_setup": "/opt/ros/jazzy/setup.bash"},
])
def test_ssh_bad_paths_and_flags_are_readable_errors(kwargs):
    code, out, err = _run(_args("cursor", **kwargs), _which({"ssh": "/usr/bin/ssh"}))
    assert code == 1
    assert out == ""
    assert err.startswith("Error:")


@pytest.mark.parametrize("target", [
    "-oProxyCommand=touch /tmp/x", "user@", "@host", "a b", "host;id", "", "user@host:2222",
])
def test_ssh_target_that_is_not_a_host_is_refused(target):
    with pytest.raises(ValueError, match="--ssh"):
        mcp_clients.check_ssh_target(target)


@pytest.mark.parametrize("target", ["robot", "ubuntu@robot.local", "pi@10.0.0.7", "me@fe80::1"])
def test_ssh_target_accepts_hosts(target):
    assert mcp_clients.check_ssh_target(target) == target


def test_ssh_without_ssh_on_path_uses_bare_name_and_says_so():
    code, out, err = _run(_args("cursor", cfg_ssh="robot"), _which({}))
    assert code == 0
    assert _ssh_argv(out)[0] == "ssh"
    assert "ssh was not found" in err
    assert "mock://turtlebot4" in err  # no --robot: say what the robot will run


def test_ssh_claude_code_command_is_one_shell_line():
    code, out, _ = _run(
        _args("claude-code", cfg_ssh="robot", cfg_robot="mock://tb"),
        _which({"ssh": "/usr/bin/ssh"}),
    )
    assert code == 0
    argv = shlex.split(out)
    assert argv[:5] == ["claude", "mcp", "add", "apyrobo", "--"]
    assert argv[-1] == "apyrobo mcp --robot mock://tb"


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell stand-in for ssh")
def test_ssh_command_survives_the_remote_shell(tmp_path):
    """ssh joins its arguments and a shell re-parses them: run exactly that."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    apyrobo = bin_dir / "apyrobo"
    apyrobo.write_text(f'#!/bin/sh\nexec {sys.executable} -m apyrobo.cli "$@"\n')
    apyrobo.chmod(0o755)
    setup = tmp_path / "my setup.sh"
    setup.write_text(f'export PATH="{bin_dir}:$PATH"\n')

    flags = mcp_clients.server_args(
        robot="mock://tb", no_go=["2,2,4,4"], policy_file=None, max_speed=0.5, remote=True,
    )
    remote = mcp_clients.remote_command(flags, setup=str(setup))
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    init = {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    }}
    done = subprocess.run(
        ["/bin/sh", "-c", remote],
        input=json.dumps(init) + "\n" + json.dumps(request) + "\n",
        capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 0, done.stderr
    replies = [json.loads(line) for line in done.stdout.splitlines()]
    names = {tool["name"] for tool in replies[-1]["result"]["tools"]}
    assert "move_to" in names
    assert "robot=mock://tb" in done.stderr


def test_ros2_config_says_to_source_ros_and_pin_uvxs_python(tmp_path, before_pypi):
    """`apyrobo mcp config --robot ros2://…` warns about the two uvx traps."""
    uvx = tmp_path / "uvx"
    uvx.write_text("")
    code, out, err = _run(
        _args("claude-desktop", cfg_robot="ros2://burger"), _which({"uvx": str(uvx)}),
    )
    assert code == 0
    assert "setup.bash" in err
    assert "--no-managed-python --python 3.12" in err
    assert "transforms3d" in err
    assert "docs/mcp.md" in err


def test_the_uvx_pin_note_is_left_out_without_uvx(tmp_path, before_pypi):
    """The installed executable sees the system packages, so it needs no pin."""
    apyrobo = tmp_path / "apyrobo"
    apyrobo.write_text("")
    code, out, err = _run(
        _args("claude-desktop", cfg_robot="ros2://burger", launcher="apyrobo"),
        _which({"apyrobo": str(apyrobo)}),
    )
    assert code == 0
    assert "setup.bash" in err
    assert "transforms3d" not in err


def test_mock_configs_say_nothing_about_ros(tmp_path, before_pypi):
    uvx = tmp_path / "uvx"
    uvx.write_text("")
    code, out, err = _run(
        _args("claude-desktop", cfg_robot="mock://tb"), _which({"uvx": str(uvx)}),
    )
    assert code == 0
    assert "setup.bash" not in err
    assert "transforms3d" not in err


def test_ssh_uvx_config_also_gets_the_pin_note(tmp_path, before_pypi):
    ssh = tmp_path / "ssh"
    ssh.write_text("")
    code, out, err = _run(
        _args("claude-desktop", cfg_ssh="ubuntu@robot.local", cfg_robot="ros2://burger",
              launcher="uvx"),
        _which({"ssh": str(ssh)}),
    )
    assert code == 0
    assert mcp_clients.launches_uvx(
        Launch("ssh", [*mcp_clients.SSH_OPTIONS, "host", "uvx --from x apyrobo mcp"])
    )
    assert "--with 'numpy<2'" in err
