"""``apyrobo mcp config --ssh`` gives ROS 2 sourcing advice that fits ssh.

Over ssh the server runs on the robot and ``--ssh-setup`` sources ROS 2
there. The local-config advice (wrap the command in ``bash -c 'source …'``)
would have the laptop source a ROS 2 it may not have, so it must not be
printed for an ssh config.
"""
from __future__ import annotations

import io
from types import SimpleNamespace

from apyrobo import mcp_clients

WRAP = '"command": "bash"'


def _run(**kw):
    base = dict(client="claude-desktop", launcher="apyrobo", name="apyrobo",
                cfg_robot="ros2://myrobot", cfg_policy=None, cfg_policy_file=None,
                cfg_max_speed=0.5, cfg_no_go=["2,2,4,4"])
    base.update(kw)
    out, err = io.StringIO(), io.StringIO()
    which = {"ssh": "/usr/bin/ssh", "apyrobo": "/usr/bin/apyrobo"}.get
    code = mcp_clients.main(SimpleNamespace(**base), stdout=out, stderr=err, which=which)
    return code, out.getvalue(), err.getvalue()


def test_documented_ssh_line_does_not_ask_to_wrap_the_laptop_command():
    # docs/mcp.md, "Client on your laptop, robot elsewhere"
    code, out, err = _run(cfg_ssh="ubuntu@robot.local",
                          cfg_ssh_setup="/opt/ros/jazzy/setup.bash")
    assert code == 0
    assert ". /opt/ros/jazzy/setup.bash && exec apyrobo mcp" in out
    assert WRAP not in err
    assert "<distro>" not in err


def test_ssh_without_setup_points_at_ssh_setup():
    code, out, err = _run(cfg_ssh="ubuntu@robot.local")
    assert code == 0
    assert WRAP not in err
    assert "--ssh-setup /opt/ros/<distro>/setup.bash" in err
    assert "on the robot" in err


def test_ssh_with_mock_robot_says_nothing_about_ros():
    code, out, err = _run(cfg_ssh="ubuntu@robot.local", cfg_robot="mock://tb")
    assert code == 0
    assert "setup.bash" not in err


def test_local_ros2_config_still_asks_to_wrap_the_command():
    code, out, err = _run()
    assert code == 0
    assert WRAP in err
    assert "--ssh-setup" not in err


def test_ssh_uvx_launcher_keeps_the_uvx_note(monkeypatch):
    monkeypatch.setattr(mcp_clients, "__version__", "4.1.0")
    code, out, err = _run(cfg_ssh="ubuntu@robot.local", launcher="uvx",
                          cfg_ssh_setup="/opt/ros/jazzy/setup.bash")
    assert code == 0
    assert "--with 'numpy<2'" in err
    assert WRAP not in err


def test_ros2_hints_defaults_keep_the_local_advice():
    launch = mcp_clients.Launch("/usr/bin/apyrobo", ["mcp"])
    lines = mcp_clients.ros2_hints("ros2://burger", launch)
    assert any(WRAP in ln for ln in lines)


def test_launches_uvx_sees_through_the_ssh_setup_wrapper():
    wrapped = mcp_clients.remote_command(["--robot", "ros2://b"], "uvx",
                                         "/opt/ros/jazzy/setup.bash")
    assert mcp_clients.launches_uvx(
        mcp_clients.Launch("ssh", [*mcp_clients.SSH_OPTIONS, "host", wrapped]))
    plain = mcp_clients.remote_command(["--robot", "ros2://b"], "apyrobo",
                                       "/opt/ros/jazzy/setup.bash")
    assert not mcp_clients.launches_uvx(
        mcp_clients.Launch("ssh", [*mcp_clients.SSH_OPTIONS, "host", plain]))
