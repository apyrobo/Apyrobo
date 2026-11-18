"""``apyrobo mcp config`` refuses what the server would refuse at start.

Docs (docs/mcp.md, "Let apyrobo print the snippet") say config checks the
flags it passes on. A printed config the server then rejects shows up in the
client as a dead server with no reason given, so the check has to match.
"""

from __future__ import annotations

import io
import json
import math
from types import SimpleNamespace

import pytest

from apyrobo import mcp_clients
from apyrobo.mcp_server import load_policy_file


def _run(**kw):
    base = dict(client="cursor", launcher="apyrobo", name="apyrobo", cfg_robot=None,
                cfg_policy=None, cfg_policy_file=None, cfg_max_speed=None, cfg_no_go=None)
    base.update(kw)
    out, err = io.StringIO(), io.StringIO()
    code = mcp_clients.main(SimpleNamespace(**base), stdout=out, stderr=err,
                            which=lambda name: None)
    return code, out.getvalue(), err.getvalue()


@pytest.mark.parametrize("speed", [math.nan, math.inf, -math.inf, float("1e309")])
def test_non_finite_max_speed_is_refused(speed):
    code, out, err = _run(cfg_max_speed=speed)
    assert code == 1
    assert out == ""
    assert err.startswith("Error:") and "--max-speed must be positive" in err


def test_finite_max_speed_still_passes():
    code, out, _ = _run(cfg_max_speed=0.5)
    assert code == 0
    assert "--max-speed" in out and "0.5" in out


@pytest.mark.parametrize("content, message", [
    ('{"no_go_zones": [[1, 1, 2, 2]]}', "no_go_zones[0]"),
    ('{"max_speed": NaN}', "max_speed must be a positive number"),
    ('{"max_sped": 0.5}', "max_sped"),
    ("{not json", "not valid JSON"),
    ("[]", "mapping"),
    ('{"allowed_areas": []}', "allowed_areas"),
])
def test_policy_file_the_server_would_refuse_is_refused(tmp_path, content, message):
    policy = tmp_path / "policy.json"
    policy.write_text(content)
    with pytest.raises(ValueError):  # the server's own reader refuses it
        load_policy_file(policy)
    code, out, err = _run(cfg_policy_file=str(policy))
    assert code == 1
    assert out == ""
    assert err.startswith("Error:") and message in err


def test_good_policy_file_still_passes_as_an_absolute_path(tmp_path):
    policy = tmp_path / "policy.yaml"
    policy.write_text("max_speed: 0.3\nno_go_zones:\n  - 2,2,4,4\n")
    code, out, _ = _run(cfg_policy_file=str(policy))
    assert code == 0
    assert str(policy.resolve()) in json.dumps(out)


def test_ssh_policy_file_is_still_not_read_here():
    # The path is the robot's; config cannot see that file.
    flags = mcp_clients.server_args(policy_file="/etc/apyrobo/missing.yaml", remote=True)
    assert flags == ["--policy-file", "/etc/apyrobo/missing.yaml"]


@pytest.mark.parametrize("robot, message", [
    ("notauri", "Invalid robot URI: 'notauri'"),
    ("ros2://", "Invalid robot URI: 'ros2://'"),
    ("bogus://bot", "No adapter registered for scheme 'bogus'"),
])
def test_robot_the_server_would_refuse_is_refused(robot, message):
    code, out, err = _run(cfg_robot=robot)
    assert code == 1
    assert out == ""
    assert err.startswith("Error:") and message in err


def test_unknown_scheme_names_the_available_ones():
    _, _, err = _run(cfg_robot="bogus://bot")
    assert "Available:" in err and "mock" in err and "ros2" in err


@pytest.mark.parametrize("robot", ["mock://turtlebot4", "ros2://turtlebot4", "http://bot"])
def test_known_robot_schemes_still_pass(robot):
    code, out, _ = _run(cfg_robot=robot)
    assert code == 0
    assert robot in out


def test_robot_before_config_is_checked_too():
    code, out, err = _run(robot="bogus://bot")
    assert code == 1 and out == ""
    assert "No adapter registered for scheme 'bogus'" in err


def test_ssh_robot_scheme_is_not_checked_here_but_its_shape_is():
    # The robot may have adapter packages this machine does not.
    assert mcp_clients.server_args(robot="spot://bot", remote=True) == ["--robot", "spot://bot"]
    with pytest.raises(ValueError, match="Invalid robot URI"):
        mcp_clients.server_args(robot="spot", remote=True)


def test_server_refuses_what_config_refuses():
    # The two messages must stay about the same thing.
    from apyrobo.core.robot import Robot

    for robot in ("notauri", "bogus://bot"):
        with pytest.raises(ValueError) as server:
            Robot.discover(robot)
        with pytest.raises(ValueError) as config:
            mcp_clients.server_args(robot=robot)
        assert str(server.value).split("(")[0].split(".")[0].rstrip(" ,") in str(config.value)


def test_every_robot_uri_in_the_docs_has_a_name():
    # docs/first-real-robot.md once told users to run `--robot ros2://`, which
    # the server refuses at start.
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    pages = [root / "README.md", *sorted((root / "docs").rglob("*.md"))]
    seen = 0
    for page in pages:
        for uri in re.findall(r"--robot[ =]([a-z0-9_+.-]+://\S*)", page.read_text("utf-8")):
            uri = uri.rstrip("`\"',)")
            seen += 1
            assert uri.partition("://")[2], f"{page.name}: --robot {uri} has no robot name"
    assert seen > 10
