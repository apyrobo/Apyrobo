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
