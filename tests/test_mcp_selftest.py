"""``apyrobo mcp selftest``: checks limits on mock:// and never touches a robot."""

from __future__ import annotations

import io
import json
from argparse import Namespace

import pytest

from apyrobo import cli, mcp_selftest
from apyrobo.mcp_server import build_server


def _args(**kw):
    base = dict(robot="mock://turtlebot4", policy=None, policy_file=None, max_speed=None,
                no_go=None, confirm_moves=False, st_robot=None, st_policy=None,
                st_policy_file=None, st_max_speed=None, st_no_go=None)
    base.update(kw)
    return Namespace(**base)


def _run(args):
    out, err = io.StringIO(), io.StringIO()
    code = mcp_selftest.main(args, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_defaults_pass_and_say_no_zones_are_set():
    code, out, err = _run(_args())
    assert code == 0
    assert err == ""
    assert "PASS  tools listed" in out
    assert "PASS  normal move" in out
    assert "PASS  over-limit move" in out
    assert "no no-go zones set" in out
    assert "3/3 checks passed" in out


def test_clamp_reports_the_users_limit():
    code, out, _ = _run(_args(st_max_speed=0.5))
    assert code == 0
    assert "requested 1 m/s, clamped to 0.5 m/s" in out


def test_each_zone_is_refused_and_a_zone_on_the_usual_target_is_avoided():
    code, out, _ = _run(_args(st_no_go=["2,2,4,4", "0.5,-0.5,1.5,0.5"]))
    assert code == 0
    assert "PASS  no-go zone #1: move to (3, 3) refused" in out
    assert "PASS  no-go zone #2: move to (1, 0) refused" in out
    assert "5/5 checks passed" in out


def test_policy_file_zone_names_appear(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(
        "name: dock\nmax_speed: 0.4\n"
        "no_go_zones:\n  - {name: stairs, x_min: 5, y_min: 5, x_max: 6, y_max: 6}\n"
    )
    code, out, _ = _run(_args(st_policy_file=str(path)))
    assert code == 0
    assert "policy=dock max_speed=0.4" in out
    assert "PASS  no-go zone stairs" in out


def test_flags_before_the_subcommand_count_too():
    code, out, _ = _run(_args(max_speed=0.3, no_go=["2,2,4,4"], st_no_go=["7,7,8,8"]))
    assert code == 0
    assert "max_speed=0.3 no_go_zones=2" in out


def test_zones_covering_every_near_target_still_find_a_free_target():
    zones = [{"x_min": -3.0, "y_min": -3.0, "x_max": 3.0, "y_max": 3.0}]
    assert mcp_selftest.free_targets(zones) == [(4.0, 0.0), (5.0, 0.0)]
    code, out, _ = _run(_args(st_no_go=["-3,-3,3,3"]))
    assert code == 0, out


@pytest.mark.parametrize("flag", ["st_robot", "robot"])
def test_refuses_anything_but_mock_before_building(monkeypatch, flag):
    def boom(**_):
        raise AssertionError("must not build a server for a real robot")

    monkeypatch.setattr("apyrobo.mcp_server.build_server", boom)
    code, out, err = _run(_args(**{flag: "ros2://robot"}))
    assert code == 1
    assert out == ""
    assert "only runs on mock://" in err


def test_bad_policy_file_is_a_readable_error(tmp_path):
    code, out, err = _run(_args(st_policy_file=str(tmp_path / "missing.yaml")))
    assert code == 1
    assert out == ""
    assert err.startswith("Error: cannot read policy file")


def test_confirm_moves_is_reported_as_not_tested():
    code, out, _ = _run(_args(confirm_moves=True))
    assert code == 0
    assert "--confirm-moves needs a client" in out


def test_real_moves_reach_the_mock_at_the_clamped_speed():
    server = build_server(max_speed=0.5, no_go_zones=[])
    checks = mcp_selftest.run_checks(server)
    assert all(c.ok for c in checks)
    history = server.toolbox.robot._adapter.move_history
    assert [m["speed"] for m in history] == [0.25, 0.5]


def test_failed_clamp_is_a_failure_and_exit_one(monkeypatch):
    real = mcp_selftest._Client.tool

    def unclamped(self, name, **arguments):
        is_error, text = real(self, name, **arguments)
        if name == "move_to" and not is_error:
            payload = json.loads(text)
            payload["safety_interventions"] = []
            text = json.dumps(payload)
        return is_error, text

    monkeypatch.setattr(mcp_selftest._Client, "tool", unclamped)
    code, out, _ = _run(_args())
    assert code == 1
    assert "FAIL  over-limit move" in out
    assert "2/3 checks passed" in out


def test_zone_that_lets_the_move_through_fails(monkeypatch):
    real = mcp_selftest._Client.tool

    def allowed(self, name, **arguments):
        is_error, text = real(self, name, **arguments)
        return (False, text) if name == "move_to" and is_error else (is_error, text)

    monkeypatch.setattr(mcp_selftest._Client, "tool", allowed)
    code, out, _ = _run(_args(st_no_go=["2,2,4,4"]))
    assert code == 1
    assert "FAIL  no-go zone #1" in out


def test_cli_wires_the_subcommand(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["apyrobo", "mcp", "selftest", "--max-speed", "0.5"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert "clamped to 0.5 m/s" in capsys.readouterr().out
