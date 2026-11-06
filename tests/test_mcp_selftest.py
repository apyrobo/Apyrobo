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
                no_go=None, allow_area=None, confirm_moves=False, st_robot=None, st_policy=None,
                st_policy_file=None, st_max_speed=None, st_no_go=None, st_allow_area=None)
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
    assert "PASS  move across no-go zone #1" in out
    assert "PASS  move across no-go zone #2" in out
    assert "7/7 checks passed" in out


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


def _warnings(out):
    return [line for line in out.splitlines() if line.startswith("warn  ")]


def test_defaults_warn_about_everything_left_unset_without_failing():
    code, out, _ = _run(_args())
    assert code == 0
    warnings = _warnings(out)
    assert len(warnings) == 5
    assert "no no-go zones set" in warnings[0]
    assert "no allowed area set" in warnings[1]
    assert "built-in 1.5 m/s" in warnings[2]
    assert "built-in 2 rad/s" in warnings[3]
    assert "--confirm-moves is off" in warnings[4]


def test_a_full_policy_has_no_warnings_but_still_says_it_cannot_see_the_map(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "max_speed": 0.5,
        "max_angular_speed": 0.5,
        "no_go_zones": [{"name": "dock", "x_min": 2, "y_min": 2, "x_max": 4, "y_max": 4}],
        "allowed_areas": [{"x_min": -10, "y_min": -10, "x_max": 10, "y_max": 10}],
    }))
    code, out, _ = _run(_args(st_policy_file=str(policy), st_confirm_moves=True))
    assert code == 0
    assert _warnings(out) == []
    assert "cannot know what is missing from your map" in out


def test_each_warning_goes_away_on_its_own():
    _, out, _ = _run(_args(st_no_go=["2,2,4,4"]))
    assert not any("no no-go zones" in w for w in _warnings(out))
    assert len(_warnings(out)) == 4

    _, out, _ = _run(_args(st_allow_area=["-5,-5,5,5"]))
    assert not any("allowed area" in w for w in _warnings(out))
    assert len(_warnings(out)) == 4

    _, out, _ = _run(_args(st_max_speed=0.5))
    assert not any("m/s" in w for w in _warnings(out))
    assert len(_warnings(out)) == 4

    _, out, _ = _run(_args(confirm_moves=True))
    assert not any("--confirm-moves" in w for w in _warnings(out))
    assert len(_warnings(out)) == 4


def test_warnings_do_not_hide_a_failure(monkeypatch):
    failing = [mcp_selftest.Check(False, "normal move", "broken")]
    monkeypatch.setattr(mcp_selftest, "run_checks", lambda server: failing)
    code, out, _ = _run(_args())
    assert code == 1
    assert "FAIL  normal move" in out
    assert _warnings(out)


def test_strict_base_policy_is_not_called_the_built_in_default():
    _, out, _ = _run(_args(st_policy="strict"))
    assert not any("built-in" in w for w in _warnings(out))


def test_confirm_moves_flag_works_on_the_selftest_subcommand(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["apyrobo", "mcp", "selftest", "--confirm-moves"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0
    assert "--confirm-moves is off" not in capsys.readouterr().out


def test_allowed_area_is_exercised_and_targets_stay_inside_it():
    code, out, _ = _run(_args(st_allow_area=["10,10,20,20"], st_no_go=["14,14,16,16"]))
    assert code == 0, out
    assert "PASS  move outside the allowed areas" in out
    assert "PASS  normal move" in out
    assert "PASS  move across no-go zone" in out


def test_crossing_check_is_not_passed_by_the_area_refusing_the_target():
    # The zone fills the far end of the area: nothing past it is allowed.
    code, out, _ = _run(_args(st_allow_area=["0,-1,5,1"], st_no_go=["3,-1,5,1"]))
    assert code == 0, out
    assert "not tried: no target past it inside the allowed areas" in out


def test_area_that_lets_the_move_through_fails(monkeypatch):
    real = mcp_selftest._Client.tool

    def lenient(self, name, **arguments):
        if name == "move_to":
            self._server.toolbox.enforcer.policy.allowed_areas = []
        return real(self, name, **arguments)

    monkeypatch.setattr(mcp_selftest._Client, "tool", lenient)
    code, out, _ = _run(_args(st_allow_area=["-5,-5,5,5"]))
    assert code == 1
    assert "FAIL  move outside the allowed areas" in out


def test_bad_allow_area_is_a_readable_error():
    code, _, err = _run(_args(st_allow_area=["1,2,3"]))
    assert code == 1
    assert "allowed area must be x_min,y_min,x_max,y_max" in err


def test_a_move_between_separate_areas_is_tried_and_refused():
    code, out, _ = _run(_args(st_allow_area=["-1,-1,1,1", "5,5,6,6"]))
    assert code == 0
    assert "PASS  move between allowed areas" in out
    assert "refused, robot stayed put" in out


def test_touching_areas_leave_no_such_move_to_try():
    code, out, _ = _run(_args(st_allow_area=["-1,-1,1,1", "1,-1,3,1"]))
    assert code == 0
    assert "not tried: no straight line from here leaves the areas" in out


def test_one_area_prints_no_between_line():
    _, out, _ = _run(_args(st_allow_area=["-5,-5,5,5"]))
    assert "between allowed areas" not in out


def test_a_gap_that_lets_the_move_through_fails(monkeypatch):
    real = mcp_selftest._Client.tool

    def lenient(self, name, **arguments):
        if name == "move_to" and (arguments["x"], arguments["y"]) == (5.0, 5.0):
            self._server.toolbox.enforcer.policy.allowed_areas = []
        return real(self, name, **arguments)

    monkeypatch.setattr(mcp_selftest._Client, "tool", lenient)
    code, out, _ = _run(_args(st_allow_area=["-1,-1,1,1", "5,5,6,6"]))
    assert code == 1
    assert "FAIL  move between allowed areas" in out


def test_an_audit_log_the_server_cannot_open_fails(tmp_path):
    path = tmp_path / "missing" / "a.log"
    code, out, _ = _run(_args(st_audit_log=str(path)))
    assert code == 1
    assert f"FAIL  audit log: cannot open {path} for append" in out
    assert "apyrobo mcp would exit at start" in out
    assert "3/4 checks passed" in out
    assert not path.parent.exists()


def test_an_existing_audit_log_passes_and_is_left_as_it_was(tmp_path):
    path = tmp_path / "a.log"
    path.write_bytes(b'{"decision":"ran"}\n')
    code, out, _ = _run(_args(st_audit_log=str(path)))
    assert code == 0
    assert f"PASS  audit log: {path} can be opened for append; nothing was written" in out
    assert "4/4 checks passed" in out
    assert path.read_bytes() == b'{"decision":"ran"}\n'


def test_a_new_audit_log_passes_and_leaves_no_file_behind(tmp_path):
    path = tmp_path / "a.log"
    code, out, _ = _run(_args(st_audit_log=str(path)))
    assert code == 0
    assert "PASS  audit log" in out
    assert not path.exists()


def test_a_directory_as_audit_log_fails(tmp_path):
    code, out, _ = _run(_args(st_audit_log=str(tmp_path)))
    assert code == 1
    assert "FAIL  audit log" in out


def test_no_audit_log_means_no_audit_check():
    _, out, _ = _run(_args())
    assert "audit log" not in out


@pytest.mark.parametrize("where", ["before", "after"])
def test_audit_log_flag_works_on_either_side_of_selftest(monkeypatch, capsys, tmp_path, where):
    bad = str(tmp_path / "missing" / "a.log")
    argv = (["apyrobo", "mcp", "--audit-log", bad, "selftest"] if where == "before"
            else ["apyrobo", "mcp", "selftest", "--audit-log", bad])
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert "FAIL  audit log" in capsys.readouterr().out
