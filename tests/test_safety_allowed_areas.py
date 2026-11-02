"""Allowed areas: once any are set, a move target outside all of them is refused."""

from __future__ import annotations

import argparse
import json

import pytest

from apyrobo.cli import cmd_mcp
from apyrobo.core.robot import Robot
from apyrobo.mcp_clients import server_args
from apyrobo.mcp_server import build_server, load_policy_file, parse_zone
from apyrobo.safety.enforcer import SafetyEnforcer, SafetyPolicy, SafetyViolation

AREA = {"x_min": 0.0, "y_min": 0.0, "x_max": 10.0, "y_max": 6.0}


def enforcer(**policy) -> SafetyEnforcer:
    return SafetyEnforcer(
        Robot.discover("mock://area_bot"), policy=SafetyPolicy(name="t", **policy)
    )


def call(server, name, arguments=None, msg_id=1):
    resp = server.handle({
        "jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    })
    return resp["result"]


class TestEnforcer:
    def test_unset_accepts_any_target_as_before(self):
        guard = enforcer()
        guard.move(1000.0, -1000.0)
        assert guard.get_position() == (1000.0, -1000.0)

    def test_target_inside_runs(self):
        guard = enforcer(allowed_areas=[AREA])
        guard.move(5.0, 5.0)
        assert guard.get_position() == (5.0, 5.0)

    @pytest.mark.parametrize("x,y", [(0.0, 0.0), (10.0, 6.0), (0.0, 6.0)])
    def test_the_edge_counts_as_inside(self, x, y):
        guard = enforcer(allowed_areas=[AREA])
        guard.move(x, y)
        assert guard.get_position() == (x, y)

    def test_target_outside_is_refused_recorded_and_the_robot_stays(self):
        guard = enforcer(allowed_areas=[AREA])
        before = guard.get_position()
        with pytest.raises(SafetyViolation, match="outside every allowed area"):
            guard.move(11.0, 3.0)
        assert guard.get_position() == before
        assert guard.violations[-1]["type"] == "outside_allowed_area"

    def test_any_of_several_areas_will_do(self):
        far = {"x_min": 20.0, "y_min": 20.0, "x_max": 30.0, "y_max": 30.0}
        guard = enforcer(allowed_areas=[AREA, far])
        guard.move(25.0, 25.0)
        assert guard.get_position() == (25.0, 25.0)
        with pytest.raises(SafetyViolation):
            guard.move(15.0, 15.0)

    def test_a_zone_still_wins_inside_an_area(self):
        zone = {"x_min": 2.0, "y_min": 2.0, "x_max": 4.0, "y_max": 4.0}
        guard = enforcer(allowed_areas=[AREA], collision_zones=[zone])
        with pytest.raises(SafetyViolation, match="collision zone"):
            guard.move(3.0, 3.0)

    def test_a_robot_outside_may_drive_in(self):
        guard = enforcer(allowed_areas=[{"x_min": 5, "y_min": 5, "x_max": 6, "y_max": 6}])
        guard.move(5.5, 5.5)
        assert guard.get_position() == (5.5, 5.5)


class TestServer:
    def test_flag_refuses_with_a_readable_error_and_reports_the_areas(self):
        srv = build_server(allowed_areas=[AREA])
        res = call(srv, "move_to", {"x": 20, "y": 0})
        assert res["isError"] is True
        assert "REFUSED" in res["content"][0]["text"]
        assert "outside every allowed area" in res["content"][0]["text"]
        report = json.loads(call(srv, "safety_report")["content"][0]["text"])
        assert report["safety_policy"]["allowed_areas"] == [AREA]

    def test_policy_name_marks_a_flag_override(self):
        assert build_server(allowed_areas=[AREA]).toolbox.enforcer.policy.name == "default+cli"

    def test_flags_add_to_the_file(self, tmp_path):
        policy = tmp_path / "p.yaml"
        policy.write_text("allowed_areas:\n  - name: floor\n    x_min: 0\n    y_min: 0\n"
                          "    x_max: 1\n    y_max: 1\n  - 5,5,6,6\n")
        far = parse_zone("20,20,21,21")
        areas = build_server(policy_file=policy, allowed_areas=[far]).toolbox.enforcer.policy
        assert [a.get("name") for a in areas.allowed_areas] == ["floor", None, None]
        assert areas.allowed_areas[-1] == far

    def test_confirmation_is_not_asked_for_a_move_that_will_be_refused(self):
        toolbox = build_server(allowed_areas=[AREA]).toolbox
        assert toolbox.confirmation_prompt("move_to", {"x": 20, "y": 0}) is None
        assert toolbox.confirmation_prompt("move_to", {"x": 1, "y": 1}) is not None

    def test_a_task_step_outside_the_area_is_refused(self):
        srv = build_server(allowed_areas=[{"x_min": 5, "y_min": 5, "x_max": 6, "y_max": 6}])
        res = call(srv, "execute_task", {"task": "go to 1 1"})
        assert "REFUSED" in res["content"][0]["text"]
        assert "outside_allowed_area" in res["content"][0]["text"]
        position = json.loads(call(srv, "robot_status")["content"][0]["text"])["position"]
        assert (position["x"], position["y"]) == (0.0, 0.0)


class TestPolicyFile:
    @pytest.mark.parametrize("value", ["[]", "null", "'0,0,1,1'", "{}"])
    def test_empty_or_non_list_is_an_error_not_anywhere(self, tmp_path, value):
        policy = tmp_path / "p.yaml"
        policy.write_text(f"allowed_areas: {value}\n")
        with pytest.raises(ValueError, match="allowed_areas must list at least one area"):
            load_policy_file(policy)

    def test_absent_key_stays_absent(self, tmp_path):
        policy = tmp_path / "p.yaml"
        policy.write_text("max_speed: 0.5\n")
        assert "allowed_areas" not in load_policy_file(policy)

    def test_bad_area_names_its_key(self, tmp_path):
        policy = tmp_path / "p.yaml"
        policy.write_text("allowed_areas:\n  - x_min: 0\n")
        with pytest.raises(ValueError, match=r"allowed_areas\[0\] is missing"):
            load_policy_file(policy)
        policy.write_text("allowed_areas:\n  - '5,0,1,1'\n")
        with pytest.raises(ValueError, match=r"allowed_areas\[0\]: allowed area has min > max"):
            load_policy_file(policy)

    @pytest.mark.parametrize("key", ["no_go_zones", "allowed_areas"])
    @pytest.mark.parametrize("bad", [".nan", ".inf", "-.inf"])
    def test_non_finite_corners_are_refused(self, tmp_path, key, bad):
        policy = tmp_path / "p.yaml"
        policy.write_text(
            f"{key}:\n  - x_min: {bad}\n    y_min: 0\n    x_max: 1\n    y_max: 1\n"
        )
        with pytest.raises(ValueError, match="finite"):
            load_policy_file(policy)


@pytest.mark.parametrize("spec", ["nan,0,1,1", "0,0,inf,1", "-inf,0,1,1"])
def test_parse_zone_refuses_non_finite(spec):
    with pytest.raises(ValueError, match="finite"):
        parse_zone(spec)


def test_a_nan_string_in_a_policy_file_is_refused(tmp_path):
    policy = tmp_path / "p.yaml"
    policy.write_text("no_go_zones:\n  - nan,0,1,1\n")
    with pytest.raises(ValueError, match="finite"):
        load_policy_file(policy)


def test_cli_passes_the_flag_to_the_server(monkeypatch):
    seen = {}

    class StopError(Exception):
        pass

    def fake_build(**kwargs):
        seen.update(kwargs)
        raise StopError

    monkeypatch.setattr("apyrobo.mcp_server.build_server", fake_build)
    args = argparse.Namespace(
        mcp_command=None, robot="mock://x", policy=None, max_speed=None, no_go=None,
        allow_area=["0,0,10,6"], provider="rule", policy_file=None, confirm_moves=False,
    )
    with pytest.raises(StopError):
        cmd_mcp(args)
    assert seen["allowed_areas"] == [AREA]


def test_bad_flag_is_a_readable_error_at_start(monkeypatch, capsys):
    args = argparse.Namespace(
        mcp_command=None, robot="mock://x", policy=None, max_speed=None, no_go=None,
        allow_area=["0,0,10"], provider="rule", policy_file=None, confirm_moves=False,
    )
    with pytest.raises(SystemExit) as exc:
        cmd_mcp(args)
    assert exc.value.code == 1
    assert "allowed area must be" in capsys.readouterr().err


def test_client_config_passes_it_through_and_checks_it_now():
    assert server_args(allow_area=["0, 0, 10, 6"]) == ["--allow-area", "0,0,10,6"]
    with pytest.raises(ValueError, match="allowed area"):
        server_args(allow_area=["nope"])
