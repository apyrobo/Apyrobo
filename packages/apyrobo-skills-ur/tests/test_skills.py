"""Tests for apyrobo-skills-ur skill implementations."""
from __future__ import annotations

import math
import sys
import os

# Allow running without pip install by inserting the package root onto sys.path
_pkg_root = os.path.join(os.path.dirname(__file__), "..")
if _pkg_root not in sys.path:
    sys.path.insert(0, os.path.abspath(_pkg_root))

import pytest

from apyrobo_skills_ur.motion import move_joints, move_linear, move_home, set_tcp
from apyrobo_skills_ur.manipulation import pick, place, get_pose


# ---------------------------------------------------------------------------
# move_joints
# ---------------------------------------------------------------------------

class TestMoveJoints:
    def test_returns_bool(self):
        assert isinstance(move_joints([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), bool)

    def test_returns_true_on_success(self):
        assert move_joints([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) is True

    def test_prints_joint_positions(self, capsys):
        positions = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
        move_joints(positions)
        out = capsys.readouterr().out
        assert "move_joints" in out
        assert "0.100" in out

    def test_default_speed(self, capsys):
        move_joints([0.0] * 6)
        out = capsys.readouterr().out
        assert "0.50" in out

    def test_custom_speed(self, capsys):
        move_joints([0.0] * 6, speed=0.25)
        out = capsys.readouterr().out
        assert "0.25" in out

    def test_speed_clamped_high(self, capsys):
        move_joints([0.0] * 6, speed=99.0)
        out = capsys.readouterr().out
        assert "1.00" in out

    def test_speed_clamped_low(self, capsys):
        move_joints([0.0] * 6, speed=-5.0)
        out = capsys.readouterr().out
        assert "0.00" in out

    def test_prints_completion_message(self, capsys):
        move_joints([0.0] * 6)
        out = capsys.readouterr().out
        assert "complete" in out.lower()

    def test_six_joint_count_reported(self, capsys):
        move_joints([0.0] * 6)
        out = capsys.readouterr().out
        assert "6" in out


# ---------------------------------------------------------------------------
# move_linear
# ---------------------------------------------------------------------------

class TestMoveLinear:
    def test_returns_bool(self):
        assert isinstance(move_linear(0.3, -0.2, 0.4), bool)

    def test_returns_true_on_success(self):
        assert move_linear(0.3, -0.2, 0.4) is True

    def test_prints_target_coordinates(self, capsys):
        move_linear(0.300, -0.200, 0.450)
        out = capsys.readouterr().out
        assert "0.300" in out
        assert "-0.200" in out
        assert "0.450" in out

    def test_default_speed(self, capsys):
        move_linear(0.0, 0.0, 0.5)
        out = capsys.readouterr().out
        assert "0.100" in out

    def test_custom_speed(self, capsys):
        move_linear(0.0, 0.0, 0.5, speed=0.25)
        out = capsys.readouterr().out
        assert "0.250" in out

    def test_speed_clamped_high(self, capsys):
        move_linear(0.0, 0.0, 0.5, speed=99.0)
        out = capsys.readouterr().out
        assert "1.000" in out

    def test_prints_movel_label(self, capsys):
        move_linear(0.0, 0.0, 0.5)
        out = capsys.readouterr().out
        assert "MoveL" in out


# ---------------------------------------------------------------------------
# move_home
# ---------------------------------------------------------------------------

class TestMoveHome:
    def test_returns_bool(self):
        assert isinstance(move_home(), bool)

    def test_returns_true(self):
        assert move_home() is True

    def test_prints_home_message(self, capsys):
        move_home()
        out = capsys.readouterr().out
        assert "home" in out.lower()

    def test_prints_joint_values(self, capsys):
        move_home()
        out = capsys.readouterr().out
        # Home has joints at 0.0 and -pi/2 values
        assert "0.0000" in out

    def test_prints_ready_message(self, capsys):
        move_home()
        out = capsys.readouterr().out
        assert "ready" in out.lower()


# ---------------------------------------------------------------------------
# set_tcp
# ---------------------------------------------------------------------------

class TestSetTcp:
    def test_returns_bool(self):
        assert isinstance(set_tcp(), bool)

    def test_returns_true(self):
        assert set_tcp() is True

    def test_default_offsets(self, capsys):
        set_tcp()
        out = capsys.readouterr().out
        assert "x=0.0000" in out
        assert "y=0.0000" in out
        assert "z=0.1000" in out

    def test_custom_offsets(self, capsys):
        set_tcp(x=0.05, y=0.02, z=0.15)
        out = capsys.readouterr().out
        assert "x=0.0500" in out
        assert "y=0.0200" in out
        assert "z=0.1500" in out

    def test_prints_controller_updated(self, capsys):
        set_tcp()
        out = capsys.readouterr().out
        assert "controller" in out.lower()


# ---------------------------------------------------------------------------
# pick
# ---------------------------------------------------------------------------

class TestPick:
    def test_returns_bool(self):
        assert isinstance(pick(0.5, 0.0, 0.1), bool)

    def test_returns_true_on_success(self):
        assert pick(0.5, 0.0, 0.1) is True

    def test_prints_target_pose(self, capsys):
        pick(0.500, -0.100, 0.050)
        out = capsys.readouterr().out
        assert "0.500" in out
        assert "-0.100" in out

    def test_prints_gripper_message(self, capsys):
        pick(0.5, 0.0, 0.1)
        out = capsys.readouterr().out
        assert "gripper" in out.lower()

    def test_approach_height_used(self, capsys):
        pick(0.5, 0.0, 0.0, approach_height=0.2)
        out = capsys.readouterr().out
        # Approach pose z = 0.0 + 0.2 = 0.2
        assert "0.200" in out

    def test_prints_complete_message(self, capsys):
        pick(0.5, 0.0, 0.1)
        out = capsys.readouterr().out
        assert "complete" in out.lower()

    def test_approach_height_clamped_low(self):
        # Should not raise even with zero/negative approach height
        assert pick(0.5, 0.0, 0.1, approach_height=0.0) is True


# ---------------------------------------------------------------------------
# place
# ---------------------------------------------------------------------------

class TestPlace:
    def test_returns_bool(self):
        assert isinstance(place(0.4, 0.3, 0.05), bool)

    def test_returns_true_on_success(self):
        assert place(0.4, 0.3, 0.05) is True

    def test_prints_target_pose(self, capsys):
        place(0.400, 0.300, 0.050)
        out = capsys.readouterr().out
        assert "0.400" in out
        assert "0.300" in out

    def test_prints_gripper_open_message(self, capsys):
        place(0.4, 0.3, 0.05)
        out = capsys.readouterr().out
        assert "gripper" in out.lower()

    def test_approach_height_used(self, capsys):
        place(0.4, 0.3, 0.0, approach_height=0.15)
        out = capsys.readouterr().out
        # Approach pose z = 0.0 + 0.15 = 0.15
        assert "0.150" in out

    def test_prints_clear_message(self, capsys):
        place(0.4, 0.3, 0.05)
        out = capsys.readouterr().out
        assert "clear" in out.lower()


# ---------------------------------------------------------------------------
# get_pose
# ---------------------------------------------------------------------------

class TestGetPose:
    def test_returns_dict(self):
        assert isinstance(get_pose(), dict)

    def test_has_position_keys(self):
        pose = get_pose()
        for key in ("x", "y", "z"):
            assert key in pose, f"missing key: {key}"

    def test_has_orientation_keys(self):
        pose = get_pose()
        for key in ("rx", "ry", "rz"):
            assert key in pose, f"missing key: {key}"

    def test_values_are_floats(self):
        pose = get_pose()
        for key, val in pose.items():
            assert isinstance(val, float), f"key '{key}' is not float: {type(val)}"

    def test_prints_pose_values(self, capsys):
        get_pose()
        out = capsys.readouterr().out
        assert "TCP pose" in out

    def test_z_is_positive(self):
        # Default pose should have z above the table
        pose = get_pose()
        assert pose["z"] > 0.0


# ---------------------------------------------------------------------------
# Entry-point: register()
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_runs_without_error(self, capsys):
        from apyrobo_skills_ur import register
        register()  # should not raise

    def test_register_prints_skill_count(self, capsys):
        from apyrobo_skills_ur import register
        register()
        out = capsys.readouterr().out
        assert "Registered" in out

    def test_all_seven_skills_exported(self):
        import apyrobo_skills_ur as pkg
        for name in ("move_joints", "move_linear", "move_home", "set_tcp",
                     "pick", "place", "get_pose"):
            assert hasattr(pkg, name), f"missing export: {name}"
