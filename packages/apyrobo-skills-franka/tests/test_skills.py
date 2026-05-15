"""Tests for apyrobo-skills-franka skill implementations."""
from __future__ import annotations

import math

import pytest

from apyrobo_skills_franka.arm import cartesian_sweep, move_home, move_to_pose
from apyrobo_skills_franka.gripper import grasp, impedance_control, release


# ---------------------------------------------------------------------------
# move_to_pose
# ---------------------------------------------------------------------------

class TestMoveToPose:
    def test_returns_bool(self):
        assert isinstance(move_to_pose(0.3, 0.0, 0.5), bool)

    def test_returns_true(self):
        assert move_to_pose(0.3, 0.0, 0.5) is True

    def test_prints_target_position(self, capsys):
        move_to_pose(0.3, 0.1, 0.5)
        out = capsys.readouterr().out
        assert "move_to_pose" in out
        assert "0.300" in out

    def test_default_orientation_zero(self, capsys):
        move_to_pose(0.4, 0.0, 0.6)
        out = capsys.readouterr().out
        assert "roll=0.000" in out
        assert "pitch=0.000" in out
        assert "yaw=0.000" in out

    def test_custom_orientation(self, capsys):
        move_to_pose(0.4, 0.0, 0.6, roll=1.57, pitch=0.5, yaw=-0.3)
        out = capsys.readouterr().out
        assert "1.570" in out
        assert "0.500" in out

    def test_negative_coordinates_accepted(self):
        assert move_to_pose(-0.1, -0.2, 0.3) is True

    def test_prints_trajectory_message(self, capsys):
        move_to_pose(0.3, 0.0, 0.5)
        out = capsys.readouterr().out
        assert "trajectory" in out.lower()


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

    def test_prints_joint_targets(self, capsys):
        move_home()
        out = capsys.readouterr().out
        # pi/4 ≈ 0.785 should appear in the joint configuration
        assert "0.785" in out

    def test_prints_homing_trajectory(self, capsys):
        move_home()
        out = capsys.readouterr().out
        assert "homing" in out.lower()


# ---------------------------------------------------------------------------
# cartesian_sweep
# ---------------------------------------------------------------------------

class TestCartesianSweep:
    def test_returns_bool(self):
        assert isinstance(cartesian_sweep([0.0, 0.0, 0.3], [0.3, 0.0, 0.5]), bool)

    def test_returns_true(self):
        assert cartesian_sweep([0.0, 0.0, 0.3], [0.3, 0.0, 0.5]) is True

    def test_default_steps_prints_ten_steps(self, capsys):
        cartesian_sweep([0.0, 0.0, 0.3], [0.1, 0.0, 0.4])
        out = capsys.readouterr().out
        assert "Step 10/10" in out

    def test_custom_steps(self, capsys):
        cartesian_sweep([0.0, 0.0, 0.3], [0.1, 0.0, 0.4], steps=5)
        out = capsys.readouterr().out
        assert "Step 5/5" in out

    def test_minimum_steps_clamped_to_two(self, capsys):
        cartesian_sweep([0.0, 0.0, 0.3], [0.1, 0.0, 0.4], steps=1)
        out = capsys.readouterr().out
        assert "Step 2/2" in out

    def test_prints_start_and_end_coordinates(self, capsys):
        cartesian_sweep([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
        out = capsys.readouterr().out
        assert "0.100" in out
        assert "0.600" in out

    def test_prints_sweep_complete(self, capsys):
        cartesian_sweep([0.0, 0.0, 0.3], [0.3, 0.0, 0.5])
        out = capsys.readouterr().out
        assert "complete" in out.lower()


# ---------------------------------------------------------------------------
# grasp
# ---------------------------------------------------------------------------

class TestGrasp:
    def test_returns_bool(self):
        assert isinstance(grasp(), bool)

    def test_returns_true(self):
        assert grasp() is True

    def test_default_width_in_output(self, capsys):
        grasp()
        out = capsys.readouterr().out
        assert "0.040" in out

    def test_default_force_in_output(self, capsys):
        grasp()
        out = capsys.readouterr().out
        assert "20.0" in out

    def test_custom_parameters(self, capsys):
        grasp(width=0.02, force=35.0, speed=0.05)
        out = capsys.readouterr().out
        assert "0.020" in out
        assert "35.0" in out

    def test_width_clamped_to_max(self, capsys):
        grasp(width=0.5)
        out = capsys.readouterr().out
        assert "0.080" in out

    def test_width_clamped_to_min(self, capsys):
        grasp(width=-1.0)
        out = capsys.readouterr().out
        assert "0.000" in out

    def test_force_clamped_to_max(self, capsys):
        grasp(force=9999.0)
        out = capsys.readouterr().out
        assert "70.0" in out

    def test_prints_grasped_message(self, capsys):
        grasp()
        out = capsys.readouterr().out
        assert "grasped" in out.lower()


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------

class TestRelease:
    def test_returns_bool(self):
        assert isinstance(release(), bool)

    def test_returns_true(self):
        assert release() is True

    def test_default_width_fully_open(self, capsys):
        release()
        out = capsys.readouterr().out
        assert "0.080" in out

    def test_custom_width(self, capsys):
        release(width=0.05)
        out = capsys.readouterr().out
        assert "0.050" in out

    def test_width_clamped_to_max(self, capsys):
        release(width=0.99)
        out = capsys.readouterr().out
        assert "0.080" in out

    def test_prints_open_message(self, capsys):
        release()
        out = capsys.readouterr().out
        assert "open" in out.lower()


# ---------------------------------------------------------------------------
# impedance_control
# ---------------------------------------------------------------------------

class TestImpedanceControl:
    def test_returns_bool(self):
        assert isinstance(impedance_control(duration_s=0.1), bool)

    def test_returns_true(self):
        assert impedance_control(duration_s=0.1) is True

    def test_default_stiffness_in_output(self, capsys):
        impedance_control(duration_s=0.1)
        out = capsys.readouterr().out
        assert "200.0" in out

    def test_default_damping_in_output(self, capsys):
        impedance_control(duration_s=0.1)
        out = capsys.readouterr().out
        assert "10.0" in out

    def test_custom_stiffness_and_damping(self, capsys):
        impedance_control(stiffness=500.0, damping=25.0, duration_s=0.1)
        out = capsys.readouterr().out
        assert "500.0" in out
        assert "25.0" in out

    def test_stiffness_clamped_to_max(self, capsys):
        impedance_control(stiffness=99999.0, duration_s=0.1)
        out = capsys.readouterr().out
        assert "3000.0" in out

    def test_stiffness_clamped_to_min(self, capsys):
        impedance_control(stiffness=0.0, duration_s=0.1)
        out = capsys.readouterr().out
        assert "10.0" in out

    def test_duration_clamped_to_min(self, capsys):
        impedance_control(duration_s=0.0)
        out = capsys.readouterr().out
        assert "0.10" in out

    def test_prints_stopped_message(self, capsys):
        impedance_control(duration_s=0.1)
        out = capsys.readouterr().out
        assert "stopped" in out.lower()


# ---------------------------------------------------------------------------
# Entry-point: register()
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_runs_without_error(self):
        from apyrobo_skills_franka import register
        register()  # should not raise

    def test_register_prints_skill_count(self, capsys):
        from apyrobo_skills_franka import register
        register()
        out = capsys.readouterr().out
        assert "Registered" in out

    def test_all_six_skills_exported(self):
        import apyrobo_skills_franka as pkg
        for name in ("move_to_pose", "move_home", "cartesian_sweep",
                     "grasp", "release", "impedance_control"):
            assert hasattr(pkg, name), f"missing export: {name}"
