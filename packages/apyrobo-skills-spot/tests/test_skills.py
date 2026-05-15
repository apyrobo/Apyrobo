"""Tests for apyrobo-skills-spot skill implementations."""
from __future__ import annotations

import pytest

from apyrobo_skills_spot.locomotion import walk_to, sit, stand, stair_climb
from apyrobo_skills_spot.utility import dock, capture_image, arm_pick


# ---------------------------------------------------------------------------
# walk_to
# ---------------------------------------------------------------------------

class TestWalkTo:
    def test_returns_bool(self):
        assert isinstance(walk_to(1.0, 2.0), bool)

    def test_returns_true_on_success(self):
        assert walk_to(0.0, 0.0) is True

    def test_prints_target_coordinates(self, capsys):
        walk_to(3.5, -1.2)
        out = capsys.readouterr().out
        assert "3.500" in out
        assert "-1.200" in out

    def test_custom_heading(self, capsys):
        walk_to(0.0, 0.0, heading=1.57)
        out = capsys.readouterr().out
        assert "1.570" in out

    def test_speed_clamped_low(self, capsys):
        walk_to(1.0, 1.0, speed=0.0)
        out = capsys.readouterr().out
        assert "0.1" in out

    def test_speed_clamped_high(self, capsys):
        walk_to(1.0, 1.0, speed=99.0)
        out = capsys.readouterr().out
        assert "2.0" in out

    def test_prints_arrival_message(self, capsys):
        walk_to(5.0, 5.0)
        out = capsys.readouterr().out
        assert "Arrived" in out or "arrived" in out.lower()


# ---------------------------------------------------------------------------
# sit
# ---------------------------------------------------------------------------

class TestSit:
    def test_returns_bool(self):
        assert isinstance(sit(), bool)

    def test_returns_true(self):
        assert sit() is True

    def test_prints_sit_message(self, capsys):
        sit()
        out = capsys.readouterr().out
        assert "sit" in out.lower()

    def test_prints_rest_message(self, capsys):
        sit()
        out = capsys.readouterr().out
        assert "rest" in out.lower()


# ---------------------------------------------------------------------------
# stand
# ---------------------------------------------------------------------------

class TestStand:
    def test_returns_bool(self):
        assert isinstance(stand(), bool)

    def test_returns_true(self):
        assert stand() is True

    def test_default_height_printed(self, capsys):
        stand()
        out = capsys.readouterr().out
        assert "0.52" in out

    def test_custom_height(self, capsys):
        stand(height=0.65)
        out = capsys.readouterr().out
        assert "0.65" in out

    def test_height_clamped_low(self, capsys):
        stand(height=0.0)
        out = capsys.readouterr().out
        assert "0.28" in out

    def test_height_clamped_high(self, capsys):
        stand(height=99.0)
        out = capsys.readouterr().out
        assert "0.72" in out


# ---------------------------------------------------------------------------
# stair_climb
# ---------------------------------------------------------------------------

class TestStairClimb:
    def test_returns_bool(self):
        assert isinstance(stair_climb(), bool)

    def test_returns_true(self):
        assert stair_climb() is True

    def test_default_direction_up(self, capsys):
        stair_climb()
        out = capsys.readouterr().out
        assert "Ascending" in out or "ascending" in out.lower()

    def test_direction_down(self, capsys):
        stair_climb(num_stairs=2, direction="down")
        out = capsys.readouterr().out
        assert "Descending" in out or "descending" in out.lower()

    def test_num_stairs_logged(self, capsys):
        stair_climb(num_stairs=5)
        out = capsys.readouterr().out
        assert "5" in out

    def test_num_stairs_clamped_to_one(self):
        # num_stairs <= 0 should be treated as 1, not raise
        assert stair_climb(num_stairs=0) is True

    def test_invalid_direction_defaults_to_up(self, capsys):
        stair_climb(direction="sideways")
        out = capsys.readouterr().out
        assert "Ascending" in out or "up" in out.lower()

    def test_each_step_printed(self, capsys):
        stair_climb(num_stairs=3)
        out = capsys.readouterr().out
        assert "Stair 1/3" in out
        assert "Stair 3/3" in out


# ---------------------------------------------------------------------------
# dock
# ---------------------------------------------------------------------------

class TestDock:
    def test_returns_bool(self):
        assert isinstance(dock(), bool)

    def test_returns_true(self):
        assert dock() is True

    def test_default_dock_id(self, capsys):
        dock()
        out = capsys.readouterr().out
        assert "1" in out

    def test_custom_dock_id(self, capsys):
        dock(dock_id=42)
        out = capsys.readouterr().out
        assert "42" in out

    def test_prints_charging_message(self, capsys):
        dock()
        out = capsys.readouterr().out
        assert "charg" in out.lower() or "dock" in out.lower()


# ---------------------------------------------------------------------------
# capture_image
# ---------------------------------------------------------------------------

class TestCaptureImage:
    def test_returns_str(self):
        result = capture_image()
        assert isinstance(result, str)

    def test_default_camera_in_path(self):
        result = capture_image()
        assert "frontleft" in result

    def test_custom_camera(self):
        result = capture_image(camera="back")
        assert "back" in result

    def test_custom_save_path(self):
        result = capture_image(save_path="/tmp/my_image.jpg")
        assert result == "/tmp/my_image.jpg"

    def test_invalid_camera_defaults_to_frontleft(self):
        result = capture_image(camera="nonexistent")
        assert "frontleft" in result

    def test_prints_capture_message(self, capsys):
        capture_image()
        out = capsys.readouterr().out
        assert "capture" in out.lower() or "frame" in out.lower()

    def test_empty_save_path_generates_default(self):
        result = capture_image(camera="left", save_path="")
        assert "left" in result
        assert len(result) > 0


# ---------------------------------------------------------------------------
# arm_pick
# ---------------------------------------------------------------------------

class TestArmPick:
    def test_returns_bool(self):
        assert isinstance(arm_pick(0.5, 0.0, 0.3), bool)

    def test_returns_true(self):
        assert arm_pick(0.5, 0.0, 0.3) is True

    def test_prints_target_coordinates(self, capsys):
        arm_pick(0.7, -0.1, 0.25)
        out = capsys.readouterr().out
        assert "0.700" in out
        assert "-0.100" in out
        assert "0.250" in out

    def test_prints_grasp_message(self, capsys):
        arm_pick(0.5, 0.0, 0.3)
        out = capsys.readouterr().out
        assert "grasp" in out.lower() or "gripper" in out.lower()

    def test_prints_retract_message(self, capsys):
        arm_pick(0.5, 0.0, 0.3)
        out = capsys.readouterr().out
        assert "retracting" in out.lower() or "carry" in out.lower()


# ---------------------------------------------------------------------------
# Entry-point: register()
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_runs_without_error(self):
        from apyrobo_skills_spot import register
        register()  # should not raise

    def test_register_prints_skill_count(self, capsys):
        from apyrobo_skills_spot import register
        register()
        out = capsys.readouterr().out
        assert "Registered" in out

    def test_all_seven_skills_exported(self):
        import apyrobo_skills_spot as pkg
        for name in ("walk_to", "sit", "stand", "stair_climb",
                     "dock", "capture_image", "arm_pick"):
            assert hasattr(pkg, name), f"missing export: {name}"

    def test_register_in_all_dunder(self):
        import apyrobo_skills_spot as pkg
        assert "register" in pkg.__all__
