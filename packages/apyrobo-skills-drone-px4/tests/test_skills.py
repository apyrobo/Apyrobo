"""Tests for apyrobo-skills-drone-px4 skill implementations."""
from __future__ import annotations

import pytest

from apyrobo_skills_drone_px4.flight import takeoff, land, fly_to, return_home
from apyrobo_skills_drone_px4.payload import orbit, capture_image


# ---------------------------------------------------------------------------
# takeoff
# ---------------------------------------------------------------------------

class TestTakeoff:
    def test_returns_bool(self):
        assert isinstance(takeoff(), bool)

    def test_returns_true(self):
        assert takeoff() is True

    def test_default_altitude(self, capsys):
        takeoff()
        out = capsys.readouterr().out
        assert "10.0" in out

    def test_custom_altitude(self, capsys):
        takeoff(altitude_m=30.0)
        out = capsys.readouterr().out
        assert "30.0" in out

    def test_altitude_clamped_low(self, capsys):
        takeoff(altitude_m=0.0)
        out = capsys.readouterr().out
        assert "1.0" in out  # clamped to 1 m

    def test_altitude_clamped_high(self, capsys):
        takeoff(altitude_m=9999.0)
        out = capsys.readouterr().out
        assert "120.0" in out  # clamped to 120 m

    def test_prints_arming_message(self, capsys):
        takeoff()
        out = capsys.readouterr().out
        assert "arm" in out.lower()

    def test_prints_completion(self, capsys):
        takeoff()
        out = capsys.readouterr().out
        assert "complete" in out.lower() or "loiter" in out.lower()


# ---------------------------------------------------------------------------
# land
# ---------------------------------------------------------------------------

class TestLand:
    def test_returns_bool(self):
        assert isinstance(land(), bool)

    def test_returns_true(self):
        assert land() is True

    def test_prints_land_message(self, capsys):
        land()
        out = capsys.readouterr().out
        assert "land" in out.lower()

    def test_prints_disarm_message(self, capsys):
        land()
        out = capsys.readouterr().out
        assert "disarm" in out.lower()


# ---------------------------------------------------------------------------
# fly_to
# ---------------------------------------------------------------------------

class TestFlyTo:
    def test_returns_bool(self):
        assert isinstance(fly_to(37.7749, -122.4194), bool)

    def test_returns_true(self):
        assert fly_to(37.7749, -122.4194) is True

    def test_lat_lon_in_output(self, capsys):
        fly_to(48.8566, 2.3522, alt_m=100.0)
        out = capsys.readouterr().out
        assert "48.856600" in out
        assert "2.352200" in out

    def test_default_altitude(self, capsys):
        fly_to(0.0, 0.0)
        out = capsys.readouterr().out
        assert "50.0" in out

    def test_custom_altitude(self, capsys):
        fly_to(0.0, 0.0, alt_m=200.0)
        out = capsys.readouterr().out
        assert "200.0" in out

    def test_altitude_clamped_low(self, capsys):
        fly_to(0.0, 0.0, alt_m=-10.0)
        out = capsys.readouterr().out
        assert "1.0" in out

    def test_speed_clamped_high(self, capsys):
        fly_to(0.0, 0.0, speed_ms=999.0)
        out = capsys.readouterr().out
        assert "20.0" in out  # clamped to 20 m/s

    def test_custom_speed(self, capsys):
        fly_to(0.0, 0.0, speed_ms=10.0)
        out = capsys.readouterr().out
        assert "10.0" in out


# ---------------------------------------------------------------------------
# return_home
# ---------------------------------------------------------------------------

class TestReturnHome:
    def test_returns_bool(self):
        assert isinstance(return_home(), bool)

    def test_returns_true(self):
        assert return_home() is True

    def test_prints_rtl_message(self, capsys):
        return_home()
        out = capsys.readouterr().out
        assert "rtl" in out.lower() or "home" in out.lower()

    def test_prints_disarm_message(self, capsys):
        return_home()
        out = capsys.readouterr().out
        assert "disarm" in out.lower()


# ---------------------------------------------------------------------------
# orbit
# ---------------------------------------------------------------------------

class TestOrbit:
    def test_returns_bool(self):
        assert isinstance(orbit(51.5074, -0.1278), bool)

    def test_returns_true(self):
        assert orbit(51.5074, -0.1278) is True

    def test_default_radius_in_output(self, capsys):
        orbit(0.0, 0.0)
        out = capsys.readouterr().out
        assert "20.0" in out

    def test_custom_radius(self, capsys):
        orbit(0.0, 0.0, radius_m=50.0)
        out = capsys.readouterr().out
        assert "50.0" in out

    def test_radius_clamped_low(self, capsys):
        orbit(0.0, 0.0, radius_m=0.0)
        out = capsys.readouterr().out
        assert "5.0" in out  # clamped to 5 m

    def test_multiple_loops(self, capsys):
        orbit(0.0, 0.0, loops=3)
        out = capsys.readouterr().out
        assert "loop 3/3" in out.lower()

    def test_single_loop_default(self, capsys):
        orbit(0.0, 0.0)
        out = capsys.readouterr().out
        assert "loop 1/1" in out.lower()

    def test_prints_completion(self, capsys):
        orbit(0.0, 0.0)
        out = capsys.readouterr().out
        assert "complete" in out.lower()


# ---------------------------------------------------------------------------
# capture_image
# ---------------------------------------------------------------------------

class TestCaptureImage:
    def test_returns_str(self):
        result = capture_image()
        assert isinstance(result, str)

    def test_default_path_contains_camera(self):
        result = capture_image(camera="downward")
        assert "downward" in result

    def test_custom_camera(self):
        result = capture_image(camera="forward")
        assert "forward" in result

    def test_explicit_save_path_returned(self):
        path = "/tmp/test_frame.jpg"
        result = capture_image(save_path=path)
        assert result == path

    def test_empty_save_path_generates_default(self):
        result = capture_image(camera="thermal", save_path="")
        assert result  # non-empty string
        assert "thermal" in result

    def test_prints_shutter_message(self, capsys):
        capture_image()
        out = capsys.readouterr().out
        assert "shutter" in out.lower() or "trigger" in out.lower()


# ---------------------------------------------------------------------------
# Entry-point: register()
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_runs_without_error(self, capsys):
        from apyrobo_skills_drone_px4 import register
        register()  # should not raise

    def test_register_prints_skill_count(self, capsys):
        from apyrobo_skills_drone_px4 import register
        register()
        out = capsys.readouterr().out
        assert "Registered" in out

    def test_all_six_skills_exported(self):
        import apyrobo_skills_drone_px4 as pkg
        for name in ("takeoff", "land", "fly_to", "return_home", "orbit", "capture_image"):
            assert hasattr(pkg, name), f"missing export: {name}"
