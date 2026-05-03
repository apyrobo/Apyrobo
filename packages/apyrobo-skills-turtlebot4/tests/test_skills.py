"""Tests for apyrobo-skills-turtlebot4 skill implementations."""
from __future__ import annotations

import pytest

from apyrobo_skills_turtlebot4.navigation import dock, patrol_area, undock
from apyrobo_skills_turtlebot4.inspection import check_surroundings, inspect_room
from apyrobo_skills_turtlebot4.social import follow_person


# ---------------------------------------------------------------------------
# patrol_area
# ---------------------------------------------------------------------------

class TestPatrolArea:
    def test_returns_bool(self):
        assert isinstance(patrol_area(), bool)

    def test_returns_true_on_success(self):
        assert patrol_area() is True

    def test_default_waypoints(self, capsys):
        patrol_area()
        out = capsys.readouterr().out
        assert "patrol_area" in out

    def test_custom_waypoints(self, capsys):
        wps = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]]
        result = patrol_area(waypoints=wps)
        assert result is True
        out = capsys.readouterr().out
        assert "3" in out  # 3 waypoints mentioned

    def test_multi_loop(self, capsys):
        patrol_area(waypoints=[[0, 0], [1, 0]], loops=3)
        out = capsys.readouterr().out
        assert "Loop 3/3" in out

    def test_single_waypoint(self):
        assert patrol_area(waypoints=[[5.0, 5.0]]) is True

    def test_none_waypoints_uses_default(self, capsys):
        patrol_area(waypoints=None)
        out = capsys.readouterr().out
        assert "4" in out  # default 4 waypoints


# ---------------------------------------------------------------------------
# dock
# ---------------------------------------------------------------------------

class TestDock:
    def test_returns_bool(self):
        assert isinstance(dock(), bool)

    def test_returns_true(self):
        assert dock() is True

    def test_default_station_id(self, capsys):
        dock()
        out = capsys.readouterr().out
        assert "dock_01" in out

    def test_custom_station_id(self, capsys):
        dock(dock_station_id="charging_bay_3")
        out = capsys.readouterr().out
        assert "charging_bay_3" in out

    def test_prints_docking_message(self, capsys):
        dock()
        out = capsys.readouterr().out
        assert "dock" in out.lower()


# ---------------------------------------------------------------------------
# undock
# ---------------------------------------------------------------------------

class TestUndock:
    def test_returns_bool(self):
        assert isinstance(undock(), bool)

    def test_returns_true(self):
        assert undock() is True

    def test_prints_undock_message(self, capsys):
        undock()
        out = capsys.readouterr().out
        assert "undock" in out.lower()


# ---------------------------------------------------------------------------
# inspect_room
# ---------------------------------------------------------------------------

class TestInspectRoom:
    def test_returns_bool(self):
        assert isinstance(inspect_room(), bool)

    def test_returns_true(self):
        assert inspect_room() is True

    def test_default_room_id(self, capsys):
        inspect_room()
        out = capsys.readouterr().out
        assert "room_01" in out

    def test_custom_room_id(self, capsys):
        inspect_room(room_id="lab_B")
        out = capsys.readouterr().out
        assert "lab_B" in out

    def test_camera_height_clamped_low(self, capsys):
        inspect_room(camera_height=0.0)
        out = capsys.readouterr().out
        assert "0.30" in out  # clamped to 0.3

    def test_camera_height_clamped_high(self, capsys):
        inspect_room(camera_height=99.0)
        out = capsys.readouterr().out
        assert "1.20" in out  # clamped to 1.2

    def test_four_frames_captured(self, capsys):
        inspect_room()
        out = capsys.readouterr().out
        assert "4 frames" in out


# ---------------------------------------------------------------------------
# check_surroundings
# ---------------------------------------------------------------------------

class TestCheckSurroundings:
    def test_returns_bool(self):
        assert isinstance(check_surroundings(), bool)

    def test_returns_true(self):
        assert check_surroundings() is True

    def test_default_radius(self, capsys):
        check_surroundings()
        out = capsys.readouterr().out
        assert "2.0" in out

    def test_custom_radius(self, capsys):
        check_surroundings(radius=5.0)
        out = capsys.readouterr().out
        assert "5.0" in out

    def test_radius_clamped_low(self, capsys):
        check_surroundings(radius=0.0)
        out = capsys.readouterr().out
        assert "0.5" in out

    def test_radius_clamped_high(self, capsys):
        check_surroundings(radius=999.0)
        out = capsys.readouterr().out
        assert "10.0" in out

    def test_360_scan_mentioned(self, capsys):
        check_surroundings()
        out = capsys.readouterr().out
        assert "360" in out


# ---------------------------------------------------------------------------
# follow_person
# ---------------------------------------------------------------------------

class TestFollowPerson:
    def test_returns_bool(self):
        assert isinstance(follow_person(duration_s=0.1), bool)

    def test_returns_true(self):
        assert follow_person(duration_s=0.1) is True

    def test_default_person_id(self, capsys):
        follow_person(duration_s=0.1)
        out = capsys.readouterr().out
        assert "person_01" in out

    def test_custom_person_id(self, capsys):
        follow_person(person_id="visitor_7", duration_s=0.1)
        out = capsys.readouterr().out
        assert "visitor_7" in out

    def test_duration_clamped_low(self, capsys):
        follow_person(duration_s=0.0)
        out = capsys.readouterr().out
        assert "1.0" in out  # clamped to 1 s

    def test_prints_completion(self, capsys):
        follow_person(duration_s=0.1)
        out = capsys.readouterr().out
        assert "complete" in out.lower()


# ---------------------------------------------------------------------------
# Entry-point: register()
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_runs_without_error(self, capsys):
        from apyrobo_skills_turtlebot4 import register
        register()  # should not raise

    def test_register_prints_skill_count(self, capsys):
        from apyrobo_skills_turtlebot4 import register
        register()
        out = capsys.readouterr().out
        assert "Registered" in out

    def test_all_six_skills_exported(self):
        import apyrobo_skills_turtlebot4 as pkg
        for name in ("patrol_area", "dock", "undock",
                     "inspect_room", "check_surroundings", "follow_person"):
            assert hasattr(pkg, name), f"missing export: {name}"
