"""Tests for apyrobo-skills-agv skill implementations."""
from __future__ import annotations

import pytest

from apyrobo_skills_agv.navigation import navigate_to, follow_route, dock_to_station
from apyrobo_skills_agv.cargo import load_cargo, unload_cargo


# ---------------------------------------------------------------------------
# navigate_to
# ---------------------------------------------------------------------------

class TestNavigateTo:
    def test_returns_bool(self):
        assert isinstance(navigate_to(1.0, 2.0), bool)

    def test_returns_true(self):
        assert navigate_to(1.0, 2.0) is True

    def test_coordinates_in_output(self, capsys):
        navigate_to(3.5, 7.2)
        out = capsys.readouterr().out
        assert "3.500" in out
        assert "7.200" in out

    def test_default_map_id(self, capsys):
        navigate_to(0.0, 0.0)
        out = capsys.readouterr().out
        assert "default" in out

    def test_custom_map_id(self, capsys):
        navigate_to(0.0, 0.0, map_id="warehouse_B")
        out = capsys.readouterr().out
        assert "warehouse_B" in out

    def test_theta_in_output(self, capsys):
        navigate_to(0.0, 0.0, theta=1.57)
        out = capsys.readouterr().out
        assert "1.570" in out

    def test_prints_arrival_message(self, capsys):
        navigate_to(5.0, 5.0)
        out = capsys.readouterr().out
        assert "arrived" in out.lower() or "complete" in out.lower()


# ---------------------------------------------------------------------------
# follow_route
# ---------------------------------------------------------------------------

class TestFollowRoute:
    def test_returns_bool(self):
        assert isinstance(follow_route("route_01"), bool)

    def test_returns_true(self):
        assert follow_route("route_01") is True

    def test_route_id_in_output(self, capsys):
        follow_route("inspection_loop")
        out = capsys.readouterr().out
        assert "inspection_loop" in out

    def test_single_pass_mode(self, capsys):
        follow_route("route_01", loop=False)
        out = capsys.readouterr().out
        assert "single-pass" in out

    def test_loop_mode_label(self, capsys):
        follow_route("route_01", loop=True)
        out = capsys.readouterr().out
        assert "looping" in out

    def test_prints_completion(self, capsys):
        follow_route("route_01")
        out = capsys.readouterr().out
        assert "complete" in out.lower()


# ---------------------------------------------------------------------------
# dock_to_station
# ---------------------------------------------------------------------------

class TestDockToStation:
    def test_returns_bool(self):
        assert isinstance(dock_to_station("station_A"), bool)

    def test_returns_true(self):
        assert dock_to_station("station_A") is True

    def test_station_id_in_output(self, capsys):
        dock_to_station("charging_bay_1")
        out = capsys.readouterr().out
        assert "charging_bay_1" in out

    def test_default_approach_speed(self, capsys):
        dock_to_station("station_A")
        out = capsys.readouterr().out
        assert "0.20" in out

    def test_custom_approach_speed(self, capsys):
        dock_to_station("station_A", approach_speed=0.1)
        out = capsys.readouterr().out
        assert "0.10" in out

    def test_approach_speed_clamped_low(self, capsys):
        dock_to_station("station_A", approach_speed=0.0)
        out = capsys.readouterr().out
        assert "0.05" in out  # clamped to 0.05 m/s

    def test_approach_speed_clamped_high(self, capsys):
        dock_to_station("station_A", approach_speed=99.0)
        out = capsys.readouterr().out
        assert "0.50" in out  # clamped to 0.5 m/s

    def test_prints_docked_message(self, capsys):
        dock_to_station("station_A")
        out = capsys.readouterr().out
        assert "docked" in out.lower() or "contact" in out.lower()


# ---------------------------------------------------------------------------
# load_cargo
# ---------------------------------------------------------------------------

class TestLoadCargo:
    def test_returns_bool(self):
        assert isinstance(load_cargo("station_A"), bool)

    def test_returns_true(self):
        assert load_cargo("station_A") is True

    def test_station_id_in_output(self, capsys):
        load_cargo("pick_station_3")
        out = capsys.readouterr().out
        assert "pick_station_3" in out

    def test_explicit_cargo_id_in_output(self, capsys):
        load_cargo("station_A", cargo_id="pallet_007")
        out = capsys.readouterr().out
        assert "pallet_007" in out

    def test_empty_cargo_id_uses_queued(self, capsys):
        load_cargo("station_A", cargo_id="")
        out = capsys.readouterr().out
        assert "queued" in out.lower() or "next" in out.lower()

    def test_prints_sensor_confirmation(self, capsys):
        load_cargo("station_A")
        out = capsys.readouterr().out
        assert "sensor" in out.lower() or "secured" in out.lower()


# ---------------------------------------------------------------------------
# unload_cargo
# ---------------------------------------------------------------------------

class TestUnloadCargo:
    def test_returns_bool(self):
        assert isinstance(unload_cargo("station_B"), bool)

    def test_returns_true(self):
        assert unload_cargo("station_B") is True

    def test_station_id_in_output(self, capsys):
        unload_cargo("drop_station_2")
        out = capsys.readouterr().out
        assert "drop_station_2" in out

    def test_explicit_cargo_id_in_output(self, capsys):
        unload_cargo("station_B", cargo_id="crate_X")
        out = capsys.readouterr().out
        assert "crate_X" in out

    def test_empty_cargo_id_uses_current(self, capsys):
        unload_cargo("station_B", cargo_id="")
        out = capsys.readouterr().out
        assert "current" in out.lower() or "load" in out.lower()

    def test_prints_delivered_message(self, capsys):
        unload_cargo("station_B")
        out = capsys.readouterr().out
        assert "delivered" in out.lower() or "cleared" in out.lower()


# ---------------------------------------------------------------------------
# Entry-point: register()
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_runs_without_error(self, capsys):
        from apyrobo_skills_agv import register
        register()  # should not raise

    def test_register_prints_skill_count(self, capsys):
        from apyrobo_skills_agv import register
        register()
        out = capsys.readouterr().out
        assert "Registered" in out

    def test_all_five_skills_exported(self):
        import apyrobo_skills_agv as pkg
        for name in ("navigate_to", "follow_route", "dock_to_station",
                     "load_cargo", "unload_cargo"):
            assert hasattr(pkg, name), f"missing export: {name}"
