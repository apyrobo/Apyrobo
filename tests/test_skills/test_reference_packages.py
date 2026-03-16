"""
SK-04: Tests for reference skill packages.

Covers:
    - All 10+ skill handlers execute correctly against MockAdapter
    - Each package passes SkillPackage.validate()
    - waypoint_tour visits all waypoints in order
    - patrol_route emits obstacle report at each waypoint
    - dock_to_charger calls robot.connect() after reaching dock position
    - Package install via CLI succeeds
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from apyrobo.core.robot import Robot
from apyrobo.skills.handlers import dispatch, _HANDLERS
from apyrobo.skills.package import SkillPackage
from apyrobo.skills.registry import SkillRegistry
from apyrobo.skills.skill import Skill


SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_robot() -> Robot:
    return Robot.discover("mock://ref_pkg_bot")


@pytest.fixture
def robot_mock() -> MagicMock:
    """Fine-grained mock for assertions."""
    m = MagicMock()
    m.gripper_close.return_value = True
    m.gripper_open.return_value = True
    caps_mock = MagicMock()
    caps_mock.name = "test_robot"
    caps_mock.capabilities = []
    caps_mock.sensors = [MagicMock(), MagicMock()]
    m.capabilities.return_value = caps_mock
    health_mock = {"battery_pct": 85, "status": "ok"}
    m.get_health.return_value = health_mock
    return m


# ===========================================================================
# Package validation
# ===========================================================================


class TestPackageValidation:
    """Each reference package passes SkillPackage.validate()."""

    @pytest.mark.parametrize("pkg_name", [
        "builtin-skills",
        "navigation-extras",
        "inspection",
    ])
    def test_package_validates(self, pkg_name: str) -> None:
        pkg = SkillPackage.load(SKILLS_DIR / pkg_name)
        errors = pkg.validate()
        assert errors == [], f"{pkg_name} validation errors: {errors}"

    @pytest.mark.parametrize("pkg_name", [
        "builtin-skills",
        "navigation-extras",
        "inspection",
    ])
    def test_package_has_readme(self, pkg_name: str) -> None:
        assert (SKILLS_DIR / pkg_name / "README.md").exists()

    @pytest.mark.parametrize("pkg_name", [
        "builtin-skills",
        "navigation-extras",
        "inspection",
    ])
    def test_package_has_apache_license(self, pkg_name: str) -> None:
        pkg = SkillPackage.load(SKILLS_DIR / pkg_name)
        assert pkg.license == "Apache-2.0"


# ===========================================================================
# Built-in handlers (7 skills in builtin-skills)
# ===========================================================================


class TestBuiltinSkillHandlers:
    """All handlers in builtin-skills execute correctly."""

    def test_navigate_to(self, robot_mock: MagicMock) -> None:
        result = dispatch("navigate_to", robot_mock, {"x": 5.0, "y": 3.0, "speed": 0.8})
        assert result is True
        robot_mock.move.assert_called_once_with(x=5.0, y=3.0, speed=0.8)

    def test_stop(self, robot_mock: MagicMock) -> None:
        result = dispatch("stop", robot_mock, {})
        assert result is True
        robot_mock.stop.assert_called_once()

    def test_rotate(self, robot_mock: MagicMock) -> None:
        result = dispatch("rotate", robot_mock, {"angle_rad": 1.57, "speed": 0.5})
        assert result is True
        robot_mock.rotate.assert_called_once_with(angle_rad=1.57, speed=0.5)

    def test_pick_object(self, robot_mock: MagicMock) -> None:
        result = dispatch("pick_object", robot_mock, {})
        assert result is True
        robot_mock.gripper_close.assert_called_once()

    def test_place_object(self, robot_mock: MagicMock) -> None:
        result = dispatch("place_object", robot_mock, {})
        assert result is True
        robot_mock.gripper_open.assert_called_once()

    def test_report_status(self, robot_mock: MagicMock) -> None:
        result = dispatch("report_status", robot_mock, {})
        assert result is True
        robot_mock.capabilities.assert_called_once()

    def test_report_battery_status(self, robot_mock: MagicMock) -> None:
        result = dispatch("report_battery_status", robot_mock, {})
        assert result is True
        robot_mock.get_health.assert_called_once()


# ===========================================================================
# Navigation-extras handlers
# ===========================================================================


class TestNavigationExtrasHandlers:
    """waypoint_tour and dock_to_charger handlers."""

    def test_waypoint_tour_visits_all_waypoints(self, robot_mock: MagicMock) -> None:
        """waypoint_tour visits all waypoints in order."""
        waypoints = [
            {"x": 1.0, "y": 0.0},
            {"x": 3.0, "y": 2.0},
            {"x": 0.0, "y": 4.0},
        ]
        result = dispatch("waypoint_tour", robot_mock, {
            "waypoints": waypoints,
            "speed": 0.5,
            "loops": 1,
        })
        assert result is True
        assert robot_mock.move.call_count == 3
        robot_mock.move.assert_any_call(x=1.0, y=0.0, speed=0.5)
        robot_mock.move.assert_any_call(x=3.0, y=2.0, speed=0.5)
        robot_mock.move.assert_any_call(x=0.0, y=4.0, speed=0.5)

        # Verify order
        calls = robot_mock.move.call_args_list
        assert calls[0] == call(x=1.0, y=0.0, speed=0.5)
        assert calls[1] == call(x=3.0, y=2.0, speed=0.5)
        assert calls[2] == call(x=0.0, y=4.0, speed=0.5)

    def test_waypoint_tour_loops(self, robot_mock: MagicMock) -> None:
        """waypoint_tour repeats waypoints for specified loops."""
        waypoints = [{"x": 1.0, "y": 0.0}, {"x": 2.0, "y": 0.0}]
        dispatch("waypoint_tour", robot_mock, {
            "waypoints": waypoints,
            "speed": 0.3,
            "loops": 2,
        })
        assert robot_mock.move.call_count == 4

    def test_dock_to_charger_navigates_then_connects(self, robot_mock: MagicMock) -> None:
        """dock_to_charger navigates to dock position then calls connect()."""
        result = dispatch("dock_to_charger", robot_mock, {
            "dock_x": 10.0,
            "dock_y": 5.0,
        })
        assert result is True
        robot_mock.move.assert_called_once_with(x=10.0, y=5.0, speed=0.3)
        robot_mock.connect.assert_called_once()

        # Verify order: move before connect
        move_call_idx = None
        connect_call_idx = None
        for i, c in enumerate(robot_mock.method_calls):
            if c[0] == "move" and move_call_idx is None:
                move_call_idx = i
            if c[0] == "connect" and connect_call_idx is None:
                connect_call_idx = i
        assert move_call_idx is not None
        assert connect_call_idx is not None
        assert move_call_idx < connect_call_idx


# ===========================================================================
# Inspection handlers
# ===========================================================================


class TestInspectionHandlers:
    """patrol_route and scan_area handlers."""

    def test_patrol_route_scans_at_each_waypoint(
        self, robot_mock: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """patrol_route moves + rotates + logs at each waypoint."""
        waypoints = [
            {"x": 1.0, "y": 0.0},
            {"x": 3.0, "y": 2.0},
        ]
        with caplog.at_level(logging.INFO):
            result = dispatch("patrol_route", robot_mock, {
                "waypoints": waypoints,
                "speed": 0.5,
                "loops": 1,
            })
        assert result is True

        # Should move to each waypoint
        assert robot_mock.move.call_count == 2
        # Should rotate at each waypoint for scanning
        assert robot_mock.rotate.call_count == 2
        # Should log checkpoint at each waypoint
        assert robot_mock.capabilities.call_count == 2

        # Verify log messages mention checkpoint coordinates
        checkpoint_logs = [r for r in caplog.records if "checkpoint" in r.message.lower()]
        assert len(checkpoint_logs) == 2

    def test_scan_area_performs_full_rotation(self, robot_mock: MagicMock) -> None:
        """scan_area rotates for the specified number of full rotations."""
        result = dispatch("scan_area", robot_mock, {
            "rotation_speed": 0.3,
            "full_rotations": 2,
        })
        assert result is True
        assert robot_mock.rotate.call_count == 2
        robot_mock.rotate.assert_called_with(angle_rad=6.283, speed=0.3)

    def test_scan_area_default_params(self, robot_mock: MagicMock) -> None:
        """scan_area with defaults does 1 rotation."""
        dispatch("scan_area", robot_mock, {})
        assert robot_mock.rotate.call_count == 1


# ===========================================================================
# Install reference package via registry
# ===========================================================================


class TestReferencePackageInstall:
    """Install reference packages into registry."""

    def test_install_builtin_skills(self, tmp_path: Path) -> None:
        """Install builtin-skills and verify 7 skills listed."""
        reg = SkillRegistry(tmp_path / "registry")
        pkg = reg.install(SKILLS_DIR / "builtin-skills")
        assert pkg.name == "builtin-skills"
        assert len(pkg.skills) == 7
        assert "navigate_to" in pkg.skill_ids
        assert "report_battery_status" in pkg.skill_ids

    def test_install_navigation_extras(self, tmp_path: Path) -> None:
        reg = SkillRegistry(tmp_path / "registry")
        pkg = reg.install(SKILLS_DIR / "navigation-extras")
        assert pkg.name == "navigation-extras"
        assert "waypoint_tour" in pkg.skill_ids
        assert "dock_to_charger" in pkg.skill_ids

    def test_install_inspection(self, tmp_path: Path) -> None:
        reg = SkillRegistry(tmp_path / "registry")
        pkg = reg.install(SKILLS_DIR / "inspection")
        assert pkg.name == "inspection"
        assert "patrol_route" in pkg.skill_ids
        assert "scan_area" in pkg.skill_ids

    def test_install_all_three_packages(self, tmp_path: Path) -> None:
        """All three packages install into the same registry."""
        reg = SkillRegistry(tmp_path / "registry")
        reg.install(SKILLS_DIR / "builtin-skills")
        reg.install(SKILLS_DIR / "navigation-extras")
        reg.install(SKILLS_DIR / "inspection")
        assert reg.package_count == 3
        all_skills = reg.all_skills()
        assert len(all_skills) >= 11  # 7 + 2 + 2
