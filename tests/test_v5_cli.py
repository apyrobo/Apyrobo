"""Tests for v5.0.0 Five-Minute Success CLI additions.

Covers: apyrobo init, apyrobo shell, apyrobo tutorial,
        enhanced apyrobo test-skill (capability mismatch).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ns(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


# ---------------------------------------------------------------------------
# apyrobo init
# ---------------------------------------------------------------------------

class TestCmdInit:
    def test_creates_directory_structure(self, tmp_path):
        from apyrobo.cli import cmd_init
        args = _ns(name="my-bot", description="Test bot", author="", directory=str(tmp_path / "out"), force=False)
        cmd_init(args)
        out = tmp_path / "out"
        assert (out / "pyproject.toml").exists()
        assert (out / "README.md").exists()
        assert (out / "src" / "apyrobo_skills_my_bot").is_dir()
        assert (out / "src" / "apyrobo_skills_my_bot" / "__init__.py").exists()
        assert (out / "src" / "apyrobo_skills_my_bot" / "skills.py").exists()
        assert (out / "tests" / "test_skills.py").exists()
        assert (out / ".github" / "workflows" / "ci.yml").exists()

    def test_pyproject_has_entry_point(self, tmp_path):
        from apyrobo.cli import cmd_init
        args = _ns(name="spot", description="", author="", directory=str(tmp_path / "out"), force=False)
        cmd_init(args)
        content = (tmp_path / "out" / "pyproject.toml").read_text()
        assert "apyrobo.skills" in content
        assert "apyrobo_skills_spot" in content

    def test_pyproject_has_author_when_provided(self, tmp_path):
        from apyrobo.cli import cmd_init
        args = _ns(name="mybot", description="", author="Alice", directory=str(tmp_path / "out"), force=False)
        cmd_init(args)
        content = (tmp_path / "out" / "pyproject.toml").read_text()
        assert "Alice" in content
        # Should be valid TOML inline table, not double braces
        assert "{{" not in content

    def test_no_author_field_when_empty(self, tmp_path):
        from apyrobo.cli import cmd_init
        args = _ns(name="mybot", description="", author="", directory=str(tmp_path / "out"), force=False)
        cmd_init(args)
        content = (tmp_path / "out" / "pyproject.toml").read_text()
        assert "authors" not in content

    def test_kebab_name_normalised_to_snake_module(self, tmp_path):
        from apyrobo.cli import cmd_init
        args = _ns(name="my-robot", description="", author="", directory=str(tmp_path / "out"), force=False)
        cmd_init(args)
        assert (tmp_path / "out" / "src" / "apyrobo_skills_my_robot").is_dir()

    def test_refuses_existing_dir_without_force(self, tmp_path):
        from apyrobo.cli import cmd_init
        out = tmp_path / "out"
        out.mkdir()
        args = _ns(name="bot", description="", author="", directory=str(out), force=False)
        with pytest.raises(SystemExit):
            cmd_init(args)

    def test_force_overwrites_existing_dir(self, tmp_path):
        from apyrobo.cli import cmd_init
        out = tmp_path / "out"
        out.mkdir()
        (out / "stale.txt").write_text("old")
        args = _ns(name="bot", description="", author="", directory=str(out), force=True)
        cmd_init(args)  # must not raise
        assert (out / "pyproject.toml").exists()

    def test_skills_py_contains_skill_decorator(self, tmp_path):
        from apyrobo.cli import cmd_init
        args = _ns(name="ur5", description="", author="", directory=str(tmp_path / "out"), force=False)
        cmd_init(args)
        content = (tmp_path / "out" / "src" / "apyrobo_skills_ur5" / "skills.py").read_text()
        assert "@skill" in content
        assert "def register" in content

    def test_tests_py_imports_module(self, tmp_path):
        from apyrobo.cli import cmd_init
        args = _ns(name="tb4", description="", author="", directory=str(tmp_path / "out"), force=False)
        cmd_init(args)
        content = (tmp_path / "out" / "tests" / "test_skills.py").read_text()
        assert "apyrobo_skills_tb4" in content

    def test_ci_workflow_uses_apyrobo(self, tmp_path):
        from apyrobo.cli import cmd_init
        args = _ns(name="mybot", description="", author="", directory=str(tmp_path / "out"), force=False)
        cmd_init(args)
        content = (tmp_path / "out" / ".github" / "workflows" / "ci.yml").read_text()
        assert "apyrobo" in content
        assert "pytest" in content

    def test_uses_default_directory_from_name(self, tmp_path, monkeypatch):
        from apyrobo.cli import cmd_init
        monkeypatch.chdir(tmp_path)
        args = _ns(name="demo-bot", description="", author="", directory=None, force=False)
        cmd_init(args)
        assert (tmp_path / "demo-bot" / "pyproject.toml").exists()

    def test_underscore_name_normalised(self, tmp_path):
        from apyrobo.cli import cmd_init
        args = _ns(name="my_bot", description="", author="", directory=str(tmp_path / "out"), force=False)
        cmd_init(args)
        content = (tmp_path / "out" / "pyproject.toml").read_text()
        # package name should use hyphens
        assert "apyrobo-skills-my-bot" in content


# ---------------------------------------------------------------------------
# apyrobo shell
# ---------------------------------------------------------------------------

class TestCmdShell:
    def test_shell_launches_interact_with_robot_in_namespace(self):
        from apyrobo.cli import cmd_shell
        captured_ns: dict = {}
        captured_banner: list[str] = []

        def fake_interact(banner="", local=None, exitmsg=""):
            captured_banner.append(banner)
            captured_ns.update(local or {})

        args = _ns(robot="mock://turtlebot4", provider="rule")
        with patch("code.interact", side_effect=fake_interact):
            cmd_shell(args)

        assert "robot" in captured_ns
        assert "agent" in captured_ns
        assert "Robot" in captured_ns
        assert "BUILTIN_SKILLS" in captured_ns

    def test_shell_banner_mentions_robot_and_provider(self):
        from apyrobo.cli import cmd_shell
        banners: list[str] = []

        def fake_interact(banner="", local=None, exitmsg=""):
            banners.append(banner)

        args = _ns(robot="mock://turtlebot4", provider="rule")
        with patch("code.interact", side_effect=fake_interact):
            cmd_shell(args)

        assert banners
        banner = banners[0]
        assert "rule" in banner
        assert "mock://turtlebot4" in banner

    def test_shell_exits_on_bad_robot(self):
        from apyrobo.cli import cmd_shell
        args = _ns(robot="nonexistent://bad-robot", provider="rule")
        with pytest.raises(SystemExit):
            cmd_shell(args)

    def test_shell_exits_on_bad_provider(self):
        from apyrobo.cli import cmd_shell
        args = _ns(robot="mock://turtlebot4", provider="__invalid_provider__")
        with patch("code.interact"):
            with pytest.raises((SystemExit, ValueError)):
                cmd_shell(args)


# ---------------------------------------------------------------------------
# apyrobo tutorial
# ---------------------------------------------------------------------------

class TestCmdTutorial:
    def test_tutorial_non_interactive_completes(self, capsys):
        from apyrobo.cli import cmd_tutorial
        args = _ns(non_interactive=True)
        cmd_tutorial(args)
        out = capsys.readouterr().out
        assert "APYROBO Interactive Tutorial" in out
        assert "Tutorial complete" in out

    def test_tutorial_prints_all_step_titles(self, capsys):
        from apyrobo.cli import cmd_tutorial, _TUTORIAL_STEPS
        args = _ns(non_interactive=True)
        cmd_tutorial(args)
        out = capsys.readouterr().out
        for step in _TUTORIAL_STEPS:
            assert step["title"] in out

    def test_tutorial_prints_next_steps_at_end(self, capsys):
        from apyrobo.cli import cmd_tutorial
        args = _ns(non_interactive=True)
        cmd_tutorial(args)
        out = capsys.readouterr().out
        assert "apyrobo shell" in out
        assert "apyrobo init" in out

    def test_tutorial_interactive_quit(self, capsys, monkeypatch):
        from apyrobo.cli import cmd_tutorial
        monkeypatch.setattr("builtins.input", lambda _: "q")
        args = _ns(non_interactive=False)
        cmd_tutorial(args)  # must not raise
        out = capsys.readouterr().out
        assert "APYROBO Interactive Tutorial" in out

    def test_tutorial_interactive_enter_continues(self, capsys, monkeypatch):
        from apyrobo.cli import cmd_tutorial, _TUTORIAL_STEPS
        responses = iter([""] * len(_TUTORIAL_STEPS))
        monkeypatch.setattr("builtins.input", lambda _: next(responses, ""))
        args = _ns(non_interactive=False)
        cmd_tutorial(args)
        out = capsys.readouterr().out
        assert "Tutorial complete" in out


# ---------------------------------------------------------------------------
# apyrobo test-skill — capability mismatch detection
# ---------------------------------------------------------------------------

class TestCmdTestSkillCapabilityMismatch:
    """Enhanced test-skill now emits structured warnings for missing capabilities."""

    def _run_test_skill(self, skill_id, robot_uri="mock://turtlebot4", params="{}", repeat=1):
        from apyrobo.cli import cmd_test_skill
        args = _ns(skill=skill_id, robot=robot_uri, params=params, repeat=repeat)
        cmd_test_skill(args)

    def test_passes_for_known_builtin_skill(self, capsys):
        self._run_test_skill("navigate_to")
        out = capsys.readouterr().out
        assert "Passed: 1/1" in out

    def test_capability_mismatch_warns_for_missing_cap(self, capsys, monkeypatch):
        from apyrobo.cli import cmd_test_skill
        from apyrobo.skills.skill import Skill
        from apyrobo.core.schemas import CapabilityType

        # pick_object requires PICK — mock turtlebot4 only has navigate/rotate,
        # so this triggers the mismatch warning
        fake_skill = Skill(
            skill_id="fake_manip_op",
            name="fake manipulation op",
            description="needs manipulation",
            required_capability=CapabilityType.MANIPULATE,
        )
        monkeypatch.setattr(
            "apyrobo.skills.skill.BUILTIN_SKILLS",
            {"fake_manip_op": fake_skill},
        )

        from apyrobo.skills.handlers import _DEFAULT_REGISTRY
        _DEFAULT_REGISTRY._handlers["fake_manip_op"] = lambda robot, p: True

        args = _ns(skill="fake_manip_op", robot="mock://turtlebot4", params="{}", repeat=1)
        cmd_test_skill(args)

        out = capsys.readouterr().out
        assert "Capability mismatch" in out

        del _DEFAULT_REGISTRY._handlers["fake_manip_op"]

    def test_failure_prints_structured_hint_for_attribute_error(self, capsys, monkeypatch):
        from apyrobo.cli import cmd_test_skill
        from apyrobo.skills.handlers import _DEFAULT_REGISTRY

        def bad_handler(robot, p):
            raise AttributeError("robot has no attribute 'gripper_open'")

        _DEFAULT_REGISTRY._handlers["bad_skill_xyz"] = bad_handler
        args = _ns(skill="bad_skill_xyz", robot="mock://turtlebot4", params="{}", repeat=1)
        with pytest.raises(SystemExit):
            cmd_test_skill(args)
        out = capsys.readouterr().out
        assert "Failure summary" in out

        del _DEFAULT_REGISTRY._handlers["bad_skill_xyz"]

    def test_repeat_counts_all_runs(self, capsys):
        self._run_test_skill("navigate_to", repeat=3)
        out = capsys.readouterr().out
        assert "Passed: 3/3" in out
        assert "Run 1" in out
        assert "Run 3" in out

    def test_structured_error_on_unknown_skill(self):
        from apyrobo.cli import cmd_test_skill
        args = _ns(skill="__nonexistent_skill_abc123__", robot="mock://turtlebot4", params="{}", repeat=1)
        with pytest.raises(SystemExit):
            cmd_test_skill(args)


# ---------------------------------------------------------------------------
# apyrobo dashboard — RobotDashboard unit tests
# ---------------------------------------------------------------------------

class TestRobotDashboard:
    @pytest.fixture
    def dash(self):
        from apyrobo.core.robot import Robot
        from apyrobo.dashboard import RobotDashboard
        robot = Robot.discover("mock://turtlebot4")
        return RobotDashboard(robot, robot_uri="mock://turtlebot4")

    def test_get_robot_status_connected(self, dash):
        status = dash.get_robot_status()
        assert status["connected"] is True
        assert status["name"]
        assert isinstance(status["capabilities"], list)

    def test_record_skill_appears_in_history(self, dash):
        dash.record_skill("navigate_to", "ok", 150.0)
        history = dash.get_skill_history()
        assert len(history) == 1
        assert history[0]["skill_id"] == "navigate_to"
        assert history[0]["status"] == "ok"
        assert history[0]["elapsed_ms"] == 150.0

    def test_history_newest_first(self, dash):
        dash.record_skill("skill_a", "ok", 1.0)
        dash.record_skill("skill_b", "ok", 2.0)
        history = dash.get_skill_history()
        assert history[0]["skill_id"] == "skill_b"
        assert history[1]["skill_id"] == "skill_a"

    def test_record_safety_event_appears(self, dash):
        dash.record_safety_event("watchdog.fired", "navigate_to timed out")
        events = dash.get_safety_events()
        assert len(events) == 1
        assert events[0]["event_type"] == "watchdog.fired"

    def test_get_available_skills_nonempty(self, dash):
        skills = dash.get_available_skills()
        assert len(skills) > 0
        assert all("skill_id" in s for s in skills)

    def test_status_partial_html_contains_robot_name(self, dash):
        from apyrobo.dashboard import _render_status_partial
        status = dash.get_robot_status()
        html = _render_status_partial(status)
        assert status["name"] in html

    def test_history_partial_empty_state(self, dash):
        from apyrobo.dashboard import _render_history_partial
        html = _render_history_partial([])
        assert "No skills" in html

    def test_history_partial_with_data(self, dash):
        from apyrobo.dashboard import _render_history_partial
        dash.record_skill("navigate_to", "ok", 123.0)
        html = _render_history_partial(dash.get_skill_history())
        assert "navigate_to" in html
        assert "123" in html

    def test_safety_partial_empty_ok(self, dash):
        from apyrobo.dashboard import _render_safety_partial
        html = _render_safety_partial([])
        assert "No safety events" in html

    def test_safety_partial_with_event(self, dash):
        from apyrobo.dashboard import _render_safety_partial
        dash.record_safety_event("watchdog.fired", "detail text")
        html = _render_safety_partial(dash.get_safety_events())
        assert "watchdog.fired" in html
        assert "detail text" in html

    def test_live_dashboard_html_has_htmx(self, dash):
        from apyrobo.dashboard import _render_live_dashboard_html
        html = _render_live_dashboard_html("mock://turtlebot4")
        assert "htmx" in html
        assert "hx-get" in html
        assert "hx-trigger" in html
        assert "mock://turtlebot4" in html

    def test_fastapi_app_creates_routes(self, dash):
        pytest.importorskip("fastapi")
        app = dash.create_fastapi_app()
        routes = [r.path for r in app.routes]
        assert "/" in routes
        assert "/health" in routes
        assert "/api/status" in routes
        assert "/partials/status" in routes
        assert "/partials/history" in routes
        assert "/partials/safety" in routes

    def test_disconnected_status_partial(self):
        from apyrobo.dashboard import _render_status_partial
        html = _render_status_partial({"connected": False, "error": "timeout"})
        assert "Not connected" in html or "timeout" in html

    def test_history_bounded_by_max(self):
        from apyrobo.core.robot import Robot
        from apyrobo.dashboard import RobotDashboard, _MAX_HISTORY
        robot = Robot.discover("mock://turtlebot4")
        dash = RobotDashboard(robot)
        for i in range(_MAX_HISTORY + 10):
            dash.record_skill(f"skill_{i}", "ok", float(i))
        assert len(dash.get_skill_history()) == _MAX_HISTORY


# ---------------------------------------------------------------------------
# mock_fleet demo module
# ---------------------------------------------------------------------------

class TestMockFleet:
    def test_module_importable(self):
        from apyrobo.demo import mock_fleet
        assert hasattr(mock_fleet, "run")

    def test_run_iterates_and_exits(self):
        from apyrobo.demo import mock_fleet
        # run() loops forever — call with max_iterations to limit
        mock_fleet.run(max_iterations=1)
