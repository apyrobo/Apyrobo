"""Tests for `apyrobo test-skill` CLI command."""
from __future__ import annotations

import sys
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from apyrobo.cli import cmd_test_skill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs) -> SimpleNamespace:
    defaults = {
        "skill": "navigate_to",
        "robot": "mock://turtlebot4",
        "params": "{}",
        "repeat": 1,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _run(args: SimpleNamespace, expect_exit: int | None = None) -> str:
    buf = StringIO()
    if expect_exit is not None:
        with patch("sys.stdout", buf), pytest.raises(SystemExit) as exc_info:
            cmd_test_skill(args)
        assert exc_info.value.code == expect_exit
    else:
        with patch("sys.stdout", buf):
            cmd_test_skill(args)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Successful run
# ---------------------------------------------------------------------------

class TestTestSkillSuccess:
    def test_successful_run_prints_checkmark(self):
        out = _run(_make_args(skill="navigate_to"))
        assert "✅" in out

    def test_passed_count_shown(self):
        out = _run(_make_args(skill="navigate_to", repeat=3))
        assert "Passed: 3/3" in out

    def test_repeat_runs_correct_number(self):
        out = _run(_make_args(skill="navigate_to", repeat=5))
        for i in range(1, 6):
            assert f"Run {i}" in out

    def test_skill_name_shown_in_header(self):
        out = _run(_make_args(skill="navigate_to"))
        assert "navigate_to" in out

    def test_robot_uri_shown_in_header(self):
        out = _run(_make_args(skill="navigate_to", robot="mock://turtlebot4"))
        assert "mock://turtlebot4" in out

    def test_timing_stats_shown(self):
        out = _run(_make_args(skill="navigate_to", repeat=2))
        assert "Avg:" in out
        assert "Min:" in out
        assert "Max:" in out


# ---------------------------------------------------------------------------
# Failing run (handler returns False)
# ---------------------------------------------------------------------------

class TestTestSkillFailure:
    def test_failing_skill_shows_cross(self, tmp_path):
        skill_file = tmp_path / "bad_skill.py"
        skill_file.write_text(
            "from apyrobo import skill\n\n"
            "@skill(description='always fails', capability='navigate')\n"
            "def bad_skill() -> bool:\n"
            "    return False\n"
        )
        out = _run(_make_args(skill=str(skill_file)), expect_exit=1)
        assert "❌" in out

    def test_failed_count_shown(self, tmp_path):
        skill_file = tmp_path / "bad_skill2.py"
        skill_file.write_text(
            "from apyrobo import skill\n\n"
            "@skill(description='always fails', capability='navigate')\n"
            "def bad_skill2() -> bool:\n"
            "    return False\n"
        )
        out = _run(_make_args(skill=str(skill_file), repeat=2), expect_exit=1)
        assert "Passed: 0/2" in out

    def test_exception_in_handler_shown_as_failure(self, tmp_path):
        skill_file = tmp_path / "exc_skill.py"
        skill_file.write_text(
            "from apyrobo import skill\n\n"
            "@skill(description='raises', capability='navigate')\n"
            "def exc_skill() -> bool:\n"
            "    raise RuntimeError('boom')\n"
        )
        out = _run(_make_args(skill=str(skill_file)), expect_exit=1)
        assert "❌" in out
        assert "boom" in out


# ---------------------------------------------------------------------------
# Bad skill ID
# ---------------------------------------------------------------------------

class TestTestSkillBadId:
    def test_unknown_skill_id_exits_with_error(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cmd_test_skill(_make_args(skill="definitely_not_a_skill_xyz"))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_missing_skill_file_exits_with_error(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cmd_test_skill(_make_args(skill="nonexistent_file.py"))
        assert exc_info.value.code == 1

    def test_invalid_json_params_exits_with_error(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cmd_test_skill(_make_args(skill="navigate_to", params="{not json}"))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "JSON" in captured.err or "json" in captured.err.lower()


# ---------------------------------------------------------------------------
# --params
# ---------------------------------------------------------------------------

class TestTestSkillParams:
    def test_params_passed_to_handler(self, tmp_path):
        skill_file = tmp_path / "param_skill.py"
        skill_file.write_text(
            "from apyrobo import skill\n\n"
            "received = {}\n\n"
            "@skill(description='records params', capability='navigate')\n"
            "def param_skill(x: float = 0.0, y: float = 0.0) -> bool:\n"
            "    received['x'] = x\n"
            "    received['y'] = y\n"
            "    return True\n"
        )
        out = _run(_make_args(skill=str(skill_file), params='{"x": 3.5, "y": 7.0}'))
        assert "✅" in out

    def test_empty_params_json_is_valid(self):
        out = _run(_make_args(skill="navigate_to", params="{}"))
        assert "✅" in out


# ---------------------------------------------------------------------------
# --repeat
# ---------------------------------------------------------------------------

class TestTestSkillRepeat:
    def test_default_repeat_is_one(self):
        out = _run(_make_args(skill="navigate_to"))
        assert "Runs:     1" in out
        assert "Run 1" in out

    def test_repeat_n_runs_n_times(self):
        out = _run(_make_args(skill="navigate_to", repeat=4))
        assert "Runs:     4" in out
        assert "Passed: 4/4" in out


# ---------------------------------------------------------------------------
# Skill file loading
# ---------------------------------------------------------------------------

class TestTestSkillFileLoading:
    def test_skill_loaded_from_file(self, tmp_path):
        skill_file = tmp_path / "my_skill.py"
        skill_file.write_text(
            "from apyrobo import skill\n\n"
            "@skill(description='file skill', capability='navigate')\n"
            "def my_skill() -> bool:\n"
            "    return True\n"
        )
        out = _run(_make_args(skill=str(skill_file)))
        assert "✅" in out

    def test_nonexistent_py_file_treated_as_id(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cmd_test_skill(_make_args(skill="nonexistent_skill.py"))
        assert exc_info.value.code == 1
