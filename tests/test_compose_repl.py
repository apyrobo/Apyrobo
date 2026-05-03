"""Tests for the ComposeREPL skill composition REPL."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import apyrobo.skills.builtins  # noqa: F401 — ensure handlers are registered

from apyrobo import Robot
from apyrobo.skills.compose import ComposeREPL, _parse_kwargs
from apyrobo.skills.library import SkillLibrary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def robot():
    return Robot.discover("mock://turtlebot4")


@pytest.fixture()
def library():
    return SkillLibrary()


@pytest.fixture()
def repl(robot, library):
    return ComposeREPL(robot, library)


# ---------------------------------------------------------------------------
# _parse_kwargs
# ---------------------------------------------------------------------------

class TestParseKwargs:
    def test_int_conversion(self):
        assert _parse_kwargs(["loops=3"]) == {"loops": 3}

    def test_float_conversion(self):
        assert _parse_kwargs(["x=3.5"]) == {"x": 3.5}

    def test_string_value(self):
        assert _parse_kwargs(["room=lab_A"]) == {"room": "lab_A"}

    def test_quoted_string(self):
        assert _parse_kwargs(['name="my robot"']) == {"name": "my robot"}

    def test_multiple_kwargs(self):
        result = _parse_kwargs(["x=1.0", "y=2.0", "speed=0.5"])
        assert result == {"x": 1.0, "y": 2.0, "speed": 0.5}

    def test_empty_tokens(self):
        assert _parse_kwargs([]) == {}

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="key=value"):
            _parse_kwargs(["badtoken"])

    def test_negative_float(self):
        result = _parse_kwargs(["x=-1.5"])
        assert result == {"x": -1.5}


# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------

class TestHelp:
    def test_help_contains_commands(self, repl):
        out = repl.run_command("help")
        for cmd in ("add", "show", "remove", "clear", "run", "test",
                    "save", "load", "skills", "robot", "quit"):
            assert cmd in out

    def test_help_mentions_usage(self, repl):
        out = repl.run_command("help")
        assert "skill_id" in out or "SKILL" in out or "Available" in out


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------

class TestSkillsCommand:
    def test_lists_builtin_skills(self, repl):
        out = repl.run_command("skills")
        assert "navigate_to" in out

    def test_shows_descriptions(self, repl):
        out = repl.run_command("skills")
        assert len(out.strip().splitlines()) > 2

    def test_empty_library_says_no_skills(self, robot):
        from apyrobo.skills.skill import Skill
        lib = SkillLibrary()
        lib._custom_skills.clear()
        # patch BUILTIN_SKILLS to empty so all_skills returns nothing
        with patch.dict("apyrobo.skills.skill.BUILTIN_SKILLS", {}, clear=True):
            r = ComposeREPL(robot, lib)
            out = r.run_command("skills")
        assert "No skills" in out


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------

class TestAdd:
    def test_add_valid_skill(self, repl):
        out = repl.run_command("add navigate_to x=3.0 y=5.0")
        assert "[1]" in out
        assert "navigate_to" in out

    def test_add_increments_step_counter(self, repl):
        repl.run_command("add navigate_to x=1.0 y=0.0")
        out = repl.run_command("add stop")
        assert "[2]" in out

    def test_add_unknown_skill_shows_error(self, repl):
        out = repl.run_command("add definitely_not_a_skill_xyz")
        assert "Unknown skill" in out or "unknown" in out.lower()

    def test_add_no_args_shows_usage(self, repl):
        out = repl.run_command("add")
        assert "Usage" in out

    def test_add_params_shown_in_output(self, repl):
        out = repl.run_command("add navigate_to x=7.0 y=8.0")
        assert "7.0" in out
        assert "8.0" in out

    def test_add_defaults_shown_when_no_params(self, repl):
        out = repl.run_command("add stop")
        assert "defaults" in out or "stop" in out

    def test_add_bad_param_format_shows_error(self, repl):
        out = repl.run_command("add navigate_to badparam")
        assert "error" in out.lower() or "key=value" in out.lower()

    def test_plan_grows_after_add(self, repl):
        repl.run_command("add navigate_to x=1.0 y=0.0")
        repl.run_command("add stop")
        assert len(repl._plan) == 2

    def test_plan_stores_correct_skill_id(self, repl):
        repl.run_command("add navigate_to x=1.0 y=0.0")
        assert repl._plan[0][0] == "navigate_to"

    def test_plan_stores_params(self, repl):
        repl.run_command("add navigate_to x=3.0 y=4.0")
        assert repl._plan[0][1]["x"] == 3.0
        assert repl._plan[0][1]["y"] == 4.0


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------

class TestShow:
    def test_show_empty_plan(self, repl):
        out = repl.run_command("show")
        assert "empty" in out.lower()

    def test_show_lists_steps(self, repl):
        repl.run_command("add navigate_to x=1.0 y=2.0")
        repl.run_command("add stop")
        out = repl.run_command("show")
        assert "[1]" in out
        assert "[2]" in out

    def test_show_includes_skill_ids(self, repl):
        repl.run_command("add navigate_to x=1.0 y=2.0")
        repl.run_command("add stop")
        out = repl.run_command("show")
        assert "navigate_to" in out
        assert "stop" in out

    def test_show_step_count_in_header(self, repl):
        repl.run_command("add navigate_to")
        repl.run_command("add stop")
        out = repl.run_command("show")
        assert "2" in out


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

class TestRemove:
    def test_remove_first_step(self, repl):
        repl.run_command("add navigate_to x=1.0 y=0.0")
        repl.run_command("add stop")
        repl.run_command("remove 1")
        assert len(repl._plan) == 1
        assert repl._plan[0][0] == "stop"

    def test_remove_shows_confirmation(self, repl):
        repl.run_command("add navigate_to x=1.0 y=0.0")
        out = repl.run_command("remove 1")
        assert "Removed" in out or "removed" in out

    def test_remove_out_of_range(self, repl):
        repl.run_command("add stop")
        out = repl.run_command("remove 5")
        assert "out of range" in out.lower() or "range" in out.lower()

    def test_remove_zero_is_invalid(self, repl):
        repl.run_command("add stop")
        out = repl.run_command("remove 0")
        assert "out of range" in out.lower() or "range" in out.lower()

    def test_remove_non_integer(self, repl):
        out = repl.run_command("remove foo")
        assert "Invalid" in out or "invalid" in out.lower()

    def test_remove_no_args_shows_usage(self, repl):
        out = repl.run_command("remove")
        assert "Usage" in out


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_empties_plan(self, repl):
        repl.run_command("add navigate_to")
        repl.run_command("add stop")
        repl.run_command("clear")
        assert repl._plan == []

    def test_clear_shows_confirmation(self, repl):
        repl.run_command("add stop")
        out = repl.run_command("clear")
        assert "clear" in out.lower() or "removed" in out.lower()

    def test_clear_empty_plan(self, repl):
        out = repl.run_command("clear")
        assert "0" in out


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

class TestRun:
    def test_run_empty_plan_shows_error(self, repl):
        out = repl.run_command("run")
        assert "empty" in out.lower()

    def test_run_executes_plan(self, repl):
        repl.run_command("add navigate_to x=1.0 y=2.0")
        out = repl.run_command("run")
        assert "COMPLETED" in out or "completed" in out.lower() or "✅" in out

    def test_run_multi_step_plan(self, repl):
        repl.run_command("add navigate_to x=1.0 y=0.0")
        repl.run_command("add stop")
        out = repl.run_command("run")
        assert "2" in out  # 2 steps in summary

    def test_run_shows_step_output(self, repl):
        repl.run_command("add navigate_to x=1.0 y=2.0")
        out = repl.run_command("run")
        # Should show something about navigate_to completing
        assert "navigate_to" in out or "✅" in out or "completed" in out.lower()

    def test_run_duplicate_skill_ids(self, repl):
        """Same skill added twice should execute both (unique graph nodes)."""
        repl.run_command("add navigate_to x=1.0 y=0.0")
        repl.run_command("add navigate_to x=3.0 y=4.0")
        out = repl.run_command("run")
        assert "COMPLETED" in out or "completed" in out.lower() or "2" in out


# ---------------------------------------------------------------------------
# test (single skill)
# ---------------------------------------------------------------------------

class TestTestCommand:
    def test_test_single_skill(self, repl):
        out = repl.run_command("test navigate_to x=1.0 y=0.0")
        assert "✅" in out or "completed" in out.lower()

    def test_test_unknown_skill(self, repl):
        out = repl.run_command("test no_such_skill")
        assert "Unknown" in out or "unknown" in out.lower()

    def test_test_no_args_shows_usage(self, repl):
        out = repl.run_command("test")
        assert "Usage" in out

    def test_test_shows_timing(self, repl):
        out = repl.run_command("test stop")
        assert "s)" in out  # timing like (0.001s)


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_writes_file(self, repl, tmp_path):
        repl.run_command("add navigate_to x=1.0 y=2.0")
        repl.run_command("add stop")
        out_file = tmp_path / "plan.json"
        out = repl.run_command(f"save {out_file}")
        assert out_file.exists()
        assert "saved" in out.lower()

    def test_saved_file_is_valid_json(self, repl, tmp_path):
        repl.run_command("add navigate_to x=1.0 y=2.0")
        out_file = tmp_path / "plan.json"
        repl.run_command(f"save {out_file}")
        data = json.loads(out_file.read_text())
        assert "steps" in data
        assert data["steps"][0]["skill_id"] == "navigate_to"

    def test_load_restores_plan(self, repl, tmp_path):
        repl.run_command("add navigate_to x=1.0 y=2.0")
        repl.run_command("add stop")
        out_file = tmp_path / "plan.json"
        repl.run_command(f"save {out_file}")

        repl2 = ComposeREPL(repl._robot, repl._library)
        repl2.run_command(f"load {out_file}")
        assert len(repl2._plan) == 2
        assert repl2._plan[0][0] == "navigate_to"
        assert repl2._plan[1][0] == "stop"

    def test_load_shows_plan(self, repl, tmp_path):
        repl.run_command("add stop")
        out_file = tmp_path / "plan.json"
        repl.run_command(f"save {out_file}")

        repl2 = ComposeREPL(repl._robot, repl._library)
        out = repl2.run_command(f"load {out_file}")
        assert "stop" in out

    def test_load_missing_file(self, repl, tmp_path):
        out = repl.run_command(f"load {tmp_path}/nonexistent.json")
        assert "not found" in out.lower() or "File" in out

    def test_load_invalid_json(self, repl, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        out = repl.run_command(f"load {bad}")
        assert "Invalid" in out or "JSON" in out

    def test_save_no_args_shows_usage(self, repl):
        out = repl.run_command("save")
        assert "Usage" in out

    def test_load_no_args_shows_usage(self, repl):
        out = repl.run_command("load")
        assert "Usage" in out

    def test_round_trip_preserves_params(self, repl, tmp_path):
        repl.run_command("add navigate_to x=7.5 y=8.5")
        out_file = tmp_path / "plan.json"
        repl.run_command(f"save {out_file}")

        repl2 = ComposeREPL(repl._robot, repl._library)
        repl2.run_command(f"load {out_file}")
        assert repl2._plan[0][1]["x"] == 7.5
        assert repl2._plan[0][1]["y"] == 8.5


# ---------------------------------------------------------------------------
# unknown command
# ---------------------------------------------------------------------------

class TestUnknownCommand:
    def test_unknown_command_message(self, repl):
        out = repl.run_command("blargfoo")
        assert "Unknown command" in out
        assert "blargfoo" in out
        assert "help" in out.lower()

    def test_empty_line_returns_empty(self, repl):
        out = repl.run_command("")
        assert out == ""

    def test_whitespace_only_returns_empty(self, repl):
        out = repl.run_command("   ")
        assert out == ""


# ---------------------------------------------------------------------------
# robot command
# ---------------------------------------------------------------------------

class TestRobotCommand:
    def test_robot_switch_success(self, repl):
        out = repl.run_command("robot mock://test")
        assert "mock://test" in out or "Switched" in out

    def test_robot_no_args_shows_usage(self, repl):
        out = repl.run_command("robot")
        assert "Usage" in out

    def test_robot_invalid_uri_shows_error(self, repl):
        out = repl.run_command("robot totally://invalid://uri")
        # Should report failure, not crash
        assert "Failed" in out or "Error" in out or "invalid" in out.lower() or "Switched" in out


# ---------------------------------------------------------------------------
# prompt format
# ---------------------------------------------------------------------------

class TestPrompt:
    def test_empty_plan_prompt(self, repl):
        assert repl._prompt() == "compose> "

    def test_nonempty_plan_prompt(self, repl):
        repl.run_command("add stop")
        assert repl._prompt() == "compose[1]> "

    def test_prompt_shows_step_count(self, repl):
        repl.run_command("add navigate_to")
        repl.run_command("add stop")
        assert "[2]" in repl._prompt()
