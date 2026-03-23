"""
Comprehensive tests for apyrobo/skills/library.py — targeting missing coverage lines.

Covers:
- SkillLibrary() without dir
- SkillLibrary(dir_that_exists) auto-loads
- load_all when no dir
- load_all from temp dir with valid json skill files
- load_file success
- load_json
- save_skill (to temp path, to default dir)
- get (custom/registry/builtin)
- all_skills(), custom_skills()
- remove (existing/nonexisting)
- load_errors property
- __len__, __contains__, __repr__
"""

from __future__ import annotations

import json
import pytest

from apyrobo.skills.library import SkillLibrary
from apyrobo.skills.skill import BUILTIN_SKILLS, Skill


# ---------------------------------------------------------------------------
# Minimal skill JSON format
# ---------------------------------------------------------------------------

PATROL_JSON = {
    "skill_id": "custom_patrol",
    "name": "Patrol",
    "description": "Patrol area",
    "required_capability": "navigate",
    "parameters": {},
}

DELIVER_JSON = {
    "skill_id": "custom_deliver",
    "name": "Deliver",
    "description": "Deliver item",
    "required_capability": "navigate",
    "parameters": {"destination": "room_1"},
}


def write_skill_file(directory, data: dict) -> None:
    path = directory / f"{data['skill_id']}.json"
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# SkillLibrary init
# ---------------------------------------------------------------------------

class TestSkillLibraryInit:
    def test_init_without_dir(self):
        lib = SkillLibrary()
        assert lib._skills_dir is None
        assert lib._custom_skills == {}

    def test_init_with_nonexistent_dir(self, tmp_path):
        nonexistent = tmp_path / "no_such_dir"
        lib = SkillLibrary(skills_dir=nonexistent)
        # Dir doesn't exist -> no auto-load
        assert len(lib._custom_skills) == 0

    def test_init_with_existing_dir_auto_loads(self, tmp_path):
        write_skill_file(tmp_path, PATROL_JSON)
        lib = SkillLibrary(skills_dir=tmp_path)
        assert "custom_patrol" in lib._custom_skills

    def test_init_with_existing_dir_loads_multiple(self, tmp_path):
        write_skill_file(tmp_path, PATROL_JSON)
        write_skill_file(tmp_path, DELIVER_JSON)
        lib = SkillLibrary(skills_dir=tmp_path)
        assert "custom_patrol" in lib._custom_skills
        assert "custom_deliver" in lib._custom_skills

    def test_init_with_registry(self, tmp_path):
        from unittest.mock import MagicMock
        registry = MagicMock()
        registry.get_skill.return_value = (None, None)
        registry.all_skills.return_value = {}
        lib = SkillLibrary(registry=registry)
        assert lib._registry is registry


# ---------------------------------------------------------------------------
# load_all
# ---------------------------------------------------------------------------

class TestLoadAll:
    def test_load_all_no_dir_returns_zero(self):
        lib = SkillLibrary()
        result = lib.load_all()
        assert result == 0

    def test_load_all_nonexistent_dir_returns_zero(self, tmp_path):
        lib = SkillLibrary(skills_dir=tmp_path / "missing")
        result = lib.load_all()
        assert result == 0

    def test_load_all_from_dir_with_json_files(self, tmp_path):
        write_skill_file(tmp_path, PATROL_JSON)
        write_skill_file(tmp_path, DELIVER_JSON)
        lib = SkillLibrary(skills_dir=tmp_path)
        # Auto-loaded at init, reset and reload
        lib._custom_skills = {}
        result = lib.load_all()
        assert result == 2

    def test_load_all_ignores_non_json_files(self, tmp_path):
        write_skill_file(tmp_path, PATROL_JSON)
        (tmp_path / "readme.txt").write_text("not a skill")
        lib = SkillLibrary()
        lib._skills_dir = tmp_path
        result = lib.load_all()
        assert result == 1

    def test_load_all_records_errors_for_invalid_json(self, tmp_path):
        (tmp_path / "bad_skill.json").write_text("{invalid json}")
        lib = SkillLibrary()
        lib._skills_dir = tmp_path
        result = lib.load_all()
        assert result == 0
        assert len(lib.load_errors) == 1

    def test_load_all_records_errors_for_missing_fields(self, tmp_path):
        (tmp_path / "incomplete.json").write_text('{"name": "only_name"}')
        lib = SkillLibrary()
        lib._skills_dir = tmp_path
        lib.load_all()
        assert len(lib.load_errors) == 1

    def test_load_all_resets_errors_on_each_call(self, tmp_path):
        (tmp_path / "bad.json").write_text("{bad}")
        lib = SkillLibrary()
        lib._skills_dir = tmp_path
        lib.load_all()
        assert len(lib.load_errors) == 1
        # Now fix the file
        (tmp_path / "bad.json").write_text(json.dumps(PATROL_JSON))
        lib.load_all()
        assert len(lib.load_errors) == 0


# ---------------------------------------------------------------------------
# load_file
# ---------------------------------------------------------------------------

class TestLoadFile:
    def test_load_file_success(self, tmp_path):
        write_skill_file(tmp_path, PATROL_JSON)
        lib = SkillLibrary()
        skill = lib.load_file(tmp_path / "custom_patrol.json")
        assert skill.skill_id == "custom_patrol"
        assert "custom_patrol" in lib._custom_skills

    def test_load_file_returns_skill(self, tmp_path):
        write_skill_file(tmp_path, DELIVER_JSON)
        lib = SkillLibrary()
        skill = lib.load_file(tmp_path / "custom_deliver.json")
        assert isinstance(skill, Skill)
        assert skill.name == "Deliver"

    def test_load_file_with_string_path(self, tmp_path):
        write_skill_file(tmp_path, PATROL_JSON)
        lib = SkillLibrary()
        skill = lib.load_file(str(tmp_path / "custom_patrol.json"))
        assert skill.skill_id == "custom_patrol"


# ---------------------------------------------------------------------------
# load_json
# ---------------------------------------------------------------------------

class TestLoadJson:
    def test_load_json_success(self):
        lib = SkillLibrary()
        skill = lib.load_json(json.dumps(PATROL_JSON))
        assert skill.skill_id == "custom_patrol"
        assert "custom_patrol" in lib._custom_skills

    def test_load_json_adds_to_custom_skills(self):
        lib = SkillLibrary()
        lib.load_json(json.dumps(PATROL_JSON))
        lib.load_json(json.dumps(DELIVER_JSON))
        assert len(lib._custom_skills) == 2


# ---------------------------------------------------------------------------
# save_skill
# ---------------------------------------------------------------------------

class TestSaveSkill:
    def test_save_skill_to_explicit_path(self, tmp_path):
        lib = SkillLibrary()
        skill = Skill.from_dict(PATROL_JSON)
        save_path = tmp_path / "saved_patrol.json"
        result = lib.save_skill(skill, path=save_path)
        assert result == save_path
        assert save_path.exists()

    def test_save_skill_creates_parent_dirs(self, tmp_path):
        lib = SkillLibrary()
        skill = Skill.from_dict(PATROL_JSON)
        nested_path = tmp_path / "nested" / "dir" / "patrol.json"
        lib.save_skill(skill, path=nested_path)
        assert nested_path.exists()

    def test_save_skill_to_default_dir(self, tmp_path):
        lib = SkillLibrary(skills_dir=tmp_path)
        # Remove auto-loaded files to start clean
        skill = Skill.from_dict(PATROL_JSON)
        result = lib.save_skill(skill)
        expected = tmp_path / "custom_patrol.json"
        assert result == expected
        assert expected.exists()

    def test_save_skill_raises_without_dir_and_no_path(self):
        lib = SkillLibrary()
        skill = Skill.from_dict(PATROL_JSON)
        with pytest.raises(ValueError, match="No skills directory"):
            lib.save_skill(skill)

    def test_save_skill_file_contains_valid_json(self, tmp_path):
        lib = SkillLibrary()
        skill = Skill.from_dict(PATROL_JSON)
        save_path = tmp_path / "patrol.json"
        lib.save_skill(skill, path=save_path)
        data = json.loads(save_path.read_text())
        assert data["skill_id"] == "custom_patrol"

    def test_save_skill_accepts_string_path(self, tmp_path):
        lib = SkillLibrary()
        skill = Skill.from_dict(PATROL_JSON)
        str_path = str(tmp_path / "patrol.json")
        lib.save_skill(skill, path=str_path)
        assert (tmp_path / "patrol.json").exists()


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_custom_skill(self):
        lib = SkillLibrary()
        lib.load_json(json.dumps(PATROL_JSON))
        skill = lib.get("custom_patrol")
        assert skill is not None
        assert skill.skill_id == "custom_patrol"

    def test_get_builtin_skill(self):
        lib = SkillLibrary()
        skill = lib.get("navigate_to")
        assert skill is not None
        assert skill.skill_id == "navigate_to"

    def test_get_nonexistent_returns_none(self):
        lib = SkillLibrary()
        skill = lib.get("does_not_exist")
        assert skill is None

    def test_get_custom_overrides_builtin(self):
        # If a custom skill has the same ID as a builtin, custom wins
        lib = SkillLibrary()
        custom_nav = {
            "skill_id": "navigate_to",
            "name": "Custom Navigate",
            "description": "Overridden",
            "required_capability": "navigate",
            "parameters": {},
        }
        lib.load_json(json.dumps(custom_nav))
        skill = lib.get("navigate_to")
        assert skill.name == "Custom Navigate"

    def test_get_from_registry(self):
        from unittest.mock import MagicMock
        registry_skill = Skill.from_dict(PATROL_JSON)
        registry = MagicMock()
        registry.get_skill.return_value = (registry_skill, "some_package")
        lib = SkillLibrary(registry=registry)
        skill = lib.get("custom_patrol")
        assert skill is not None
        assert skill.skill_id == "custom_patrol"

    def test_get_registry_returns_none_falls_to_builtin(self):
        from unittest.mock import MagicMock
        registry = MagicMock()
        registry.get_skill.return_value = (None, None)
        lib = SkillLibrary(registry=registry)
        skill = lib.get("navigate_to")
        assert skill is not None


# ---------------------------------------------------------------------------
# all_skills / custom_skills
# ---------------------------------------------------------------------------

class TestAllSkills:
    def test_all_skills_includes_builtins(self):
        lib = SkillLibrary()
        all_s = lib.all_skills()
        for builtin_id in BUILTIN_SKILLS:
            assert builtin_id in all_s

    def test_all_skills_includes_custom(self):
        lib = SkillLibrary()
        lib.load_json(json.dumps(PATROL_JSON))
        all_s = lib.all_skills()
        assert "custom_patrol" in all_s

    def test_all_skills_with_registry(self):
        from unittest.mock import MagicMock
        registry_skill = Skill.from_dict(DELIVER_JSON)
        registry = MagicMock()
        registry.all_skills.return_value = {"custom_deliver": registry_skill}
        lib = SkillLibrary(registry=registry)
        all_s = lib.all_skills()
        assert "custom_deliver" in all_s

    def test_custom_skills_only_custom(self):
        lib = SkillLibrary()
        lib.load_json(json.dumps(PATROL_JSON))
        custom = lib.custom_skills()
        assert "custom_patrol" in custom
        # Should NOT include builtins
        for builtin_id in BUILTIN_SKILLS:
            assert builtin_id not in custom

    def test_custom_skills_returns_copy(self):
        lib = SkillLibrary()
        lib.load_json(json.dumps(PATROL_JSON))
        c1 = lib.custom_skills()
        c2 = lib.custom_skills()
        assert c1 is not c2


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

class TestRemove:
    def test_remove_existing_custom_skill(self):
        lib = SkillLibrary()
        lib.load_json(json.dumps(PATROL_JSON))
        result = lib.remove("custom_patrol")
        assert result is True
        assert "custom_patrol" not in lib._custom_skills

    def test_remove_nonexistent_returns_false(self):
        lib = SkillLibrary()
        result = lib.remove("does_not_exist")
        assert result is False

    def test_remove_builtin_returns_false(self):
        # Cannot remove built-in skills
        lib = SkillLibrary()
        result = lib.remove("navigate_to")
        assert result is False


# ---------------------------------------------------------------------------
# load_errors property
# ---------------------------------------------------------------------------

class TestLoadErrors:
    def test_load_errors_initially_empty(self):
        lib = SkillLibrary()
        assert lib.load_errors == []

    def test_load_errors_after_failed_load(self, tmp_path):
        (tmp_path / "broken.json").write_text("not json at all!")
        lib = SkillLibrary()
        lib._skills_dir = tmp_path
        lib.load_all()
        errors = lib.load_errors
        assert len(errors) == 1
        assert "file" in errors[0]
        assert "error" in errors[0]

    def test_load_errors_returns_copy(self, tmp_path):
        lib = SkillLibrary()
        e1 = lib.load_errors
        e2 = lib.load_errors
        assert e1 is not e2


# ---------------------------------------------------------------------------
# __len__ / __contains__ / __repr__
# ---------------------------------------------------------------------------

class TestDunder:
    def test_len_includes_builtins(self):
        lib = SkillLibrary()
        assert len(lib) >= len(BUILTIN_SKILLS)

    def test_len_increases_with_custom(self):
        lib = SkillLibrary()
        before = len(lib)
        lib.load_json(json.dumps(PATROL_JSON))
        assert len(lib) == before + 1

    def test_contains_builtin(self):
        lib = SkillLibrary()
        assert "navigate_to" in lib

    def test_contains_custom(self):
        lib = SkillLibrary()
        lib.load_json(json.dumps(PATROL_JSON))
        assert "custom_patrol" in lib

    def test_not_contains_nonexistent(self):
        lib = SkillLibrary()
        assert "nonexistent_skill" not in lib

    def test_repr_contains_counts(self):
        lib = SkillLibrary()
        lib.load_json(json.dumps(PATROL_JSON))
        r = repr(lib)
        assert "SkillLibrary" in r
        assert "custom=" in r
        assert "total=" in r
        assert "builtin=" in r
