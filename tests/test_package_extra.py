"""
Extra coverage tests for apyrobo/skills/package.py.

Targets missing lines:
  - validate_version: valid and invalid semver
  - validate_package_name: valid and invalid names
  - parse_version_tuple: valid and invalid
  - check_version_constraint: all operators (>=, >, <=, <, ==, bare)
  - SkillPackage constructor: invalid name, invalid version
  - SkillPackage.remove_skill: found and not found
  - SkillPackage.get_skill: found and not found
  - SkillPackage.skill_ids property
  - SkillPackage.to_manifest()
  - SkillPackage.to_manifest_json()
  - SkillPackage.from_manifest()
  - SkillPackage.save() to temp dir
  - SkillPackage.load() happy path and missing manifest
  - SkillPackage._load_skill_handlers()
  - SkillPackage.pack() to archive
  - SkillPackage.from_archive()
  - SkillPackage.validate(): no skills, duplicate IDs, valid
  - SkillPackage.__repr__()
  - SkillPackage.__eq__()
  - SkillPackage.init()
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from apyrobo.skills.package import (
    SkillPackage,
    validate_version,
    validate_package_name,
    parse_version_tuple,
    check_version_constraint,
    MANIFEST_FILE,
)
from apyrobo.skills.skill import BUILTIN_SKILLS, Skill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _builtin(skill_id: str) -> Skill:
    """Return a copy of a builtin skill for use in tests."""
    return BUILTIN_SKILLS[skill_id]


def _make_pkg(name: str = "test-pkg", version: str = "1.0.0") -> SkillPackage:
    return SkillPackage(name=name, version=version, description="Test pkg", author="Tester")


# ===========================================================================
# validate_version
# ===========================================================================

class TestValidateVersion:
    @pytest.mark.parametrize("ver", [
        "0.0.1", "1.0.0", "2.3.4", "10.20.30",
        "1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-0.3.7",
    ])
    def test_valid(self, ver: str) -> None:
        assert validate_version(ver) is True

    @pytest.mark.parametrize("ver", [
        "", "1", "1.0", "1.0.0.0", "v1.0.0",
        "1.0.x", "1.0.0-", "a.b.c",
    ])
    def test_invalid(self, ver: str) -> None:
        assert validate_version(ver) is False


# ===========================================================================
# validate_package_name
# ===========================================================================

class TestValidatePackageName:
    @pytest.mark.parametrize("name", [
        "a", "my-pkg", "warehouse-logistics", "nav2",
        "abc", "a1b2c3", "test-pkg-v2",
    ])
    def test_valid(self, name: str) -> None:
        assert validate_package_name(name) is True

    @pytest.mark.parametrize("name", [
        "", "A", "My-Pkg", "123", "-start", "end-",
        "has space", "has.dot", "has_under",
        "a" * 65,  # too long
    ])
    def test_invalid(self, name: str) -> None:
        assert validate_package_name(name) is False


# ===========================================================================
# parse_version_tuple
# ===========================================================================

class TestParseVersionTuple:
    def test_simple(self) -> None:
        assert parse_version_tuple("1.2.3") == (1, 2, 3, "")

    def test_with_prerelease(self) -> None:
        t = parse_version_tuple("2.0.0-alpha")
        assert t == (2, 0, 0, "alpha")

    def test_zeros(self) -> None:
        assert parse_version_tuple("0.0.0") == (0, 0, 0, "")

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid semver"):
            parse_version_tuple("not-a-version")

    def test_missing_patch_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_version_tuple("1.0")


# ===========================================================================
# check_version_constraint
# ===========================================================================

class TestCheckVersionConstraint:
    # >= operator
    def test_gte_satisfied(self) -> None:
        assert check_version_constraint("1.5.0", ">=1.0.0") is True

    def test_gte_equal(self) -> None:
        assert check_version_constraint("1.0.0", ">=1.0.0") is True

    def test_gte_not_satisfied(self) -> None:
        assert check_version_constraint("0.9.0", ">=1.0.0") is False

    # > operator
    def test_gt_satisfied(self) -> None:
        assert check_version_constraint("2.0.0", ">1.0.0") is True

    def test_gt_equal_not_satisfied(self) -> None:
        assert check_version_constraint("1.0.0", ">1.0.0") is False

    def test_gt_not_satisfied(self) -> None:
        assert check_version_constraint("0.5.0", ">1.0.0") is False

    # <= operator
    def test_lte_satisfied(self) -> None:
        assert check_version_constraint("0.9.0", "<=1.0.0") is True

    def test_lte_equal(self) -> None:
        assert check_version_constraint("1.0.0", "<=1.0.0") is True

    def test_lte_not_satisfied(self) -> None:
        assert check_version_constraint("1.1.0", "<=1.0.0") is False

    # < operator
    def test_lt_satisfied(self) -> None:
        assert check_version_constraint("0.9.9", "<1.0.0") is True

    def test_lt_equal_not_satisfied(self) -> None:
        assert check_version_constraint("1.0.0", "<1.0.0") is False

    def test_lt_not_satisfied(self) -> None:
        assert check_version_constraint("2.0.0", "<1.0.0") is False

    # == operator
    def test_eq_satisfied(self) -> None:
        assert check_version_constraint("1.2.3", "==1.2.3") is True

    def test_eq_not_satisfied(self) -> None:
        assert check_version_constraint("1.2.4", "==1.2.3") is False

    # bare version (exact match)
    def test_bare_satisfied(self) -> None:
        assert check_version_constraint("1.0.0", "1.0.0") is True

    def test_bare_not_satisfied(self) -> None:
        assert check_version_constraint("1.0.1", "1.0.0") is False

    # compound constraint
    def test_compound_both_satisfied(self) -> None:
        assert check_version_constraint("1.5.0", ">=1.0.0,<2.0.0") is True

    def test_compound_upper_bound_violated(self) -> None:
        assert check_version_constraint("2.0.0", ">=1.0.0,<2.0.0") is False

    def test_compound_lower_bound_violated(self) -> None:
        assert check_version_constraint("0.9.0", ">=1.0.0,<2.0.0") is False


# ===========================================================================
# SkillPackage constructor validation
# ===========================================================================

class TestSkillPackageConstructor:
    def test_valid_construction(self) -> None:
        pkg = _make_pkg()
        assert pkg.name == "test-pkg"
        assert pkg.version == "1.0.0"

    def test_invalid_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid package name"):
            SkillPackage(name="Invalid_Name", version="1.0.0")

    def test_invalid_version_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid version"):
            SkillPackage(name="valid-name", version="not-a-version")

    def test_defaults(self) -> None:
        pkg = SkillPackage(name="mypkg", version="0.1.0")
        assert pkg.skills == []
        assert pkg.dependencies == {}
        assert pkg.tags == []
        assert pkg.required_capabilities == []


# ===========================================================================
# Skill management
# ===========================================================================

class TestSkillManagement:
    def test_add_and_get_skill(self) -> None:
        pkg = _make_pkg()
        skill = _builtin("navigate_to")
        pkg.add_skill(skill)
        found = pkg.get_skill("navigate_to")
        assert found is not None
        assert found.skill_id == "navigate_to"

    def test_get_skill_not_found(self) -> None:
        pkg = _make_pkg()
        assert pkg.get_skill("nonexistent") is None

    def test_remove_skill_found(self) -> None:
        pkg = _make_pkg()
        pkg.add_skill(_builtin("navigate_to"))
        result = pkg.remove_skill("navigate_to")
        assert result is True
        assert pkg.get_skill("navigate_to") is None

    def test_remove_skill_not_found(self) -> None:
        pkg = _make_pkg()
        result = pkg.remove_skill("ghost_skill")
        assert result is False

    def test_add_skill_replaces_existing(self) -> None:
        pkg = _make_pkg()
        pkg.add_skill(_builtin("navigate_to"))
        pkg.add_skill(_builtin("navigate_to"))
        assert len(pkg.skills) == 1

    def test_skill_ids_property(self) -> None:
        pkg = _make_pkg()
        pkg.add_skill(_builtin("navigate_to"))
        pkg.add_skill(_builtin("rotate"))
        ids = pkg.skill_ids
        assert "navigate_to" in ids
        assert "rotate" in ids
        assert len(ids) == 2

    def test_skill_ids_empty(self) -> None:
        pkg = _make_pkg()
        assert pkg.skill_ids == []


# ===========================================================================
# Serialisation
# ===========================================================================

class TestManifestSerialisation:
    def test_to_manifest(self) -> None:
        pkg = SkillPackage(
            name="test-pkg",
            version="1.2.3",
            description="My package",
            author="Me",
            required_capabilities=["navigate"],
            tags=["nav", "indoor"],
            dependencies={"base-nav": ">=1.0.0"},
        )
        pkg.add_skill(_builtin("navigate_to"))
        m = pkg.to_manifest()
        assert m["name"] == "test-pkg"
        assert m["version"] == "1.2.3"
        assert m["description"] == "My package"
        assert m["author"] == "Me"
        assert "navigate_to" in m["skills"]
        assert m["tags"] == ["nav", "indoor"]
        assert m["dependencies"] == {"base-nav": ">=1.0.0"}
        assert m["required_capabilities"] == ["navigate"]

    def test_to_manifest_json(self) -> None:
        pkg = _make_pkg()
        json_str = pkg.to_manifest_json()
        data = json.loads(json_str)
        assert data["name"] == "test-pkg"
        assert data["version"] == "1.0.0"

    def test_from_manifest(self) -> None:
        manifest = {
            "name": "nav-skills",
            "version": "2.0.0",
            "description": "Navigation skills",
            "author": "Acme",
            "license": "MIT",
            "homepage": "https://example.com",
            "required_capabilities": ["navigate"],
            "min_apyrobo_version": "0.1.0",
            "skills": [],
            "dependencies": {},
            "tags": ["nav"],
        }
        pkg = SkillPackage.from_manifest(manifest)
        assert pkg.name == "nav-skills"
        assert pkg.version == "2.0.0"
        assert pkg.author == "Acme"
        assert pkg.license == "MIT"
        assert pkg.tags == ["nav"]

    def test_from_manifest_with_skills(self) -> None:
        skill = _builtin("navigate_to")
        manifest = {
            "name": "nav-pkg",
            "version": "1.0.0",
            "skills": ["navigate_to"],
        }
        pkg = SkillPackage.from_manifest(manifest, skills=[skill])
        assert len(pkg.skills) == 1
        assert pkg.skills[0].skill_id == "navigate_to"

    def test_from_manifest_minimal_keys(self) -> None:
        # from_manifest should tolerate missing optional keys
        manifest = {"name": "minimal-pkg", "version": "0.1.0"}
        pkg = SkillPackage.from_manifest(manifest)
        assert pkg.name == "minimal-pkg"
        assert pkg.description == ""
        assert pkg.dependencies == {}


# ===========================================================================
# Filesystem: save / load
# ===========================================================================

class TestSaveLoad:
    def test_save_creates_manifest(self, tmp_path: Path) -> None:
        pkg = _make_pkg()
        pkg.add_skill(_builtin("navigate_to"))
        pkg.save(tmp_path)
        assert (tmp_path / MANIFEST_FILE).exists()

    def test_save_creates_skill_files(self, tmp_path: Path) -> None:
        pkg = _make_pkg()
        pkg.add_skill(_builtin("navigate_to"))
        pkg.add_skill(_builtin("rotate"))
        pkg.save(tmp_path)
        assert (tmp_path / "skills" / "navigate_to.json").exists()
        assert (tmp_path / "skills" / "rotate.json").exists()

    def test_save_returns_directory(self, tmp_path: Path) -> None:
        pkg = _make_pkg()
        result = pkg.save(tmp_path)
        assert result == tmp_path

    def test_load_happy_path(self, tmp_path: Path) -> None:
        pkg = _make_pkg()
        pkg.add_skill(_builtin("navigate_to"))
        pkg.save(tmp_path)
        loaded = SkillPackage.load(tmp_path)
        assert loaded.name == "test-pkg"
        assert loaded.version == "1.0.0"
        assert loaded.get_skill("navigate_to") is not None

    def test_load_missing_manifest_raises(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match=MANIFEST_FILE):
            SkillPackage.load(empty_dir)

    def test_load_skill_file_missing_logs_warning(self, tmp_path: Path) -> None:
        pkg = _make_pkg()
        pkg.add_skill(_builtin("navigate_to"))
        pkg.save(tmp_path)
        # Remove the skill file
        (tmp_path / "skills" / "navigate_to.json").unlink()
        # Should not raise — just warn and skip
        loaded = SkillPackage.load(tmp_path)
        assert loaded.get_skill("navigate_to") is None

    def test_save_no_skills(self, tmp_path: Path) -> None:
        pkg = _make_pkg()
        pkg.save(tmp_path)
        assert (tmp_path / MANIFEST_FILE).exists()
        # skills dir should be empty
        skills_dir = tmp_path / "skills"
        assert skills_dir.exists()
        assert list(skills_dir.iterdir()) == []


# ===========================================================================
# Handler loading
# ===========================================================================

class TestLoadSkillHandlers:
    def test_no_handlers_no_error(self) -> None:
        pkg = _make_pkg()
        pkg.add_skill(_builtin("navigate_to"))  # no handler_module set
        pkg._load_skill_handlers()  # should not raise

    def test_handler_module_load_error_is_swallowed(self) -> None:
        pkg = _make_pkg()
        skill = Skill(
            skill_id="custom_skill",
            name="Custom",
            handler_module="nonexistent.module.path",
        )
        pkg.add_skill(skill)
        # Should not raise, only warns
        pkg._load_skill_handlers()


# ===========================================================================
# Archive: pack / from_archive
# ===========================================================================

class TestArchive:
    def test_pack_creates_file(self, tmp_path: Path) -> None:
        src = tmp_path / "mypkg"
        pkg = _make_pkg()
        pkg.add_skill(_builtin("navigate_to"))
        archive = pkg.pack(src)
        assert archive.exists()
        assert archive.suffix == ".skillpkg"

    def test_pack_with_explicit_output(self, tmp_path: Path) -> None:
        src = tmp_path / "mypkg"
        out = tmp_path / "output.skillpkg"
        pkg = _make_pkg()
        pkg.pack(src, output_path=out)
        assert out.exists()

    def test_from_archive_roundtrip(self, tmp_path: Path) -> None:
        src = tmp_path / "mypkg"
        pkg = _make_pkg()
        pkg.add_skill(_builtin("navigate_to"))
        archive = pkg.pack(src)
        loaded = SkillPackage.from_archive(archive, extract_to=tmp_path / "extracted")
        assert loaded.name == "test-pkg"
        assert loaded.version == "1.0.0"
        assert loaded.get_skill("navigate_to") is not None

    def test_from_archive_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Archive not found"):
            SkillPackage.from_archive(tmp_path / "ghost.skillpkg")

    def test_from_archive_default_extract_to(self, tmp_path: Path) -> None:
        src = tmp_path / "mypkg"
        pkg = _make_pkg()
        pkg.add_skill(_builtin("rotate"))
        archive = pkg.pack(src)
        # No extract_to — should extract alongside archive
        loaded = SkillPackage.from_archive(archive)
        assert loaded.name == "test-pkg"


# ===========================================================================
# Validation
# ===========================================================================

class TestValidate:
    def test_no_skills_error(self) -> None:
        pkg = _make_pkg()
        errors = pkg.validate()
        assert any("skill" in e.lower() for e in errors)

    def test_duplicate_skill_ids(self) -> None:
        pkg = _make_pkg()
        skill1 = _builtin("navigate_to")
        # Manually inject duplicate
        pkg.skills = [skill1, skill1]
        errors = pkg.validate()
        assert any("duplicate" in e.lower() for e in errors)

    def test_valid_package(self) -> None:
        pkg = _make_pkg()
        pkg.add_skill(_builtin("navigate_to"))
        errors = pkg.validate()
        assert errors == []

    def test_multiple_skills_valid(self) -> None:
        pkg = _make_pkg()
        pkg.add_skill(_builtin("navigate_to"))
        pkg.add_skill(_builtin("rotate"))
        pkg.add_skill(_builtin("stop"))
        errors = pkg.validate()
        assert errors == []


# ===========================================================================
# Dunder methods
# ===========================================================================

class TestDunderMethods:
    def test_repr(self) -> None:
        pkg = _make_pkg()
        pkg.add_skill(_builtin("navigate_to"))
        r = repr(pkg)
        assert "test-pkg" in r
        assert "1.0.0" in r
        assert "skills=1" in r

    def test_repr_with_deps(self) -> None:
        pkg = SkillPackage(
            name="test-pkg",
            version="1.0.0",
            dependencies={"base-nav": ">=1.0.0"},
        )
        r = repr(pkg)
        assert "deps=1" in r

    def test_eq_same(self) -> None:
        pkg1 = _make_pkg("test-pkg", "1.0.0")
        pkg2 = _make_pkg("test-pkg", "1.0.0")
        assert pkg1 == pkg2

    def test_eq_different_version(self) -> None:
        pkg1 = _make_pkg("test-pkg", "1.0.0")
        pkg2 = _make_pkg("test-pkg", "2.0.0")
        assert pkg1 != pkg2

    def test_eq_different_name(self) -> None:
        pkg1 = _make_pkg("pkg-a", "1.0.0")
        pkg2 = _make_pkg("pkg-b", "1.0.0")
        assert pkg1 != pkg2

    def test_eq_non_package(self) -> None:
        pkg = _make_pkg()
        result = pkg.__eq__("not a package")
        assert result is NotImplemented


# ===========================================================================
# init class method
# ===========================================================================

class TestInit:
    def test_init_without_directory(self) -> None:
        pkg = SkillPackage.init("nav-pkg", version="0.2.0", description="Nav", author="Me")
        assert pkg.name == "nav-pkg"
        assert pkg.version == "0.2.0"
        assert pkg.skills == []

    def test_init_with_directory(self, tmp_path: Path) -> None:
        pkg = SkillPackage.init("nav-pkg", directory=tmp_path)
        assert (tmp_path / MANIFEST_FILE).exists()
        assert pkg.name == "nav-pkg"
