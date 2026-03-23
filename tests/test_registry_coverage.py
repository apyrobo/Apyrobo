"""
Comprehensive tests for apyrobo/skills/registry.py.

Covers: SkillRegistry install (dir, archive, conflict, force, validation errors,
unmet deps), install_from_dir, remove, get, get_info, is_installed, list_packages,
search, all_skills, get_skill, check_dependencies, _get_dependents, clear,
package_count, registry_dir, __repr__.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apyrobo.skills.registry import SkillRegistry, PackageConflict
from apyrobo.skills.package import SkillPackage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_valid_package(tmp_path: Path, name: str = "test-pkg",
                         version: str = "1.0.0",
                         description: str = "Test package",
                         author: str = "testuser") -> tuple[SkillPackage, Path]:
    """Create a valid SkillPackage with at least one skill and return (pkg, pkg_dir)."""
    from apyrobo.skills.skill import BUILTIN_SKILLS
    pkg_dir = tmp_path / name
    pkg = SkillPackage.init(
        name=name,
        version=version,
        description=description,
        author=author,
        directory=str(pkg_dir),
    )
    # Add a builtin skill so the package passes validation
    skill = list(BUILTIN_SKILLS.values())[0]
    pkg.add_skill(skill)
    pkg.save(str(pkg_dir))
    return pkg, pkg_dir


def pack_package(pkg: SkillPackage, pkg_dir: Path, output_dir: Path) -> Path:
    """Pack a package and return the archive path."""
    archive_path = pkg.pack(str(pkg_dir), str(output_dir / f"{pkg.name}-{pkg.version}.skillpkg"))
    return Path(archive_path)


# ---------------------------------------------------------------------------
# SkillRegistry basic setup
# ---------------------------------------------------------------------------

class TestSkillRegistrySetup:
    def test_registry_uses_custom_dir(self, tmp_path):
        registry = SkillRegistry(tmp_path / "myregistry")
        assert registry.registry_dir == tmp_path / "myregistry"
        assert (tmp_path / "myregistry").exists()

    def test_registry_dir_property(self, tmp_path):
        registry = SkillRegistry(tmp_path)
        assert registry.registry_dir == tmp_path

    def test_package_count_empty(self, tmp_path):
        registry = SkillRegistry(tmp_path)
        assert registry.package_count == 0

    def test_repr(self, tmp_path):
        registry = SkillRegistry(tmp_path)
        r = repr(registry)
        assert "SkillRegistry" in r
        assert "packages=0" in r


# ---------------------------------------------------------------------------
# Install from directory
# ---------------------------------------------------------------------------

class TestInstallFromDir:
    def test_install_from_dir_success(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "my-pkg")
        result = registry.install(pkg_dir)
        assert result.name == "my-pkg"
        assert registry.is_installed("my-pkg")

    def test_install_from_dir_alias(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "my-pkg2")
        result = registry.install_from_dir(pkg_dir)
        assert result.name == "my-pkg2"

    def test_install_conflict_raises(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "my-pkg")
        registry.install(pkg_dir)
        # Second install without force should raise
        with pytest.raises(PackageConflict, match="already installed"):
            registry.install(pkg_dir)

    def test_install_conflict_with_force(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "my-pkg")
        registry.install(pkg_dir)
        # Second install with force should succeed
        result = registry.install(pkg_dir, force=True)
        assert result.name == "my-pkg"

    def test_install_nonexistent_raises(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        with pytest.raises(FileNotFoundError):
            registry.install(tmp_path / "nonexistent-path")

    def test_install_increments_package_count(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        assert registry.package_count == 0
        pkg, pkg_dir = create_valid_package(tmp_path, "my-pkg")
        registry.install(pkg_dir)
        assert registry.package_count == 1


# ---------------------------------------------------------------------------
# Install from archive (.skillpkg)
# ---------------------------------------------------------------------------

class TestInstallFromArchive:
    def test_install_from_archive(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "archive-pkg")
        archive_path = pack_package(pkg, pkg_dir, tmp_path)
        result = registry.install(archive_path)
        assert result.name == "archive-pkg"
        assert registry.is_installed("archive-pkg")

    def test_install_archive_conflict_with_force(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "archive-pkg2")
        archive_path = pack_package(pkg, pkg_dir, tmp_path)
        registry.install(archive_path)
        # Second install with force
        result = registry.install(archive_path, force=True)
        assert result.name == "archive-pkg2"


# ---------------------------------------------------------------------------
# _install_package with validation errors
# ---------------------------------------------------------------------------

class TestInstallPackageValidation:
    def test_install_invalid_package_raises(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        # Create a package with no skills — validate() will return an error
        bad_pkg = SkillPackage(name="bad-pkg", version="1.0.0")
        # No skills added → validate() returns "Package must contain at least one skill"
        with pytest.raises(ValueError, match="Invalid package"):
            registry._install_package(bad_pkg)

    def test_install_with_unmet_dependencies_logs_warning(self, tmp_path, caplog):
        import logging
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "dep-pkg")
        # Add a dependency to the manifest that isn't installed (keep skills entry intact)
        manifest = pkg_dir / "skill-package.json"
        data = json.loads(manifest.read_text())
        data["dependencies"] = {"missing-dep": ">=1.0.0"}
        manifest.write_text(json.dumps(data))
        dep_pkg = SkillPackage.load(pkg_dir)
        assert len(dep_pkg.skills) > 0, "Package must have skills for this test"
        with caplog.at_level(logging.WARNING):
            result = registry._install_package(dep_pkg)
        assert result.name == "dep-pkg"
        assert registry.is_installed("dep-pkg")


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------

class TestRemove:
    def test_remove_installed(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "remove-pkg")
        registry.install(pkg_dir)
        result = registry.remove("remove-pkg")
        assert result is True
        assert not registry.is_installed("remove-pkg")

    def test_remove_not_found(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        result = registry.remove("nonexistent-pkg")
        assert result is False

    def test_remove_decrements_package_count(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "rm-pkg2")
        registry.install(pkg_dir)
        assert registry.package_count == 1
        registry.remove("rm-pkg2")
        assert registry.package_count == 0


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_installed(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "get-pkg")
        registry.install(pkg_dir)
        result = registry.get("get-pkg")
        assert result is not None
        assert result.name == "get-pkg"

    def test_get_not_installed_returns_none(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        result = registry.get("nonexistent")
        assert result is None

    def test_get_missing_dir_returns_none(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "dir-pkg")
        registry.install(pkg_dir)
        # Remove the package directory while keeping the index entry
        import shutil
        pkg_path = registry._packages_dir / "dir-pkg"
        if pkg_path.exists():
            shutil.rmtree(pkg_path)
        result = registry.get("dir-pkg")
        assert result is None


# ---------------------------------------------------------------------------
# get_info, is_installed, list_packages
# ---------------------------------------------------------------------------

class TestQueryMethods:
    def test_get_info(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "info-pkg",
                                            description="My description")
        registry.install(pkg_dir)
        info = registry.get_info("info-pkg")
        assert info is not None
        assert info["description"] == "My description"

    def test_get_info_missing(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        assert registry.get_info("nonexistent") is None

    def test_is_installed_true(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "is-pkg")
        registry.install(pkg_dir)
        assert registry.is_installed("is-pkg") is True

    def test_is_installed_false(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        assert registry.is_installed("nonexistent") is False

    def test_list_packages_empty(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        assert registry.list_packages() == []

    def test_list_packages_with_packages(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg1, dir1 = create_valid_package(tmp_path / "p1", "pkg-aaa")
        pkg2, dir2 = create_valid_package(tmp_path / "p2", "pkg-bbb")
        registry.install(dir1)
        registry.install(dir2)
        packages = registry.list_packages()
        assert len(packages) == 2
        names = [p["name"] for p in packages]
        assert "pkg-aaa" in names
        assert "pkg-bbb" in names


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_by_name(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "navigation-pkg",
                                            description="Robot navigation")
        registry.install(pkg_dir)
        results = registry.search("navigation")
        assert len(results) > 0
        assert results[0]["name"] == "navigation-pkg"

    def test_search_by_description(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "basic-pkg",
                                            description="warehouse logistics skills")
        registry.install(pkg_dir)
        results = registry.search("warehouse")
        assert len(results) > 0

    def test_search_by_tags(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "tagged-pkg")
        # Add tags to manifest (skills entry preserved since create_valid_package saved them)
        manifest = pkg_dir / "skill-package.json"
        data = json.loads(manifest.read_text())
        data["tags"] = ["robotics", "navigation"]
        manifest.write_text(json.dumps(data))
        result = registry.install(pkg_dir)
        assert result is not None
        results = registry.search("robotics")
        assert len(results) > 0

    def test_search_by_skill_id(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "skill-pkg")
        registry.install(pkg_dir)
        # Search for skill IDs in index
        results = registry.search("skill-pkg")
        assert len(results) > 0

    def test_search_no_results(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        results = registry.search("zzznomatchzzz")
        assert results == []


# ---------------------------------------------------------------------------
# all_skills, get_skill
# ---------------------------------------------------------------------------

class TestSkillLookup:
    def test_all_skills_empty(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        skills = registry.all_skills()
        assert isinstance(skills, dict)

    def test_all_skills_after_install(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "skill-pkg")
        registry.install(pkg_dir)
        skills = registry.all_skills()
        assert isinstance(skills, dict)

    def test_get_skill_not_found(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        skill, pkg_name = registry.get_skill("nonexistent_skill")
        assert skill is None
        assert pkg_name is None


# ---------------------------------------------------------------------------
# check_dependencies
# ---------------------------------------------------------------------------

class TestCheckDependencies:
    def test_no_dependencies(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "no-dep-pkg")
        unmet = registry.check_dependencies(pkg)
        assert unmet == []

    def test_unmet_dependency_not_installed(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "dep-pkg")
        # Inject a dependency
        manifest = pkg_dir / "skill-package.json"
        data = json.loads(manifest.read_text())
        data["dependencies"] = {"missing-dep": ">=1.0.0"}
        manifest.write_text(json.dumps(data))
        dep_pkg = SkillPackage.load(pkg_dir)
        unmet = registry.check_dependencies(dep_pkg)
        assert len(unmet) == 1
        assert "missing-dep" in unmet[0]

    def test_dependency_version_mismatch(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        # Install a dep with version 1.0.0
        dep_pkg, dep_dir = create_valid_package(tmp_path / "dep", "dep-base",
                                                version="1.0.0")
        registry.install(dep_dir)

        # Create a package directly with dependencies (no need to load from disk)
        from apyrobo.skills.skill import BUILTIN_SKILLS
        skill = list(BUILTIN_SKILLS.values())[0]
        main_pkg = SkillPackage(name="main-pkg", version="1.0.0",
                                dependencies={"dep-base": ">=2.0.0"},
                                skills=[skill])

        unmet = registry.check_dependencies(main_pkg)
        assert len(unmet) == 1
        assert "dep-base" in unmet[0]

    def test_dependency_met(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        # Install a dep with version 2.0.0
        dep_pkg, dep_dir = create_valid_package(tmp_path / "dep", "dep-base",
                                                version="2.0.0")
        registry.install(dep_dir)

        # Create a package directly with dependencies
        from apyrobo.skills.skill import BUILTIN_SKILLS
        skill = list(BUILTIN_SKILLS.values())[0]
        main_pkg = SkillPackage(name="main-pkg", version="1.0.0",
                                dependencies={"dep-base": ">=1.0.0"},
                                skills=[skill])

        unmet = registry.check_dependencies(main_pkg)
        assert unmet == []


# ---------------------------------------------------------------------------
# _get_dependents
# ---------------------------------------------------------------------------

class TestGetDependents:
    def test_get_dependents_none(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg, pkg_dir = create_valid_package(tmp_path, "solo-pkg")
        registry.install(pkg_dir)
        deps = registry._get_dependents("solo-pkg")
        assert deps == []

    def test_get_dependents_found(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        # Install base package
        base_pkg, base_dir = create_valid_package(tmp_path / "base", "base-pkg",
                                                  version="1.0.0")
        registry.install(base_dir)

        # Install a package that depends on base-pkg (with skills so it validates)
        from apyrobo.skills.skill import BUILTIN_SKILLS
        skill = list(BUILTIN_SKILLS.values())[0]
        dep_pkg = SkillPackage(name="dep-pkg", version="1.0.0",
                               dependencies={"base-pkg": ">=1.0.0"},
                               skills=[skill])
        dep_dir = tmp_path / "dep" / "dep-pkg"
        dep_pkg.save(str(dep_dir))
        registry.install(dep_dir)

        dependents = registry._get_dependents("base-pkg")
        assert "dep-pkg" in dependents


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_removes_all(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg1, dir1 = create_valid_package(tmp_path / "p1", "pkg-a")
        pkg2, dir2 = create_valid_package(tmp_path / "p2", "pkg-b")
        registry.install(dir1)
        registry.install(dir2)
        assert registry.package_count == 2
        count = registry.clear()
        assert count == 2
        assert registry.package_count == 0

    def test_clear_empty_registry(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        count = registry.clear()
        assert count == 0

    def test_registry_usable_after_clear(self, tmp_path):
        registry = SkillRegistry(tmp_path / "registry")
        pkg1, dir1 = create_valid_package(tmp_path / "p1", "pkg-a")
        registry.install(dir1)
        registry.clear()
        pkg2, dir2 = create_valid_package(tmp_path / "p2", "pkg-b")
        registry.install(dir2)
        assert registry.is_installed("pkg-b")
        assert not registry.is_installed("pkg-a")


# ---------------------------------------------------------------------------
# Index persistence
# ---------------------------------------------------------------------------

class TestIndexPersistence:
    def test_index_persists_across_instances(self, tmp_path):
        registry_dir = tmp_path / "registry"
        registry1 = SkillRegistry(registry_dir)
        pkg, pkg_dir = create_valid_package(tmp_path, "persist-pkg")
        registry1.install(pkg_dir)

        # Create a new instance pointing to the same directory
        registry2 = SkillRegistry(registry_dir)
        assert registry2.is_installed("persist-pkg")

    def test_remove_updates_index(self, tmp_path):
        registry_dir = tmp_path / "registry"
        registry1 = SkillRegistry(registry_dir)
        pkg, pkg_dir = create_valid_package(tmp_path, "rm-persist-pkg")
        registry1.install(pkg_dir)
        registry1.remove("rm-persist-pkg")

        registry2 = SkillRegistry(registry_dir)
        assert not registry2.is_installed("rm-persist-pkg")
