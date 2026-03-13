"""
Skill Registry — local package index for installed skill packages.

The registry tracks which packages are installed, their versions,
and where their files live on disk. It provides install, remove,
search, and dependency resolution.

Architecture:
    ~/.apyrobo/registry/
        index.json              ← registry index (metadata for all installed packages)
        packages/
            warehouse-logistics/
                skill-package.json
                skills/
                    patrol_route.json
                    pick_and_place.json
            navigation-base/
                ...

Usage:
    registry = SkillRegistry()                     # uses ~/.apyrobo/registry
    registry = SkillRegistry("/custom/path")       # custom location

    registry.install("/path/to/pkg-1.0.0.skillpkg")
    registry.install_from_dir("/path/to/my-skills")

    pkg = registry.get("warehouse-logistics")
    all_pkgs = registry.list_packages()
    results = registry.search("warehouse")
    registry.remove("warehouse-logistics")

    # Get all skills from all installed packages
    skills = registry.all_skills()
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from apyrobo.skills.package import (
    SkillPackage,
    ARCHIVE_EXT,
    check_version_constraint,
    parse_version_tuple,
)
from apyrobo.skills.skill import Skill

logger = logging.getLogger(__name__)

# Default registry location
DEFAULT_REGISTRY_DIR = Path.home() / ".apyrobo" / "registry"
INDEX_FILE = "index.json"
PACKAGES_DIR = "packages"


class PackageConflict(Exception):
    """Raised when installing a package would conflict with an existing one."""


class DependencyError(Exception):
    """Raised when a package's dependencies cannot be satisfied."""


class SkillRegistry:
    """
    Local filesystem registry for installed skill packages.

    Provides install, remove, search, listing, and dependency checking.
    """

    def __init__(self, registry_dir: str | Path | None = None) -> None:
        self._root = Path(registry_dir) if registry_dir else DEFAULT_REGISTRY_DIR
        self._root.mkdir(parents=True, exist_ok=True)
        self._packages_dir = self._root / PACKAGES_DIR
        self._packages_dir.mkdir(exist_ok=True)
        self._index_path = self._root / INDEX_FILE
        self._index: dict[str, dict[str, Any]] = self._load_index()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    # ------------------------------------------------------------------
    # Install
    # ------------------------------------------------------------------

    def install(self, source: str | Path, force: bool = False) -> SkillPackage:
        """
        Install a skill package from an archive (.skillpkg) or directory.

        Args:
            source: Path to .skillpkg archive or package directory.
            force: If True, overwrite existing package with same name.

        Returns the installed SkillPackage.
        Raises PackageConflict if already installed and not force.
        """
        source = Path(source)

        if source.is_file() and source.suffix == ARCHIVE_EXT:
            # Extract to temp location, then install
            pkg = SkillPackage.from_archive(source, extract_to=self._root / "_staging")
            try:
                return self._install_package(pkg, force=force)
            finally:
                staging = self._root / "_staging"
                if staging.exists():
                    shutil.rmtree(staging)
        elif source.is_dir():
            pkg = SkillPackage.load(source)
            return self._install_package(pkg, force=force)
        else:
            raise FileNotFoundError(f"Not a .skillpkg or directory: {source}")

    def install_from_dir(self, directory: str | Path, force: bool = False) -> SkillPackage:
        """Convenience alias: install from a package directory."""
        return self.install(Path(directory), force=force)

    def _install_package(self, pkg: SkillPackage, force: bool = False) -> SkillPackage:
        """Core installation logic."""
        # Validate
        errors = pkg.validate()
        if errors:
            raise ValueError(f"Invalid package: {'; '.join(errors)}")

        # Check conflicts
        if pkg.name in self._index and not force:
            existing_ver = self._index[pkg.name]["version"]
            raise PackageConflict(
                f"Package {pkg.name!r} already installed (version {existing_ver}). "
                f"Use force=True to overwrite."
            )

        # Check dependencies
        unmet = self.check_dependencies(pkg)
        if unmet:
            logger.warning(
                "Package %s has unmet dependencies: %s (installing anyway)",
                pkg.name, unmet,
            )

        # Copy to registry
        dest = self._packages_dir / pkg.name
        if dest.exists():
            shutil.rmtree(dest)
        pkg.save(dest)

        # Update index
        self._index[pkg.name] = {
            "version": pkg.version,
            "description": pkg.description,
            "author": pkg.author,
            "license": pkg.license,
            "skills": pkg.skill_ids,
            "dependencies": pkg.dependencies,
            "tags": pkg.tags,
            "required_capabilities": pkg.required_capabilities,
        }
        self._save_index()

        logger.info("Installed %s@%s (%d skills)", pkg.name, pkg.version, len(pkg.skills))
        return pkg

    # ------------------------------------------------------------------
    # Remove
    # ------------------------------------------------------------------

    def remove(self, package_name: str) -> bool:
        """
        Remove an installed package.

        Returns True if removed, False if not found.
        """
        if package_name not in self._index:
            return False

        # Check reverse dependencies
        dependents = self._get_dependents(package_name)
        if dependents:
            logger.warning(
                "Removing %s which is depended on by: %s",
                package_name, dependents,
            )

        pkg_dir = self._packages_dir / package_name
        if pkg_dir.exists():
            shutil.rmtree(pkg_dir)

        del self._index[package_name]
        self._save_index()

        logger.info("Removed package: %s", package_name)
        return True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, package_name: str) -> SkillPackage | None:
        """Get a full SkillPackage by name, or None if not installed."""
        if package_name not in self._index:
            return None
        pkg_dir = self._packages_dir / package_name
        if not pkg_dir.exists():
            return None
        return SkillPackage.load(pkg_dir)

    def get_info(self, package_name: str) -> dict[str, Any] | None:
        """Get index metadata for a package (fast — no disk reads)."""
        return self._index.get(package_name)

    def is_installed(self, package_name: str) -> bool:
        return package_name in self._index

    def list_packages(self) -> list[dict[str, Any]]:
        """List all installed packages with metadata."""
        result = []
        for name, info in sorted(self._index.items()):
            entry = {"name": name}
            entry.update(info)
            result.append(entry)
        return result

    def search(self, query: str) -> list[dict[str, Any]]:
        """
        Search installed packages by name, description, tags, or skill IDs.
        """
        query_lower = query.lower()
        results = []
        for name, info in self._index.items():
            score = 0
            if query_lower in name.lower():
                score += 10
            if query_lower in info.get("description", "").lower():
                score += 5
            if any(query_lower in tag.lower() for tag in info.get("tags", [])):
                score += 8
            if any(query_lower in sid.lower() for sid in info.get("skills", [])):
                score += 6
            if score > 0:
                entry = {"name": name, "score": score}
                entry.update(info)
                results.append(entry)
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    def all_skills(self) -> dict[str, Skill]:
        """Get all skills from all installed packages, keyed by skill_id."""
        skills: dict[str, Skill] = {}
        for name in self._index:
            pkg = self.get(name)
            if pkg:
                for skill in pkg.skills:
                    skills[skill.skill_id] = skill
        return skills

    def get_skill(self, skill_id: str) -> tuple[Skill | None, str | None]:
        """
        Find a skill across all installed packages.

        Returns (skill, package_name) or (None, None).
        """
        for name, info in self._index.items():
            if skill_id in info.get("skills", []):
                pkg = self.get(name)
                if pkg:
                    skill = pkg.get_skill(skill_id)
                    if skill:
                        return skill, name
        return None, None

    # ------------------------------------------------------------------
    # Dependencies
    # ------------------------------------------------------------------

    def check_dependencies(self, pkg: SkillPackage) -> list[str]:
        """
        Check whether a package's dependencies are satisfied.

        Returns a list of unmet dependency descriptions.
        """
        unmet: list[str] = []
        for dep_name, constraint in pkg.dependencies.items():
            if dep_name not in self._index:
                unmet.append(f"{dep_name} ({constraint}) — not installed")
            else:
                installed_ver = self._index[dep_name]["version"]
                if not check_version_constraint(installed_ver, constraint):
                    unmet.append(
                        f"{dep_name} ({constraint}) — installed {installed_ver}"
                    )
        return unmet

    def _get_dependents(self, package_name: str) -> list[str]:
        """Find packages that depend on *package_name*."""
        dependents = []
        for name, info in self._index.items():
            if name == package_name:
                continue
            if package_name in info.get("dependencies", {}):
                dependents.append(name)
        return dependents

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def clear(self) -> int:
        """Remove all installed packages. Returns count removed."""
        count = len(self._index)
        if self._packages_dir.exists():
            shutil.rmtree(self._packages_dir)
        self._packages_dir.mkdir(exist_ok=True)
        self._index = {}
        self._save_index()
        return count

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    @property
    def package_count(self) -> int:
        return len(self._index)

    @property
    def registry_dir(self) -> Path:
        return self._root

    def __repr__(self) -> str:
        return (
            f"<SkillRegistry dir={self._root} "
            f"packages={self.package_count}>"
        )
