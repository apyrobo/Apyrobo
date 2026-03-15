"""
Skill Package — defines the distributable unit for APYROBO skills.

A skill package bundles one or more skills with metadata, version info,
dependency declarations, and an optional README into a portable archive
(.skillpkg — a renamed zip file).

Manifest format (skill-package.json):

    {
        "name": "warehouse-logistics",
        "version": "1.0.0",
        "description": "Pick, place, and delivery skills for warehouse robots",
        "author": "ACME Robotics",
        "license": "Apache-2.0",
        "homepage": "https://github.com/acme/warehouse-skills",
        "required_capabilities": ["navigate", "pick", "place"],
        "min_apyrobo_version": "0.1.0",
        "skills": ["patrol_route", "scan_area"],
        "dependencies": {
            "navigation-base": ">=1.0.0"
        },
        "tags": ["warehouse", "logistics", "delivery"]
    }

Usage:
    # Create a package
    pkg = SkillPackage.init("my-skills", "1.0.0", author="Me")
    pkg.add_skill(patrol_skill)
    pkg.save("/workspace/my-skills")

    # Pack into distributable archive
    archive = pkg.pack("/workspace/my-skills", "/tmp/my-skills-1.0.0.skillpkg")

    # Install from archive
    pkg = SkillPackage.from_archive("/tmp/my-skills-1.0.0.skillpkg")
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

from apyrobo.skills.skill import Skill, BUILTIN_SKILLS
from apyrobo.skills.handlers import load_handler_module

logger = logging.getLogger(__name__)

# Manifest file name inside a package directory
MANIFEST_FILE = "skill-package.json"
# Archive extension
ARCHIVE_EXT = ".skillpkg"
# Subdirectory for skill JSON files within a package
SKILLS_DIR = "skills"

# Semantic version pattern (major.minor.patch with optional pre-release)
_SEMVER_RE = re.compile(
    r"^(\d+)\.(\d+)\.(\d+)(?:-([\w.]+))?$"
)

# Package name pattern: lowercase alphanumeric + hyphens
_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*[a-z0-9]$|^[a-z]$")


def validate_version(version: str) -> bool:
    """Check whether *version* is a valid semver string."""
    return bool(_SEMVER_RE.match(version))


def validate_package_name(name: str) -> bool:
    """Package names must be lowercase alphanumeric with hyphens."""
    return bool(_NAME_RE.match(name)) and len(name) <= 64


def parse_version_tuple(version: str) -> tuple[int, int, int, str]:
    """Parse a semver string into a comparable tuple."""
    m = _SEMVER_RE.match(version)
    if not m:
        raise ValueError(f"Invalid semver: {version!r}")
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4) or "")


def check_version_constraint(version: str, constraint: str) -> bool:
    """
    Check if *version* satisfies a simple constraint string.

    Supports: ">=1.0.0", "==1.2.3", ">=1.0.0,<2.0.0"
    """
    parts = [c.strip() for c in constraint.split(",")]
    ver = parse_version_tuple(version)
    for part in parts:
        if part.startswith(">="):
            if ver < parse_version_tuple(part[2:]):
                return False
        elif part.startswith(">"):
            if ver <= parse_version_tuple(part[1:]):
                return False
        elif part.startswith("<="):
            if ver > parse_version_tuple(part[2:]):
                return False
        elif part.startswith("<"):
            if ver >= parse_version_tuple(part[1:]):
                return False
        elif part.startswith("=="):
            if ver != parse_version_tuple(part[2:]):
                return False
        else:
            # Bare version means exact match
            if ver != parse_version_tuple(part):
                return False
    return True


# ---------------------------------------------------------------------------
# Skill Package
# ---------------------------------------------------------------------------

class SkillPackage:
    """
    A versioned, distributable bundle of skills.

    Attributes:
        name:        Package name (lowercase, hyphenated).
        version:     Semver version string.
        description: One-line summary.
        author:      Author name or org.
        license:     SPDX license identifier.
        homepage:    Project URL.
        required_capabilities: Capability types the skills need.
        min_apyrobo_version:   Minimum framework version.
        skills:      Ordered list of Skill objects in this package.
        dependencies: {package_name: version_constraint}.
        tags:        Searchable tags.
    """

    def __init__(
        self,
        name: str,
        version: str,
        description: str = "",
        author: str = "",
        license: str = "Apache-2.0",
        homepage: str = "",
        required_capabilities: list[str] | None = None,
        min_apyrobo_version: str = "0.1.0",
        skills: list[Skill] | None = None,
        dependencies: dict[str, str] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        if not validate_package_name(name):
            raise ValueError(
                f"Invalid package name: {name!r}. "
                "Must be lowercase alphanumeric with hyphens, 1-64 chars."
            )
        if not validate_version(version):
            raise ValueError(
                f"Invalid version: {version!r}. Must be semver (e.g. 1.0.0)."
            )
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.license = license
        self.homepage = homepage
        self.required_capabilities = required_capabilities or []
        self.min_apyrobo_version = min_apyrobo_version
        self.skills: list[Skill] = list(skills or [])
        self.dependencies: dict[str, str] = dict(dependencies or {})
        self.tags: list[str] = list(tags or [])

    # ------------------------------------------------------------------
    # Skill management
    # ------------------------------------------------------------------

    def add_skill(self, skill: Skill) -> None:
        """Add a skill to this package."""
        # Replace if same ID already present
        self.skills = [s for s in self.skills if s.skill_id != skill.skill_id]
        self.skills.append(skill)

    def remove_skill(self, skill_id: str) -> bool:
        """Remove a skill by ID. Returns True if found."""
        before = len(self.skills)
        self.skills = [s for s in self.skills if s.skill_id != skill_id]
        return len(self.skills) < before

    def get_skill(self, skill_id: str) -> Skill | None:
        for s in self.skills:
            if s.skill_id == skill_id:
                return s
        return None

    @property
    def skill_ids(self) -> list[str]:
        return [s.skill_id for s in self.skills]

    # ------------------------------------------------------------------
    # Manifest serialisation
    # ------------------------------------------------------------------

    def to_manifest(self) -> dict[str, Any]:
        """Serialise to a manifest dict (for skill-package.json)."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "homepage": self.homepage,
            "required_capabilities": self.required_capabilities,
            "min_apyrobo_version": self.min_apyrobo_version,
            "skills": self.skill_ids,
            "dependencies": self.dependencies,
            "tags": self.tags,
        }

    def to_manifest_json(self) -> str:
        return json.dumps(self.to_manifest(), indent=2)

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any], skills: list[Skill] | None = None) -> SkillPackage:
        """Create a SkillPackage from a manifest dict + skill objects."""
        return cls(
            name=manifest["name"],
            version=manifest["version"],
            description=manifest.get("description", ""),
            author=manifest.get("author", ""),
            license=manifest.get("license", "Apache-2.0"),
            homepage=manifest.get("homepage", ""),
            required_capabilities=manifest.get("required_capabilities", []),
            min_apyrobo_version=manifest.get("min_apyrobo_version", "0.1.0"),
            skills=skills,
            dependencies=manifest.get("dependencies", {}),
            tags=manifest.get("tags", []),
        )

    # ------------------------------------------------------------------
    # Filesystem operations
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """
        Save the package to a directory.

        Creates:
            <directory>/skill-package.json   (manifest)
            <directory>/skills/<id>.json     (one per skill)
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        skills_dir = directory / SKILLS_DIR
        skills_dir.mkdir(exist_ok=True)

        # Write manifest
        manifest_path = directory / MANIFEST_FILE
        with open(manifest_path, "w") as f:
            f.write(self.to_manifest_json())

        # Write each skill
        for skill in self.skills:
            skill_path = skills_dir / f"{skill.skill_id}.json"
            with open(skill_path, "w") as f:
                f.write(skill.to_json())

        logger.info("Saved package %s@%s to %s (%d skills)",
                     self.name, self.version, directory, len(self.skills))
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> SkillPackage:
        """Load a package from a directory containing skill-package.json."""
        directory = Path(directory)
        manifest_path = directory / MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"No {MANIFEST_FILE} found in {directory}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Load skill files
        skills_dir = directory / SKILLS_DIR
        skills: list[Skill] = []
        if skills_dir.exists():
            for skill_id in manifest.get("skills", []):
                skill_path = skills_dir / f"{skill_id}.json"
                if skill_path.exists():
                    with open(skill_path) as f:
                        skills.append(Skill.from_dict(json.load(f)))
                else:
                    logger.warning("Skill file not found: %s", skill_path)

        pkg = cls.from_manifest(manifest, skills=skills)
        pkg._load_skill_handlers()
        return pkg

    # ------------------------------------------------------------------
    # Handler loading
    # ------------------------------------------------------------------

    def _load_skill_handlers(self) -> None:
        """Import handler modules declared by skills in this package."""
        for skill in self.skills:
            if skill.handler_module:
                try:
                    load_handler_module(skill.handler_module)
                except Exception:
                    logger.warning(
                        "Could not load handler module %s for skill %s",
                        skill.handler_module, skill.skill_id,
                    )

    # ------------------------------------------------------------------
    # Archive operations (.skillpkg)
    # ------------------------------------------------------------------

    def pack(self, source_dir: str | Path, output_path: str | Path | None = None) -> Path:
        """
        Pack the package directory into a .skillpkg archive.

        If output_path is None, creates <name>-<version>.skillpkg
        in the parent of source_dir.
        """
        source_dir = Path(source_dir)
        if output_path is None:
            output_path = source_dir.parent / f"{self.name}-{self.version}{ARCHIVE_EXT}"
        else:
            output_path = Path(output_path)

        # Ensure the source is saved first
        self.save(source_dir)

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(source_dir):
                for fname in files:
                    filepath = Path(root) / fname
                    arcname = filepath.relative_to(source_dir)
                    zf.write(filepath, arcname)

        logger.info("Packed %s@%s → %s", self.name, self.version, output_path)
        return output_path

    @classmethod
    def from_archive(cls, archive_path: str | Path, extract_to: str | Path | None = None) -> SkillPackage:
        """
        Load a package from a .skillpkg archive.

        If extract_to is None, extracts to a temp directory alongside the archive.
        """
        archive_path = Path(archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        if extract_to is None:
            extract_to = archive_path.parent / archive_path.stem
        extract_to = Path(extract_to)

        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)

        return cls.load(extract_to)

    # ------------------------------------------------------------------
    # Init helper
    # ------------------------------------------------------------------

    @classmethod
    def init(
        cls,
        name: str,
        version: str = "0.1.0",
        description: str = "",
        author: str = "",
        directory: str | Path | None = None,
    ) -> SkillPackage:
        """
        Create and optionally save a new empty package.

        If directory is provided, writes the manifest to disk.
        """
        pkg = cls(name=name, version=version, description=description, author=author)
        if directory:
            pkg.save(directory)
        return pkg

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Validate the package. Returns a list of error strings (empty = valid).
        """
        errors: list[str] = []
        if not self.name:
            errors.append("Package name is required")
        if not self.version:
            errors.append("Package version is required")
        if not self.skills:
            errors.append("Package must contain at least one skill")

        # Check skill IDs match manifest
        for skill in self.skills:
            if not skill.skill_id:
                errors.append(f"Skill has empty skill_id: {skill.name}")
            if not skill.name:
                errors.append(f"Skill {skill.skill_id} has empty name")

        # Check for duplicate skill IDs
        ids = [s.skill_id for s in self.skills]
        dupes = [sid for sid in ids if ids.count(sid) > 1]
        if dupes:
            errors.append(f"Duplicate skill IDs: {set(dupes)}")

        return errors

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<SkillPackage {self.name}@{self.version} "
            f"skills={len(self.skills)} deps={len(self.dependencies)}>"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SkillPackage):
            return NotImplemented
        return self.name == other.name and self.version == other.version
