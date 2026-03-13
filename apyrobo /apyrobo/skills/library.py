"""
Skill Library — manages a collection of skills from JSON files.

Loads skills from a directory, validates them, and makes them available
for the agent to plan with. Supports hot-reloading.

Usage:
    library = SkillLibrary("/workspace/skills")
    library.load_all()
    skill = library.get("custom_patrol")
    all_skills = library.all_skills()  # built-in + custom
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from apyrobo.skills.skill import Skill, BUILTIN_SKILLS

logger = logging.getLogger(__name__)


class SkillLibrary:
    """
    Manages built-in and custom skills.

    Custom skills are loaded from JSON files in a directory.
    Built-in skills are always available.
    """

    def __init__(self, skills_dir: str | Path | None = None) -> None:
        self._custom_skills: dict[str, Skill] = {}
        self._skills_dir = Path(skills_dir) if skills_dir else None
        self._load_errors: list[dict[str, Any]] = []

        if self._skills_dir and self._skills_dir.exists():
            self.load_all()

    def load_all(self) -> int:
        """Load all .json skill files from the skills directory."""
        if self._skills_dir is None or not self._skills_dir.exists():
            logger.warning("Skills directory not set or doesn't exist")
            return 0

        loaded = 0
        self._load_errors = []

        for path in sorted(self._skills_dir.glob("*.json")):
            try:
                self.load_file(path)
                loaded += 1
            except Exception as e:
                self._load_errors.append({"file": str(path), "error": str(e)})
                logger.error("Failed to load skill %s: %s", path.name, e)

        logger.info("Loaded %d custom skills from %s", loaded, self._skills_dir)
        return loaded

    def load_file(self, path: str | Path) -> Skill:
        """Load a single skill from a JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        skill = Skill.from_dict(data)
        self._custom_skills[skill.skill_id] = skill
        logger.debug("Loaded skill: %s from %s", skill.skill_id, path.name)
        return skill

    def load_json(self, json_str: str) -> Skill:
        """Load a skill from a JSON string."""
        skill = Skill.from_json(json_str)
        self._custom_skills[skill.skill_id] = skill
        return skill

    def save_skill(self, skill: Skill, path: str | Path | None = None) -> Path:
        """Save a skill to a JSON file."""
        if path is None:
            if self._skills_dir is None:
                raise ValueError("No skills directory set and no path provided")
            path = self._skills_dir / f"{skill.skill_id}.json"
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(skill.to_json())

        logger.info("Saved skill %s to %s", skill.skill_id, path)
        return path

    def get(self, skill_id: str) -> Skill | None:
        """Get a skill by ID (checks custom first, then built-in)."""
        return self._custom_skills.get(skill_id) or BUILTIN_SKILLS.get(skill_id)

    def all_skills(self) -> dict[str, Skill]:
        """All available skills (built-in + custom). Custom overrides built-in."""
        merged = dict(BUILTIN_SKILLS)
        merged.update(self._custom_skills)
        return merged

    def custom_skills(self) -> dict[str, Skill]:
        """Only custom-loaded skills."""
        return dict(self._custom_skills)

    def remove(self, skill_id: str) -> bool:
        """Remove a custom skill. Cannot remove built-ins."""
        if skill_id in self._custom_skills:
            del self._custom_skills[skill_id]
            return True
        return False

    @property
    def load_errors(self) -> list[dict[str, Any]]:
        return list(self._load_errors)

    def __len__(self) -> int:
        return len(self.all_skills())

    def __contains__(self, skill_id: str) -> bool:
        return self.get(skill_id) is not None

    def __repr__(self) -> str:
        return (
            f"<SkillLibrary builtin={len(BUILTIN_SKILLS)} "
            f"custom={len(self._custom_skills)} "
            f"total={len(self)}>"
        )
