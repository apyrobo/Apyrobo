"""
Skill Library — manages a collection of skills from JSON files and packages.

Loads skills from a directory and/or the SkillRegistry, validates them,
and makes them available for the agent to plan with. Supports hot-reloading.

Usage:
    library = SkillLibrary("/workspace/skills")
    library.load_all()
    skill = library.get("custom_patrol")
    all_skills = library.all_skills()  # built-in + custom + registry

    # With registry integration:
    from apyrobo.skills.registry import SkillRegistry
    registry = SkillRegistry()
    library = SkillLibrary("/workspace/skills", registry=registry)
    # Now all_skills() includes skills from installed packages
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
    Manages built-in, custom, and registry-installed skills.

    Custom skills are loaded from JSON files in a directory.
    Registry skills come from installed packages.
    Built-in skills are always available.
    """

    @classmethod
    def from_decorated(
        cls,
        skills_dir: "str | Path | None" = None,
        registry: "Any | None" = None,
    ) -> "SkillLibrary":
        """Build a library pre-populated with all ``@skill``-decorated skills.

        Scans the global decorated-skill registry, registers each Skill's
        metadata, and wires its execution handler into the global HandlerRegistry
        so the SkillExecutor can dispatch it at runtime.

        Example::

            @skill(description="Inspect the shelf")
            def inspect_shelf(shelf_id: str) -> bool: ...

            lib = SkillLibrary.from_decorated()
            agent = Agent(provider="rule", library=lib)
        """
        import inspect as _inspect
        from apyrobo.skills.decorators import get_decorated_skills
        from apyrobo.skills.handlers import _DEFAULT_REGISTRY

        instance = cls(skills_dir=skills_dir, registry=registry)
        for sid, (skill_def, fn) in get_decorated_skills().items():
            instance.register(skill_def)

            # Build a (robot, params) -> bool handler from the decorated fn.
            # The fn takes its own keyword args; params is the runtime dict.
            accepted = set(_inspect.signature(fn).parameters)

            def _make_handler(f: Any, ok: set) -> Any:
                def _handler(robot: Any, params: dict) -> bool:
                    filtered = {k: v for k, v in params.items() if k in ok}
                    result = f(**filtered)
                    return bool(result) if result is not None else True
                return _handler

            _DEFAULT_REGISTRY.add(sid, _make_handler(fn, accepted))
        return instance

    def __init__(self, skills_dir: str | Path | None = None,
                 registry: Any | None = None) -> None:
        self._custom_skills: dict[str, Skill] = {}
        self._skills_dir = Path(skills_dir) if skills_dir else None
        self._load_errors: list[dict[str, Any]] = []
        self._registry = registry  # SkillRegistry or None

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

    def register(self, skill: Skill) -> None:
        """Register a skill in-memory without touching the filesystem.

        Use this to inject custom skills into an Agent's planning context
        without needing a skills directory on disk.

            lib = SkillLibrary()
            lib.register(my_skill)
            agent = Agent(provider="rule", library=lib)
        """
        self._custom_skills[skill.skill_id] = skill
        logger.debug("Registered in-memory skill: %s", skill.skill_id)

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
        """Get a skill by ID (checks custom first, then registry, then built-in)."""
        if skill_id in self._custom_skills:
            return self._custom_skills[skill_id]
        if self._registry is not None:
            skill, _pkg = self._registry.get_skill(skill_id)
            if skill is not None:
                return skill
        return BUILTIN_SKILLS.get(skill_id)

    def all_skills(self) -> dict[str, Skill]:
        """All available skills (built-in + registry + custom). Custom overrides registry overrides built-in."""
        merged = dict(BUILTIN_SKILLS)
        if self._registry is not None:
            merged.update(self._registry.all_skills())
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
