"""apyrobo-skills-agv — Generic AGV skill pack for APYROBO."""
from __future__ import annotations

from apyrobo_skills_agv.navigation import navigate_to, follow_route, dock_to_station
from apyrobo_skills_agv.cargo import load_cargo, unload_cargo

__all__ = [
    "navigate_to",
    "follow_route",
    "dock_to_station",
    "load_cargo",
    "unload_cargo",
    "register",
]

# Skill IDs as defined by each function's @skill decorator
_SKILL_FUNCTIONS = [
    navigate_to,
    follow_route,
    dock_to_station,
    load_cargo,
    unload_cargo,
]


def register() -> None:
    """Register all AGV skills with the active SkillLibrary.

    Called automatically via the ``apyrobo.skills`` entry-point when the
    package is installed, or manually::

        import apyrobo_skills_agv
        apyrobo_skills_agv.register()
    """
    from apyrobo.skills.library import SkillLibrary
    from apyrobo.skills.decorators import get_decorated_skills

    lib = SkillLibrary.from_decorated()
    decorated = get_decorated_skills()
    registered = []
    for fn in _SKILL_FUNCTIONS:
        sid = fn.__name__
        if sid in decorated:
            registered.append(sid)

    if registered:
        print(f"[apyrobo-skills-agv] Registered {len(registered)} skill(s): "
              + ", ".join(registered))
