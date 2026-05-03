"""apyrobo-skills-turtlebot4 — TurtleBot 4 skill pack for APYROBO."""
from __future__ import annotations

from apyrobo_skills_turtlebot4.navigation import dock, patrol_area, undock
from apyrobo_skills_turtlebot4.inspection import check_surroundings, inspect_room
from apyrobo_skills_turtlebot4.social import follow_person

__all__ = [
    "patrol_area",
    "dock",
    "undock",
    "inspect_room",
    "check_surroundings",
    "follow_person",
    "register",
]

# Skill IDs as defined by each function's @skill decorator
_SKILL_FUNCTIONS = [
    patrol_area,
    dock,
    undock,
    inspect_room,
    check_surroundings,
    follow_person,
]


def register() -> None:
    """Register all TurtleBot 4 skills with the active SkillLibrary.

    Called automatically via the ``apyrobo.skills`` entry-point when the
    package is installed, or manually::

        import apyrobo_skills_turtlebot4
        apyrobo_skills_turtlebot4.register()
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
        print(f"[apyrobo-skills-turtlebot4] Registered {len(registered)} skill(s): "
              + ", ".join(registered))
