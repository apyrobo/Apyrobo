"""apyrobo-skills-spot — Boston Dynamics Spot skill pack for APYROBO."""
from __future__ import annotations

from apyrobo_skills_spot.locomotion import walk_to, sit, stand, stair_climb
from apyrobo_skills_spot.utility import dock, capture_image, arm_pick

__all__ = [
    "walk_to",
    "sit",
    "stand",
    "stair_climb",
    "dock",
    "capture_image",
    "arm_pick",
    "register",
]

# Skill functions as defined by each function's @skill decorator
_SKILL_FUNCTIONS = [
    walk_to,
    sit,
    stand,
    stair_climb,
    dock,
    capture_image,
    arm_pick,
]


def register() -> None:
    """Register all Spot skills with the active SkillLibrary.

    Called automatically via the ``apyrobo.skills`` entry-point when the
    package is installed, or manually::

        import apyrobo_skills_spot
        apyrobo_skills_spot.register()
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
        print(f"[apyrobo-skills-spot] Registered {len(registered)} skill(s): "
              + ", ".join(registered))
