"""apyrobo-skills-drone-px4 — PX4-based drone skill pack for APYROBO."""
from __future__ import annotations

from apyrobo_skills_drone_px4.flight import takeoff, land, fly_to, return_home
from apyrobo_skills_drone_px4.payload import orbit, capture_image

__all__ = [
    "takeoff",
    "land",
    "fly_to",
    "return_home",
    "orbit",
    "capture_image",
    "register",
]

# Skill IDs as defined by each function's @skill decorator
_SKILL_FUNCTIONS = [
    takeoff,
    land,
    fly_to,
    return_home,
    orbit,
    capture_image,
]


def register() -> None:
    """Register all PX4 drone skills with the active SkillLibrary.

    Called automatically via the ``apyrobo.skills`` entry-point when the
    package is installed, or manually::

        import apyrobo_skills_drone_px4
        apyrobo_skills_drone_px4.register()
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
        print(f"[apyrobo-skills-drone-px4] Registered {len(registered)} skill(s): "
              + ", ".join(registered))
