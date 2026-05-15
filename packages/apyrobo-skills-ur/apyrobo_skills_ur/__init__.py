"""apyrobo-skills-ur — Universal Robots UR3/UR5/UR10/UR16 skill pack for APYROBO."""
from __future__ import annotations

from apyrobo_skills_ur.motion import move_joints, move_linear, move_home, set_tcp
from apyrobo_skills_ur.manipulation import pick, place, get_pose

__all__ = [
    "move_joints",
    "move_linear",
    "move_home",
    "set_tcp",
    "pick",
    "place",
    "get_pose",
    "register",
]

# Skill IDs as defined by each function's @skill decorator
_SKILL_FUNCTIONS = [
    move_joints,
    move_linear,
    move_home,
    set_tcp,
    pick,
    place,
    get_pose,
]


def register() -> None:
    """Register all Universal Robots skills with the active SkillLibrary.

    Called automatically via the ``apyrobo.skills`` entry-point when the
    package is installed, or manually::

        import apyrobo_skills_ur
        apyrobo_skills_ur.register()
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
        print(f"[apyrobo-skills-ur] Registered {len(registered)} skill(s): "
              + ", ".join(registered))
