"""apyrobo-skills-spot — Boston Dynamics Spot skill pack for APYROBO.

⚠️  REFERENCE SCAFFOLD — NOT A HARDWARE DRIVER.

These skills print the motion they *would* perform and return
success; they do not talk to hardware. Wire each skill to the Boston Dynamics `bosdyn` SDK
to drive a real robot. The per-function docstrings describe the
*intended* behavior; the current bodies are stubs.

For skills that actually move a robot today, see **apyrobo-skills-ros-nav** (real Nav2 actions) and the **ros2:// adapter**, which drives a TurtleBot3 in Gazebo (`tests/integration/test_gazebo_turtlebot.py`).
"""
from __future__ import annotations

import warnings

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
    warnings.warn(
        "apyrobo-skills-spot is a REFERENCE SCAFFOLD: its skills print "
        "intended actions and return success without driving hardware. "
        "Wire them to the bosdyn SDK for real motion.",
        stacklevel=2,
    )
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
