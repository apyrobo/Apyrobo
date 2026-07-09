"""apyrobo-skills-drone-px4 — PX4-based drones skill pack for APYROBO.

⚠️  REFERENCE SCAFFOLD — NOT A HARDWARE DRIVER.

These skills print the motion they *would* perform and return
success; they do not talk to hardware. Wire each skill to MAVSDK-Python or `pymavlink` (MAVLink)
to drive a real robot. The per-function docstrings describe the
*intended* behavior; the current bodies are stubs.

For skills that actually move a robot today, see **apyrobo-skills-ros-nav** (real Nav2 actions) and the **ros2:// adapter**, which drives a TurtleBot3 in Gazebo (`tests/integration/test_gazebo_turtlebot.py`).
"""
from __future__ import annotations

import warnings

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
    warnings.warn(
        "apyrobo-skills-drone-px4 is a REFERENCE SCAFFOLD: its skills print "
        "intended actions and return success without driving hardware. "
        "Wire them to MAVSDK / pymavlink for real motion.",
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
        print(f"[apyrobo-skills-drone-px4] Registered {len(registered)} skill(s): "
              + ", ".join(registered))
