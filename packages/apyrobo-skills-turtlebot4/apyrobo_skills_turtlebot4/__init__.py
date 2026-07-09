"""apyrobo-skills-turtlebot4 — TurtleBot 4 skill pack for APYROBO.

⚠️  REFERENCE SCAFFOLD — NOT A HARDWARE DRIVER.

These skills print the motion they *would* perform and return
success; they do not talk to hardware. Wire each skill to the real `ros2://` adapter — `Robot.discover("ros2://turtlebot4")`, which already drives a TurtleBot3 in Gazebo (`tests/integration/test_gazebo_turtlebot.py`). This package is a skill-template only
to drive a real robot. The per-function docstrings describe the
*intended* behavior; the current bodies are stubs.

For skills that actually move a robot today, see **apyrobo-skills-ros-nav** (real Nav2 actions) and the **ros2:// adapter**, which drives a TurtleBot3 in Gazebo (`tests/integration/test_gazebo_turtlebot.py`).
"""
from __future__ import annotations

import warnings

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
    warnings.warn(
        "apyrobo-skills-turtlebot4 is a REFERENCE SCAFFOLD: its skills print "
        "intended actions and return success without driving hardware. "
        "Wire them to the ros2:// adapter (Robot.discover('ros2://turtlebot4')) for real motion.",
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
        print(f"[apyrobo-skills-turtlebot4] Registered {len(registered)} skill(s): "
              + ", ".join(registered))
