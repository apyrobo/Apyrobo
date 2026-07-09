"""apyrobo-skills-ur — Universal Robots (UR3/UR5/UR10/UR16) skill pack for APYROBO.

⚠️  REFERENCE SCAFFOLD — NOT A HARDWARE DRIVER.

These skills print the motion they *would* perform and return
success; they do not talk to hardware. Wire each skill to `ur_rtde` (RTDE) or the [Universal_Robots_ROS2_Driver](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver)
to drive a real robot. The per-function docstrings describe the
*intended* behavior; the current bodies are stubs.

For skills that actually move a robot today, see **apyrobo-skills-ros-nav** (real Nav2 actions) and the **ros2:// adapter**, which drives a TurtleBot3 in Gazebo (`tests/integration/test_gazebo_turtlebot.py`).
"""
from __future__ import annotations

import warnings

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
    warnings.warn(
        "apyrobo-skills-ur is a REFERENCE SCAFFOLD: its skills print "
        "intended actions and return success without driving hardware. "
        "Wire them to ur_rtde / the UR ROS 2 driver for real motion.",
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
        print(f"[apyrobo-skills-ur] Registered {len(registered)} skill(s): "
              + ", ".join(registered))
