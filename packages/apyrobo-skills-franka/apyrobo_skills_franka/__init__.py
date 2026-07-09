"""apyrobo-skills-franka — Franka Panda skill pack for APYROBO.

⚠️  REFERENCE SCAFFOLD — NOT A HARDWARE DRIVER.

These skills print the motion they *would* perform and return
success; they do not talk to hardware. Wire each skill to `franky` / `libfranka` / `franka_ros2`
to drive a real robot. The per-function docstrings describe the
*intended* behavior; the current bodies are stubs.

For skills that actually move a robot today, see **apyrobo-skills-ros-nav** (real Nav2 actions) and the **ros2:// adapter**, which drives a TurtleBot3 in Gazebo (`tests/integration/test_gazebo_turtlebot.py`).
"""
from __future__ import annotations

import warnings

from apyrobo_skills_franka.arm import cartesian_sweep, move_home, move_to_pose
from apyrobo_skills_franka.gripper import grasp, impedance_control, release

__all__ = [
    "move_to_pose",
    "move_home",
    "cartesian_sweep",
    "grasp",
    "release",
    "impedance_control",
    "register",
]

# Skill IDs as defined by each function's @skill decorator
_SKILL_FUNCTIONS = [
    move_to_pose,
    move_home,
    cartesian_sweep,
    grasp,
    release,
    impedance_control,
]


def register() -> None:
    """Register all Franka Panda skills with the active SkillLibrary.

    Called automatically via the ``apyrobo.skills`` entry-point when the
    package is installed, or manually::

        import apyrobo_skills_franka
        apyrobo_skills_franka.register()
    """
    warnings.warn(
        "apyrobo-skills-franka is a REFERENCE SCAFFOLD: its skills print "
        "intended actions and return success without driving hardware. "
        "Wire them to franky / libfranka for real motion.",
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
        print(f"[apyrobo-skills-franka] Registered {len(registered)} skill(s): "
              + ", ".join(registered))
