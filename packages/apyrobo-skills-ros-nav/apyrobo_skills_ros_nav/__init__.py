"""apyrobo-skills-ros-nav — APYROBO skill wrappers for ROS 2 Nav2 navigation stack.

Skills are registered via the ``apyrobo.skills`` entry-point automatically
when the package is installed, or manually::

    import apyrobo_skills_ros_nav
    apyrobo_skills_ros_nav.register()

The module imports cleanly even without ROS 2 installed; skills raise a
clear ``ImportError`` with installation instructions at *call* time if
``rclpy`` is missing.
"""
from __future__ import annotations

from apyrobo_skills_ros_nav.skills import (
    navigate_to_pose,
    follow_path,
    clear_costmaps,
    nav2_recover,
)

__all__ = [
    "navigate_to_pose",
    "follow_path",
    "clear_costmaps",
    "nav2_recover",
    "register",
]

_SKILL_FUNCTIONS = [
    navigate_to_pose,
    follow_path,
    clear_costmaps,
    nav2_recover,
]


def register() -> None:
    """Register all ROS 2 Nav2 skills with the active SkillLibrary.

    Called automatically via the ``apyrobo.skills`` entry-point when the
    package is installed, or manually::

        import apyrobo_skills_ros_nav
        apyrobo_skills_ros_nav.register()
    """
    from apyrobo.skills.library import SkillLibrary  # noqa: F401
    from apyrobo.skills.decorators import get_decorated_skills

    SkillLibrary.from_decorated()
    decorated = get_decorated_skills()
    registered = [
        fn.__name__
        for fn in _SKILL_FUNCTIONS
        if fn.__name__ in decorated
    ]

    if registered:
        print(
            f"[apyrobo-skills-ros-nav] Registered {len(registered)} skill(s): "
            + ", ".join(registered)
        )
