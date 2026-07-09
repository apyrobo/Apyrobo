# apyrobo-skills-turtlebot4

> ⚠️ **Reference scaffold — not a hardware driver.**
> These skills print the motion they *would* perform and return
> `True`. They are a template to wire to the vendor SDK, not working
> TurtleBot 4 support.

TurtleBot 4 skill pack for APYROBO.

## Status

| | |
|---|---|
| Hardware I/O | ❌ none — prints intended actions, returns `True` |
| To make it real | wire each skill to the real `ros2://` adapter — `Robot.discover("ros2://turtlebot4")`, which already drives a TurtleBot3 in Gazebo (`tests/integration/test_gazebo_turtlebot.py`). This package is a skill-template only |
| Works today | `apyrobo-skills-ros-nav` (real Nav2) · the `ros2://` adapter (drives a TurtleBot3 in Gazebo) |

Skills (stubs): `follow_person`, `inspect_room`, `check_surroundings`, `patrol_area`, `dock`, `undock`

The per-function docstrings describe the *intended* hardware behavior; the current implementations are stubs. Replace each skill body in `apyrobo_skills_turtlebot4/` with real SDK calls to drive hardware.
