# apyrobo-skills-ur

> ⚠️ **Reference scaffold — not a hardware driver.**
> These skills print the motion they *would* perform and return
> `True`. They are a template to wire to the vendor SDK, not working
> Universal Robots (UR3/UR5/UR10/UR16) support.

Universal Robots (UR3/UR5/UR10/UR16) skill pack for APYROBO.

## Status

| | |
|---|---|
| Hardware I/O | ❌ none — prints intended actions, returns `True` |
| To make it real | wire each skill to `ur_rtde` (RTDE) or the [Universal_Robots_ROS2_Driver](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver) |
| Works today | `apyrobo-skills-ros-nav` (real Nav2) · the `ros2://` adapter (drives a TurtleBot3 in Gazebo) |

Skills (stubs): `pick`, `place`, `get_pose`, `move_joints`, `move_linear`, `move_home`, `set_tcp`

The per-function docstrings describe the *intended* hardware behavior; the current implementations are stubs. Replace each skill body in `apyrobo_skills_ur/` with real SDK calls to drive hardware.
