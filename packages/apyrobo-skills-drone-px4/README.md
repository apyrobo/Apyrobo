# apyrobo-skills-drone-px4

> ⚠️ **Reference scaffold — not a hardware driver.**
> These skills print the motion they *would* perform and return
> `True`. They are a template to wire to the vendor SDK, not working
> PX4-based drones support.

PX4-based drones skill pack for APYROBO.

## Status

| | |
|---|---|
| Hardware I/O | ❌ none — prints intended actions, returns `True` |
| To make it real | wire each skill to MAVSDK-Python or `pymavlink` (MAVLink) |
| Works today | `apyrobo-skills-ros-nav` (real Nav2) · the `ros2://` adapter (drives a TurtleBot3 in Gazebo) |

Skills (stubs): `takeoff`, `land`, `fly_to`, `return_home`, `orbit`, `capture_image`

The per-function docstrings describe the *intended* hardware behavior; the current implementations are stubs. Replace each skill body in `apyrobo_skills_drone_px4/` with real SDK calls to drive hardware.
