# apyrobo-skills-spot

> ⚠️ **Reference scaffold — not a hardware driver.**
> These skills print the motion they *would* perform and return
> `True`. They are a template to wire to the vendor SDK, not working
> Boston Dynamics Spot support.

Boston Dynamics Spot skill pack for APYROBO.

## Status

| | |
|---|---|
| Hardware I/O | ❌ none — prints intended actions, returns `True` |
| To make it real | wire each skill to the Boston Dynamics `bosdyn` SDK |
| Works today | `apyrobo-skills-ros-nav` (real Nav2) · the `ros2://` adapter (drives a TurtleBot3 in Gazebo) |

Skills (stubs): `dock`, `capture_image`, `arm_pick`, `walk_to`, `sit`, `stand`, `stair_climb`

The per-function docstrings describe the *intended* hardware behavior; the current implementations are stubs. Replace each skill body in `apyrobo_skills_spot/` with real SDK calls to drive hardware.
