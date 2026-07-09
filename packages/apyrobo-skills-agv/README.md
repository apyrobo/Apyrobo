# apyrobo-skills-agv

> ⚠️ **Reference scaffold — not a hardware driver.**
> These skills print the motion they *would* perform and return
> `True`. They are a template to wire to the vendor SDK, not working
> generic AGVs support.

generic AGVs skill pack for APYROBO.

## Status

| | |
|---|---|
| Hardware I/O | ❌ none — prints intended actions, returns `True` |
| To make it real | wire each skill to your fleet manager's API (e.g. the VDA5050 interface) |
| Works today | `apyrobo-skills-ros-nav` (real Nav2) · the `ros2://` adapter (drives a TurtleBot3 in Gazebo) |

Skills (stubs): `load_cargo`, `unload_cargo`, `navigate_to`, `follow_route`, `dock_to_station`

The per-function docstrings describe the *intended* hardware behavior; the current implementations are stubs. Replace each skill body in `apyrobo_skills_agv/` with real SDK calls to drive hardware.
