# Pilot Reference Stack

> Locked decisions for the APYROBO pilot stack (fixed at the v0.2
> milestone and still current). When anyone suggests "what about Spot?"
> or "what about ROS Iron?", the answer is: **not in the pilot stack**.

| Decision          | Choice                                                       | Rationale                                                    |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Robot**         | TurtleBot4 (hardware target); CI's sim proof currently drives a TurtleBot3 burger via the same `ros2://` adapter | `ros-humble-turtlebot4-simulator` already in Dockerfile; TB3 is the lighter headless-CI model |
| **ROS version**   | Humble                                                       | `FROM ros:humble-desktop` in base image                      |
| **Navigation**    | Nav2 + SLAM Toolbox (live map building, no pre-built map)    | Simplest path to autonomous navigation; no map prep needed   |
| **Environment**   | Docker via `docker/docker-compose.yml`                       | Existing Dockerfile is production-ready                      |
| **LLM provider**  | Rule-based default; LLM configurable via `APYROBO_CONFIG`   | Zero external dependencies for CI; opt-in LLM for demos      |
| **Safety policy** | `strict` preset active by default; explicit opt-out required | Safety-first: no accidental unguarded execution               |

## Non-goals for the pilot stack

These are explicitly out of scope for the pilot. They may appear in a later milestone.

- Alternative robots (Spot, Stretch, custom URDF)
- ROS distributions other than Humble (Iron, Jazzy, Rolling)
- Pre-built map navigation (AMCL without SLAM)
- Cloud-hosted LLM as a hard requirement
- Multi-machine swarm (single-host Docker only)

## File references

| Artifact                  | Path                                  |
| ------------------------- | ------------------------------------- |
| Dockerfile                | `docker/Dockerfile`                   |
| Compose file              | `docker/docker-compose.yml`           |
| Safety enforcer           | `apyrobo/safety/enforcer.py`          |
| Strict policy definition  | `apyrobo/safety/enforcer.py:205`      |
| Golden task suite         | `tests/golden/golden_tasks.py`        |
| Agent entry point         | `apyrobo/skills/agent.py`            |
