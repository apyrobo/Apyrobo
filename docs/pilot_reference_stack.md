# Pilot Reference Stack — v0.2

> Locked decisions for the APYROBO v0.2 pilot.
> When anyone suggests "what about Spot?" or "what about ROS Iron?",
> the answer is: **v0.2**.

| Decision          | Choice                                                       | Rationale                                                    |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Robot**         | TurtleBot4                                                   | `ros-humble-turtlebot4-simulator` already in Dockerfile      |
| **ROS version**   | Humble                                                       | `FROM ros:humble-desktop` in base image                      |
| **Navigation**    | Nav2 + SLAM Toolbox (live map building, no pre-built map)    | Simplest path to autonomous navigation; no map prep needed   |
| **Environment**   | Docker via `docker/docker-compose.yml`                       | Existing Dockerfile is production-ready                      |
| **LLM provider**  | Rule-based default; LLM configurable via `APYROBO_CONFIG`   | Zero external dependencies for CI; opt-in LLM for demos      |
| **Safety policy** | `strict` preset active by default; explicit opt-out required | Safety-first: no accidental unguarded execution               |

## Non-goals for v0.2

These are explicitly out of scope. They may appear in v0.3+.

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
