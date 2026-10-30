# Design note: porting the Gazebo stack to Jazzy + Gazebo Harmonic

**Status:** proposal, nothing ported yet. The CI jobs described below still
run ROS 2 Humble + Gazebo Classic and stay unchanged until a replacement is
green beside them.

## Why

Every physics-sim proof in this repo (the `gazebo`, `gazebo-nav`,
`gazebo-nav-interop` and `gazebo-nav-demo` Compose profiles, the three
Gazebo jobs in `.github/workflows/integration.yml`, and the recorded
[demos/nav2_gazebo](../demos/nav2_gazebo) demo) runs on ROS 2 Humble and
Gazebo Classic. Gazebo Classic has reached end of life, and Ubuntu 24.04
users on ROS 2 Jazzy can't run the flagship demo natively. Arc 3 of the
[roadmap](../ROADMAP.md) already commits to the port.

The adapter side is mostly ready: `ros2_bridge.py` accepts
`ROS_DISTRO=jazzy` (line 108), and `ROS2Adapter` only needs `/cmd_vel`,
`/odom` and Nav2's `NavigateToPose`, all of which exist on Jazzy. The work is
in the containers and boot scripts.

## Where Gazebo Classic is wired in today

| File | Classic / Humble coupling |
|------|---------------------------|
| `docker/Dockerfile.gazebo` | `FROM ros:humble-ros-base`; apt installs `ros-humble-gazebo-ros-pkgs`, `ros-humble-turtlebot3-gazebo`; `PYTHONPATH` and the launch_testing strip script hard-code `/opt/ros/humble/lib/python3.10`; `GAZEBO_MODEL_PATH` |
| `tests/integration/gazebo_boot.sh` | `gzserver` with `libgazebo_ros_init.so` / `libgazebo_ros_factory.so`, waits on `/spawn_entity`, spawns with `gazebo_ros spawn_entity.py` |
| `tests/integration/gazebo_nav_boot.sh` | same as above, plus `turtlebot3_world.world` path, `robot_state_publisher` fed the Humble URDF with the `${namespace}` placeholder stripped, `nav2_bringup` in SLAM mode |
| `docker/docker-compose.yml` | healthchecks `source /opt/ros/humble/setup.bash` |
| `.github/workflows/integration.yml` | three jobs build and run the profiles above |
| `tests/integration/README_gazebo.md` | documents the Classic setup |

## Plan

Port beside, not in place, so the green Humble jobs keep guarding `main`:

1. **New image** `docker/Dockerfile.gz-jazzy` from `ros:jazzy-ros-base`
   with the Jazzy equivalents: `ros-jazzy-ros-gz` (Gazebo Harmonic and the
   `ros_gz_sim` / `ros_gz_bridge` packages), `ros-jazzy-turtlebot3-gazebo`,
   `ros-jazzy-nav2-bringup`, `ros-jazzy-slam-toolbox`. Ubuntu 24.04 ships
   Python 3.12 and marks the system interpreter as externally managed, so
   paths move to `python3.12` and `pip install` needs a venv with
   `--system-site-packages` or `--break-system-packages`.
2. **New boot script** `tests/integration/gz_nav_boot.sh`: start
   `gz sim -s -r` (server only, headless) on the TurtleBot3 world, spawn
   with `ros_gz_sim create`, and run `ros_gz_bridge parameter_bridge` for
   `/clock`, `/cmd_vel`, `/odom`, `/scan`, `/tf` and `/joint_states`. Prefer
   the TurtleBot3 package's own Jazzy launch file over hand-rolled steps if
   it covers headless mode.
3. **New Compose profile** `gz-jazzy-nav` reusing
   `tests/integration/test_gazebo_nav2_e2e.py` unchanged. The test only
   talks to `ros2://`, so it is the acceptance check: the same assertion on
   a different simulator.
4. **New CI job** `gz-jazzy-nav-e2e`, non-blocking (`continue-on-error`)
   until it has been green on `main` for a week, then required.
5. **Then** port `gazebo`, `gazebo-nav-interop` and the demo recording,
   update `README_gazebo.md` and the roadmap, and retire the Classic jobs in
   a separate PR.

## Open questions for the spike

These are unverified and must be settled by running the stack, not by
reading docs:

- **`/cmd_vel` message type.** `ROS2Adapter` publishes `geometry_msgs/Twist`
  (`ros2_bridge.py`, line 323). Nav2 announced it keeps `Twist` as the
  default on Jazzy and switches to `TwistStamped` (parameter
  `enable_stamped_cmd_vel`) from Kilted on
  ([notice](https://discourse.openrobotics.org/t/notice-nav2-migrated-to-twiststamped-to-replace-twist-for-cmd-vel-topics/40944)).
  What the Jazzy TurtleBot3 bridge expects on `/cmd_vel` needs checking; if
  it is `TwistStamped`, the adapter needs a configurable message type
  before the cmd_vel fallback path can work.
- **Which TurtleBot3 world and model paths** the Jazzy package installs,
  and whether its URDF still carries the `${namespace}` placeholder that
  `gazebo_nav_boot.sh` strips.
- **Startup timing.** The current healthcheck waits for `navigate_to_pose`
  and a first `/map`; Harmonic's startup may need a longer `start_period`.
- **Runner cost.** GitHub-hosted runners have no GPU; confirm Harmonic's
  server-only mode runs within the current job time budget.
