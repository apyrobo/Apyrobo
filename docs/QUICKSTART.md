# APYROBO — Simulation and ROS 2 setup

Two paths beyond `mock://`: the simulated Nav2 demo that CI runs, and a real
ROS 2 robot. To try APYROBO without ROS first, see
[quickstart_5min.md](quickstart_5min.md) or the `apyrobo mcp` try-out in
[mcp.md](mcp.md).

---

## Simulated Nav2 in Gazebo (Linux, Docker)

A TurtleBot3 in `turtlebot3_world`, Nav2 in SLAM mode, and APYROBO's rule
planner sending a real `NavigateToPose` goal. It is the same stack CI checks
on every commit ([test_gazebo_nav2_e2e.py](../tests/integration/test_gazebo_nav2_e2e.py)).
Gazebo Classic does not run under Docker Desktop on macOS, so this is a Linux
target.

```bash
# From the repo root: start gzserver + robot_state_publisher + Nav2 (SLAM)
docker compose -f docker/docker-compose.yml --profile gazebo-nav up -d gazebo-nav-sim

# Run the narrated demo against it
docker compose -f docker/docker-compose.yml --profile gazebo-nav \
    run --rm gazebo-nav-demo
```

More detail, and how to re-record it: [demos/nav2_gazebo](../demos/nav2_gazebo/README.md)
and [tests/integration/README_gazebo.md](../tests/integration/README_gazebo.md).

---

## ROS 2 / Real Robot

The `ros2://` adapter talks to ROS 2 through `rclpy` in APYROBO's own process,
so APYROBO has to run where ROS 2 is sourced: on the robot, or on a machine on
the robot's `ROS_DOMAIN_ID`. It needs `rclpy`, `nav2_msgs` and
`tf_transformations`:
`sudo apt install ros-$ROS_DISTRO-nav2-msgs ros-$ROS_DISTRO-tf-transformations`.

The `apyrobo-api` image in `docker/docker-compose.yml` does **not** include
ROS 2: it is a slim Python image for the REST gateway. The images that bundle
ROS 2 (`docker/Dockerfile.integration`, `docker/Dockerfile.gazebo`) are what CI
and the Gazebo demos use.

### Install next to ROS 2

A virtualenv with `--system-site-packages` sees ROS 2's `rclpy` and keeps pip
away from the system Python:

```bash
source /opt/ros/jazzy/setup.bash     # or humble
python3 -m venv --system-site-packages ~/apyrobo-venv
. ~/apyrobo-venv/bin/activate
pip install "git+https://github.com/apyrobo/Apyrobo"
python3 -c "import rclpy, nav2_msgs, tf_transformations; print('ROS 2 OK')"
```

### Connect from Python

```python
from apyrobo import Robot, Agent

robot = Robot.discover("ros2://turtlebot4")
print(robot.capabilities())

agent = Agent(provider="llm", model="claude-sonnet-4-20250514")
result = agent.execute("navigate to the charging dock", robot)
print(result.status)
```

For `apyrobo mcp` on a real robot (policy file, `--confirm-moves`, running over
`--ssh` from your laptop), follow [docs/first-real-robot.md](first-real-robot.md).

---

## Troubleshooting

**`ros2://` is unavailable / `No module named 'rclpy'`:** ROS 2 is not sourced
in this shell, or APYROBO is installed in a Python that cannot see it. Source
`/opt/ros/<distro>/setup.bash` and use a `--system-site-packages` virtualenv as
above.

**The robot never moves:** the most common cause is a different
`ROS_DOMAIN_ID` from the robot's. See the troubleshooting section of
[mcp.md](mcp.md).

**Docker build fails:** give Docker at least 8 GB of RAM and run
`docker system prune` to free space.
