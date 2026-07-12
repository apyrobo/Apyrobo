# Gazebo integration — driving a physics-simulated robot

The `integration` profile tests the `ros2://` adapter against
`fake_turtlebot4.py`, a *kinematic* stand-in. The **`gazebo` profile** goes
further: it runs the same adapter against a **TurtleBot3 burger in headless
Gazebo Classic** — real physics, real diff-drive, real odometry.

There is no code difference in the adapter. `ROS2Adapter` already speaks
`/cmd_vel` + `/odom`, which is exactly what the burger's `gazebo_ros`
diff-drive plugin exposes. So this profile is a *proof*, not new plumbing:
APYROBO commands make a physics-simulated robot actually move.

## Run it (Linux, or CI)

```bash
docker compose -f docker/docker-compose.yml --profile gazebo up \
    --build --abort-on-container-exit --exit-code-from gazebo-test
```

- `gazebo-sim` runs [`gazebo_boot.sh`](gazebo_boot.sh): headless `gzserver`
  + a spawned burger. It's healthy once `/odom` flows.
- `gazebo-test` runs [`test_gazebo_turtlebot.py`](test_gazebo_turtlebot.py):
  discovers `ros2://burger`, issues `move()`, and asserts the robot travelled
  (with friction and inertia in the loop), then that `stop()` is safe.

CI runs this on Linux as the **Gazebo TurtleBot3 Integration** job in
`.github/workflows/integration.yml`.

## The `gazebo-nav` profile — NL → Nav2 end-to-end

The **`gazebo-nav` profile** is the flagship-stack proof: the full pipeline
from a natural-language task to a navigating physics robot.

```bash
docker compose -f docker/docker-compose.yml --profile gazebo-nav up \
    --build --abort-on-container-exit --exit-code-from gazebo-nav-test
```

- `gazebo-nav-sim` runs [`gazebo_nav_boot.sh`](gazebo_nav_boot.sh): the
  `gazebo` setup plus `robot_state_publisher` and **Nav2 in SLAM mode**
  (`slam_toolbox` builds the map live in `turtlebot3_world` — no pre-built
  map, no AMCL initial pose). Healthy once the `navigate_to_pose` action
  server is up.
- `gazebo-nav-test` runs
  [`test_gazebo_nav2_e2e.py`](test_gazebo_nav2_e2e.py):
  `Agent(provider="rule").execute("navigate to (-1.2, -0.5)", robot)` — the
  rule agent extracts the coordinates into a `navigate_to` skill, the
  executor calls `robot.move()`, and `ROS2Adapter` takes its **preferred
  Nav2 `NavigateToPose` path** (asserted — no cmd_vel fallback). The test
  then asserts the robot actually reached the goal region.

CI runs this as the **Nav2-in-Gazebo NL End-to-End** job.

## Notes

- **macOS:** Gazebo Classic doesn't run under Docker Desktop reliably; this
  is a Linux/CI target. On a Linux host with Docker, the commands above work
  as-is.
- **No Nav2 in the `gazebo` profile** by design — the minimal sim keeps the
  moving parts down, so `move()` uses the adapter's `cmd_vel` proportional
  controller. The `gazebo-nav` profile is the full-navigation-stack variant.
- **Gazebo Classic is EOL** (maintained through 2025). A port to modern
  Gazebo (`gz-sim`) / `ros_gz` is the forward path but is a larger change
  ([roadmap](../../ROADMAP.md)).
