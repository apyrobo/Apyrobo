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

## Notes

- **macOS:** Gazebo Classic doesn't run under Docker Desktop reliably; this
  is a Linux/CI target. On a Linux host with Docker, the command above works
  as-is.
- **No Nav2 here** by design — the minimal sim keeps the moving parts down,
  so `move()` uses the adapter's `cmd_vel` proportional controller. A Nav2 +
  full-navigation-stack variant is a natural follow-up.
- **Gazebo Classic is EOL** (maintained through 2025). A port to modern
  Gazebo (`gz-sim`) / `ros_gz` is the forward path but is a larger change.
