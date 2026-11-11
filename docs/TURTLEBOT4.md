# Hello TurtleBot4

Get APYROBO running on a TurtleBot4 (real or simulated) in under 15 minutes.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| ROS 2 | Humble or Iron | Jazzy also works; older distros untested |
| apyrobo | current main | installed next to ROS 2 (step 1) |
| nav2_msgs | bundled with Nav2 | Required for NavigateToPose action |

If you only want to **develop or test without hardware**, skip to [Path B — mock adapter](#path-b--mock-adapter-no-hardware-needed).

---

## Path A — Real robot via `ros2://`

The `ros2://` adapter speaks Nav2 actions and `cmd_vel` directly.

### 1. Install apyrobo next to ROS 2

`ros2://` talks to ROS 2 through `rclpy` in APYROBO's own process, so install
it on the robot (or a machine on its `ROS_DOMAIN_ID`) where ROS 2 is sourced.
The `apyrobo-api` image in `docker/docker-compose.yml` does not include ROS 2.

```bash
source /opt/ros/humble/setup.bash
sudo apt install ros-$ROS_DISTRO-nav2-msgs ros-$ROS_DISTRO-tf-transformations
python3 -m venv --system-site-packages ~/apyrobo-venv
. ~/apyrobo-venv/bin/activate
pip install "git+https://github.com/apyrobo/Apyrobo"
```

### 2. Use the same shell for every step below

Each new terminal needs `source /opt/ros/humble/setup.bash` and
`. ~/apyrobo-venv/bin/activate` again.

### 3. Verify ROS 2 is available

```bash
python3 -c "import rclpy; print('rclpy OK')"
ros2 topic list   # should show /odom, /cmd_vel, etc.
```

If `rclpy` is missing, importing the `ros2://` adapter warns and `Robot.discover('ros2://...')` raises `RuntimeError`. Source the ROS 2 environment:

```bash
source /opt/ros/humble/setup.bash
```

### 4. Run a navigation script

```python
#!/usr/bin/env python3
"""Navigate to a waypoint on a real TurtleBot4."""

from apyrobo import Robot, Agent

# Discovers the robot via ros2:// — requires rclpy + Nav2 running
robot = Robot.discover("ros2://turtlebot4")

agent = Agent(provider="rule")
result = agent.execute(
    task="navigate to position 2.5 3.0",
    robot=robot,
    on_event=lambda e: print(f"  [{e.status.value}] {e.skill_id}: {e.message}"),
)

print(f"Status: {result.status.value}  ({result.steps_completed}/{result.steps_total} steps)")
```

Save as `navigate_waypoint.py` and run it in that shell:

```bash
python3 navigate_waypoint.py
```

### 5. Configuration knobs

The adapter reads a YAML file and/or `kwargs` to override defaults:

```python
robot = Robot.discover(
    "ros2://turtlebot4",
    namespace="/tb4",           # prefix all topic names with /tb4
    nav_timeout_sec=90.0,       # how long to wait for navigation to complete
    odom_reliability="reliable", # or "best_effort" (default)
    config_yaml="/app/config/ros2.yaml",
)
```

Default topics (TurtleBot4-compatible):

| Key | Default topic |
|---|---|
| `cmd_vel` | `/cmd_vel` |
| `odom` | `/odom` |
| `scan` | `/scan` |
| `camera` | `/oakd/rgb/preview/image_raw` |
| `depth` | `/oakd/stereo/image_raw` |
| `imu` | `/imu` |
| `nav2_action` | `navigate_to_pose` |

---

## Path B — Mock adapter (no hardware needed)

Use `mock://` for developing skills and testing logic locally — no ROS 2 or Docker required.

```python
#!/usr/bin/env python3
"""Develop and test a skill without a physical robot."""

from apyrobo import skill, SkillLibrary, Robot, Agent


@skill(description="Navigate to a shelf and scan it for items", capability="navigate")
def inspect_shelf(shelf_id: str = "A3") -> bool:
    print(f"  → moving to shelf {shelf_id}...")
    print(f"  → scanning... found 7 items")
    return True


robot = Robot.discover("mock://turtlebot4")
agent = Agent(provider="rule", library=SkillLibrary.from_decorated())

result = agent.execute("inspect the shelf", robot=robot)
print(f"Status: {result.status.value}  ({result.steps_completed}/{result.steps_total} steps)")
```

Run anywhere with just:

```bash
pip install -e .        # from the repo root, one time
python3 inspect_shelf.py
```

Expected output:

```
  → moving to shelf A3...
  → scanning... found 7 items
Status: completed  (1/1 steps)
```

---

## What works right now vs what requires real hardware

| Feature | `mock://` | `ros2://` (real/sim) |
|---|---|---|
| Skill planning & dispatch | ✅ | ✅ |
| Agent natural-language routing | ✅ | ✅ |
| Custom `@skill` decorator | ✅ | ✅ |
| Safety enforcer & confidence gating | ✅ | ✅ |
| Navigation to (x, y) | simulated | ✅ via Nav2 |
| Nav2 fallback (cmd_vel) | simulated | ✅ automatic |
| Odometry feedback | static (0, 0) | ✅ from `/odom` |
| LIDAR / camera sensors | not wired | ✅ topic introspection |
| E-stop / reset_estop | no-op | ✅ |
| SLAM trigger | no-op | ✅ slam_toolbox |
| Multi-floor map switching | no-op | ✅ map_server service |

---

## Navigation internals

When you call `agent.execute("navigate to 2.5 3.0", robot=robot)`, the ROS 2 adapter:

1. **Tries Nav2 first** — sends a `NavigateToPose` action goal with the target pose in the `map` frame, pointing toward the goal.
2. **Falls back to cmd_vel** if Nav2 is not available — runs a proportional controller loop at 10 Hz until within 0.25 m of the target.
3. **Waits up to `nav_timeout_sec`** (default 120 s) for completion.
4. **Reports pose** via the `/odom` subscription using best-effort QoS (configurable).

---

## Common errors and fixes

### `ROS 2 import failed` warning on import

```
UserWarning: ROS 2 import failed (No module named 'rclpy') — the ros2:// adapter
will not be available. ...
```

**Fix:** source your ROS 2 environment before starting Python:

```bash
source /opt/ros/humble/setup.bash
python3 your_script.py
```

If the module named is `nav2_msgs` or `tf_transformations`, install it with
`sudo apt install ros-$ROS_DISTRO-nav2-msgs ros-$ROS_DISTRO-tf-transformations`.

### `RuntimeError: The ros2:// adapter requires rclpy`

You called `Robot.discover("ros2://...")` outside a ROS 2 environment. Use `mock://` for local development or source ROS 2 first.

### Nav2 not found — using cmd_vel fallback

```
WARNING: Nav2 action server not found — using cmd_vel fallback
```

Nav2 isn't running. Start it on the robot or in Gazebo:

```bash
ros2 launch nav2_bringup navigation_launch.py
```

The adapter will automatically retry Nav2 on the next `Robot.discover()` call.

### No odometry received within 5.0s

```
WARNING: No odometry received within 5.0s — position may be stale
```

The `/odom` topic isn't publishing. Check the robot's odometry node:

```bash
ros2 topic hz /odom
```

If the QoS profile mismatches (common with TurtleBot4), set `odom_reliability="reliable"` in `Robot.discover()`.

### `UnknownSkillError` with `@skill` decorator

If you use `@skill` but build the library manually (`SkillLibrary()`), the execution handler won't be wired. Always use `SkillLibrary.from_decorated()` when your skills are defined with `@skill`.

---

## Connection Resilience

The `ros2://` adapter automatically monitors the `/odom` subscription for timeouts and recovers from transient failures — network drops, node crashes, or hardware resets — without any extra code.

**Defaults:** 5 s timeout · 1 s initial backoff doubling to 30 s · unlimited retries · ±20% jitter.

### Registering handlers

```python
robot = Robot.discover("ros2://turtlebot4")
robot.connect()

robot.health.on_disconnect(lambda: print("Robot lost! Pausing tasks..."))
robot.health.on_reconnect(lambda: print("Robot back online."))
```

### Give-up callback

```python
robot.health.on_give_up(lambda: print("Gave up after all retries — check hardware"))
```

### Customising the monitor

```python
from apyrobo.core.health import ConnectionHealth

robot = Robot.discover("ros2://turtlebot4", auto_health=False)
robot.connect()

health = ConnectionHealth(
    robot._adapter,
    timeout_seconds=3.0,
    backoff_base=0.5,
    backoff_max=20.0,
    max_retries=10,
)
health.on_disconnect(lambda: logger.warning("Lost contact"))
health.on_reconnect(lambda: logger.info("Back online"))
health.on_give_up(lambda: logger.error("Unreachable after 10 attempts"))
health.start()
```

### Disabling the monitor

```python
robot = Robot.discover("ros2://turtlebot4", auto_health=False)
robot.connect()
# health property returns None; no background thread runs
```

---

## Gazebo simulation

A robot simulated in Gazebo through its ROS 2 bridge is driven with the same
`ros2://` URI as the real one: start the simulation, source ROS 2 in the shell,
and run your code unchanged. CI does exactly this with a TurtleBot3 in Gazebo
(see [conformance_results.md](conformance_results.md)).

```python
# Real robot, or the same robot simulated in a running Gazebo
robot = Robot.discover("ros2://turtlebot4")
```

`gazebo://` is not an alias for `ros2://`: it selects `GazeboAdapter`, an
in-memory stand-in with simulated delays that never connects to Gazebo (it logs
a warning saying so). Use it only for tests that need no simulator.

---

## Next steps

- [Skill Authoring Guide](skill_authoring.md) — write production skills
- [Architecture Overview](architecture.md) — how the planner and executor interact
- [API Reference](api_reference.md) — full adapter and robot API
