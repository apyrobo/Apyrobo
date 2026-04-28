# ROS 2 Adapter Integration Tests

Self-contained end-to-end tests that prove the `ros2://` adapter works without
a physical robot.  A fake TurtleBot4 node (`fake_turtlebot4.py`) stands in for
the real hardware.

---

## What it proves

| Claim | How it's tested |
|-------|----------------|
| Adapter discovers the Nav2 action server | `adapter._has_nav2 == True` after `Robot.discover()` |
| Adapter receives odometry | `adapter._has_odom == True`; initial position ≈ (0, 0) |
| `navigate_to` via apyrobo Agent completes | `result.status == "completed"` |
| Position tracking is accurate | `adapter.position` ≈ target after navigation |
| Direct `robot.move()` also works | `nav_state == SUCCEEDED` and position matches |
| Capability introspection reports NAVIGATE | returned `RobotCapability` includes `CapabilityType.NAVIGATE` |

## What it does NOT prove

- Real Nav2 path planning (no costmap, no obstacle avoidance).
- Real sensor data (`/scan`, `/imu`, `/oakd/*` are not published by the fake node).
- Network latency or DDS QoS edge cases.
- Behaviour under E-STOP or connection loss.

---

## Running with Docker (recommended)

Requires Docker with the Compose plugin.

```bash
docker compose -f docker/docker-compose.yml --profile integration \
    up --build --abort-on-container-exit
```

This spins up two containers:

| Service | Image | Role |
|---------|-------|------|
| `fake-robot` | `apyrobo-integration` | Runs `fake_turtlebot4.py` |
| `integration-test` | `apyrobo-integration` | Runs `pytest -m integration` |

Both share `ROS_DOMAIN_ID=42` on the `ros2-test-net` bridge network.  The test
container waits for the `fake-robot` healthcheck (which polls `/odom`) before
starting pytest.

The exit code of the `integration-test` container mirrors the pytest exit code,
so CI can gate on it:

```bash
docker compose ... up --abort-on-container-exit
echo "Exit: $?"
```

### Docker Desktop (Mac/Windows)

FastDDS multicast discovery may not propagate between containers on Docker
Desktop because of the Linux VM networking layer.  If the tests fail with
`Nav2 action server not found`, add `network_mode: host` to **both** services
in `docker-compose.yml` and remove their `networks:` stanzas.

---

## Running locally (ROS 2 Humble installed)

Open two terminals.

**Terminal 1 — start the fake robot:**

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
python tests/integration/fake_turtlebot4.py
```

**Terminal 2 — run the tests:**

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
pytest -m integration tests/integration/ -v
```

Or let the test fixture launch the fake robot automatically (no need for
Terminal 1 when `FAKE_ROBOT_EXTERNAL` is not set):

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
pytest -m integration tests/integration/ -v
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROS_DOMAIN_ID` | `42` | ROS 2 DDS domain; must match on all participants |
| `FAKE_ROBOT_EXTERNAL` | _(unset)_ | Set to `1` when the fake robot is already running (Docker mode). Unset = fixture launches a subprocess. |

---

## File overview

```
tests/integration/
├── fake_turtlebot4.py      Fake robot node (rclpy) — runs standalone or as subprocess
├── test_ros2_adapter.py    pytest tests (marked @pytest.mark.integration)
└── README.md               This file
docker/
├── Dockerfile.integration  ROS 2 Humble + apyrobo image for both services
└── docker-compose.yml      Adds 'integration' profile (fake-robot + integration-test)
```
