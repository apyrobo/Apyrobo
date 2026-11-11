# Pilot Quickstart — Natural Language to Nav2 in Gazebo

One path, in order: a natural-language task, planned by APYROBO's rule agent
(no LLM, no API key), executed through the real Nav2 `NavigateToPose` action
on a physics-simulated TurtleBot3 in `turtlebot3_world`.

It is the same stack CI checks on every commit
([test_gazebo_nav2_e2e.py](../tests/integration/test_gazebo_nav2_e2e.py)).

**Stack:** Docker, ROS 2 Humble, Gazebo Classic, Nav2 (SLAM mode), APYROBO.

---

## Prerequisites

| Requirement | Why |
|---|---|
| **Linux with Docker** (Compose v2) | Gazebo Classic does not run under Docker Desktop on macOS |
| **8 GB RAM** for Docker | Gazebo + Nav2 are memory-hungry |
| **Git** | To clone the repo |

No local ROS 2 install and no GPU are needed: everything runs headless in the
`apyrobo-gazebo` image (`docker/Dockerfile.gazebo`).

To try APYROBO without ROS first, see [quickstart_5min.md](quickstart_5min.md)
or the `apyrobo mcp` try-out in [mcp.md](mcp.md).

---

## Step 1: Clone the repo

```bash
git clone https://github.com/apyrobo/Apyrobo.git
cd Apyrobo
```

---

## Step 2: Build the Gazebo image

```bash
docker compose -f docker/docker-compose.yml --profile gazebo-nav build
```

The first build downloads ROS 2 Humble, Gazebo, Nav2 and TurtleBot3 packages;
later builds are cached.

---

## Step 3: Start the simulator

```bash
docker compose -f docker/docker-compose.yml --profile gazebo-nav up -d gazebo-nav-sim
```

This starts gzserver, robot_state_publisher and Nav2 with slam_toolbox.
The container reports healthy once the `navigate_to_pose` action server is up
and a map has been published (allow a few minutes on the first start):

```bash
docker inspect --format '{{.State.Health.Status}}' apyrobo-gazebo-nav-sim
```

---

## Step 4: Run the task

```bash
docker compose -f docker/docker-compose.yml --profile gazebo-nav \
    run --rm gazebo-nav-demo
```

The demo ([demos/nav2_gazebo/demo.py](../demos/nav2_gazebo/demo.py)) sends
`"navigate to (-1.2, -0.5)"`, prints the discovered robot, the plan, live
position ticks while Nav2 drives, and whether the goal region was reached.
The recording in [demos/nav2_gazebo](../demos/nav2_gazebo/README.md) shows a
full run.

---

## Step 5: Run CI's end-to-end check yourself

CI starts from a fresh simulator, so stop the one from Step 3 first:

```bash
docker compose -f docker/docker-compose.yml --profile gazebo-nav down
docker compose -f docker/docker-compose.yml --profile gazebo-nav up \
    --build --abort-on-container-exit --exit-code-from gazebo-nav-test
```

This is the Nav2-in-Gazebo job CI runs: it exits 0 when every test in
`tests/integration/test_gazebo_nav2_e2e.py` passes.

Stop everything afterwards with:

```bash
docker compose -f docker/docker-compose.yml --profile gazebo-nav down
```

---

## What's next

| Goal | Where |
|------|-------|
| Drive a real ROS 2 robot | [first-real-robot.md](first-real-robot.md) |
| ROS 2 setup outside Docker | [QUICKSTART.md](QUICKSTART.md) |
| Let an MCP client (Claude, Cursor) drive it | [mcp.md](mcp.md) |
| Write a custom skill | [skill_authoring.md](skill_authoring.md) |
| Write a custom adapter | [adapter_authoring.md](adapter_authoring.md) |
| Understand the architecture | [architecture.md](architecture.md) |

---

## Troubleshooting

**The demo waits for the simulator:**
`gazebo-nav-demo` starts only after `gazebo-nav-sim` is healthy. Check
`docker compose -f docker/docker-compose.yml --profile gazebo-nav logs gazebo-nav-sim`.

**Nav2 is slow to start:**
On a slow machine Nav2 and SLAM can take a few minutes. The healthcheck
allows a 2-minute start period, then retries every 10 s, 18 times.

**Docker build fails:**
- Make sure Docker has at least 8 GB RAM.
- Free disk space: `docker system prune`.

**On macOS:**
Gazebo Classic does not run under Docker Desktop. Use a Linux machine or VM,
or start with `mock://` ([quickstart_5min.md](quickstart_5min.md)).
