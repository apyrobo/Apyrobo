# APYROBO — From Zero to MVP Demo

Step-by-step guide to get the full demo running. Total time: ~15 minutes.

---

## Prerequisites

- **Docker Desktop** (Apple Silicon or x86) — [download](https://www.docker.com/products/docker-desktop/)
- **XQuartz** (macOS only, for Gazebo GUI) — `brew install --cask xquartz`
- **Git**

---

## Step 1: Clone and Build (5 min)

```bash
git clone https://github.com/YOUR_USERNAME/apyrobo.git
cd apyrobo

# Build the Docker container (downloads ROS 2 + Gazebo + deps)
docker compose -f docker/docker-compose.yml build
```

This takes 5-10 minutes on first build (downloading ~3GB of ROS 2 packages).
Subsequent builds are cached and take seconds.

---

## Step 2: Enable GUI (macOS only)

```bash
# Open XQuartz, go to Preferences → Security → check "Allow network clients"
# Then restart XQuartz and run:
xhost +localhost
```

On Linux:
```bash
xhost +local:docker
```

---

## Step 3: Launch Everything (2 min)

```bash
# Start the container
docker compose -f docker/docker-compose.yml up -d

# Open a shell
docker compose -f docker/docker-compose.yml exec apyrobo bash

# Inside the container — launch Gazebo + Nav2 + APYROBO
bash scripts/launch.sh
```

This opens a tmux session with 4 panes:
- **Pane 0**: Gazebo simulator (takes 30-60s to start)
- **Pane 1**: Nav2 navigation stack
- **Pane 2**: APYROBO shell (ready for your commands)
- **Pane 3**: Sensor topic monitor

Wait ~40 seconds for everything to initialise. You'll see the Gazebo window
appear with a TurtleBot4 in a warehouse environment.

---

## Step 4: Run Tests (1 min)

In the APYROBO pane (Pane 2):

```bash
# Quick sanity check — runs 72 mock tests
python3 run_tests.py

# Full integration test against live Gazebo
python3 scripts/integration_test.py
```

The integration test validates: ROS 2 connectivity, robot discovery,
capability detection, odometry, sensor pipeline, navigation, safety
enforcement, skill execution, and agent planning.

---

## Step 5: Run the Demo (2 min)

```bash
# Default demo: deliver a package
python3 scripts/demo_gazebo.py

# Custom task
python3 scripts/demo_gazebo.py --task "go to (3, 2)"

# Interactive mode — type tasks one by one
python3 scripts/demo_gazebo.py --interactive
```

---

## Step 6: Record the Demo (2 min)

```bash
# Record screen + terminal + metrics
python3 scripts/record_demo.py

# Record with benchmark (20 trials)
python3 scripts/record_demo.py --benchmark 20

# Headless (no screen capture, just metrics)
python3 scripts/record_demo.py --no-screen --benchmark 20
```

Recordings go to `recordings/demo_YYYYMMDD_HHMMSS/`:
- `screen_capture.mp4` — Gazebo screen recording
- `terminal_output.log` — full terminal output
- `metrics.json` — execution metrics (timing, steps, events)
- `benchmark.json` — aggregate results if --benchmark used

---

## Step 7: Review Results

```bash
# View the benchmark
cat recordings/demo_*/benchmark.json | python3 -m json.tool
```

The benchmark report gives you:
- **Task success rate** — % of tasks completed without abort
- **Average duration** — seconds per task
- **Per-trial results** — individual pass/fail with timing

This is what you include in grant applications and the README.

---

## Tmux Controls

| Key | Action |
|-----|--------|
| `Ctrl+B` then arrow | Switch pane |
| `Ctrl+B` then `d` | Detach (keeps running) |
| `tmux attach -t apyrobo` | Reattach |
| `Ctrl+B` then `z` | Zoom current pane |
| `Ctrl+C` | Stop current process in pane |

---

## Troubleshooting

**Gazebo doesn't open / black screen:**
- macOS: Make sure XQuartz is running and `xhost +localhost` was run
- Linux: Make sure `xhost +local:docker` was run
- Try headless: `bash scripts/launch.sh --no-gui`

**Nav2 fails to start:**
- It needs a map. Try the TurtleBot4-specific launch:
  `ros2 launch turtlebot4_navigation nav2.launch.py`
- Or use SLAM: `ros2 launch turtlebot4_navigation slam.launch.py`

**"No odometry received":**
- Gazebo might still be loading. Wait 30-60 seconds.
- Check: `ros2 topic echo /odom --once`

**Tests pass in mock but fail in Gazebo:**
- This is expected for timing-sensitive tests. Increase `--timeout`.
- Navigation in Gazebo is slower than mock — goals take real time.

**Docker build fails:**
- Check Docker has at least 8GB RAM allocated
- Run `docker system prune` to free space

---

## What's Next After the Demo

1. **Tag the release**: `git tag v0.1.0-mvp && git push --tags`
2. **Upload the recording** to YouTube or the README
3. **Publish benchmark results** in the README
4. **Apply for grants** with the demo video + metrics
5. **Open the repo** and announce on Discord / Reddit / HN
