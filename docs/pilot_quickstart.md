# Pilot Quickstart — Zero to First Task in 30 Minutes

One path. No options. Follow every step in order.

**Target:** You will command a simulated TurtleBot4 to deliver a package
across a warehouse — planned by an AI agent, enforced by safety constraints,
with live sensor data.

**Stack:** Docker, ROS 2 Humble, Gazebo, Nav2, APYROBO.

---

## Prerequisites

| Requirement | Why |
|---|---|
| **Docker Desktop** | Everything runs in a container — no local ROS 2 install needed |
| **8 GB RAM** allocated to Docker | Gazebo + Nav2 are memory-hungry |
| **Git** | To clone the repo |

Optional (for the Gazebo GUI window):
- **macOS:** XQuartz (`brew install --cask xquartz`)
- **Linux:** X11 is already there

> No GPU required. Gazebo runs in software rendering mode by default.

---

## Step 1: Clone the repo

```bash
git clone https://github.com/nicobrenner/apyrobo.git
cd apyrobo
```

---

## Step 2: Build the Docker image

```bash
docker compose -f docker/docker-compose.yml build
```

First build downloads ~3 GB (ROS 2 + Gazebo + Nav2 + Python deps).
Takes 15–20 minutes on a typical connection. Subsequent builds are cached.

---

## Step 3: Start the container and get a shell

```bash
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml exec apyrobo bash
```

You are now inside the container at `/workspace` (the repo root).

---

## Step 4: Run the preflight checker

```bash
python3 scripts/preflight.py --ros
```

Expected output (all green before launching Gazebo):

```
  APYROBO Preflight Checklist
  ========================================

  [✓] Python import
  [✓] Mock pipeline
  [✓] Config file
  [✓] Safety policy
  [✓] ROS 2 available
  [X] Gazebo topics: missing /odom, /cmd_vel, /scan
      Run: bash scripts/launch.sh
  [X] Nav2 ready: navigate_to_pose action server not responding
      Run: bash scripts/launch.sh  (wait ~40s for Nav2)
  [-] LLM reachable: no LLM configured (using rule-based agent)
```

The first four checks must be green. Gazebo/Nav2 failures are expected —
you haven't launched them yet. That's the next step.

---

## Step 5: Launch Gazebo + Nav2

```bash
bash scripts/launch.sh
```

This opens a 4-pane tmux session:

| Pane | What's running |
|------|---------------|
| 0 | Gazebo + TurtleBot4 simulator |
| 1 | Nav2 navigation stack |
| 2 | APYROBO shell (your command line) |
| 3 | Sensor topic monitor |

**Wait 40–60 seconds** for Gazebo to fully load. You'll see the warehouse
world appear (if GUI is enabled) and Nav2 will print "Lifecycle node
active" when ready.

Tmux controls:
- `Ctrl+B` then arrow keys — switch panes
- `Ctrl+B` then `d` — detach (everything keeps running)
- `tmux attach -t apyrobo` — reattach later

---

## Step 6: Verify the ROS 2 stack

In the APYROBO pane (Pane 2):

```bash
python3 scripts/integration_test.py
```

This runs 9 tests against the live simulator: ROS 2 connectivity, robot
discovery, capabilities, odometry, sensors, navigation, safety, skills,
and agent planning. All should pass.

If navigation tests fail, increase the timeout:

```bash
python3 scripts/integration_test.py --timeout 60
```

---

## Step 7: Run your first real task

```bash
python3 scripts/demo_gazebo.py
```

Default task: *"deliver package from (1, 0) to (3, 0)"*

You'll see the agent plan the task, the robot navigate with live status
updates, and the safety enforcer clamp speeds. Output looks like:

```
  ✓ Robot: turtlebot4
  ✓ Sensor pipeline online (4 subscribers)
  ✓ Safety enforcer active: max_speed=0.8 m/s
  ✓ Agent ready

  ══════════════════════════════════════
    Task: deliver package from (1, 0) to (3, 0)
  ══════════════════════════════════════

  ⏳ [  0.0s] navigate_to: Planning route...
  🔄 [  1.2s] navigate_to: Moving to (1.0, 0.0)
  ✅ [  8.4s] navigate_to: Arrived
  ⏳ [  8.5s] report_status: Generating report...
  ✅ [  9.0s] report_status: Complete

  ✓ COMPLETED in 9.0s (2/2 steps)
  ✓ No safety violations
  ✓ Demo complete!
```

Try a custom task:

```bash
python3 scripts/demo_gazebo.py --task "go to (3, 2)"
```

Or interactive mode — type tasks one at a time:

```bash
python3 scripts/demo_gazebo.py --interactive
```

---

## You're done

You just:

1. Built the full APYROBO stack from source
2. Validated the environment with the preflight checker
3. Launched a simulated robot with navigation
4. Verified the entire pipeline end-to-end
5. Commanded a robot with a natural-language task

---

## What's next

| Goal | Command / Path |
|------|---------------|
| Record a demo video | `python3 scripts/record_demo.py` |
| Run benchmarks | `python3 scripts/record_demo.py --benchmark 20` |
| Write a custom skill | See `docs/skill_authoring.md` |
| Write a custom adapter | See `docs/adapter_authoring.md` |
| Understand the architecture | See `docs/architecture.md` |
| Full API reference | See `docs/api_reference.md` |

---

## Troubleshooting

**Preflight fails on "Python import":**
You're probably outside the container. Run `docker compose -f docker/docker-compose.yml exec apyrobo bash` first.

**Gazebo doesn't open / black screen:**
- macOS: Open XQuartz, Preferences > Security > check "Allow network clients", restart XQuartz, then run `xhost +localhost` before `docker compose up`.
- Linux: Run `xhost +local:docker` before `docker compose up`.
- Skip the GUI entirely: `bash scripts/launch.sh --no-gui`

**Nav2 fails to start:**
- It may need a map. Try: `ros2 launch turtlebot4_navigation slam.launch.py` in Pane 1.
- Or wait longer — Nav2 can take 30–60s to initialize.

**"No odometry received" / integration test hangs:**
- Gazebo is still loading. Wait 60 seconds and retry.
- Check: `ros2 topic echo /odom --once`

**Docker build fails:**
- Ensure Docker has at least 8 GB RAM (Docker Desktop > Settings > Resources).
- Free disk space: `docker system prune`

**LLM not working:**
- By default APYROBO uses a rule-based agent (no LLM needed).
- To use an LLM: `export OPENAI_API_KEY=sk-...` and set `agent.provider: llm` in `apyrobo.yaml`.
