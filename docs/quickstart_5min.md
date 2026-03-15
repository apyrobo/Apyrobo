# APYROBO — 5-Minute Quickstart

Get a mock robot running and execute your first AI-planned task — no ROS 2, no Docker, no hardware required.

---

## Install (30 seconds)

```bash
pip install -e .
```

Or from the repo:

```bash
git clone https://github.com/apyrobo/apyrobo.git
cd apyrobo
pip install -e ".[dev]"
```

---

## Step 1: Discover a Mock Robot (1 minute)

```python
from apyrobo import Robot

# No ROS 2 needed — the mock adapter simulates a TurtleBot4
robot = Robot.discover("mock://turtlebot4")

# What can it do?
caps = robot.capabilities()
print(f"Robot: {caps.name}")
print(f"Capabilities: {[c.name for c in caps.capabilities]}")
# → ['navigate_to', 'rotate', 'pick_object', 'place_object']

# Move it
robot.move(x=2.0, y=3.0, speed=0.5)
print(f"Position: {robot.get_position()}")
# → Position: (2.0, 3.0)

robot.stop()
```

---

## Step 2: Execute a Skill Graph (2 minutes)

```python
from apyrobo import Robot, SkillExecutor, SkillGraph, BUILTIN_SKILLS

robot = Robot.discover("mock://turtlebot4")

# Build a task plan: navigate → pick → navigate → place
graph = SkillGraph()
graph.add_skill(BUILTIN_SKILLS["navigate_to"],
                parameters={"x": 3.0, "y": 4.0, "speed": 0.5})
graph.add_skill(BUILTIN_SKILLS["pick_object"],
                depends_on=["navigate_to"])

# Execute with event streaming
executor = SkillExecutor(robot)
executor.on_event(lambda e: print(f"  [{e.status.value}] {e.skill_id}: {e.message}"))

result = executor.execute_graph(graph)
print(f"\nResult: {result.status.value} ({result.steps_completed}/{result.steps_total} steps)")
# → Result: completed (2/2 steps)
```

---

## Step 3: Use an AI Agent (2 minutes)

```python
from apyrobo import Robot, Agent

robot = Robot.discover("mock://turtlebot4")

# Rule-based agent (no API key needed)
agent = Agent(provider="rule")

# Natural language → skill plan → execution
result = agent.execute("go to position 5, 3 then pick up the object", robot)
print(f"Task: {result.task_name}")
print(f"Status: {result.status}")
print(f"Steps: {result.steps_completed}/{result.steps_total}")
```

To use an LLM provider:

```python
# With OpenAI (requires OPENAI_API_KEY env var)
agent = Agent(provider="llm", model="gpt-4")

# With Anthropic
agent = Agent(provider="llm", model="claude-sonnet-4-20250514")

# With any LiteLLM-supported model
agent = Agent(provider="llm", model="ollama/llama3")
```

---

## Step 4: Add Safety Enforcement

```python
from apyrobo import Robot, SafetyEnforcer, SafetyPolicy

robot = Robot.discover("mock://turtlebot4")

# Safety enforcer wraps the robot — transparent to agents
enforcer = SafetyEnforcer(robot, policy="default")

# Speed is automatically clamped
enforcer.move(x=5.0, y=5.0, speed=100.0)  # → clamped to 1.5 m/s
print(f"Interventions: {enforcer.interventions}")

# Collision zones block movement
enforcer.add_collision_zone({"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2})
try:
    enforcer.move(x=1.0, y=1.0)
except Exception as e:
    print(f"Blocked: {e}")
```

---

## Step 5: Multi-Robot Swarm

```python
from apyrobo import Robot, SwarmBus, SwarmCoordinator, Agent

# Create a swarm
bus = SwarmBus()
bus.register(Robot.discover("mock://robot_alpha"))
bus.register(Robot.discover("mock://robot_beta"))

print(f"Swarm: {bus.robot_count} robots")

# Coordinator splits tasks across robots
coordinator = SwarmCoordinator(bus)
agent = Agent(provider="rule")

result = coordinator.execute_task("deliver package to dock 3", agent)
print(f"Result: {result.status}")
```

---

## What's Next

| Goal | Guide |
|------|-------|
| Write custom skills | [Skill Authoring Guide](docs/skill_authoring.md) |
| Add new robot hardware | [Adapter Authoring Guide](docs/adapter_authoring.md) |
| Run with real ROS 2 | [Full Docker Setup](docs/QUICKSTART.md) |
| Understand the architecture | [Architecture Overview](docs/architecture.md) |
| API reference | [API Docs](docs/api_reference.md) |

---

## Full Docker Setup (with ROS 2 + Gazebo)

For the full simulation experience with Gazebo:

```bash
# Build the Docker container (downloads ROS 2 + Gazebo + deps)
docker compose -f docker/docker-compose.yml build

# Start and enter the container
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml exec apyrobo bash

# Launch Gazebo + Nav2 + APYROBO
bash scripts/launch.sh

# Run integration tests against live Gazebo
python3 scripts/integration_test.py
```

See the [full Docker quickstart](docs/QUICKSTART.md) for GUI setup and troubleshooting.
