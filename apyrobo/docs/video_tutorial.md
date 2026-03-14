# APYROBO Video Tutorial Script

Production guide for a 12-minute "Build Your First Robot AI" video tutorial.

---

## Overview

| Field | Value |
|-------|-------|
| **Title** | Build Your First Robot AI in 12 Minutes with APYROBO |
| **Target audience** | Robotics engineers, ROS 2 developers, AI/ML engineers |
| **Prerequisites** | Python 3.10+, basic ROS 2 familiarity (optional) |
| **Format** | Screencast with voiceover, code editor + terminal split |
| **Software** | VS Code, terminal, Python 3.10+, pip |

---

## Script

### 0:00–0:30 — Hook

**Visual:** Robot arm picking up an object in Gazebo, then a terminal showing 5 lines of Python.

**Voiceover:**
> "What if you could go from a natural language command to a robot executing a multi-step task — with safety constraints, observability, and crash recovery — in under 50 lines of Python? That's APYROBO. Let's build it."

---

### 0:30–1:30 — What Is APYROBO?

**Visual:** Architecture diagram from docs/architecture.md (simplified).

**Voiceover:**
> "APYROBO is an AI orchestration layer for robotics. It sits on top of ROS 2 and adds four things that ROS 2 doesn't give you out of the box:
> 1. Model-agnostic AI planning — use any LLM or no LLM at all
> 2. A skill graph engine — chain robot actions with preconditions, timeouts, and retries
> 3. Hard safety enforcement — speed clamping, collision zones, human proximity checks
> 4. Multi-robot coordination — task splitting, failure reassignment, deadlock detection
>
> Think of it as the brain between the LLM and the robot."

---

### 1:30–3:00 — Install & First Robot

**Visual:** Terminal, typing commands.

```bash
pip install apyrobo
python3
```

```python
from apyrobo import Robot

# Discover a mock robot (no hardware needed)
robot = Robot.discover("mock://my_bot")
print(robot.robot_id)          # "my_bot"
print(robot.capabilities())    # navigation, rotation, gripper...

# Move it
robot.move(x=3.0, y=4.0, speed=0.5)
print(robot.get_position())    # (3.0, 4.0)
```

**Voiceover:**
> "Install with pip, import Robot, and discover a mock robot. The mock adapter runs entirely in-memory — no ROS, no simulator, no hardware. Perfect for development and testing.
>
> The `discover()` method uses URI schemes. `mock://` for testing, `gazebo://` for simulation, `mqtt://` for IoT robots, `http://` for REST APIs. You can write your own adapter for any protocol."

---

### 3:00–5:00 — Skill Graphs

**Visual:** VS Code editor, building a skill graph.

```python
from apyrobo import Robot, SkillGraph, SkillExecutor, BUILTIN_SKILLS

robot = Robot.discover("mock://warehouse_bot")

# Build a pick-and-place task
graph = SkillGraph()
graph.add_skill(BUILTIN_SKILLS["navigate_to"],
                parameters={"x": 5.0, "y": 3.0, "speed": 0.8})
graph.add_skill(BUILTIN_SKILLS["pick_object"],
                depends_on=["navigate_to"])
graph.add_skill(BUILTIN_SKILLS["navigate_to"],
                parameters={"x": 0.0, "y": 0.0, "speed": 0.5})
graph.add_skill(BUILTIN_SKILLS["place_object"],
                depends_on=["navigate_to"])

# See the execution plan
for layer in graph.get_execution_layers():
    print([s.name for s in layer])

# Execute it
executor = SkillExecutor(robot)
result = executor.execute_graph(graph)
print(result.status)  # "completed"
```

**Voiceover:**
> "Skills are the building blocks. Each one has preconditions, postconditions, parameters, timeouts, and retry counts.
>
> You chain them into a directed acyclic graph — a skill graph. The executor runs them in topological order, respecting dependencies. Independent skills can run in parallel.
>
> APYROBO ships with built-in skills for navigation, rotation, pick/place, and status reporting. You can write your own — see the skill authoring guide."

---

### 5:00–7:00 — AI Planning

**Visual:** Code editor, using the Agent.

```python
from apyrobo import Agent, Robot

robot = Robot.discover("mock://warehouse_bot")

# Rule-based agent (no API key needed)
agent = Agent(provider="rule")
result = agent.execute("deliver package to dock 3", robot)
print(result.status)       # "completed"
print(result.skills_run)   # ["navigate_to", "pick_object", ...]

# LLM-powered agent (any provider via LiteLLM)
agent = Agent(provider="llm", model="gpt-4o")
result = agent.execute("scan the warehouse perimeter", robot)
```

**Voiceover:**
> "The Agent takes a natural language task, plans a skill graph, and executes it. The rule-based provider works offline with no API key — great for deterministic tasks.
>
> Switch to an LLM provider for open-ended planning. APYROBO uses LiteLLM under the hood, so you can use OpenAI, Anthropic, or any local model. The agent generates the skill graph, the executor runs it — the LLM never talks directly to the robot."

---

### 7:00–9:00 — Safety Enforcement

**Visual:** Code editor + terminal showing safety violations.

```python
from apyrobo import Robot, SafetyEnforcer, SafetyPolicy

robot = Robot.discover("mock://warehouse_bot")

# Default safety policy
enforcer = SafetyEnforcer(robot)

# Speed is clamped automatically
enforcer.move(x=10.0, y=10.0, speed=99.0)
# Actually moves at 1.5 m/s (policy max)

# Add a no-go zone
enforcer.add_collision_zone({
    "x_min": 4.0, "x_max": 6.0,
    "y_min": 4.0, "y_max": 6.0,
})
# enforcer.move(x=5.0, y=5.0)  # raises SafetyViolation!

# Strict policy for sensitive environments
enforcer_strict = SafetyEnforcer(robot, policy="strict")
# max_speed=0.5, human_proximity_limit=1.0
```

**Voiceover:**
> "Every command goes through the safety enforcer before reaching the robot. No agent — AI or human — can bypass it.
>
> Speed clamping, collision zones, human proximity checks, battery awareness, odometry watchdog, and human escalation. All configurable via policies, and you can hot-swap policies at runtime.
>
> This is the killer feature for real-world deployment. An LLM hallucination might plan a crazy speed — the enforcer catches it."

---

### 9:00–10:30 — Multi-Robot Swarm

**Visual:** Code showing SwarmCoordinator with multiple robots.

```python
from apyrobo import Robot, SwarmBus, SwarmCoordinator, Agent

# Create a fleet
robots = [
    Robot.discover("mock://bot_1"),
    Robot.discover("mock://bot_2"),
    Robot.discover("mock://bot_3"),
]

# Set up swarm
bus = SwarmBus()
for r in robots:
    bus.register(r)

coordinator = SwarmCoordinator(bus, robots)
agent = Agent(provider="rule")

# Split and execute across the fleet
result = coordinator.execute_task(
    "deliver packages to docks 1, 2, and 3",
    agent,
)
print(result.status)  # "completed"
print(coordinator.assignments)  # which robot got which sub-task
```

**Voiceover:**
> "APYROBO is swarm-first. The SwarmCoordinator splits tasks across robots based on capabilities, monitors progress, and reassigns on failure.
>
> The SwarmBus handles inter-robot messaging — targeted or broadcast. SwarmSafety adds proximity checks and deadlock detection.
>
> This is something neither RAI nor ROS-LLM can do."

---

### 10:30–11:30 — Observability & Persistence

**Visual:** Terminal showing metrics, then a dashboard screenshot.

```python
from apyrobo.observability import MetricsCollector, on_event

# Plug into the event bus
collector = MetricsCollector()
on_event(collector.handle_event)

# After running tasks...
print(collector.prometheus_text())
# apyrobo_skills_total{skill_id="navigate_to"} 5
# apyrobo_skill_duration_ms{skill_id="navigate_to"} 234.5
# apyrobo_skill_failures_total{skill_id="pick_object"} 1

# State persistence for crash recovery
from apyrobo.persistence import SQLiteStateStore
store = SQLiteStateStore("robot_state.db")
store.begin_task("task_001", {"description": "deliver"}, "bot_1", 3)
# If the process crashes, get_interrupted_tasks() returns it
```

**Voiceover:**
> "Every skill execution emits events. Plug in a MetricsCollector for Prometheus, an OTel exporter for distributed tracing, or an AlertManager for threshold-based alerts.
>
> State persistence means crash recovery. If your process dies mid-task, the SQLite or Redis backend knows exactly where you left off."

---

### 11:30–12:00 — Next Steps

**Visual:** Links to documentation.

**Voiceover:**
> "That's APYROBO in 12 minutes. You've seen discovery, skills, AI planning, safety, swarm coordination, and observability.
>
> To go deeper:
> - Read the 5-minute quickstart for a hands-on walkthrough
> - Check the skill authoring guide to write custom skills
> - Read the adapter guide to connect your own hardware
> - Browse the API reference for the full surface area
>
> Star us on GitHub, file issues, and join the community. Happy building."

---

## Production Notes

### Recording Setup
- **Screen resolution:** 1920×1080
- **Font size:** 18pt in editor, 16pt in terminal
- **Theme:** Dark (VS Code Dark+)
- **Terminal:** Split pane (editor left, terminal right)
- **Zoom:** 125% browser zoom for dashboard shots

### Editing
- Cut dead air, aim for <13 minutes total
- Add chapter markers for YouTube
- Highlight code blocks with yellow box overlay when referencing
- Speed up pip install and long outputs at 4x

### Chapters (YouTube)
```
0:00 Hook
0:30 What Is APYROBO?
1:30 Install & First Robot
3:00 Skill Graphs
5:00 AI Planning
7:00 Safety Enforcement
9:00 Multi-Robot Swarm
10:30 Observability & Persistence
11:30 Next Steps
```

### Thumbnail
- Text: "Robot AI in 12 min"
- Visual: Robot arm + Python code snippet + APYROBO logo
- Colors: Dark background, green/blue accent
