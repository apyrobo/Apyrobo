# APYROBO Architecture

## Design Principles

1. **Build on ROS 2, don't replace it.** ROS 2 handles robot communication, drivers, navigation, and motion planning. APYROBO adds the semantic intelligence layer above it.

2. **Capability abstraction over hardware abstraction.** ROS 2 already provides hardware abstraction at the communication layer. APYROBO adds *semantic* capability discovery — knowing what a robot can do, not just what topics it publishes.

3. **Model-agnostic AI.** APYROBO works with any LLM provider (OpenAI, Anthropic, local models via LiteLLM). Swap providers without changing robot code.

4. **Safety is infrastructure, not application logic.** Safety constraints are enforced at the framework level. No AI agent can bypass them.

5. **Swarm-first.** Multi-robot coordination is a first-class module, not an afterthought.

## Module Map

| Module | Purpose | Phase |
|--------|---------|-------|
| `core/` | Capability abstraction, robot discovery, ROS 2 bridge | 1 |
| `skills/` | Skill definitions, graph engine, agent integration | 2 |
| `safety/` | Safety policies, enforcement, speed caps, collision zones | 3 |
| `swarm/` | Multi-robot messaging, task distribution, deadlock detection | 4 |
| `sensors/` | Sensor pipelines, fusion, world-state representation | 5 |
| `sim/` | Simulation connectors (Gazebo, future: Isaac Sim, MuJoCo) | 0-1 |

## Data Flow

```
User / AI Agent
       │
       ▼
  TaskRequest (natural language → skill plan)
       │
       ▼
  SkillExecutor (runs skills in sequence)
       │
       ▼
  SafetyEnforcer (intercepts every command)
       │
       ▼
  CapabilityAdapter (translates to ROS 2)
       │
       ▼
  ROS 2 (Nav2, MoveIt, cmd_vel, etc.)
       │
       ▼
  Robot Hardware / Simulator
```
