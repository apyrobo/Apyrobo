# APYROBO Architecture

## Design Principles

1. **Build on ROS 2, don't replace it.** ROS 2 handles robot communication, drivers, navigation, and motion planning. APYROBO adds the semantic intelligence layer above it.

2. **Capability abstraction over hardware abstraction.** ROS 2 already provides hardware abstraction at the communication layer. APYROBO adds *semantic* capability discovery — knowing what a robot can do, not just what topics it publishes.

3. **Model-agnostic AI.** APYROBO works with any LLM provider (OpenAI, Anthropic, local models via LiteLLM). Swap providers without changing robot code.

4. **Safety is infrastructure, not application logic.** Safety constraints are enforced at the framework level. No AI agent can bypass them.

5. **Swarm-first.** Multi-robot coordination is a first-class module, not an afterthought.

---

## System Architecture Diagram

```
                        ┌────────────────────────┐
                        │   User / Application   │
                        │  "deliver to dock 3"   │
                        └───────────┬────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────┐
│                     APYROBO LAYER                             │
│                                                               │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────┐   │
│  │ AI Agent │───>│ Inference    │    │   Observability   │   │
│  │ (any LLM)│    │ Router       │    │                   │   │
│  │          │    │ ┌──────────┐ │    │  Metrics  Traces  │   │
│  │ rule /   │    │ │ edge     │ │    │  Alerts   Replay  │   │
│  │ llm /    │    │ │ cloud    │ │    │  Dashboard        │   │
│  │ tool_call│    │ │ fallback │ │    └───────────────────┘   │
│  └────┬─────┘    │ └──────────┘ │                            │
│       │          └──────────────┘                            │
│       ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                   Skill Graph Engine                  │    │
│  │                                                      │    │
│  │   navigate_to ──> pick_object ──> navigate_to        │    │
│  │       │                │              │              │    │
│  │  preconditions   postconditions   parameters         │    │
│  │  timeout/retry   state tracking   parallel exec      │    │
│  └──────────┬───────────────────────────────────────────┘    │
│             │                                                │
│             ▼                                                │
│  ┌──────────────────┐    ┌──────────────────────────────┐   │
│  │ Safety Enforcer   │    │     Swarm Coordinator        │   │
│  │                   │    │                              │   │
│  │ speed clamping    │    │  task splitting              │   │
│  │ collision zones   │    │  capability matching         │   │
│  │ watchdog (SF-05)  │    │  failure reassignment        │   │
│  │ battery check     │    │  proximity safety            │   │
│  │ escalation        │    │  deadlock detection          │   │
│  │ policy hot-swap   │    │  message bus                 │   │
│  └────────┬─────────┘    └──────────┬───────────────────┘   │
│           │                         │                        │
│           ▼                         ▼                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Capability Adapters                      │   │
│  │                                                      │   │
│  │  mock://     gazebo://    mqtt://     http://         │   │
│  │  (testing)   (sim)        (IoT)       (REST)         │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────┴───────────────────────────────┐   │
│  │              Sensor Pipelines                         │   │
│  │                                                      │   │
│  │  LiDAR  →  fusion  →  WorldState  →  obstacle map   │   │
│  │  Camera →           →  detections →  human tracking  │   │
│  │  IMU    →           →  odometry                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              State Persistence                        │   │
│  │                                                      │   │
│  │  JSON file  │  SQLite (OB-06)  │  Redis (OB-07)     │   │
│  │  (dev)      │  (production)    │  (distributed)      │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────┐
│                    ROS 2 MIDDLEWARE                            │
│                                                               │
│  Nav2        MoveIt       cmd_vel      tf2       DDS          │
│  (nav)       (manip)      (velocity)   (frames)  (transport)  │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│              SIMULATORS / HARDWARE                            │
│                                                               │
│  Gazebo    Isaac Sim    TurtleBot4    Custom ARM    Drones    │
└───────────────────────────────────────────────────────────────┘
```

---

## Module Map

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `core/` | Capability abstraction, robot discovery | `Robot`, `CapabilityAdapter`, `RobotCapability` |
| `skills/` | Skill graph engine, agent integration | `Skill`, `SkillGraph`, `SkillExecutor`, `Agent` |
| `safety/` | Hard safety constraints | `SafetyEnforcer`, `SafetyPolicy`, `SpeedProfile` |
| `swarm/` | Multi-robot coordination | `SwarmBus`, `SwarmCoordinator`, `SwarmSafety` |
| `sensors/` | Sensor fusion, world state | `SensorPipeline`, `WorldState` |
| `inference/` | LLM routing, circuit breakers | `InferenceRouter`, `TokenBudget`, `PlanCache` |
| `observability.py` | Metrics, tracing, alerting | `MetricsCollector`, `OTelExporter`, `AlertManager` |
| `persistence.py` | State persistence | `StateStore`, `SQLiteStateStore`, `RedisStateStore` |
| `dashboard.py` | HTTP API for monitoring | `Dashboard`, `create_app()` |

---

## Data Flow

```
User / AI Agent
       │
       ▼
  TaskRequest (natural language → skill plan)
       │
       ▼
  InferenceRouter (edge → cloud → fallback)
       │
       ▼
  Agent.plan() (LLM generates SkillGraph)
       │
       ▼
  SkillExecutor (runs skills in topological order)
       │
       ├── trace_context() wraps entire execution (OB-03)
       ├── emit_event() on each skill completion (OB-01)
       │
       ▼
  SafetyEnforcer (intercepts every command)
       │
       ├── speed clamping, collision zones, proximity
       ├── watchdog: odometry vs commanded position
       ├── battery check before each move
       │
       ▼
  CapabilityAdapter (translates to robot protocol)
       │
       ▼
  ROS 2 / Simulator / Hardware
```

---

## Safety Architecture

The safety enforcer is a **transparent wrapper** around the robot. It intercepts every command and enforces constraints before forwarding to the adapter.

```
Agent calls:   enforcer.move(x=5, y=5, speed=10)
                    │
                    ▼
               Battery check (SF-10)
                    │ PASS
                    ▼
               Speed clamping (SF-06) → speed = 1.5
                    │
                    ▼
               Collision zones (SF-07)
                    │ PASS
                    ▼
               Human proximity (SF-02)
                    │ PASS
                    ▼
               Start timeout timer (SF-01)
                    │
                    ▼
               robot.move(x=5, y=5, speed=1.5)  ← actual robot command
                    │
                    ▼
               Watchdog monitors odometry (SF-05)
```

If any check fails:
- **Violation** → `SafetyViolation` exception, command blocked
- **Intervention** → command modified (speed clamped), logged
- **Escalation** → robot stopped, webhook sent, wait for human ACK (SF-03)

---

## Swarm Architecture

```
┌─────────────────────────────────────────┐
│           SwarmCoordinator              │
│                                         │
│  task → split → assign → execute        │
│         │         │         │           │
│    capability   round     parallel      │
│    matching     robin     execution     │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│             SwarmBus                    │
│                                         │
│  robot_a ◄──── messages ────► robot_b   │
│     │                           │       │
│     └──── broadcast ────────────┘       │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│           SwarmSafety                   │
│                                         │
│  proximity checks   deadlock detection  │
│  position tracking  wait graph cycles   │
└─────────────────────────────────────────┘
```
