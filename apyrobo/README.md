<p align="center">
  <strong>APYROBO</strong><br>
  <em>The open-source AI orchestration layer for robotics</em>
</p>

<p align="center">
  <a href="https://github.com/apyrobo/apyrobo/actions/workflows/ci.yml"><img src="https://github.com/apyrobo/apyrobo/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/apyrobo/apyrobo/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python"></a>
  <a href="https://docs.ros.org/en/humble/"><img src="https://img.shields.io/badge/ROS%202-Humble-green.svg" alt="ROS 2"></a>
  <a href="#"><img src="https://img.shields.io/badge/status-pre--alpha-orange.svg" alt="Status"></a>
</p>

---

**APYROBO gives AI agents the runtime to act in the physical world.** It sits on top of [ROS 2](https://docs.ros.org/en/humble/) and provides capability discovery, skill orchestration, swarm coordination, and safety enforcement. One layer, any hardware, any LLM.

```
"deliver the package to dock 3"
        │
        ▼
   ┌─────────┐    ┌───────────┐    ┌──────────┐    ┌─────────┐
   │ AI Agent │ →  │ Skill     │ →  │ Safety   │ →  │ ROS 2 / │
   │ (any LLM)│    │ Graph     │    │ Enforcer │    │ Hardware │
   └─────────┘    └───────────┘    └──────────┘    └─────────┘
```

### 30-Second Demo

```python
from apyrobo import Robot, Agent

robot = Robot.discover("mock://turtlebot4")        # No ROS 2 needed
agent = Agent(provider="rule")                      # No API key needed
result = agent.execute("go to 3, 4 and pick up the object", robot)
print(result.status)  # → completed
```

---

## Why APYROBO

| Challenge | How APYROBO Solves It |
|-----------|----------------------|
| LLMs can't control robots safely | Safety enforcer wraps every command with hard constraints |
| Robot code is tied to hardware | Capability adapters abstract any robot behind a semantic API |
| Multi-robot coordination is ad-hoc | Swarm bus + coordinator handles task splitting natively |
| Skill composition is manual | Skill graph engine chains skills with pre/postconditions |
| No standard AI-robotics interface | Model-agnostic agent layer works with any LLM provider |

## Features

- **Capability Abstraction** — Semantic API that knows *what* robots can do, not just what topics they publish
- **AI Agent Orchestration** — Natural language to verified skill execution with any LLM (OpenAI, Anthropic, local)
- **Skill Graph Engine** — DAG-based task plans with precondition/postcondition verification and retry logic
- **Safety Enforcement** — Speed clamping, collision zones, watchdog, battery checks, escalation — agents can't bypass
- **Swarm Coordination** — Multi-robot task splitting, proximity safety, deadlock detection
- **Observability** — Structured JSON logging, Prometheus metrics, OpenTelemetry traces, execution replay
- **Multiple Persistence Backends** — JSON file, SQLite, Redis for state that survives crashes

---

## Quick Start (5 minutes, no ROS 2)

### Install

```bash
git clone https://github.com/apyrobo/apyrobo.git
cd apyrobo
pip install -e ".[dev]"
```

### Discover and Command

```python
from apyrobo import Robot

robot = Robot.discover("mock://turtlebot4")
caps = robot.capabilities()
print(f"Capabilities: {[c.name for c in caps.capabilities]}")
# → ['navigate_to', 'rotate', 'pick_object', 'place_object']

robot.move(x=2.0, y=3.0, speed=0.5)
print(robot.get_position())  # → (2.0, 3.0)
```

### Execute a Skill Graph

```python
from apyrobo import Robot, SkillExecutor, SkillGraph, BUILTIN_SKILLS

robot = Robot.discover("mock://turtlebot4")
graph = SkillGraph()
graph.add_skill(BUILTIN_SKILLS["navigate_to"], parameters={"x": 3.0, "y": 4.0})
graph.add_skill(BUILTIN_SKILLS["pick_object"], depends_on=["navigate_to"])

executor = SkillExecutor(robot)
result = executor.execute_graph(graph)
print(f"{result.status.value}: {result.steps_completed}/{result.steps_total} steps")
# → completed: 2/2 steps
```

### Use an AI Agent

```python
from apyrobo import Robot, Agent

robot = Robot.discover("mock://turtlebot4")
agent = Agent(provider="rule")  # or "llm" with any LiteLLM model
result = agent.execute("go to 5, 3 then pick up the object", robot)
```

See the [full quickstart guide](docs/quickstart_5min.md) for safety enforcement, swarm coordination, and LLM setup.

---

## Architecture

```
┌─────────────────────────────────────────┐
│         Foundation Models / LLMs        │  Any provider (OpenAI, Anthropic, local)
├─────────────────────────────────────────┤
│  ┌─────────────────────────────────┐    │
│  │    APYROBO Orchestration Layer  │    │  This project
│  │                                 │    │
│  │  Capability   Skill     Swarm   │    │
│  │  Adapters     Graph     Coord   │    │
│  │                                 │    │
│  │  Sensor       Safety    Agent   │    │
│  │  Pipelines    Enforcer  Runtime │    │
│  │                                 │    │
│  │  Inference    Observ-   State   │    │
│  │  Router       ability   Store   │    │
│  └─────────────────────────────────┘    │
├─────────────────────────────────────────┤
│     ROS 2 (DDS, Nav2, MoveIt, TF2)     │  Industry standard, not replaced
├─────────────────────────────────────────┤
│   Simulators (Gazebo, Isaac Sim)        │
├─────────────────────────────────────────┤
│        Physical Hardware                │
└─────────────────────────────────────────┘
```

---

## Project Structure

```
apyrobo/
├── apyrobo/              # Main package
│   ├── core/             # Capability abstraction, robot discovery, adapters
│   ├── skills/           # Skill graph engine, executor, AI agent integration
│   ├── safety/           # Safety policy enforcement, watchdog, escalation
│   ├── swarm/            # Multi-robot bus, coordinator, proximity/deadlock
│   ├── sensors/          # Sensor pipelines, fusion, world state
│   ├── inference/        # Multi-tier LLM routing, circuit breakers
│   ├── observability.py  # Structured logging, metrics, tracing, alerting
│   ├── persistence.py    # State store (JSON, SQLite, Redis)
│   └── dashboard.py      # FastAPI metrics/health dashboard
├── tests/                # 123+ pytest tests (skills, safety, swarm, chaos)
├── docs/                 # Guides, architecture, API reference
├── docker/               # Dockerfile + docker-compose (ROS 2 + Gazebo)
├── examples/             # Usage examples and demos
└── .github/workflows/    # CI pipeline, nightly builds
```

---

## Adapters

APYROBO works with any robot through capability adapters:

| Adapter | URI Scheme | Use Case |
|---------|-----------|----------|
| `MockAdapter` | `mock://` | Unit testing, development |
| `GazeboAdapter` | `gazebo://` | Simulation with physics |
| `MQTTAdapter` | `mqtt://` | IoT / remote robots |
| `HTTPAdapter` | `http://` | REST-based robot APIs |

Write your own: see the [Adapter Authoring Guide](docs/adapter_authoring.md).

---

## Documentation

| Document | Description |
|----------|-------------|
| [5-Minute Quickstart](docs/quickstart_5min.md) | Install + mock robot + first task |
| [Full Docker Setup](docs/QUICKSTART.md) | ROS 2 + Gazebo simulation |
| [Architecture](docs/architecture.md) | Design principles and data flow |
| [Skill Authoring Guide](docs/skill_authoring.md) | Write, test, and publish custom skills |
| [Adapter Authoring Guide](docs/adapter_authoring.md) | Add support for new hardware |
| [API Reference](docs/api_reference.md) | Auto-generated from docstrings |
| [APYROBO vs Alternatives](docs/comparison.md) | Comparison with RAI, ROS-LLM |
| [Roadmap](ROADMAP.md) | Public milestones and contribution areas |

---

## Roadmap

| Phase | Module | Status |
|-------|--------|--------|
| 1 | Core — Capability Abstraction Layer | Done |
| 2 | Skills — Skill Graph + Agent Integration | Done |
| 3 | Execution + Safety Enforcement | Done |
| 4 | Swarm — Multi-Robot Coordination | Done |
| 5 | Sensor Pipelines + Inference Routing | Done |
| 6 | Observability + Persistence | Done |
| 7 | CI/CD + Testing Infrastructure | Done |
| 8 | Documentation + Developer Experience | In progress |
| 9 | MVP Demo + Launch | Planned |

See [ROADMAP.md](ROADMAP.md) for the full technical roadmap with contribution opportunities.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

```bash
# Run the test suite
pip install -e ".[dev]"
pytest tests/ -v -o "addopts="
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

<p align="center">
  <strong>APYROBO</strong> · Built on ROS 2 · Open Source · Any Hardware · Any LLM
</p>
