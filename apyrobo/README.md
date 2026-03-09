# APYROBO

**The open-source AI orchestration layer for robotics.**

APYROBO sits on top of [ROS 2](https://docs.ros.org/en/humble/) and gives AI agents the runtime to act in the physical world — capability discovery, skill orchestration, swarm coordination, and safety enforcement. One layer, any hardware.

> **Status: Pre-alpha.** We're building in the open. Star the repo to follow along.

---

## What APYROBO Is (and Isn't)

**APYROBO is not a replacement for ROS 2.** ROS 2 is the industry-standard middleware for robot communication, and its ecosystem of drivers, navigation stacks, and motion planners represents over 15 years of community investment.

**APYROBO is the semantic intelligence layer above ROS 2.** It provides what ROS 2 doesn't:

- **Capability Abstraction** — A semantic API that knows *what* a robot can do (navigate, pick, place, scan), not just what topics it publishes.
- **AI Agent Orchestration** — Turn natural language into verified skill execution with any LLM provider (OpenAI, Anthropic, local models).
- **Skill Graph Engine** — Chain reusable robot skills into complex task plans with precondition/postcondition verification.
- **Swarm Coordination** — Multi-robot task distribution as a first-class concern, not an afterthought.
- **Safety Enforcement** — Hard constraints at the framework level that no AI agent can bypass.

---

## Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Apple Silicon or x86)
- [XQuartz](https://www.xquartz.org/) (macOS only, for Gazebo GUI)
- Git

### Build & Run

```bash
git clone https://github.com/apyrobo/apyrobo.git
cd apyrobo

# Build the development container
docker compose -f docker/docker-compose.yml build

# Start the container
docker compose -f docker/docker-compose.yml up -d

# Open a shell inside the container
docker compose -f docker/docker-compose.yml exec apyrobo bash

# Run tests
pytest
```

### Try It (Mock Mode — No ROS 2 Required)

```python
from apyrobo.core.robot import Robot

# Discover a mock robot (works without ROS 2)
robot = Robot.discover("mock://turtlebot4")

# Query what it can do
caps = robot.capabilities()
print(f"Robot: {caps.name}")
print(f"Capabilities: {[c.name for c in caps.capabilities]}")
print(f"Sensors: {[s.sensor_id for s in caps.sensors]}")

# Command it
robot.move(x=2.0, y=3.0, speed=0.5)
robot.stop()
```

---

## Architecture

```
┌─────────────────────────────────────────┐
│         Foundation Models / LLMs        │  ← Any provider (OpenAI, Anthropic, local)
├─────────────────────────────────────────┤
│          AI Agents (LangChain, etc.)    │
├─────────────────────────────────────────┤
│  ┌─────────────────────────────────┐    │
│  │     APYROBO Orchestration Layer │    │  ← This project
│  │  ┌───────┐ ┌───────┐ ┌───────┐ │    │
│  │  │ Caps  │ │Skills │ │Swarm  │ │    │
│  │  │  API  │ │ Graph │ │ Coord │ │    │
│  │  └───────┘ └───────┘ └───────┘ │    │
│  │  ┌───────┐ ┌───────┐ ┌───────┐ │    │
│  │  │Sensor │ │Safety │ │ Agent │ │    │
│  │  │Pipes  │ │Enforc │ │Runtime│ │    │
│  │  └───────┘ └───────┘ └───────┘ │    │
│  └─────────────────────────────────┘    │
├─────────────────────────────────────────┤
│     ROS 2 (DDS, Nav2, MoveIt, TF2)     │  ← Industry standard, not replaced
├─────────────────────────────────────────┤
│   Simulators (Gazebo, Isaac Sim, etc.)  │
├─────────────────────────────────────────┤
│        Physical Hardware                │
└─────────────────────────────────────────┘
```

---

## Project Structure

```
apyrobo/
├── apyrobo/              # Main Python package
│   ├── core/             # Phase 1: Capability abstraction layer
│   │   ├── schemas.py    #   Pydantic models (RobotCapability, TaskRequest, etc.)
│   │   ├── robot.py      #   Robot discovery and command interface
│   │   └── adapters.py   #   Capability adapters (Mock, TurtleBot4, etc.)
│   ├── skills/           # Phase 2: Skill graph engine + agent integration
│   ├── safety/           # Phase 3: Safety policy enforcement
│   ├── swarm/            # Phase 4: Multi-robot coordination
│   ├── sensors/          # Phase 5: Sensor pipelines
│   └── sim/              # Simulation connectors
├── tests/                # pytest test suite
│   └── test_core/        #   Core module tests (schemas, robot, adapters)
├── docker/               # Dockerfile + docker-compose
├── docs/                 # Architecture documentation
├── examples/             # Usage examples
├── pyproject.toml        # Python package config
├── LICENSE               # Apache 2.0
└── README.md
```

---

## Roadmap

| Phase | Module | Status |
|-------|--------|--------|
| 0 | Environment Setup (Docker, ROS 2, Gazebo) | 🟡 In progress |
| 1 | Core — Capability Abstraction Layer | 🟡 In progress |
| 2 | Skills — Skill Graph Engine + Agent Integration | ⬜ Planned |
| 3 | Execution Model + Safety Enforcement | ⬜ Planned |
| 4 | Swarm — Multi-Robot Coordination | ⬜ Planned |
| 5 | Sensor Pipelines | ⬜ Planned |
| 6 | MVP Demo + Launch | ⬜ Planned |

See `ROADMAP.md` for the full technical roadmap.

---

## Contributing

We're not yet accepting external contributions (building privately until MVP demo), but we will be soon. If you're interested, star the repo and join the [Discord](#) to follow along.

See `CONTRIBUTING.md` for guidelines once contributions open.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

**APYROBO** · Built on ROS 2 · Open Source
