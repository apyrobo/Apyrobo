# APYROBO vs RAI vs ROS-LLM: Robot AI Framework Comparison

Choosing the right AI-robotics framework depends on your use case. This comparison covers the three leading open-source options as of 2025-2026.

---

## Which robot MCP server fits you

If you want a model (through MCP) to drive a robot, these are the projects we
know of that sit between the model and ROS. Each cell is what the project's own
README or code states at the commit named, read 18 November 2026; "not stated"
means we could not find it there. Open an issue if a row has gone stale.

| Project (commit read) | Robot type | ROS version | Limits outside the model | Tested where | Licence |
|---|---|---|---|---|---|
| APYROBO `apyrobo mcp` (this repo) | Mobile robots: `mock://`, or a ROS 2 robot through Nav2 (`ros2://`); not arms | ROS 2; CI runs Humble | Speed clamp, rectangular no-go zones and allowed areas (target and route), battery floor, opt-in human confirmation per move via MCP elicitation, a stop that is always allowed, JSONL audit log; see [the MCP guide](mcp.md#what-this-does-and-doesnt-guarantee) | `mock://` and Nav2 in Gazebo in CI; not on hardware by us | Apache-2.0 |
| [ros-mcp-server](https://github.com/robotmcp/ros-mcp-server) (476591a; PyPI `ros-mcp` 3.1.2) | Any ROS robot through rosbridge, "no changes to existing robot source code" | "ROS 2 (Jazzy, Humble, and others) and ROS 1" | None stated; the README lists permissions as a wanted contribution | Integration tests in CI on melodic, noetic, humble, jazzy | Apache-2.0 |
| [ros2_mcp](https://github.com/wise-vision/ros2_mcp) by WiseVision, formerly `mcp_server_ros_2` (4f287b0) | Any ROS 2 graph: topics, services, actions | Humble and Jazzy "Supported. Tests run in CI"; "ROS 1 is not supported" | A read-only mode (`ROS2_MCP_READONLY=1`) in which mutating tools are "not registered at all"; no speed, zone or confirmation limits stated; audit not stated | CI on `ros:jazzy-ros-base` and `ros:humble-ros-base` | MPL-2.0 |
| [BotInfo](https://github.com/botinfo-ai/botinfo) (2a5dde5; 0.1.0) | Not an MCP server itself: a policy layer in front of an existing connector | Not stated, apart from third-party benchmark rows on ROS 2 Humble | Owner-listed actions with typed bounds (out of range is "refused (never clamped)"), rate limits, longest continuous motion, optional workspace box, owner confirmation with expiry, "A stop always wins", hash-chained JSONL audit | "simulation and mocks only. Nothing here has run on a real robot" | Apache-2.0 |
| [Principia Robotica](https://github.com/EngineerAbdullahBinZafar/principia-robotica) (6cd4984; 1.0.0) | Differential drive; quadrotor and swarm in its visualizer | A ROS 2 bridge (`/cmd_vel_raw` to `/cmd_vel`); distro not stated | A control-barrier velocity filter on the bridge path; the same filter is also an MCP tool the model calls; confirmation and audit not stated | Unit tests in its repo; hardware not stated | MIT |
| [armguard-mcp](https://github.com/Lynn-hh/armguard-mcp) (043187d; PyPI 0.1.1) | Robot arms, "Franka FR3 first" | ROS 2 Jazzy | Server-side YAML policy: joint limits, workspace, keep-out along the tool-tip path ("not link geometry"), speed, step, force, gripper; approval via MCP elicitation "denied by default"; software e-stop; append-only JSONL audit | Simulated FR3, and ROS 2 Jazzy "against real rclpy endpoints and a real MoveIt 2 move_group"; "not been run on a physical robot yet" | Apache-2.0 (alpha) |

In short: for an arm, look at armguard-mcp. For full access to any topic or
service, ros-mcp-server or WiseVision's ros2_mcp. For a mobile base where the model
must stay inside speed limits and zones you set, APYROBO, and you can try those
limits on `mock://` before connecting the robot.

---

## TL;DR

| Feature | APYROBO | RAI (Robotec) | ROS-LLM |
|---------|---------|---------------|---------|
| **Focus** | AI orchestration layer | Conversational robot AI | LLM-to-ROS bridge |
| **ROS 2 Relationship** | Builds on top (not replacing) | Tightly integrated | Direct bridge |
| **LLM Providers** | Any (via LiteLLM) | OpenAI, Anthropic | OpenAI mainly |
| **Safety Enforcement** | Built-in, framework-level | Limited | None |
| **Multi-Robot** | First-class swarm module | No | No |
| **Skill System** | DAG-based graph engine | Action-based | Function calling |
| **Hardware Agnostic** | Yes (adapter pattern) | ROS 2 robots | ROS 2 robots |
| **Observability** | Prometheus, OTel, replay | Logging | Logging |
| **State Persistence** | JSON / SQLite / Redis | No | No |
| **License** | Apache 2.0 | Apache 2.0 | MIT |

---

## Detailed Comparison

### 1. Architecture Philosophy

**APYROBO** is a *semantic intelligence layer*. It doesn't replace ROS 2 — it adds capability abstraction, skill composition, safety enforcement, and swarm coordination above the ROS 2 middleware. The core insight: LLMs need a structured runtime to act safely in the physical world.

**RAI** (by Robotec.AI) is a *conversational AI framework for ROS 2*. It focuses on natural language interaction with robots, using ROS 2 actions and services as the execution backend. RAI is tightly coupled to ROS 2 and uses a conversational agent pattern.

**ROS-LLM** is a *direct bridge* from LLM function calls to ROS 2 topics and services. It's lightweight and focused on the "LLM calls a ROS action" pattern without additional orchestration.

### 2. Safety

| Aspect | APYROBO | RAI | ROS-LLM |
|--------|---------|-----|---------|
| Speed clamping | Yes (per-policy) | No | No |
| Collision zones | Yes (static + dynamic) | No | No |
| Watchdog (odometry check) | Yes (SF-05) | No | No |
| Battery awareness | Yes (SF-10) | No | No |
| Human proximity | Yes (SF-02) | No | No |
| Escalation workflow | Yes (SF-03) | No | No |
| Policy hot-swap | Yes (SF-11) | No | No |
| Formal verification export | Yes (TLA+, UPPAAL) | No | No |

APYROBO treats safety as infrastructure. Every command passes through the `SafetyEnforcer` before reaching the robot — agents cannot bypass it. This is critical for real-world deployment where an LLM hallucination could cause physical damage.

### 3. Skill Composition

**APYROBO** uses a **Directed Acyclic Graph (DAG)** of skills with preconditions, postconditions, timeouts, and retry logic. Skills can be chained into complex plans with parallel execution layers. The executor handles state flow between skills automatically.

**RAI** uses ROS 2 actions directly. Tasks are expressed as natural language and mapped to individual actions. There's no built-in skill composition or dependency tracking.

**ROS-LLM** uses function calling to invoke ROS services/actions. Each call is independent — there's no skill graph or composition engine.

### 4. Multi-Robot Support

| Feature | APYROBO | RAI | ROS-LLM |
|---------|---------|-----|---------|
| Multi-robot messaging | SwarmBus | No | No |
| Task splitting | SwarmCoordinator | No | No |
| Capability-based assignment | Yes | No | No |
| Failure reassignment | Yes | No | No |
| Proximity safety | Yes | No | No |
| Deadlock detection | Yes | No | No |

APYROBO is **swarm-first**. The `SwarmCoordinator` splits tasks across robots based on capabilities, redistributes on failure, and monitors proximity/deadlock.

### 5. LLM Integration

**APYROBO** is model-agnostic via LiteLLM:
- OpenAI (GPT-4, GPT-4o)
- Anthropic (Claude)
- Local models (Ollama, vLLM)
- Multi-tier routing with circuit breakers
- Plan caching and token budgets
- Built-in rule-based provider (no API key needed)

**RAI** supports OpenAI and Anthropic directly with its own agent framework.

**ROS-LLM** primarily targets OpenAI's function calling API.

### 6. Observability

| Feature | APYROBO | RAI | ROS-LLM |
|---------|---------|-----|---------|
| Structured JSON logging | Yes | Basic | No |
| Trace context (correlation IDs) | Yes (OB-03) | No | No |
| Prometheus metrics | Yes (OB-02) | No | No |
| OpenTelemetry export | Yes (OB-08) | No | No |
| Execution replay | Yes (OB-10) | No | No |
| Health dashboard | Yes (OB-04) | No | No |
| Alerting | Yes (OB-11) | No | No |
| Time-series storage | Yes (OB-12) | No | No |

### 7. State Persistence

APYROBO provides three storage backends:
- **JSON file** — simple, inspectable, no dependencies
- **SQLite** — concurrent access, production use
- **Redis** — distributed deployments, shared state

RAI and ROS-LLM have no built-in state persistence.

### 8. Robot MCP servers

APYROBO also runs as an MCP server (`apyrobo mcp`), so a model in Claude
Desktop, Claude Code or another MCP client can drive a robot through the same
`SafetyEnforcer`. It is not the only robot MCP server with safety limits:
[armguard-mcp](https://github.com/Lynn-hh/armguard-mcp)
([PyPI](https://pypi.org/project/armguard-mcp/)) is a safety-first MCP server
for ROS 2 robot arms (Jazzy, MoveIt 2, Franka FR3), with joint, workspace and
keep-out envelopes, speed caps, human approval through MCP elicitation, an
e-stop and an audit log. It is Apache-2.0 and, by its own README, alpha and not
yet run on hardware.

The two cover different robots: armguard-mcp guards manipulators; APYROBO
guards mobile robots (`mock://`, or ROS 2 with Nav2 through `ros2://`). For a
robot arm, use armguard-mcp. For a mobile base or a fleet where the model must
stay inside speed limits and no-go zones you set, use APYROBO. The
[MCP guide](mcp.md#how-apyrobo-fits-with-other-robot-mcp-servers) compares
these and other robot MCP servers side by side.

A gateway is a different layer and works alongside APYROBO rather than
instead of it. [Governing an LLM-Driven Robot Fleet with
agentgateway](https://webofmike.com/llm-robot-fleet-agentgateway/) (Mike
Moore) puts agentgateway between the model and the robots' MCP servers: it
allows or denies whole MCP tools per robot identity, denies by default, uses
CEL rules and keeps an audit log, shown on a simulated fleet without ROS. It
does not check a tool's arguments. APYROBO checks the arguments of the tools
it allows (speed against `--max-speed`, targets against no-go zones). We have
not tested the two together.

---

## When to Use Each

### Use APYROBO when:
- You need **safety enforcement** for real-world deployment
- You're building a **multi-robot system**
- You want **model-agnostic LLM** support (swap providers freely)
- You need **production observability** (metrics, traces, alerts)
- You're building a **skill-based system** with complex task composition
- You need **crash recovery** (state persistence)

### Use RAI when:
- You want a **conversational interface** to a single ROS 2 robot
- Your robot tasks are **simple and sequential**
- You're already deeply invested in the **ROS 2 action/service** pattern
- You prefer a **tightly integrated** ROS 2 solution

### Use ROS-LLM when:
- You want the **simplest possible** LLM-to-ROS bridge
- Your use case is **prototyping** or **research**
- You only need **one LLM provider** (OpenAI)
- You don't need safety, swarm, or production features

---

## Migration Path

### From RAI to APYROBO

1. Replace RAI agent with `Agent(provider="llm")`
2. Convert ROS 2 actions to APYROBO skills
3. Add safety policies
4. Optionally add swarm coordination

### From ROS-LLM to APYROBO

1. Replace function call handlers with skill definitions
2. Use `CapabilityAdapter` to wrap your ROS 2 nodes
3. Build skill graphs for complex tasks
4. Add `SafetyEnforcer` wrapper

---

## Links

- **APYROBO**: [github.com/apyrobo/apyrobo](https://github.com/apyrobo/apyrobo)
- **RAI**: [github.com/RobotecAI/rai](https://github.com/RobotecAI/rai)
- **ROS-LLM**: [github.com/autonohm/ros_llm](https://github.com/autonohm/ros_llm)
- **armguard-mcp**: [github.com/Lynn-hh/armguard-mcp](https://github.com/Lynn-hh/armguard-mcp)
- **agentgateway fleet post**: [webofmike.com/llm-robot-fleet-agentgateway](https://webofmike.com/llm-robot-fleet-agentgateway/)
