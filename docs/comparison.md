# APYROBO vs RAI vs ROS-LLM: Robot AI Framework Comparison

Choosing the right AI-robotics framework depends on your use case. This comparison covers the three leading open-source options as of 2025-2026.

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
