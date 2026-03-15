# APYROBO Roadmap

Public roadmap for the APYROBO project. Items are grouped by milestone and roughly ordered by priority within each group.

**Legend:** :white_check_mark: Done | :construction: In Progress | :clipboard: Planned | :bulb: Exploring

---

## v0.1 — Foundation (Current)

Core framework with mock adapter support and offline planning.

| Status | Item | Description |
|--------|------|-------------|
| :white_check_mark: | Capability adapter pattern | `mock://`, `gazebo://`, `mqtt://`, `http://` URI schemes |
| :white_check_mark: | Skill graph engine | DAG-based skill composition with preconditions/postconditions |
| :white_check_mark: | Skill executor | Sequential + parallel execution with timeout and retry |
| :white_check_mark: | Rule-based agent | Offline planning with no API key required |
| :white_check_mark: | LLM agent | Model-agnostic planning via LiteLLM |
| :white_check_mark: | Safety enforcer | Speed clamping, collision zones, watchdog, escalation |
| :white_check_mark: | Swarm coordination | Task splitting, failure reassignment, deadlock detection |
| :white_check_mark: | Observability | Prometheus metrics, OTel export, alerting, replay |
| :white_check_mark: | State persistence | JSON, SQLite, Redis backends with crash recovery |
| :white_check_mark: | Test suite | 120+ tests including property-based and chaos tests |

---

## v0.2 — Production Hardening

Focus: reliability, performance, and real hardware support.

| Status | Item | Description | Contribution Welcome? |
|--------|------|-------------|----------------------|
| :construction: | Nav2 adapter | Full ROS 2 Nav2 integration for real navigation stacks | Yes |
| :construction: | MoveIt adapter | ROS 2 MoveIt 2 integration for manipulation | Yes |
| :clipboard: | Gazebo adapter improvements | Spawn/delete models, reset world, sensor streams | Yes |
| :clipboard: | Connection resilience | Auto-reconnect, connection pooling, health probes | Yes |
| :clipboard: | Skill retry policies | Exponential backoff, jitter, circuit breaker per skill | No |
| :clipboard: | Execution checkpointing | Resume skill graphs from last successful step | No |
| :clipboard: | Config file support | YAML/TOML config for policies, adapters, inference | Yes |
| :clipboard: | Performance profiling | Identify and fix bottlenecks in executor hot path | Yes |

---

## v0.3 — Intelligence

Focus: smarter planning, learning from execution, multi-modal input.

| Status | Item | Description | Contribution Welcome? |
|--------|------|-------------|----------------------|
| :clipboard: | Plan caching | Cache and reuse LLM-generated plans for repeated tasks | No |
| :clipboard: | Plan validation | LLM plans checked against capability model before execution | No |
| :clipboard: | Multi-turn agent | Clarification dialogue when task is ambiguous | No |
| :clipboard: | Tool-calling agent | Function-calling LLM directly invokes skills | No |
| :clipboard: | Vision integration | Camera feeds as context for LLM planning | Yes |
| :clipboard: | Execution feedback loop | Feed execution results back to planner for re-planning | No |
| :clipboard: | Skill discovery | Agents discover available skills at runtime | Yes |
| :bulb: | Learning from demonstrations | Record human teleoperation as new skills | Yes |

---

## v0.4 — Fleet & Cloud

Focus: multi-robot fleet management, cloud deployment, enterprise features.

| Status | Item | Description | Contribution Welcome? |
|--------|------|-------------|----------------------|
| :clipboard: | Fleet manager | Centralized fleet dashboard with task queue | Yes |
| :clipboard: | Cloud deployment | Docker Compose / Kubernetes deployment templates | Yes |
| :clipboard: | REST API gateway | HTTP API for external systems to submit tasks | Yes |
| :clipboard: | Role-based access | Per-robot and per-task permission model | No |
| :clipboard: | Audit trail | Immutable log of all commands, decisions, and violations | No |
| :clipboard: | Multi-site support | Coordinate robots across separate physical locations | No |
| :bulb: | Edge inference | Run small models on robot hardware for low-latency decisions | Yes |
| :bulb: | Digital twin sync | Sync physical robot state to simulation in real-time | Yes |

---

## v1.0 — Stable Release

Focus: API stability, backwards compatibility, comprehensive documentation.

| Status | Item | Description |
|--------|------|-------------|
| :clipboard: | API freeze | No breaking changes to public API |
| :clipboard: | Migration guide | v0.x to v1.0 upgrade documentation |
| :clipboard: | Certification support | Export safety proofs for regulatory compliance |
| :clipboard: | Plugin system | Third-party skills, adapters, and providers as pip packages |
| :clipboard: | Long-term support | 12-month security and bug fix window |

---

## How to Contribute

### Pick an Item

Items marked **"Contribution Welcome? Yes"** are good candidates for community contributions. To get started:

1. Open an issue referencing the roadmap item
2. Discuss your approach in the issue before writing code
3. Submit a PR against the `main` branch
4. Include tests and update relevant documentation

### Suggest New Items

Have an idea not on this roadmap? Open a **Feature Request** issue with:
- **Use case:** What problem does it solve?
- **Proposed approach:** How would you implement it?
- **Alternatives:** What else did you consider?

### Priority Requests

If a planned item is critical for your use case, comment on or react to the relevant issue. Community demand influences prioritization.

---

## Non-Goals

Things APYROBO intentionally does **not** aim to do:

- **Replace ROS 2** — APYROBO builds on ROS 2, it doesn't replace the middleware, drivers, or navigation stack
- **Be a simulator** — Use Gazebo, Isaac Sim, or Webots for simulation; APYROBO connects to them via adapters
- **Train models** — APYROBO orchestrates pre-trained LLMs; training happens elsewhere
- **Hardware drivers** — Write ROS 2 drivers for new hardware; APYROBO adapts the semantic layer above them
