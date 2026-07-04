# APYROBO Roadmap

Public roadmap for the APYROBO project. Items are grouped by milestone and roughly ordered by priority within each group.

**Legend:** :white_check_mark: Done | :construction: In Progress | :clipboard: Planned | :bulb: Exploring

**Contribution labels:**
- ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) — Great for newcomers; well-scoped, mentored
- ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) — Community contributions welcome; may require domain expertise

---

## What we're actually optimizing for

Technical excellence alone does not make a framework influential. The path from "promising architecture" to "the obvious choice" runs through four compounding unlocks — in order:

1. **Production reliability** — Teams won't build on a framework that fails unpredictably around expensive hardware. Failure handling maturity is the hidden prerequisite for trust.

2. **Five-minute success** — Developers decide in the first 10 minutes whether a tool is worth their time. `git clone → docker compose up → working demo` is the bar. Anything slower loses them.

3. **Ecosystem gravity** — Compatibility wins. Every maintained adapter, every integration, every supported simulator is a reason for a new team to pick APYROBO over starting from scratch. Network effects compound from here.

4. **Category ownership** — The goal is for "AI-native robotics orchestration" to be synonymous with APYROBO the way "container orchestration" is synonymous with Kubernetes. That requires killer demos, repeated contextual visibility, and being easy to discover, understand, and discuss online.

The milestones below are ordered by this logic, not by feature priority.

---

## v0.1.0 — Foundation (Current Release)

Core framework with mock adapter support, offline planning, and safety enforcement.

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

## v0.2.0 — Handler Registry + Voice

Focus: dynamic skill handler dispatch, voice control, reliability, and real hardware support.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| ✅ | Handler registry | Dynamic `@skill_handler` registration and dispatch — `HandlerRegistry` class with decorator, `dispatch`, introspection ([#1][i1]) | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | Voice adapter layer | STT/TTS integration — Whisper, Piper, OpenAI ([#2][i2]) | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | Nav2 adapter | Full ROS 2 Nav2 integration for real navigation stacks | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | MoveIt adapter | ROS 2 MoveIt 2 integration for manipulation | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ | Gazebo adapter improvements | Spawn/despawn models, reset world, joint states, apply forces, error handling | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ | Connection resilience | Auto-reconnect with exponential backoff, disconnect/reconnect hooks, observability events | |
| ✅ Done | Skill retry policies | Exponential backoff, jitter, circuit breaker per skill — `RetryStrategy`, `RetryPolicy`, `CircuitBreaker`, `RetryExecutor` in `apyrobo/skills/retry.py` | |
| ✅ Done | Execution checkpointing | Resume skill graphs from last successful step — `CheckpointEntry`, `CheckpointStore`, `CheckpointedExecutor` in `apyrobo/skills/checkpoint.py` | |
| ✅ | Increase test coverage to 90% | Add unit tests for voice, handler registry, and edge cases ([#3][i3]) — **achieved 92% (2076 tests)** | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ | Config file support | YAML/TOML config for policies, adapters, inference — auto-detect format, `from_toml_file()`, `to_toml()` | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |

---

## v0.3.0 — Memory + VLM

Focus: persistent agent memory, vision-language model integration, smarter planning.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| ✅ | Episodic memory | SQLite-backed task execution history — `EpisodicStore`, queryable by time/robot/outcome ([#4][i4]) | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ | Semantic memory | Key-value fact store with cosine similarity vector recall — `SemanticStore` with numpy embeddings | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ Done | VLM integration | Vision-language models for camera-informed planning — `VLMAdapter`, `LiteLLMVLMAdapter`, `MockVLMAdapter`, `VLMRouter`, `InferenceRouter.route_vision()` ([#5][i5]) | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ | Plan caching | TTL-based plan cache with hit/miss metrics — `PlanCache` (memory + SQLite) | |
| ✅ Done | Plan validation | LLM plans checked against capability model before execution — `ValidationIssue`, `ValidationResult`, `PlanValidator` in `apyrobo/skills/plan_validator.py` | |
| ✅ Done | Multi-turn agent | Clarification dialogue when task is ambiguous — `ConversationMessage`, `ConversationHistory`, `MultiTurnAgent` with token-aware context truncation | |
| ✅ Done | Tool-calling agent | Function-calling LLM directly invokes skills — `SkillTool`, `ToolCallingAgent` with mock fallback | |
| ✅ Done | Execution feedback loop | Feed execution results back to planner for re-planning — `ExecutionResult`, `FeedbackCollector`, `AdaptiveExecutor` | |
| ✅ Done | Skill discovery | Agents discover available skills at runtime — `SkillManifest`, `SkillDiscovery`, `DiscoveryRegistry` | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ Done | MuJoCo integration | Lightweight sim adapter for MuJoCo physics — merged PR #38 | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | Formal safety verification | Export safety proofs for regulatory compliance — merged PR #38 | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | Learning from demonstrations | Record human teleoperation as new skills | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |

---

## v0.4.0 — Fleet & Cloud

Focus: multi-robot fleet management, cloud deployment, enterprise features.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| ✅ Done | Fleet manager | Centralized fleet dashboard with task queue — `RobotInfo`, `FleetManager` with load-balanced assignment and offline detection | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | Cloud deployment | Docker Compose / Kubernetes deployment templates — multi-stage `docker/Dockerfile`, `docker/docker-compose.yml`, full `k8s/` manifests with HPA, `docs/deployment.md` | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ Done | REST API gateway | HTTP API for external systems to submit tasks — FastAPI app with task/robot endpoints and API-key auth | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | Role-based access | Per-robot and per-task permission model — `RBACRole`, `ROLE_PERMISSIONS`, `RBACManager` added to `auth.py` | |
| ✅ Done | Audit trail | Immutable log of all commands, decisions, and violations — `AuditEvent`, `AuditTrail` with SHA-256 hash chain integrity verification | |
| ✅ Done | Multi-site support | Coordinate robots across separate physical locations — merged PR #37 | |
| ✅ Done | Edge inference | Run small models on robot hardware for low-latency decisions — merged PR #37 | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | Digital twin sync | Sync physical robot state to simulation in real-time — merged PR #37 | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |

---

## v1.0.0 — Hosted Registry & Stable Release

Focus: API stability, hosted skill/adapter registry, backwards compatibility, comprehensive documentation.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| ✅ | Hosted skill registry | `apyrobo/registry/` — FastAPI server + `SkillRegistryClient` for discovering and publishing skills | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ | API freeze | `API_STABILITY.md` — public API surface frozen for v1.x | |
| ✅ | Migration guide | `MIGRATION.md` — v0.x to v1.0 upgrade documentation | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ | Certification support | `apyrobo/audit.py` — immutable audit trail with SHA-256 hash chain for regulatory compliance | |
| ✅ | Plugin system | `apyrobo/plugins/` — third-party skills and adapters as pip packages via entry-points | |
| ✅ | Long-term support | `apyrobo/lts/` — LTS policy, EOL tracking, `VersionChecker`, security advisories | |
| ✅ | Changelog | `CHANGELOG.md` — complete release history from v0.1.0 to v1.0.0 | |
| ✅ | Versioning tools | `apyrobo/versioning/` — changelog parser, migration guide generator, deprecated API scanner | |

---

## v1.1.0 — Ship & Discover

Focus: get apyrobo into developers' hands. Today the package version is still `0.1.0` and `pip install apyrobo` returns nothing. This milestone closes that gap and gives real-robot users a first-class diagnostic experience.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| ✅ Done | **PyPI publish** | Bump version to `1.0.0`, publish wheel to PyPI so `pip install apyrobo` works. Wire `python-package.yml` to fire on version tags. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ Done | **`apyrobo doctor`** | CLI command that checks the local environment: Python version, rclpy availability, ROS_DOMAIN_ID, reachable adapters, API keys. Prints a pass/fail checklist with fix suggestions. Single biggest DX win for real-robot onboarding. | |
| ✅ Done | **Grafana dashboard** | Add a pre-built Grafana dashboard (`docker/grafana/`) wired to the existing Prometheus metrics. Launch with `docker compose --profile observability up`. Covers task throughput, skill latency, fleet status, error rates. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | **Version alignment** | All version strings (`pyproject.toml`, `apyrobo/__version__.py`, `CHANGELOG.md`, Docker image tags) should reflect the same value. Create a `scripts/bump_version.sh` helper. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ Done | **README badges** | Add CI status, PyPI version, coverage, and license badges to README. Makes the project look alive to anyone landing on GitHub. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ Done | **Worktree cleanup script** | `scripts/clean_worktrees.sh` — prune the 50+ leftover `.claude/worktrees/` branches and merged remote refs that accumulate during development. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |

---

## v1.2.0 — Real Robot Hardening

Focus: make the `ros2://` path reliable enough that a developer can stake production work on it. The integration test proves it works once; this milestone proves it keeps working.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| ✅ Done | **Connection health monitor** | Detect `/odom` timeouts, auto-reconnect with exponential backoff, emit `robot.disconnected` / `robot.reconnected` events. Today a silent network drop leaves the adapter stuck. | |
| ✅ Done | **`apyrobo connect --verify`** | `apyrobo connect ros2://turtlebot4 --verify` — one command that connects, reads battery + position + velocity, prints a latency measurement, and exits 0/1. Replaces the current "write a script and guess" workflow. | |
| ✅ Done | **Nav2 costmap awareness** | Pass the Nav2 costmap to the planner so it can reject goals inside obstacles before sending them, rather than waiting for Nav2 to fail. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | **Multi-robot task handoff** | When a robot fails mid-task, automatically reassign the remaining steps to the next available robot in the fleet. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | **Diagnostics export** | `apyrobo diagnose --robot ros2://turtlebot4 --out diag.json` — capture adapter state, last N tasks, error history, and hardware readings into a portable file for sharing with maintainers. | |
| ✅ Done | **Hardware-in-the-loop CI** | GitHub Actions workflow that runs the integration suite against a real TurtleBot4 on a self-hosted runner. Triggered on release tags only. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |

---

## v1.3.0 — Skill Ecosystem

Focus: make it worth publishing skills. Right now the registry infrastructure exists but there are zero community skills and no way to test a skill in isolation.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| ✅ Done | **`apyrobo-skills-turtlebot4`** | A pip-installable skill package for common TurtleBot4 tasks: `patrol_area`, `dock`, `undock`, `follow_person`, `inspect_room`. The reference implementation for third-party skill packages. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | **Skill test harness** | `apyrobo test-skill my_skill.py --robot mock://` — runs a skill against a mock robot, checks preconditions/postconditions, prints a test report. Lowers the bar for contributing skills. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ Done | **Hosted skill registry** | Deploy the `apyrobo/registry/` FastAPI server publicly (e.g. `registry.apyrobo.dev`) so `apyrobo skill search patrol` actually returns results. | |
| ✅ Done | **Skill composition CLI** | `apyrobo skill compose` — interactive REPL for chaining skills into a graph and testing the result, without writing Python. | :bulb: |

---

## v2.0.0 — Adaptive Intelligence

Focus: the planner gets smarter. Today the LLM produces a static plan and retries on failure; this milestone makes it actually adapt.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| ✅ Done | **LLM replanning loop** | When a skill fails, send the failure reason back to the LLM and ask it to replan the remaining steps. Not just retry — actually reconsider the approach. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | **VLM task verification** | After a skill completes, use the VLM adapter to check camera feed and confirm the expected state ("is the cup actually picked up?"). Flag discrepancies for human review or replan. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | **Long-horizon planning** | Break a multi-step, multi-hour goal ("restock all shelves") into a plan that spans robots, shifts, and recharges. Requires persistent plan state and checkpoint recovery. | :bulb: |
| ✅ Done | **Correction learning** | When a user overrides a plan step, record the correction and use it to bias future planning for similar tasks. Builds on the existing demonstrations system. | :bulb: |
| ✅ Done | **Sim-to-real transfer** | `apyrobo plan --simulate` — run a full task plan in Gazebo first, report any failures, then optionally deploy to the real robot. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |

---

## Where to Start Contributing

### Good First Issues

These items are well-scoped, have clear acceptance criteria, and include mentoring:

| Item | Milestone | Issue |
|------|-----------|-------|
| ✅ Increase test coverage to 90% | v0.2.0 | [#3][i3] |
| ✅ Add Kubernetes deployment template | v0.4.0 | — |
| PyPI publish + version bump to 1.0.0 | v1.1.0 | — |
| README badges (CI, PyPI, coverage) | v1.1.0 | — |
| Worktree cleanup script | v1.1.0 | — |
| Skill test harness (`apyrobo test-skill`) | v1.3.0 | — |
| `apyrobo-skills-turtlebot4` package | v1.3.0 | [#7][i7] |

### Help Wanted

These items need domain expertise (ROS 2, simulation, speech, computer vision):

| Item | Milestone | Issue |
|------|-----------|-------|
| Grafana dashboard provisioning | v1.1.0 | — |
| Nav2 costmap awareness | v1.2.0 | — |
| Hardware-in-the-loop CI (self-hosted runner) | v1.2.0 | — |
| LLM replanning loop | v2.0.0 | — |
| VLM task verification | v2.0.0 | [#5][i5] |
| Sim-to-real transfer | v2.0.0 | [#6][i6] |

### How to Pick Up an Item

1. Check the [issues list](https://github.com/apyrobo/apyrobo/issues) for the matching issue
2. Comment on the issue to claim it
3. Discuss your approach before writing code
4. Submit a PR against `main` with tests and docs
5. See [CONTRIBUTING.md](CONTRIBUTING.md) for code style and setup instructions

### Suggest New Items

Have an idea not on this roadmap? Open a **Feature Request** issue with:
- **Use case:** What problem does it solve?
- **Proposed approach:** How would you implement it?
- **Alternatives:** What else did you consider?

### Priority Requests

If a planned item is critical for your use case, comment on or :+1: the relevant issue. Community demand influences prioritization.

---

## v3.0.0 — Universal Coverage

Focus: make apyrobo the neutral AI orchestration standard. Robot-agnostic, AI-agnostic, hardware-aware, community-extensible. Anyone should be able to connect any robot, any LLM, any compute platform in under 20 minutes.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| :white_check_mark: | **Hardware auto-discovery** | `apyrobo connect ros2://ur10` detects robot type from URDF/node list, loads matching skill package, informs planner of capabilities. No config required. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :white_check_mark: | **Hardware knowledge schema** | Structured per-robot spec files (`hardware/ur10.yaml`, `hardware/spot.yaml`, etc.) — reach, payload, DoF, sensor suite, speed limits — fed to planner as context so it plans safely without hallucinating capabilities. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| :white_check_mark: | **`apyrobo-skills-ur`** | Skill package for Universal Robots UR3/UR5/UR10/UR16: `move_joints`, `move_linear`, `pick`, `place`, `move_home`, `set_tcp`, `get_pose`. Reference impl for industrial arm skills. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :white_check_mark: | **`apyrobo-skills-spot`** | Skill package for Boston Dynamics Spot: `walk_to`, `sit`, `stand`, `stair_climb`, `dock`, `capture_image`, `arm_pick`. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :white_check_mark: | **`apyrobo-skills-franka`** | Skill package for Franka Panda: `move_to_pose`, `grasp`, `release`, `move_home`, `cartesian_sweep`, `impedance_control`. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :white_check_mark: | **`apyrobo-skills-drone-px4`** | Skill package for PX4-based drones: `takeoff`, `land`, `fly_to`, `orbit`, `return_home`, `capture_image`. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :white_check_mark: | **Compute profiles** | `--profile jetson-orin`, `--profile workstation-gpu`, `--profile cloud`, `--profile cpu-only` — each pre-configures the right LLM/VLM/voice models and inference backends. Zero manual liteLLM string knowledge required. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| :white_check_mark: | **Orchestration adapter base** | `OrchestrationAdapter` abstract base + `apyrobo serve` command. Anyone wires their own Slack bot, Discord bot, web UI, ROS service, or CLI on top. Ships with a `StdioOrchestrationAdapter` as the reference. Neutral — no built-in product. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :white_check_mark: | **Workflow templates** | `examples/workflows/` — 10 ready-made multi-robot workflows: warehouse pick-and-place, facility patrol, inspection round, charging cycle, shelf restock, delivery run, quality inspection, door-to-door navigation, assembly sequence, multi-floor elevator navigation. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| :white_check_mark: | **`apyrobo-skills-agv`** | Skill package for generic AGV platforms (MiR, Omron LD, Clearpath Husky): `navigate_to`, `dock_to_station`, `follow_route`, `load_cargo`, `unload_cargo`. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |

---

## v4.0.0 — Production Hardening

**Unlock: Trust.** Teams won't build on a framework they can't rely on around expensive hardware. This milestone closes the gap between impressive demos and actual deployment. The hidden killer in robotics is failure handling — without it, orchestration frameworks stay "cool demos."

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| :white_check_mark: | **Deterministic failover** | If a skill times out or throws, the robot returns to a known-safe state automatically — not a crash, not a hang. Configurable per-skill safe-state: stop in place, home arm, dock, or e-stop. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :white_check_mark: | **Plan rollback** | When a multi-step plan fails partway through, undo committed state changes in reverse order (e.g. re-open gripper, navigate back to start). `RollbackExecutor` wraps `SkillExecutor`. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :white_check_mark: | **Degraded operation mode** | If the LLM provider is unreachable, fall back to rule-based planning automatically — no crash, no hang, logged degradation event. `Agent(provider="auto")` already exists; this makes the fallback robust under network loss. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| :white_check_mark: | **Crash recovery** | `OrchestrationServer` persists in-flight task state to disk; on restart it resumes from the last committed checkpoint. Uses the existing `CheckpointStore`. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :white_check_mark: | **Watchdog improvements** | Per-skill-type dead-man switch with configurable timeout, escalation path, and recovery action. Current watchdog is global; this makes it surgical. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| :white_check_mark: | **Chaos test suite** | Pytest fixtures that inject: network partitions, motor fault signals, sensor dropout, LLM timeouts, concurrent skill conflicts. Pass = framework stays safe and observable under all of them. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :white_check_mark: | **Hardware-in-the-loop CI** | Self-hosted GitHub Actions runner with a physical robot (TurtleBot 4 and UR5 reference configs). Skill packages run against real hardware on every PR merge. The first public robotics framework with HIL CI. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |

---

## v5.0.0 — Five-Minute Success

**Unlock: Developer velocity.** Developers decide in the first 10 minutes whether a tool is worth their time. The bar is: `git clone → docker compose up → working robot demo in a browser` with zero prior robotics knowledge. This milestone also makes APYROBO dramatically faster to extend.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| ✅ Done | **`apyrobo-demo` docker compose environment** | `docker/docker-compose-demo.yml` — `docker compose up` spins up APYROBO, a 3-robot mock fleet, and the web dashboard. No ROS 2 install, no config. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | **`apyrobo init` project scaffold** | `apyrobo init my-robot` generates a pip-installable skill package with pyproject.toml, entry-point, stub `@skill`, test file, and CI workflow. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ Done | **`apyrobo shell` interactive REPL** | `apyrobo shell --robot mock://spot` — live Python REPL with robot, agent, and all skills pre-imported. Instant experimentation without writing a script. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ Done | **`apyrobo test-skill` harness** | Enhanced with capability mismatch detection: warns when skill requires a CapabilityType the connected robot doesn't have, with structured `pip install` fix hints. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ Done | **Structured error messages** | Capability mismatches, AttributeErrors, and timeouts in `test-skill` now emit a formatted failure summary with actionable fix hints. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ Done | **Interactive tutorial runner** | `apyrobo tutorial` — 6-step guided CLI walkthrough (discover → capabilities → plan → execute → write skill → test-skill). Runs entirely in mock mode, supports `--non-interactive` for CI. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | **Web dashboard** | `apyrobo dashboard --robot <uri> --port 8000` — FastAPI + HTMX live view: robot status, capability list, skill execution history, safety events, available skills. Auto-refreshes panels every 3–5s. `RobotDashboard` class exposes `record_skill()` / `record_safety_event()` hooks. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |

---

## v6.0.0 — Ecosystem Integrations

**Unlock: Compatibility.** Every maintained adapter is a reason for a team to pick APYROBO over starting from scratch. Infrastructure wins through compatibility layers. This milestone targets the hardware platforms and LLM backends where robotics teams actually live.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| ✅ Done | **NVIDIA Isaac Sim adapter** | `apyrobo connect isaac://my_scene` — `IsaacSimAdapter` with REST + omni SDK dual path, `step_simulation()`, `load_robot_prim()`, graceful `ImportError` when omni unavailable. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | **Unitree Go2 / H1 adapter** | `apyrobo connect unitree://go2` — `UnitreeAdapter` over DDS/SportClient: `move`, `stop`, `rotate`, `stand`, `sit_down`, `wave_hand`, H1 gripper open/close. URI parses model+host automatically. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | **`apyrobo-skills-ros-nav`** | `packages/apyrobo-skills-ros-nav` — `navigate_to_pose`, `follow_path`, `clear_costmaps`, `nav2_recover`. Wraps Nav2 ActionClient. Graceful `ImportError` with install instructions when rclpy missing. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | **Ollama first-class compute profile** | `apyrobo profiles detect` — probes Ollama at localhost:11434, nvidia-smi, and /proc/meminfo; recommends a profile with confidence level, best installed model, and fix hints. `--json` for scripting. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ Done | **OpenCV vision pipeline** | `VisionAdapter` — wraps OpenCV capture + optional YOLO/Haar inference as an APYROBO sensor source. Skills call `vision.detect("cup")` and receive `Detection` objects with label, confidence, bbox, and 3-D pose estimate. Background capture thread; CPU-only fallback to Haar cascades. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | **WebSocket orchestration adapter** | `apyrobo serve --transport websocket --ws-port 8765` — full-duplex JSON streaming over WebSocket. `WebSocketOrchestrationAdapter` broadcasts responses to all connected clients. asyncio + websockets, no FastAPI required. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | **Slack orchestration adapter** | `apyrobo serve --transport slack` — Slack Bolt adapter. `/apyrobo <task> [robot=<uri>]` slash command dispatches tasks; replies post skill plan as a threaded message. Supports Socket Mode and HTTP mode. `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET`, `SLACK_APP_TOKEN` env vars. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | **Live telemetry context** | `TelemetryContextProvider` — background thread samples robot position, battery, velocity, sensor status every 5s. Injects formatted state block into LLM planning prompts; skipped for rule-based provider. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |

---

## v7.0.0 — Category Ownership

**Unlock: Discoverability.** The goal is for "AI-native robotics orchestration" to be synonymous with APYROBO the way "container orchestration" became synonymous with Kubernetes. That requires killer demos people can share, a registry that creates ecosystem gravity, and documentation structured to surface in both search and LLM retrieval. Technical superiority alone doesn't win this — visibility compounds.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| ✅ Done | **Killer demo repos** | Three standalone `clone → run → wow` demos in `demos/`: (1) 10-drone coordinated survey, (2) warehouse multi-robot pick-and-pack via `TaskBus`, (3) humanoid task delegation with NL safety policies. Each under 200 lines, each runnable in under 30 seconds on a laptop with no hardware. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | **Skill registry CLI** | `apyrobo registry search <query>`, `apyrobo registry install <package> [--version]`, `apyrobo registry publish [--pkg-json FILE \| --name ... --token ...]` — full client CLI against the hosted registry. `SkillRegistryClient.install()` resolves via registry then calls pip. `--json` flag for scripting. | :bulb: |
| ✅ Done | **Benchmark suite** | `benchmarks/benchmark_suite.py` — APYROBO vs raw ROS 2 for 5 canonical tasks. Measures setup/plan/exec latency (ms), P95, LOC reduction (5.2× on average). JSON output for CI; GitHub Actions workflow publishes results as job summary artifact on every main push. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| ✅ Done | **LLM retrieval-optimized docs** | `docs/concepts/` — standalone pages for NL safety policies, multi-agent coordination, and adapter pattern; each has: what it is, when to use it, runnable example, comparison table, keyword footer. `llms.txt` at repo root (LLM-readable site map). | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| ✅ Done | **Multi-agent coordination** | `TaskBus` + `MultiAgentCoordinator` — multiple `Agent` instances negotiate over a shared bus. `bus.dispatch("pick cup", required_capability="PICK")` routes to the best available agent. `broadcast()` fans out to all agents. Capability-aware routing picks the least-loaded match. Thread-safe, configurable timeout. | :bulb: |
| ✅ Done | **Natural language safety policies** | `apyrobo policy add "never exceed 0.5 m/s near humans"` — regex + LLM fallback translates sentences into `NLSafetyPolicy` objects, stored in SQLite. `apyrobo policy check` enforces active policies against proposed actions. Plain English audit trail non-engineers can read. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |

---

## v8.0.0 — The Standard Protocol (July – December 2026)

**Unlock: Permanence.** v1–v7 built the features; v8 makes APYROBO the thing other people build *on*. The goal is that "AI-native robotics orchestration" has a spec, a conformance suite, and a governance story — so that adopting APYROBO (or even forking it) is cheaper than starting from scratch, and implementations in other languages strengthen the standard instead of fragmenting it. Kubernetes won because of the API contract, not the codebase. This milestone extracts APYROBO's contract.

Six months, three phases. Each phase ends in a shippable release.

### Phase 1 — Slim, Stable Core (July – August 2026)

A standard needs a trustworthy reference implementation: small, fast to install, boring to depend on.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| :construction: | **`apyrobo-core` split** | Extract the minimal kernel — adapter base, capability model, skill graph, executor, safety layer, config — into a package with near-zero hard dependencies. Everything else becomes an extra: `pip install apyrobo[fleet,voice,vision,registry]`. Targets: install < 15 s, `import apyrobo` < 500 ms, core deps ≤ 5. **Analysis done (`docs/core_split_plan.md`): slim the single package, litellm → `[llm]` extra; unused networkx already dropped.** | |
| :construction: | **APYROBO Protocol spec v1** | New `spec/` directory: language-agnostic, versioned specification of (1) the capability model, (2) the skill manifest schema, (3) the orchestration wire protocol (the JSON messages already flowing over `apyrobo serve` stdio/WebSocket). JSON Schema files + prose. This is the fork-safe artifact — the thing that outlives any one codebase. **1.0-draft published in `spec/` with drift-guard tests; freezes to 1.0 via RFC in Phase 3.** | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :white_check_mark: | **Versioning reset** | Decouple package versions from roadmap milestone numbers. Re-baseline `API_STABILITY.md` against the v4.x surface (it still said "frozen for v1.x"), publish a deprecation policy with minimum notice windows, and designate the post-split release as the first LTS line with dated support windows. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| :white_check_mark: | **Dependency & supply-chain audit** | Pin and minimize dependencies of core, generate an SBOM in release CI, and document the security posture (`SECURITY.md` with reporting process). Standards get audited; be ready before anyone asks. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |

### Phase 2 — Conformance & Interop (September – October 2026)

A spec nobody can test against is a blog post. Conformance is what turns implementations into an ecosystem.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| :clipboard: | **Conformance test suite** | `apyrobo conformance <adapter-or-server-uri>` — runs any adapter or protocol implementation (in any language) against the spec: capability declaration, skill dispatch, failure semantics, safety-stop behavior, event stream shape. Machine-readable report + a "APYROBO Conformant" badge program. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :clipboard: | **Reference non-Python client** | A minimal TypeScript client (`packages/apyrobo-client-ts`) speaking the wire protocol to `apyrobo serve` — submit task, stream events, cancel. Proves the protocol is real and language-agnostic; doubles as the SDK for web dashboards and bots. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :clipboard: | **Adapter authoring kit v2** | `apyrobo init --adapter` scaffold + rewritten `docs/adapter_authoring.md` walking from zero to a conformance-passing adapter in under an hour. The measure of "strong base" is how fast a stranger can extend it. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |
| :clipboard: | **Protocol fuzzing** | Property-based and fuzz tests over the wire protocol and skill manifest parsers: malformed JSON, out-of-order messages, capability spoofing, oversized payloads. Pass = safe rejection, never undefined behavior. | |
| :clipboard: | **Simulator demo videos** | Recorded, scripted demos with zero hardware: (1) the three `demos/` scenarios (mock fleet + dashboard, terminal-recorded), (2) TurtleBot4 navigation in Gazebo, (3) arm pick-and-place in MuJoCo. Each driven by a `demos/*/record.sh` so any contributor can re-render them after changes. Published in README, docs, and as short shareable clips. | ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) |

### Phase 3 — Governance & Gravity (November – December 2026)

Neutral governance is what lets competitors adopt the same standard. Ecosystem gravity is what makes them want to.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| :clipboard: | **RFC process** | Changes to `spec/` go through a public RFC template with a comment window; accepted designs recorded as ADRs. Signals that the protocol is a commons, not one maintainer's whim — the precondition for serious adopters. | |
| :clipboard: | **Seed the registry for real** | Publish the existing skill packages (`ur`, `spot`, `franka`, `px4`, `agv`, `turtlebot4`, `ros-nav`) to PyPI and the hosted registry so `apyrobo registry search` returns real results; curate the first 10 community skills with mentored PRs. | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :clipboard: | **Cross-language interop showcase** | One flagship demo: the TypeScript client orchestrates a Gazebo robot through `apyrobo serve`, end to end, recorded. The "it's a protocol, not a library" proof point. | |
| :clipboard: | **Spec v1.0 announcement** | Freeze protocol spec v1.0, publish the conformance suite results for all first-party adapters, and take it to the community: ROS Discourse, awesome-robotics lists, a launch post walking through the spec. Ends the six months with the standard *public*. | |

---

## Non-Goals

Things APYROBO intentionally does **not** aim to do:

- **Replace ROS 2** — APYROBO builds on ROS 2, it doesn't replace the middleware, drivers, or navigation stack
- **Be a simulator** — Use Gazebo, Isaac Sim, or Webots for simulation; APYROBO connects to them via adapters
- **Train models** — APYROBO orchestrates pre-trained LLMs; training happens elsewhere
- **Hardware drivers** — Write ROS 2 drivers for new hardware; APYROBO adapts the semantic layer above them

---

<!-- Issue references — run scripts/create_roadmap_issues.sh then update numbers -->
[i1]: https://github.com/apyrobo/apyrobo/issues
[i2]: https://github.com/apyrobo/apyrobo/issues
[i3]: https://github.com/apyrobo/apyrobo/issues
[i4]: https://github.com/apyrobo/apyrobo/issues
[i5]: https://github.com/apyrobo/apyrobo/issues
[i6]: https://github.com/apyrobo/apyrobo/issues
[i7]: https://github.com/apyrobo/apyrobo/issues
