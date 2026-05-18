# Changelog

All notable changes to apyrobo are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and apyrobo adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] — v6.0.0 Ecosystem Integrations

### Added

**`apyrobo profiles detect` — hardware auto-detection**
- New `apyrobo profiles detect` command probes the local machine and recommends the best profile
- Detects: Ollama at localhost:11434 (lists installed models, picks best by capability), nvidia-smi GPU info (name, VRAM), system RAM via /proc/meminfo or sysconf
- Recommends profile with `high`/`medium`/`low` confidence and a human-readable reason
- `--json` flag for scripting and CI integration
- `detect_profile()` and `DetectionResult` exported from `apyrobo.profiles.schema`

**`WebSocketOrchestrationAdapter` — real-time task streaming**
- `apyrobo serve --transport websocket --ws-port 8765` — starts a WebSocket server instead of stdio
- `WebSocketOrchestrationAdapter(host, port)` implements the `OrchestrationAdapter` contract over asyncio + websockets
- Background asyncio loop runs in a daemon thread; `receive()` blocks on a `queue.Queue` fed by the async handler
- `send(msg)` broadcasts JSON to all currently connected clients
- Multiple simultaneous clients supported; disconnected clients silently skipped
- Graceful `ImportError` with `pip install 'apyrobo[websocket]'` hint when websockets not installed

**`TelemetryContextProvider` — live robot state in LLM planning prompts**
- `apyrobo.skills.telemetry_context.TelemetryContextProvider(robot, refresh_interval=5.0)`
- Background thread samples robot state every N seconds: position, heading, battery, velocity, sensor status, active errors
- `get_context_string()` returns a compact `[Robot State — sampled Xs ago]` block
- Stale-data warning when snapshot is older than `max_snapshot_age` (default 30s)
- Integrated into `Agent.plan()` via `telemetry_provider=` kwarg; injected into LLM prompts only (skipped for rule-based provider)
- `TelemetrySnapshot` dataclass with per-field nullable fields

**`apyrobo-skills-ros-nav` — Nav2 skill package**
- `packages/apyrobo-skills-ros-nav/` — pip-installable, `apyrobo.skills` entry-point
- Skills: `navigate_to_pose(x, y, yaw, frame_id)`, `follow_path(waypoints)`, `clear_costmaps()`, `nav2_recover()`
- `Nav2Client` helper class manages rclpy node lifecycle for Nav2 ActionClient calls
- Graceful ImportError with ROS 2 install instructions when rclpy is unavailable
- All skills have 30s action timeout; return `False` when action server unavailable

**`IsaacSimAdapter` — NVIDIA Isaac Sim integration**
- `apyrobo/core/isaac_adapter.py` — `IsaacSimAdapter` supports both REST API path (no SDK) and omni SDK path
- REST path: HTTP endpoints for `/start`, `/stop`, `/step`, `/scene/load`, `/robot/state` against Isaac Sim server
- SDK path: `omni.isaac.core.World` lifecycle, `load_robot_prim()`, `step_simulation()` — all guarded by `_OMNI_AVAILABLE`
- URI scheme: `isaac://my_scene` or `isaac://host:port/scene`
- Graceful `ImportError` with pip install instructions when `omni` not available
- Full capability advertisement: NAVIGATE, MANIPULATE, SCAN sensors

**`UnitreeAdapter` — Unitree Go2 / H1 adapter**
- `apyrobo/core/unitree_adapter.py` — `UnitreeAdapter` over UDP/DDS using `unitree_sdk2py` SportClient
- URI parsing: `unitree://go2@192.168.1.100` → model=go2, host=192.168.1.100; `unitree://h1` → broadcast
- Go2 capabilities: NAVIGATE, ROTATE, SCAN; H1 adds MANIPULATE, PICK, PLACE
- Methods: `move(x, y)`, `stop()`, `rotate(angle_rad)`, `stand()`, `sit_down()`, `wave_hand()`, H1 `gripper_open()` / `gripper_close()`
- `UnitreeState` dataclass tracks position, yaw, velocities
- Graceful `ImportError` when `unitree_sdk2py` not installed

**`VisionAdapter` — OpenCV vision pipeline**
- `apyrobo/vision/pipeline.py` — background capture thread + on-demand detection
- Three detection backends: YOLO (ultralytics), Haar cascades (cv2 built-in), no-GPU required
- `detect(label=None, *, min_confidence=None)` → sorted `list[Detection]` by confidence descending
- `Detection` dataclass: label, confidence, `BoundingBox`, optional `pose_3d` from depth frame
- `estimate_pose(detection, depth_frame)` — reads depth at bounding box center for 3-D pose
- Skill helpers: `is_present(label)`, `count(label)`, context manager support
- `apyrobo/vision/__init__.py` exports `VisionAdapter`, `Detection`, `VisionFrame`
- Works fully on CPU; GPU accelerates YOLO but is not required

---

## [Unreleased] — v5.0.0 Five-Minute Success

### Added

**`apyrobo init` — project scaffold**
- New `apyrobo init <name>` command generates a complete pip-installable skill package
- Output includes `pyproject.toml` with `apyrobo.skills` entry-point, `src/<module>/skills.py` with a stub `@skill`-decorated function, `tests/test_skills.py`, `.github/workflows/ci.yml`
- Handles both kebab-case and snake_case names; normalises to kebab package name + snake module name
- `--author`, `--description`, `--directory`, `--force` flags

**`apyrobo shell` — interactive REPL**
- New `apyrobo shell --robot <uri>` command drops into a Python REPL
- Pre-imports: `robot`, `agent`, `Robot`, `Agent`, `SkillGraph`, `BUILTIN_SKILLS`
- Startup banner shows connected robot name, provider, and skill count
- Supports any robot URI; defaults to `mock://turtlebot4`

**`apyrobo tutorial` — guided walkthrough**
- New `apyrobo tutorial` command: 6-step interactive tour covering discovery, capabilities, planning, execution, writing a skill, and testing
- Runs entirely in mock mode — no hardware, no API key needed
- `--non-interactive` flag runs all steps without pausing (CI-friendly)
- Each step shows the code you'd type and the concept it teaches

**`apyrobo-demo` docker compose environment**
- `docker/docker-compose-demo.yml` — three services: `apyrobo-demo` (orchestration server), `dashboard` (web UI on port 8000), `mock-fleet` (3 simulated robots)
- `docker/Dockerfile.demo` — standalone image for deployment
- `apyrobo/demo/mock_fleet.py` — animated 3-robot status printer with deterministic position simulation
- `apyrobo demo mock-fleet` entry-point via `__main__`

**`apyrobo dashboard` — HTMX live view**
- New `apyrobo dashboard --robot <uri> --port 8000` command
- `RobotDashboard` class: wraps a `Robot`, buffers skill history (deque, last 50) and safety events (deque, last 100)
- HTMX panels auto-refresh: status every 5s, skill history every 3s, safety every 5s
- Partial endpoints: `GET /partials/status`, `/partials/history`, `/partials/safety`, `/partials/skills`
- JSON API: `GET /api/status`, `/api/skills/history`, `/api/skills/available`, `/api/safety/events`
- Dark terminal-themed UI (consistent with APYROBO's aesthetic)
- `record_skill()` and `record_safety_event()` hooks for integration with skill executor

**Enhanced `apyrobo test-skill`**
- Capability mismatch detection: checks `skill_meta.required_capability` against the robot's `capabilities()` before running
- Emits structured warning with the missing `CapabilityType`, what the robot provides, and a `pip install` fix hint
- Failure summary section: groups distinct error messages and surfaces actionable hints for common patterns (gripper AttributeError, timeout, capability errors)
- Separator width expanded from 38 to 50 characters for readability

---

## [3.0.0] - 2026-05-15

Universal Coverage — any robot, any LLM, any hardware, zero configuration.

### Added

**Skill packages**
- `apyrobo-skills-ur` — Universal Robots UR3/UR5/UR10/UR16: `move_joints`, `move_linear`, `pick`, `place`, `move_home`, `set_tcp`, `get_pose`
- `apyrobo-skills-spot` — Boston Dynamics Spot: `walk_to`, `sit`, `stand`, `stair_climb`, `dock`, `capture_image`, `arm_pick`
- `apyrobo-skills-franka` — Franka Panda: `move_to_pose`, `grasp`, `release`, `move_home`, `cartesian_sweep`, `impedance_control`
- `apyrobo-skills-drone-px4` — PX4-based drones: `takeoff`, `land`, `fly_to`, `orbit`, `return_home`, `capture_image`
- `apyrobo-skills-agv` — MiR / Omron LD / Clearpath Husky: `navigate_to`, `follow_route`, `dock_to_station`, `load_cargo`, `unload_cargo`

**Compute profiles** (`apyrobo/compute_profiles/`)
- `--profile jetson-orin`, `--profile workstation-gpu`, `--profile cloud`, `--profile cpu-only`
- Each profile pre-configures LLM, VLM, STT/TTS models and inference backends
- `apyrobo profiles` CLI command — list all profiles or inspect one; `--json` flag
- `--profile` flag added to `apyrobo exec` and `apyrobo plan`

**Hardware knowledge schema** (`apyrobo/hardware/`)
- Per-robot YAML spec files: reach, payload, DoF, sensor suite, speed limits
- `HardwareRegistry` — discovers and loads specs at runtime
- Auto-discovery: `apyrobo connect ros2://ur10` detects robot type from URDF/node list

**Orchestration server** (`apyrobo/orchestration/`)
- `OrchestrationAdapter` ABC — pluggable interface for any front-end (Slack, Discord, web UI, ROS service)
- `OrchestrationServer` — receive → plan → send loop with `max_iterations` guard
- `StdioOrchestrationAdapter` — JSON-over-stdin/stdout reference implementation
- `apyrobo serve` CLI command — run the orchestration server over stdio

**Workflow templates** (`examples/workflows/`)
- 10 ready-made multi-robot workflow scripts: patrol loop, pick-and-place, inspection round, charging cycle, shelf restock, delivery run, multi-floor navigation, quality inspection, fleet coordination, voice-commanded robot

---


## [1.0.0] - 2026-04-29

First stable release. Covers all work from PRs #32–#45.

### Added

**Real-hardware ROS 2 support**
- **ROS 2 bridge** (`apyrobo/core/ros2_bridge.py`) — production `ROS2Adapter` with Nav2 `NavigateToPose` action client, cmd_vel proportional-control fallback, BEST_EFFORT odometry subscription, configurable QoS, namespace support, feedback hooks, SLAM trigger, multi-floor map switching (PRs #32–#34)
- **Nav2 adapter** (`apyrobo/nav2.py`) — `Nav2Adapter` / `MockNav2Adapter` with `get_position()`, `cancel_navigation()`, `set_initial_pose()`, stub mode without rclpy (#35)
- **MoveIt 2 adapter** (`apyrobo/moveit.py`) — `MoveItAdapter` / `MockMoveItAdapter` with `plan_motion()` / `execute_motion()` separation, `home_arm()`, live `get_joint_states()` from `/joint_states`, stub mode (#35)

**Voice control**
- **Voice adapters** (`apyrobo/voice.py`) — `WhisperAdapter` (offline STT), `PiperAdapter` (offline TTS), `OpenAIVoiceAdapter` (cloud STT+TTS), `MockVoiceAdapter`, `VoiceAgent` (STT → plan → execute → TTS in one call), `WhisperAdapter.transcribe(bytes|str)` (#36)

**Agent and skill improvements**
- **`@skill` decorator** (`apyrobo/skills/decorators.py`) — annotate plain functions as skills; `Skill.simple()` factory; `SkillLibrary.from_decorated()` auto-registers decorated functions as runtime handlers (#43)
- **Skill discovery** (`apyrobo/skills/discovery.py`) — `SkillManifest`, `SkillDiscovery`, `DiscoveryRegistry` with robot capability matching (#40)
- **Skill retry policies** (`apyrobo/skills/retry.py`) — `RetryStrategy`, `RetryPolicy`, `CircuitBreaker`, `RetryExecutor` (#38)
- **Execution checkpointing** (`apyrobo/skills/checkpoint.py`) — `CheckpointedExecutor` resumes skill graphs from last successful step (#38)
- **Learning from demonstrations** (`apyrobo/skills/demonstrations.py`) — `DemonstrationRecorder`, `DemonstrationStore`, `DemonstrationReplayer`, `SkillLearner` (#41)

**Simulation**
- **MuJoCo adapter** (`apyrobo/sim/`) — lightweight physics sim adapter (#38)
- **Formal safety verification** — export safety proofs for regulatory compliance (#38)

**Fleet and cloud**
- **Fleet manager** (`apyrobo/fleet/`) — load-balanced task assignment, heartbeat monitoring, offline detection (#37)
- **REST API gateway** (`apyrobo/api/`) — FastAPI task/robot endpoints with API-key auth (#37)
- **Role-based access** (`apyrobo/auth.py`) — `RBACRole`, `ROLE_PERMISSIONS`, `RBACManager` (#37)
- **Audit trail** (`apyrobo/audit.py`) — immutable SQLite event log with SHA-256 hash-chain integrity (#37)
- **Multi-site and edge inference** support (#37)
- **Digital twin sync** — real-time state mirroring to simulation (#37)
- **Kubernetes and Docker Compose** deployment templates (`k8s/`, `docker/`) (#37)

**Developer experience**
- **Handler registry** — `@skill_handler` decorator for dynamic dispatch (#33)
- **YAML / TOML config** — file-based adapter and policy configuration (#34)
- **Connection resilience** — exponential-backoff auto-reconnect in base adapter (#34)
- **Integration test suite** — `tests/integration/` with `fake_turtlebot4.py` ROS 2 stub, Docker Compose `integration` profile, GitHub Actions workflow (#44)
- **Plugin system** (`apyrobo/plugins/`) — installable third-party bundles via entry-points (#45)
- **Skill registry server** (`apyrobo/registry/`) — FastAPI registry + `SkillRegistryClient` (#45)
- **Versioning tools** (`apyrobo/versioning/`) — changelog parser, migration guide generator, deprecated API scanner (#45)
- **LTS policy** (`apyrobo/lts/`) — EOL tracking, security advisory lookup (#45)
- Runnable example scripts (`examples/`) — hello robot, custom skill, fleet (#42)
- `MIGRATION.md`, `API_STABILITY.md`, `docs/TURTLEBOT4.md` reference documents (#42, #44)

### Changed
- Public API surface frozen for v1.x (see `API_STABILITY.md`)
- Test coverage: 92 % across 2 076 tests (#33–#36)
- `Development Status` classifier promoted from Alpha → Production/Stable

---

## [0.4.0] - 2026-03-23

### Added
- **REST API gateway** (`apyrobo/api/`) — FastAPI endpoints for task submission, robot listing, and skill execution with `X-API-Key` authentication
- **Fleet manager** (`apyrobo/fleet/`) — load-balanced task assignment, heartbeat monitoring, offline robot detection
- **Audit trail** (`apyrobo/audit.py`) — immutable SQLite event log with SHA-256 cryptographic hash chaining and `verify_chain()` integrity check
- **RBAC** — `RBACRole`, `ROLE_PERMISSIONS`, and `RBACManager` added to `apyrobo/auth.py` (admin / operator / viewer roles)

---

## [0.3.0] - 2026-03-23

### Added
- **VLM integration** (`apyrobo/inference/vlm.py`) — `VLMAdapter` base class, `LiteLLMVLMAdapter` (GPT-4V / Claude Vision), `MockVLMAdapter` for testing
- **Multi-turn agent** (`apyrobo/agents/multiturn.py`) — `ConversationHistory` with token-budget truncation, `MultiTurnAgent` with mock-LLM fallback
- **Tool-calling agent** (`apyrobo/agents/tool_agent.py`) — converts apyrobo skills into LLM tool definitions; handles parallel tool calls
- **Execution feedback loop** (`apyrobo/skills/feedback.py`) — `FeedbackCollector` with rolling success-rate tracking, `AdaptiveExecutor` with dynamic retry strategy
- **Skill discovery** (`apyrobo/skills/discovery.py`) — `SkillManifest`, `SkillDiscovery`, `DiscoveryRegistry` with robot capability matching
- **Memory system** (`apyrobo/memory/`) — episodic memory (SQLite), semantic memory (numpy cosine similarity), TTL plan cache
- **Handler registry** (`apyrobo/skills/registry.py`) — `@skill_handler` decorator for dynamic dispatch
- **YAML/TOML config** — file-based robot/skill/agent configuration support
- **Connection resilience** — exponential-backoff auto-reconnect in base adapter
- **Gazebo improvements** — spawn/despawn models, joint states, apply forces, world reset

---

## [0.2.0] - 2026-03-01

### Added
- Voice adapter (Whisper STT, OpenAI TTS, Piper TTS)
- Nav2 adapter for ROS 2 navigation
- MoveIt adapter for arm manipulation
- Skill execution checkpointing
- Retry and circuit-breaker patterns

---

## [0.1.0] - 2026-01-15

### Added
- Initial release — apyrobo MVP
- Skill graph, SkillExecutor, safety layer, swarm coordinator
- Observability (metrics, traces), state persistence
- REST API for operations (task submission, scheduling)
- Preflight checker, pilot quickstart guide
- Test coverage: 92% (2076 tests)
