# APYROBO Roadmap

Public roadmap for the APYROBO project: where the product is, what is being
built now, and the big arcs that come next.

**Legend:** :white_check_mark: Done | :construction: In Progress | :clipboard: Planned | :bulb: Exploring

> **Milestone numbers (v0.1.0 … v8.0.0) are planning labels, not package
> versions.** The published `apyrobo` package advances by
> [SemVer](https://semver.org) independently — PyPI is currently the **4.x**
> line, whose `4.0.0` release delivered the v4–v7 milestones together
> ([CHANGELOG.md](CHANGELOG.md)). [API_STABILITY.md](API_STABILITY.md) defines
> the mapping and the compatibility guarantees.

---

## What we're actually optimizing for

Technical excellence alone does not make a framework influential. The path from "promising architecture" to "the obvious choice" runs through four compounding unlocks — in order:

1. **Production reliability** — Teams won't build on a framework that fails unpredictably around expensive hardware. Failure handling maturity is the hidden prerequisite for trust.

2. **Five-minute success** — Developers decide in the first 10 minutes whether a tool is worth their time. `git clone → docker compose up → working demo` is the bar. Anything slower loses them.

3. **Ecosystem gravity** — Compatibility wins. Every maintained adapter, every integration, every supported simulator is a reason for a new team to pick APYROBO over starting from scratch. Network effects compound from here.

4. **Category ownership** — The goal is for "AI-native robotics orchestration" to be synonymous with APYROBO the way "container orchestration" is synonymous with Kubernetes. That requires killer demos, repeated contextual visibility, and being easy to discover, understand, and discuss online.

The milestones below are ordered by this logic, not by feature priority.

---

## Shipped — v0.1 through v7 (January – July 2026)

Fifteen milestones, compressed here because they're done; the per-item record
lives in [CHANGELOG.md](CHANGELOG.md).

| Milestones | Unlock | What landed |
|------------|--------|-------------|
| **v0.1–v0.4** Foundation | — | Capability adapter pattern, skill-graph engine + executor, rule-based and LLM agents, safety enforcer, swarm coordination, observability, state persistence, memory/VLM, fleet & cloud deployment |
| **v1.x** Ship & harden | — | PyPI publishing, `apyrobo doctor`/`diagnose`, `ros2://` hardening (reconnect, health, retry/circuit-breaker), skill packaging + registry tooling |
| **v2** Adaptive intelligence | — | LLM replanning loop, VLM task verification, adaptive execution with feedback |
| **v3** Universal coverage | — | Any robot, any LLM, any compute profile in under 20 minutes; hardware auto-discovery; compute profiles |
| **v4–v7** (shipped together as package **4.0.0**) | Trust → Velocity → Compatibility → Discoverability | Deterministic failover + rollback + crash recovery + chaos tests; five-minute onboarding (`docker compose up`, `apyrobo init`, REPL, dashboard); ecosystem integrations (Isaac, Unitree, OpenCV, Ollama, WebSocket, Slack); killer demos, registry CLI, benchmark suite, LLM-optimized docs, multi-agent coordination, NL safety policies |

---

## Now — v8.0.0: The Standard Protocol (July – December 2026)

**Unlock: Permanence.** v1–v7 built the features; v8 makes APYROBO the thing other people build *on*. The goal is that "AI-native robotics orchestration" has a spec, a conformance suite, and a governance story — so that adopting APYROBO (or even forking it) is cheaper than starting from scratch, and implementations in other languages strengthen the standard instead of fragmenting it. Kubernetes won because of the API contract, not the codebase. This milestone extracts APYROBO's contract.

### Done so far (Phases 1 & 2, July 2026)

- :white_check_mark: **Slim, stable core** — bare install: 2 hard deps, seconds not minutes; `llm`/`full` extras; bare-install CI guard; SBOM in release CI; supply-chain audit ([docs/core_split_plan.md](docs/core_split_plan.md))
- :white_check_mark: **Protocol spec 1.0 — frozen** — language-agnostic spec (capability model, adapter contract, wire protocol, skill manifests) with JSON Schemas and drift-guard tests; all changes now RFC-gated ([spec/](spec/))
- :white_check_mark: **Conformance suite + badge** — `apyrobo conformance <target>` runs any adapter or server, in any language, against the spec ([docs/conformance.md](docs/conformance.md))
- :white_check_mark: **Reference TypeScript client** — zero-dependency wire-protocol client; CI runs it against the Python server on every commit ([packages/apyrobo-client-ts](packages/apyrobo-client-ts))
- :white_check_mark: **Adapter authoring kit** — `apyrobo init --adapter` scaffolds a conformance-passing package; entry-point scheme resolution ([docs/adapter_authoring.md](docs/adapter_authoring.md))
- :white_check_mark: **Protocol fuzzing** — Hypothesis property tests over parsers, server loop, URIs, and the capability-spoofing invariant; caught three real bugs
- :white_check_mark: **Flagship proof, verified and recorded** — NL task → rule agent → Nav2 `NavigateToPose` → physics TurtleBot3 in Gazebo, green in CI on every commit, with a committed recording ([demos/nav2_gazebo](demos/nav2_gazebo)); versioning reset; honesty pass over sim adapters and deployment claims

### Remaining (Phase 3 — Governance & Gravity)

Neutral governance is what lets competitors adopt the same standard. Ecosystem gravity is what makes them want to.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| :white_check_mark: | **RFC process** | Changes to `spec/` go through a public RFC template with a comment window; accepted designs recorded as ADRs. Signals that the protocol is a commons, not one maintainer's whim — the precondition for serious adopters. **Landed: RFC issue template, lifecycle doc ([docs/rfc_process.md](docs/rfc_process.md)), ADR directory ([docs/adr/](docs/adr/)) with the bootstrap ADR, and a `spec-guard` CI check that rejects spec/ PRs lacking an accepted RFC.** | |
| :construction: | **Seed the registry for real** | Publish real packages first (`ros-nav` — live Nav2 skills; the TS client) so `apyrobo registry search` returns working results. The vendor packs (`ur`, `spot`, `franka`, `px4`, `agv`, `turtlebot4`) are **reference scaffolds** ([packages/README.md](packages/README.md)) — each must be wired to its vendor SDK (and pass hardware-in-the-loop or a sim) before it ships as anything but a template. Then curate the first 10 community skills with mentored PRs. **Landed: `apyrobo registry search` now returns working results out of the box — a bundled seed index of the two real packages (with a fixed, verified-installable `ros-nav` build) serves as offline fallback, and `registry start --seed` boots a non-empty registry. Remaining: the hosted service and the mentored community skills.** | ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) |
| :white_check_mark: | **Cross-language interop showcase** | One flagship demo: the TypeScript client orchestrates a Gazebo robot through `apyrobo serve`, end to end, recorded. The "it's a protocol, not a library" proof point. **Landed: `apyrobo serve --execute` (plans *and* runs the task, outcome in `metadata.execution` — within frozen spec 1.0), the recorded [demos/ts_interop](demos/ts_interop) demo green in CI on every commit, and the `gazebo-nav-interop` CI job green — the TypeScript client drives the physics TurtleBot3 through Nav2 over WebSocket, nothing but spec-1.0 JSON crossing the boundary.** | |
| :construction: | **Spec v1.0 announcement** | Spec v1.0 is frozen; remaining: publish the conformance suite results for all first-party adapters and take it to the community: ROS Discourse, awesome-robotics lists, a launch post walking through the spec. Ends the six months with the standard *public*. **Landed: conformance results published ([docs/conformance_results.md](docs/conformance_results.md)) and kept current by a new `conformance` CI job (6 local targets) plus the suite now running against the live Gazebo robot in the `gazebo` CI job; launch post drafted ([docs/announcing_spec_1_0.md](docs/announcing_spec_1_0.md)). Remaining: review the draft and post it — publishing is a human decision.** | |
| :white_check_mark: | **Remaining demo video** | Pick-and-place in MuJoCo. **Landed, on top of `mujoco://` becoming a real physics bridge** ([apyrobo/sim/mujoco_bridge.py](apyrobo/sim/mujoco_bridge.py), pulled forward from Arc 3 — live MuJoCo stepping, blocking Nav2-style moves, suction-style grasp via a runtime weld, 21/21 conformance with 0 warnings): the recorded demo ([demos/mujoco_pickplace](demos/mujoco_pickplace)) renders the NL → plan → execute pipeline physically delivering the package, and the identical pipeline runs headless in CI on every commit (`tests/test_mujoco_bridge.py`). | |

---

### Pulled forward — the two user-acquisition wedges

Market research ([docs/positioning.md](docs/positioning.md)) identified two
adapter-shaped gaps where a distinct user population has no good option
today. Both are arc work started early, because each one is a *wedge* — a
reason for a specific kind of team to adopt APYROBO — not just a feature.

| Status | Item | Description | Label |
|--------|------|-------------|-------|
| :white_check_mark: | **VDA5050 master-control adapter** (`vda5050://`) | The industrial-AMR world is converging on [VDA5050](https://ottomotors.com/blog/interoperability-standard-vda5050/) (MQTT; MiR, OTTO/Rockwell, Seegrid targeting 2026 compliance). A compliant fleet needs a master controller — normally an expensive proprietary one. APYROBO as the open, NL-planning master control: publish `order`/`instantActions`, consume `state`/`factsheet`, no ROS required on the robot. The warehouse-startup wedge; also the first non-ROS base, proving the protocol sits above more than one middleware. **Landed: real paho-mqtt transport, blocking orders, cancelOrder safety, factsheet capabilities — verified against a simulated AGV + full conformance suite; physical-fleet validation is the Arc 1 gate.** | |
| :white_check_mark: | **Policy-backed skills** (VLA / LeRobot) | The VLA deployment literature is explicit that neural policies need runtime safety monitors — which is exactly APYROBO's execution model. A `Policy` protocol + runner that makes a learned policy (π0, OpenVLA, anything LeRobot-shaped) just another skill-graph node, with the safety enforcer wrapping every action. The research-lab and course wedge. Core stays torch-free; LeRobot is the user's dependency. **Landed: `PolicyRunner` + `run_policy` builtin with per-tick step bounds, fail-closed episodes, SafetyEnforcer composition ([docs/policy_skills.md](docs/policy_skills.md)); verified incl. an adversarial runaway policy. Remaining: a worked real-VLA-checkpoint example.** | |

## Next — the arcs after v8

The big picture, deliberately not dated: each arc has a **gate** — the
observable event that means it's done — rather than a feature checklist.
Order reflects dependency, not strict sequence; arcs overlap.

### Arc 1 — Proven on metal

Everything verified today is verified in simulation. This arc moves the
proof to physical hardware: the flagship stack on a real TurtleBot4, the
hardware-in-the-loop runner registered and green in CI (the workflow is
built and waiting — [hil-ci.yml](.github/workflows/hil-ci.yml)), safety
behaviors validated where they actually matter, and the first vendor
scaffold graduated into a real SDK-wired adapter.

**Gate: someone else's robot — not ours — runs the flagship stack, and HIL
CI goes green on every release.**

### Arc 2 — The second implementation

A standard exists the day someone we don't control ships against the spec.
This arc is community, not code: recruit and mentor the first ten real
users (labs, courses, startups); land APYROBO in a university curriculum;
get the first third-party adapter or client through conformance; seed the
registry with the first ten community skills; run the RFC process with
outside participants.

**Gate: an "APYROBO Conformant" badge earned by a team we've never met.**

### Arc 3 — Modern simulation, no stand-ins

Retire the simulation debt: port the Gazebo integration off EOL Gazebo
Classic to `gz-sim`/`ros_gz`; decide `gazebo_native://` and `isaac://`
(both still labeled in-memory stand-ins —
[apyrobo/sim/adapters.py](apyrobo/sim/adapters.py)). `mujoco://` already
graduated: it is a real physics bridge
([apyrobo/sim/mujoco_bridge.py](apyrobo/sim/mujoco_bridge.py)), conformant
with 0 warnings against live MuJoCo. Re-render the demo suite on the
modern stack.

**Gate: every URI scheme the docs advertise either drives a real
simulator/robot or no longer exists.**

### Arc 4 — Fleet-scale operations

The production story past one robot: a distributed executor tier on a real
broker (the honest successor to the removed `apyrobo worker` phantom), the
hosted skill registry running as a service, fleet-level orchestration and
dashboarding, and an operations runbook a team can be on-call against.

**Gate: ten robots, one operator, one incident-free shift — reproducible
by a team that isn't us.**

---

## Where to start contributing

- **Pick up a Phase 3 item** above — the registry seeding item is labeled
  help-wanted and is mentored.
- **Write an adapter** for hardware you own:
  [docs/adapter_authoring.md](docs/adapter_authoring.md) gets you from zero
  to conformance-passing in under an hour, and a real vendor-SDK adapter is
  the single highest-leverage contribution (it advances Arc 1 and Arc 2 at
  once).
- **Propose spec changes** via RFC: open an issue titled `RFC: <summary>`
  ([spec/README.md](spec/README.md) § Proposing changes).
- **Anything else:** open a Feature Request issue with the use case, the
  proposed approach, and alternatives considered. See
  [CONTRIBUTING.md](CONTRIBUTING.md) for setup and code style.

---

## Non-Goals

Things APYROBO intentionally does **not** aim to do:

- **Replace ROS 2** — APYROBO builds on ROS 2, it doesn't replace the middleware, drivers, or navigation stack
- **Be a simulator** — Use Gazebo, Isaac Sim, or Webots for simulation; APYROBO connects to them via adapters
- **Train models** — APYROBO orchestrates pre-trained LLMs; training happens elsewhere
- **Hardware drivers** — Write ROS 2 drivers for new hardware; APYROBO adapts the semantic layer above them
