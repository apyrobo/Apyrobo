# DRAFT — Announcing the APYROBO Protocol, spec 1.0

> **Status: draft, not yet published.** Written for a launch post +
> ROS Discourse thread. Review, edit, and post when ready — publishing is
> deliberately a human decision. Venues: ROS Discourse (General or
> Projects), the repo README/website, awesome-robotics PR, Hacker News
> "Show HN" if desired.

---

## AI-native robotics orchestration now has a spec

For the last six months we've been building APYROBO, an open framework
that turns natural-language tasks into safe, monitored robot execution:
an agent plans against a robot's declared capabilities, a skill-graph
executor runs the plan, and a safety enforcer wraps every command. Today
we're announcing the part we think matters most — **the contract is now a
frozen, language-agnostic specification**, with a conformance suite and
public governance.

Kubernetes won because of the API contract, not the codebase. Robotics
orchestration deserves the same shape: a spec you can implement in any
language, test mechanically, and trust not to shift under you.

## What's in spec 1.0

Four short documents plus JSON Schemas ([spec/](../spec/)):

- **Capability model** — what a "robot" is to an agent: capabilities,
  sensors, joints, task requests and results.
- **Adapter contract** — the behavioral contract every capability adapter
  must satisfy, including the `scheme://name` robot URI.
- **Wire protocol** — one JSON message shape, two transports (stdio
  NDJSON, WebSocket), for submitting tasks to an orchestration server.
- **Skill manifest** — the discovery format for skills.

Spec 1.0 is **frozen**. Every change — including prose clarifications —
goes through a [public RFC](rfc_process.md) with a ≥14-day comment window,
and accepted designs are recorded as [ADRs](adr/). CI mechanically rejects
spec edits that skip the process.

## Don't trust us — test it

`apyrobo conformance <target>` runs any adapter or server, in any
language, against the spec and emits a machine-readable report:

```bash
pip install 'apyrobo[conformance]'
apyrobo conformance mock://my-robot
apyrobo conformance "stdio:node my-server.js"
```

[Published results for every first-party target](conformance_results.md) —
including which adapters are real transports and which are labeled
in-memory stand-ins. The suite has already caught real bugs in our own
reference implementation; that's the point.

## Proof it's a protocol, not a library

- A **zero-dependency TypeScript client** speaks the wire protocol to the
  Python reference server — CI runs the cross-language tests on every
  commit, and the [interop demo](../demos/ts_interop/) plans *and
  executes* a task from Node with nothing but spec-1.0 JSON crossing the
  boundary.
- The **flagship stack** — NL task → rule agent → Nav2 `NavigateToPose` →
  a physics TurtleBot3 in Gazebo — runs green in CI on every commit, with
  a committed recording ([demos/nav2_gazebo](../demos/nav2_gazebo/)).
- The first **non-ROS base**: a real
  [VDA 5050](../apyrobo/core/vda5050_adapter.py) master-control adapter
  (MQTT), because the protocol sits above middleware, not inside one.

## What we're honest about

The wire protocol 1.0 is planning-focused (execution reporting rides in an
extension field; streaming execution status is reserved for a minor
revision, via RFC). Several simulator adapters are labeled in-memory
stand-ins until the modern-sim arc lands. Hardware validation beyond
simulation is the current arc. The [roadmap](../ROADMAP.md) states each
gate explicitly.

## Where to plug in

- **Write an adapter** for hardware you own —
  [docs/adapter_authoring.md](adapter_authoring.md) scaffolds a
  conformance-passing package in under an hour.
- **Implement the spec** in your language and run the suite against it;
  the "APYROBO Conformant" badge program is described in
  [docs/conformance.md](conformance.md).
- **Propose changes** via the [RFC process](rfc_process.md).

Repo: https://github.com/apyrobo/Apyrobo · License: Apache-2.0
