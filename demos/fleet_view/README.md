# Demo: Live Fleet View

A browser canvas showing a mixed fleet — 3 drones + 3 ground robots —
moving in real time while an APYROBO orchestration server plans and
dispatches their tasks. Everything you see travels over the standard
[wire protocol](../../spec/wire-protocol.md), rendered by the
[reference TypeScript client](../../packages/apyrobo-client-ts).

Drones (triangles) and ground robots (squares) move across a top-down world
with named zones, motion trails, and dashed lines to their current targets;
a side panel streams the planner's responses and lets you dispatch your own
tasks. Run it (below) to see it live.

## What it demonstrates

- **Real rendering**, not a terminal log: robots move on a top-down world
  with trails and target lines; drones are triangles, ground robots are
  squares.
- **Natural language → skills → motion.** Type "deliver a package to the
  dock" and the planner returns a real multi-step plan
  (`Navigate To → Pick Object → Navigate To → Place Object`); the robot
  then drives to the dock. Simpler tasks like "patrol the perimeter"
  resolve to a shorter plan. The planner maps task semantics to each
  robot's declared skills.
- **Coordination.** An auto-dispatcher keeps idle robots busy by submitting
  tasks through the same `receive → plan → respond` loop a human client
  uses; you can interleave your own tasks from the panel at any time.
- **Protocol forward-compatibility, live.** Robot positions stream as
  `status: "telemetry"` messages — an *extension* status. The spec says
  1.0 clients MUST ignore statuses they don't recognize, and the reference
  client does exactly that (telemetry drives the canvas; only
  `planned`/`error` resolve a task). So this page is also a running test of
  the rule that lets the protocol evolve without breaking clients.

## Run it

```bash
pip install -e '.[websocket]'                    # from the repo root
npm --prefix packages/apyrobo-client-ts install
npm --prefix packages/apyrobo-client-ts run build
python demos/fleet_view/server.py
# → open http://localhost:8420/demos/fleet_view/index.html
```

`./record.sh` does the build + launch and prints the URL.

## How it's wired

```
 index.html ──ApyroboClient (ws://…)──►  WebSocketOrchestrationAdapter
   canvas   ◄──── telemetry (10 Hz) ────┐         │
                                        │    FleetViewServer.plan()  ← Agent
   task box ──── submitTask() ──────────┘         │
                                          assigns movement target → sim tick (20 Hz)
```

`server.py` subclasses `OrchestrationServer`: it plans through the normal
APYROBO pipeline, then *acts the plan out* by giving the addressed robot a
target derived from the task text (a named zone, explicit `(x, y)`, or a
random waypoint). Swap the kinematic sim for real adapters and the same
page drives real hardware.
