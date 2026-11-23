"""
Demo 4: Live Fleet View
=======================
A browser canvas showing a mixed fleet (drones + ground robots) moving in
real time while an APYROBO orchestration server plans and dispatches tasks.

Everything flows over the standard wire protocol (spec/wire-protocol.md):

* the page submits tasks with the reference **TypeScript client** — the
  same JSON messages any language would send;
* plan responses are the ordinary ``status: "planned" | "error"`` messages;
* robot positions are broadcast as ``status: "telemetry"`` messages —
  an *extension* status that spec-1.0 clients MUST ignore, so this demo
  doubles as a live test of the protocol's forward-compatibility rule.

Run:
    pip install 'apyrobo[websocket]'
    npm --prefix packages/apyrobo-client-ts install
    npm --prefix packages/apyrobo-client-ts run build
    python demos/fleet_view/server.py
    → open http://localhost:8420
"""
from __future__ import annotations

import http.server
import math
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/demos/", 1)[0])  # repo root when cloned

from apyrobo import Agent, MockAdapter, Robot
from apyrobo.orchestration.adapter import (
    OrchestrationMessage,
    OrchestrationServer,
    WebSocketOrchestrationAdapter,
)

HTTP_PORT = 8420
WS_PORT = 8765
TICK_HZ = 20.0
TELEMETRY_HZ = 10.0
WORLD = (100.0, 60.0)  # metres, top-down

# Named zones drawn on the map. The rule planner reads coordinates, not
# names, so tasks give the zone's coordinates ("go to the dock at (90, 30)").
# A task the planner refuses, or one with nowhere to drive, moves no robot.
ZONES: dict[str, tuple[float, float]] = {
    "dock": (90.0, 30.0),
    "warehouse": (15.0, 12.0),
    "kitchen": (15.0, 48.0),
    "lab": (50.0, 8.0),
    "charging": (50.0, 52.0),
    "gate": (85.0, 8.0),
}

PATROL_TASKS = [
    "go to the gate at (85, 8)",
    "pick up the box at (15, 12)",
    "go to the charging station at (50, 52)",
    "deliver from the warehouse (15, 12) to the lab (50, 8)",
    "deliver from the warehouse (15, 12) to the dock (90, 30)",
    "go to the kitchen at (15, 48)",
    "go to (60, 30)",
]


# ---------------------------------------------------------------------------
# Kinematic fleet simulation
# ---------------------------------------------------------------------------

@dataclass
class SimRobot:
    robot_id: str
    kind: str            # "drone" | "ground"
    x: float
    y: float
    speed: float         # m/s
    heading: float = 0.0
    target: tuple[float, float] | None = None
    task: str = ""
    adapter: MockAdapter = field(default=None, repr=False)  # type: ignore[assignment]

    @property
    def state(self) -> str:
        return "moving" if self.target else "idle"

    def tick(self, dt: float) -> None:
        if self.target is None:
            return
        tx, ty = self.target
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        step = self.speed * dt
        if dist <= step:
            self.x, self.y = tx, ty
            self.target = None
            self.task = ""
        else:
            self.heading = math.atan2(dy, dx)
            self.x += dx / dist * step
            self.y += dy / dist * step
        # Keep the mock adapter's ground truth in sync so get_position()
        # (and anything else built on the adapter) reflects the sim.
        self.adapter._position = (self.x, self.y)
        self.adapter._orientation = self.heading


def build_fleet() -> list[SimRobot]:
    rng = random.Random(7)
    fleet: list[SimRobot] = []
    for i in range(3):
        fleet.append(SimRobot(
            robot_id=f"drone_{i:02d}", kind="drone",
            x=rng.uniform(10, 90), y=rng.uniform(10, 50), speed=9.0,
        ))
    for name in ["picker_bot", "hauler_bot", "scout_bot"]:
        fleet.append(SimRobot(
            robot_id=name, kind="ground",
            x=rng.uniform(10, 90), y=rng.uniform(10, 50), speed=3.5,
        ))
    for sim in fleet:
        sim.adapter = MockAdapter(sim.robot_id)
        sim.adapter._position = (sim.x, sim.y)
    return fleet


# ---------------------------------------------------------------------------
# Orchestration server that also *executes* plans in the sim
# ---------------------------------------------------------------------------

class FleetViewServer(OrchestrationServer):
    """Plans via the normal APYROBO pipeline, then acts the plan out by
    assigning a movement target to the addressed robot.

    A robot moves only for a plan that drives somewhere the task names. A
    task the planner refuses (Agent.last_refusal, the same check apyrobo
    mcp and apyrobo serve apply) comes back with the refusal in
    ``metadata.execution`` and no robot moves, as ``apyrobo serve
    --execute`` reports it.
    """

    def __init__(self, adapter, agent, fleet: list[SimRobot]) -> None:
        self.fleet = {sim.robot_id: sim for sim in fleet}
        first = fleet[0]
        super().__init__(
            adapter, agent,
            default_robot=Robot(f"mock://{first.robot_id}", first.adapter),
            default_robot_uri=f"mock://{first.robot_id}",
        )
        for sim in fleet:
            self._robot_cache[f"mock://{sim.robot_id}"] = Robot(
                f"mock://{sim.robot_id}", sim.adapter
            )

    def _handle(self, msg: OrchestrationMessage) -> OrchestrationMessage:
        response = super()._handle(msg)
        if response.metadata.get("status") != "planned":
            return response
        reason = self._refusal(msg, response)
        if reason is not None:
            response.metadata["execution"] = {
                "status": "failed", "steps_completed": 0, "error": reason,
            }
            return response
        self._execute(msg, response)
        return response

    def _refusal(self, msg: OrchestrationMessage, response: OrchestrationMessage) -> str | None:
        """Why this plan must not move a robot, or None."""
        if not response.metadata.get("skills"):
            return getattr(self.agent, "last_refusal", None) or (
                "Not planned: the planner returned no steps, so no robot moves."
            )
        return None

    def _execute(self, msg: OrchestrationMessage, response: OrchestrationMessage) -> None:
        robot_id = (response.robot_uri or "").partition("://")[2]
        sim = self.fleet.get(robot_id)
        if sim is None:
            return
        drives = any(
            "navigate" in s["skill_id"] for s in response.metadata.get("skills", [])
        )
        target = self._target_for(msg.task) if drives else None
        if target is None:
            # Nothing to drive to (e.g. 'report status'): the robot stays put.
            return
        sim.target = target
        sim.task = msg.task
        response.metadata["target"] = list(sim.target)

    def _target_for(self, task: str) -> tuple[float, float] | None:
        """Where the task drives: its last (x, y), else a named zone, else None.

        The last pair is the end of a delivery ('from (15, 12) to (90, 30)').
        Never a made-up waypoint.
        """
        pairs = re.findall(r"\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?", task)
        if pairs:
            x = min(max(float(pairs[-1][0]), 2.0), WORLD[0] - 2.0)
            y = min(max(float(pairs[-1][1]), 2.0), WORLD[1] - 2.0)
            return (x, y)
        lowered = task.lower()
        for zone, coords in ZONES.items():
            if zone in lowered:
                return coords
        return None


# ---------------------------------------------------------------------------
# Background threads: sim tick, telemetry broadcast, auto-tasking
# ---------------------------------------------------------------------------

def run_sim(fleet: list[SimRobot], stop: threading.Event) -> None:
    dt = 1.0 / TICK_HZ
    while not stop.is_set():
        for sim in fleet:
            sim.tick(dt)
        time.sleep(dt)


def run_telemetry(
    ws: WebSocketOrchestrationAdapter, fleet: list[SimRobot], stop: threading.Event
) -> None:
    period = 1.0 / TELEMETRY_HZ
    while not stop.is_set():
        ws.send(OrchestrationMessage(
            task="fleet telemetry",
            robot_uri="mock://fleet",
            metadata={
                "status": "telemetry",  # extension status: 1.0 clients ignore it
                "world": list(WORLD),
                "zones": {name: list(xy) for name, xy in ZONES.items()},
                "robots": [
                    {
                        "id": sim.robot_id,
                        "kind": sim.kind,
                        "x": round(sim.x, 2),
                        "y": round(sim.y, 2),
                        "heading": round(sim.heading, 3),
                        "state": sim.state,
                        "task": sim.task,
                        "target": list(sim.target) if sim.target else None,
                    }
                    for sim in fleet
                ],
            },
            source="fleet_view_demo",
        ))
        time.sleep(period)


def run_auto_tasks(
    ws: WebSocketOrchestrationAdapter, fleet: list[SimRobot], stop: threading.Event
) -> None:
    """Keep the view alive: idle robots get patrol tasks through the same
    receive→plan→respond pipeline a real client would use."""
    rng = random.Random(23)
    while not stop.is_set():
        idle = [sim for sim in fleet if sim.state == "idle"]
        if idle:
            sim = rng.choice(idle)
            ws._recv_queue.put(OrchestrationMessage(
                task=rng.choice(PATROL_TASKS),
                robot_uri=f"mock://{sim.robot_id}",
                source="auto_dispatcher",
            ))
        time.sleep(rng.uniform(1.0, 2.5))


# ---------------------------------------------------------------------------
# Static file server for the canvas page
# ---------------------------------------------------------------------------

def run_http(repo_root: Path, stop: threading.Event) -> None:
    handler = partial(
        http.server.SimpleHTTPRequestHandler, directory=str(repo_root)
    )
    handler.log_message = lambda *a, **k: None  # type: ignore[method-assign]
    with http.server.ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), handler) as httpd:
        httpd.timeout = 0.5
        while not stop.is_set():
            httpd.handle_request()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    client_dist = repo_root / "packages" / "apyrobo-client-ts" / "dist" / "index.js"
    if not client_dist.exists():
        print("The fleet view page uses the reference TypeScript client.")
        print("Build it once, then re-run:")
        print("  npm --prefix packages/apyrobo-client-ts install")
        print("  npm --prefix packages/apyrobo-client-ts run build")
        sys.exit(1)

    fleet = build_fleet()
    ws = WebSocketOrchestrationAdapter(host="127.0.0.1", port=WS_PORT)
    server = FleetViewServer(ws, Agent(provider="rule"), fleet)

    stop = threading.Event()
    threads = [
        threading.Thread(target=run_sim, args=(fleet, stop), daemon=True),
        threading.Thread(target=run_http, args=(repo_root, stop), daemon=True),
    ]
    for t in threads:
        t.start()

    print("=" * 65)
    print("  APYROBO — Live Fleet View")
    print(f"  Fleet: {len(fleet)} robots | Wire protocol: ws://127.0.0.1:{WS_PORT}")
    print(f"  Open:  http://localhost:{HTTP_PORT}/demos/fleet_view/index.html")
    print("=" * 65)

    # Telemetry + auto-tasking need the WS adapter started, which happens
    # inside server.run() — start them once the server loop is live.
    def _late_start() -> None:
        time.sleep(1.0)
        threading.Thread(
            target=run_telemetry, args=(ws, fleet, stop), daemon=True
        ).start()
        threading.Thread(
            target=run_auto_tasks, args=(ws, fleet, stop), daemon=True
        ).start()

    threading.Thread(target=_late_start, daemon=True).start()

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()


if __name__ == "__main__":
    main()
