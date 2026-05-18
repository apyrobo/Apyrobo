"""
Dashboard & Metrics API — FastAPI app for monitoring APYROBO.

Features:
    OB-02: Prometheus-compatible /metrics endpoint
    OB-04: Health dashboard serving router + adapter + skill metrics at /health
    OB-05: Route log export at GET /routes
    OB-09: Fleet utilization dashboard at /fleet

Requires: pip install fastapi uvicorn (optional — dashboard degrades gracefully)

Usage (standalone):
    from apyrobo.dashboard import create_app
    app = create_app(router=router, metrics=metrics, state_store=store)
    uvicorn.run(app, host="0.0.0.0", port=8080)

Usage (embedded):
    dashboard = Dashboard(router=router, metrics=metrics, state_store=store)
    dashboard.start(port=8080)  # runs in background thread
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dashboard core (no FastAPI dependency required)
# ---------------------------------------------------------------------------

class Dashboard:
    """
    Core dashboard that aggregates metrics from all APYROBO subsystems.

    Works standalone (returns dicts) or with FastAPI (create_app()).
    """

    def __init__(
        self,
        router: Any = None,
        metrics: Any = None,
        state_store: Any = None,
        replay: Any = None,
        alert_manager: Any = None,
        timeseries: Any = None,
    ) -> None:
        self.router = router
        self.metrics = metrics
        self.state_store = state_store
        self.replay = replay
        self.alert_manager = alert_manager
        self.timeseries = timeseries
        self._server_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # OB-02: Prometheus metrics
    # ------------------------------------------------------------------

    def get_metrics_text(self) -> str:
        """Return Prometheus-format metrics text."""
        if self.metrics:
            return self.metrics.prometheus_text()
        return "# No metrics collector configured\n"

    # ------------------------------------------------------------------
    # OB-04: Health dashboard
    # ------------------------------------------------------------------

    def get_health(self) -> dict[str, Any]:
        """Return full health report."""
        result: dict[str, Any] = {
            "status": "ok",
            "timestamp": time.time(),
        }

        # Router health
        if self.router:
            result["router"] = self.router.health_report()
            result["connectivity"] = self.router.connectivity_check()

        # Metrics summary
        if self.metrics:
            result["metrics"] = self.metrics.summary()

        # State store
        if self.state_store:
            result["state_store"] = {
                "task_count": self.state_store.task_count,
                "interrupted": len(self.state_store.get_interrupted_tasks()),
                "backend": type(self.state_store).__name__,
            }

        # Alerts
        if self.alert_manager:
            result["alerts"] = {
                "rules": self.alert_manager.rules,
                "recent_alerts": self.alert_manager.alert_log[-10:],
            }

        return result

    # ------------------------------------------------------------------
    # OB-05: Route log export
    # ------------------------------------------------------------------

    def get_routes(self, limit: int = 500) -> list[dict[str, Any]]:
        """Return the last N routing decisions."""
        if self.router:
            log = self.router.route_log
            return log[-limit:]
        return []

    # ------------------------------------------------------------------
    # OB-09: Fleet utilization dashboard
    # ------------------------------------------------------------------

    def get_fleet_utilization(self) -> dict[str, Any]:
        """
        OB-09: Aggregate fleet metrics across multiple robot IDs.

        Computes: uptime, tasks/hour, skill distribution per robot.
        """
        if not self.state_store:
            return {"error": "No state store configured"}

        recent_tasks = self.state_store.get_recent_tasks(limit=200)
        if not recent_tasks:
            return {"robots": {}, "totals": {"tasks": 0, "success_rate": 0.0}}

        # Group by robot
        robots: dict[str, dict[str, Any]] = {}
        for task in recent_tasks:
            rid = task.robot_id or "unknown"
            if rid not in robots:
                robots[rid] = {
                    "robot_id": rid,
                    "tasks_total": 0,
                    "tasks_completed": 0,
                    "tasks_failed": 0,
                    "first_task": task.created_at,
                    "last_task": task.updated_at,
                    "skills_used": {},
                }
            r = robots[rid]
            r["tasks_total"] += 1
            if task.status == "completed":
                r["tasks_completed"] += 1
            elif task.status == "failed":
                r["tasks_failed"] += 1
            r["first_task"] = min(r["first_task"], task.created_at)
            r["last_task"] = max(r["last_task"], task.updated_at)

            # Track skill distribution from metadata
            skill_id = task.metadata.get("skill_id", "unknown")
            r["skills_used"][skill_id] = r["skills_used"].get(skill_id, 0) + 1

        # Compute derived metrics
        for r in robots.values():
            total = r["tasks_total"]
            r["success_rate"] = round(r["tasks_completed"] / total, 3) if total > 0 else 0.0
            uptime_hours = max(0.001, (r["last_task"] - r["first_task"]) / 3600)
            r["tasks_per_hour"] = round(total / uptime_hours, 2)
            r["uptime_hours"] = round(uptime_hours, 2)

        # Totals
        total_tasks = sum(r["tasks_total"] for r in robots.values())
        total_completed = sum(r["tasks_completed"] for r in robots.values())

        return {
            "robots": robots,
            "totals": {
                "robot_count": len(robots),
                "tasks": total_tasks,
                "completed": total_completed,
                "success_rate": round(total_completed / total_tasks, 3) if total_tasks > 0 else 0.0,
            },
        }

    # ------------------------------------------------------------------
    # OB-10: Execution replay
    # ------------------------------------------------------------------

    def get_replay(self, trace_id: str) -> dict[str, Any]:
        """
        OB-10: Reconstruct a task execution from its trace_id.

        Returns a timeline of events for debugging.
        """
        if not self.replay:
            return {"error": "No replay recorder configured"}

        timeline = self.replay.get_timeline(trace_id)
        return {
            "trace_id": trace_id,
            "event_count": len(timeline),
            "timeline": timeline,
            "duration_ms": round(
                (timeline[-1]["timestamp"] - timeline[0]["timestamp"]) * 1000, 1
            ) if len(timeline) >= 2 else 0,
        }

    # ------------------------------------------------------------------
    # FastAPI app factory
    # ------------------------------------------------------------------

    def start(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Start the dashboard in a background thread."""
        def run() -> None:
            try:
                import uvicorn
                app = create_app(
                    router=self.router, metrics=self.metrics,
                    state_store=self.state_store, replay=self.replay,
                    alert_manager=self.alert_manager,
                )
                uvicorn.run(app, host=host, port=port, log_level="warning")
            except ImportError:
                logger.warning("uvicorn not installed — dashboard not available")
            except Exception as e:
                logger.error("Dashboard failed to start: %s", e)

        self._server_thread = threading.Thread(target=run, daemon=True)
        self._server_thread.start()
        logger.info("Dashboard starting on %s:%d", host, port)


# ---------------------------------------------------------------------------
# v5.0.0 Live Robot Dashboard — HTMX-powered
# ---------------------------------------------------------------------------

import collections

_MAX_HISTORY = 50  # max skill executions tracked
_MAX_EVENTS = 100  # max safety events tracked


class RobotDashboard:
    """Live dashboard that watches a connected Robot and buffers recent events.

    Attach to an observability event stream so ``record_skill()`` and
    ``record_safety_event()`` are called as skills run and safety events fire.
    When used standalone (e.g. in the demo compose), the dashboard polls the
    robot on each request instead.

    Usage::

        robot = Robot.discover("mock://turtlebot4")
        dash = RobotDashboard(robot)
        app = dash.create_fastapi_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """

    def __init__(self, robot: Any, robot_uri: str = "") -> None:
        self.robot = robot
        self.robot_uri = robot_uri
        self._skill_history: collections.deque = collections.deque(maxlen=_MAX_HISTORY)
        self._safety_events: collections.deque = collections.deque(maxlen=_MAX_EVENTS)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Event recording (can be called from skill executor hooks)
    # ------------------------------------------------------------------

    def record_skill(
        self,
        skill_id: str,
        status: str,
        elapsed_ms: float,
        params: dict | None = None,
    ) -> None:
        with self._lock:
            self._skill_history.append({
                "skill_id": skill_id,
                "status": status,
                "elapsed_ms": round(elapsed_ms, 1),
                "params": params or {},
                "timestamp": time.time(),
            })

    def record_safety_event(self, event_type: str, detail: str) -> None:
        with self._lock:
            self._safety_events.append({
                "event_type": event_type,
                "detail": detail,
                "timestamp": time.time(),
            })

    # ------------------------------------------------------------------
    # Data accessors
    # ------------------------------------------------------------------

    def get_robot_status(self) -> dict[str, Any]:
        try:
            caps = self.robot.capabilities()
            cap_list = [{"name": c.name, "type": c.capability_type.value} for c in caps.capabilities]
            pos = None
            try:
                pos = self.robot.get_position()
            except Exception:
                pass
            battery = None
            try:
                battery = self.robot.get_battery_level()
            except Exception:
                pass
            return {
                "name": caps.name,
                "robot_id": caps.robot_id,
                "uri": self.robot_uri,
                "max_speed": caps.max_speed,
                "capabilities": cap_list,
                "position": {"x": round(pos[0], 2), "y": round(pos[1], 2)} if pos else None,
                "battery_pct": round(battery * 100) if battery is not None else None,
                "connected": True,
            }
        except Exception as exc:
            return {"connected": False, "error": str(exc)}

    def get_skill_history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(list(self._skill_history)))

    def get_safety_events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(list(self._safety_events)))

    def get_available_skills(self) -> list[dict[str, Any]]:
        from apyrobo.skills.skill import BUILTIN_SKILLS
        return [
            {
                "skill_id": s.skill_id,
                "name": s.name,
                "description": s.description,
                "type": s.skill_type.value if hasattr(s, "skill_type") else "custom",
            }
            for s in BUILTIN_SKILLS.values()
        ]

    # ------------------------------------------------------------------
    # FastAPI app
    # ------------------------------------------------------------------

    def create_fastapi_app(self) -> Any:
        try:
            from fastapi import FastAPI
            from fastapi.responses import HTMLResponse, JSONResponse
        except ImportError:
            raise RuntimeError(
                "fastapi is required for the dashboard.\n"
                "Install with: pip install 'apyrobo[dashboard]'"
            )

        app = FastAPI(title="APYROBO Live Dashboard", version="5.0.0")

        @app.get("/health")
        def health() -> dict:
            return {"status": "ok", "timestamp": time.time()}

        @app.get("/api/status")
        def api_status() -> dict:
            return self.get_robot_status()

        @app.get("/api/skills/history")
        def api_skill_history() -> list:
            return self.get_skill_history()

        @app.get("/api/skills/available")
        def api_skills_available() -> list:
            return self.get_available_skills()

        @app.get("/api/safety/events")
        def api_safety_events() -> list:
            return self.get_safety_events()

        @app.get("/partials/status", response_class=HTMLResponse)
        def partial_status() -> str:
            s = self.get_robot_status()
            return _render_status_partial(s)

        @app.get("/partials/history", response_class=HTMLResponse)
        def partial_history() -> str:
            history = self.get_skill_history()
            return _render_history_partial(history)

        @app.get("/partials/safety", response_class=HTMLResponse)
        def partial_safety() -> str:
            events = self.get_safety_events()
            return _render_safety_partial(events)

        @app.get("/partials/skills", response_class=HTMLResponse)
        def partial_skills() -> str:
            skills = self.get_available_skills()
            return _render_skills_partial(skills)

        @app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return _render_live_dashboard_html(self.robot_uri)

        return app


def _ts(t: float) -> str:
    """Format a unix timestamp as HH:MM:SS."""
    import datetime
    return datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S")


def _render_status_partial(s: dict) -> str:
    if not s.get("connected"):
        return f'<div class="panel-error">⚠ Not connected: {s.get("error", "")}</div>'
    pos = s.get("position")
    pos_str = f"({pos['x']}, {pos['y']})" if pos else "—"
    batt = s.get("battery_pct")
    batt_str = f"{batt}%" if batt is not None else "—"
    caps = ", ".join(c["type"] for c in s.get("capabilities", []))
    return (
        f'<div class="kv"><span>Name</span><span>{s["name"]}</span></div>'
        f'<div class="kv"><span>URI</span><span>{s["uri"]}</span></div>'
        f'<div class="kv"><span>Position</span><span>{pos_str}</span></div>'
        f'<div class="kv"><span>Battery</span><span>{batt_str}</span></div>'
        f'<div class="kv"><span>Capabilities</span><span>{caps}</span></div>'
        f'<div class="kv"><span>Max speed</span><span>{s.get("max_speed", "—")} m/s</span></div>'
    )


def _render_history_partial(history: list) -> str:
    if not history:
        return '<div class="muted">No skills executed yet.</div>'
    rows = ""
    for h in history[:20]:
        icon = "✅" if h["status"] == "ok" else "❌"
        rows += (
            f'<tr><td>{_ts(h["timestamp"])}</td>'
            f'<td class="skill-id">{h["skill_id"]}</td>'
            f'<td>{icon} {h["status"]}</td>'
            f'<td>{h["elapsed_ms"]}ms</td></tr>'
        )
    return f"<table><tr><th>Time</th><th>Skill</th><th>Status</th><th>Time</th></tr>{rows}</table>"


def _render_safety_partial(events: list) -> str:
    if not events:
        return '<div class="muted ok">✅ No safety events.</div>'
    rows = ""
    for e in events[:15]:
        rows += (
            f'<tr><td>{_ts(e["timestamp"])}</td>'
            f'<td class="warn">{e["event_type"]}</td>'
            f'<td>{e["detail"]}</td></tr>'
        )
    return f"<table><tr><th>Time</th><th>Event</th><th>Detail</th></tr>{rows}</table>"


def _render_skills_partial(skills: list) -> str:
    if not skills:
        return '<div class="muted">No skills loaded.</div>'
    rows = ""
    for s in skills[:30]:
        rows += (
            f'<tr><td class="skill-id">{s["skill_id"]}</td>'
            f'<td>{s["name"]}</td>'
            f'<td class="muted">{s.get("description", "")[:60]}</td></tr>'
        )
    return (
        f"<div class='muted'>{len(skills)} built-in skills</div>"
        f"<table><tr><th>ID</th><th>Name</th><th>Description</th></tr>{rows}</table>"
    )


def _render_live_dashboard_html(robot_uri: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>APYROBO Dashboard</title>
<script src="https://unpkg.com/htmx.org@1.9.12"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Courier New', monospace; background: #0d1117; color: #c9d1d9; padding: 16px; }}
header {{ display: flex; align-items: center; gap: 12px; padding-bottom: 12px;
          border-bottom: 1px solid #21262d; margin-bottom: 16px; }}
h1 {{ color: #58a6ff; font-size: 1.1rem; }}
.dot {{ width: 10px; height: 10px; border-radius: 50%; background: #3fb950; display: inline-block; }}
.uri {{ color: #8b949e; font-size: 0.85rem; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.panel {{ background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 12px; }}
.panel h2 {{ color: #7ee787; font-size: 0.85rem; margin-bottom: 10px;
             border-bottom: 1px solid #21262d; padding-bottom: 6px; }}
.kv {{ display: flex; justify-content: space-between; padding: 3px 0;
       border-bottom: 1px solid #21262d08; font-size: 0.82rem; }}
.kv span:first-child {{ color: #8b949e; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
th {{ color: #8b949e; text-align: left; padding: 4px 6px; border-bottom: 1px solid #21262d; }}
td {{ padding: 3px 6px; border-bottom: 1px solid #21262d18; }}
.skill-id {{ color: #58a6ff; font-family: monospace; }}
.muted {{ color: #8b949e; font-size: 0.82rem; padding: 4px 0; }}
.muted.ok {{ color: #3fb950; }}
.warn {{ color: #f85149; }}
.panel-error {{ color: #f85149; font-size: 0.85rem; }}
footer {{ margin-top: 16px; color: #8b949e; font-size: 0.75rem; text-align: center; }}
a {{ color: #58a6ff; }}
</style>
</head>
<body>
<header>
  <span class="dot"></span>
  <h1>APYROBO Dashboard</h1>
  <span class="uri">{robot_uri}</span>
</header>
<div class="grid">
  <div class="panel">
    <h2>🤖 Robot Status</h2>
    <div id="status-panel"
         hx-get="/partials/status"
         hx-trigger="load, every 5s"
         hx-swap="innerHTML">
      Loading…
    </div>
  </div>
  <div class="panel">
    <h2>🛡 Safety Events</h2>
    <div id="safety-panel"
         hx-get="/partials/safety"
         hx-trigger="load, every 5s"
         hx-swap="innerHTML">
      Loading…
    </div>
  </div>
  <div class="panel">
    <h2>📋 Skill History</h2>
    <div id="history-panel"
         hx-get="/partials/history"
         hx-trigger="load, every 3s"
         hx-swap="innerHTML">
      Loading…
    </div>
  </div>
  <div class="panel">
    <h2>⚙ Available Skills</h2>
    <div id="skills-panel"
         hx-get="/partials/skills"
         hx-trigger="load"
         hx-swap="innerHTML">
      Loading…
    </div>
  </div>
</div>
<footer>
  <a href="/api/status">/api/status</a> ·
  <a href="/api/skills/history">/api/skills/history</a> ·
  <a href="/api/safety/events">/api/safety/events</a> ·
  <a href="/health">/health</a>
  — APYROBO v5.0.0
</footer>
</body>
</html>"""


def create_app(
    router: Any = None,
    metrics: Any = None,
    state_store: Any = None,
    replay: Any = None,
    alert_manager: Any = None,
) -> Any:
    """
    Create a FastAPI app for the APYROBO dashboard.

    Returns a FastAPI app instance.
    Requires: pip install fastapi
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse
    except ImportError:
        raise RuntimeError("fastapi is required for the dashboard. Install with: pip install fastapi")

    dashboard = Dashboard(
        router=router, metrics=metrics,
        state_store=state_store, replay=replay,
        alert_manager=alert_manager,
    )

    app = FastAPI(title="APYROBO Dashboard", version="0.1.0")

    # OB-02: Prometheus metrics
    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics_endpoint() -> str:
        return dashboard.get_metrics_text()

    # OB-04: Health dashboard
    @app.get("/health")
    def health_endpoint() -> dict[str, Any]:
        return dashboard.get_health()

    # OB-04: HTML health page
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        health = dashboard.get_health()
        return _render_dashboard_html(health)

    # OB-05: Route log export
    @app.get("/routes")
    def routes_endpoint(limit: int = 500) -> list[dict[str, Any]]:
        return dashboard.get_routes(limit)

    # OB-09: Fleet utilization
    @app.get("/fleet")
    def fleet_endpoint() -> dict[str, Any]:
        return dashboard.get_fleet_utilization()

    # OB-10: Execution replay
    @app.get("/replay/{trace_id}")
    def replay_endpoint(trace_id: str) -> dict[str, Any]:
        return dashboard.get_replay(trace_id)

    # Alerts
    @app.get("/alerts")
    def alerts_endpoint() -> dict[str, Any]:
        if alert_manager:
            return {"rules": alert_manager.rules, "log": alert_manager.alert_log}
        return {"rules": [], "log": []}

    return app


def _render_dashboard_html(health: dict[str, Any]) -> str:
    """Generate a simple HTML dashboard page."""
    status_color = "#4CAF50" if health.get("status") == "ok" else "#f44336"

    tiers_html = ""
    if "router" in health:
        for tier in health["router"].get("tiers", []):
            state = tier.get("circuit_state", "unknown")
            color = "#4CAF50" if state == "closed" else "#f44336" if state == "open" else "#ff9800"
            tiers_html += (
                f"<tr><td>{tier['name']}</td>"
                f"<td style='color:{color}'>{state}</td>"
                f"<td>{tier.get('avg_latency_ms', 0):.0f}ms</td>"
                f"<td>{tier.get('error_rate', 0):.1%}</td>"
                f"<td>{tier.get('total_calls', 0)}</td></tr>"
            )

    metrics_html = ""
    if "metrics" in health:
        m = health["metrics"]
        metrics_html = (
            f"<p>Skill executions: {m.get('total_skill_executions', 0)} "
            f"(success rate: {m.get('skill_success_rate', 0):.1%})</p>"
            f"<p>Graph executions: {m.get('total_graph_executions', 0)} "
            f"(success rate: {m.get('graph_success_rate', 0):.1%})</p>"
        )

    return f"""<!DOCTYPE html>
<html>
<head><title>APYROBO Dashboard</title>
<style>
body {{ font-family: monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
h1 {{ color: #00d4ff; }}
h2 {{ color: #7b68ee; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #333; padding: 8px; text-align: left; }}
th {{ background: #16213e; }}
.status {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%;
           background: {status_color}; margin-right: 8px; }}
a {{ color: #00d4ff; }}
</style></head>
<body>
<h1><span class="status"></span>APYROBO Dashboard</h1>
<p>Status: {health.get('status', 'unknown')}</p>

<h2>Inference Tiers</h2>
<table>
<tr><th>Tier</th><th>Circuit</th><th>Latency</th><th>Error Rate</th><th>Calls</th></tr>
{tiers_html}
</table>

<h2>Metrics</h2>
{metrics_html}

<h2>Links</h2>
<ul>
<li><a href="/metrics">/metrics</a> — Prometheus metrics</li>
<li><a href="/routes">/routes</a> — Routing decisions</li>
<li><a href="/fleet">/fleet</a> — Fleet utilization</li>
<li><a href="/alerts">/alerts</a> — Alert rules &amp; log</li>
</ul>
</body></html>"""
