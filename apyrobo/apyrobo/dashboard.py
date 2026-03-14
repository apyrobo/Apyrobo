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
