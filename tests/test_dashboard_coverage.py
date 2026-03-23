"""
Comprehensive tests for apyrobo/dashboard.py.

Covers:
  - Dashboard.__init__ (no args and with all subsystems)
  - Dashboard.get_metrics_text (with/without metrics)
  - Dashboard.get_health (with/without router, metrics, state_store, alert_manager)
  - Dashboard.get_routes (with/without router, custom limit)
  - Dashboard.get_fleet_utilization (with/without state_store, empty tasks, full fleet)
  - Dashboard.get_replay (with/without replay recorder, single-event timeline)
  - Dashboard.start() / stop() (background thread lifecycle, ImportError path)
  - create_app() (with FastAPI present; RuntimeError when FastAPI absent)
  - All FastAPI endpoint handlers exercised via the Dashboard methods directly
  - _render_dashboard_html helper (via create_app index route or direct call)
"""

from __future__ import annotations

import sys
import threading
import time
import types
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Helpers — build lightweight mock subsystems
# ---------------------------------------------------------------------------

def _make_router(
    route_log: list | None = None,
    health_report: dict | None = None,
    connectivity_check: dict | None = None,
) -> MagicMock:
    r = MagicMock()
    r.route_log = route_log if route_log is not None else []
    r.health_report.return_value = health_report or {"tiers": []}
    r.connectivity_check.return_value = connectivity_check or {"ok": True}
    return r


def _make_metrics(prometheus_text: str = "# metrics\n", summary: dict | None = None) -> MagicMock:
    m = MagicMock()
    m.prometheus_text.return_value = prometheus_text
    m.summary.return_value = summary or {
        "total_skill_executions": 5,
        "skill_success_rate": 0.8,
        "total_graph_executions": 2,
        "graph_success_rate": 1.0,
    }
    return m


def _make_state_store(tasks: list | None = None, task_count: int = 3) -> MagicMock:
    ss = MagicMock()
    ss.task_count = task_count
    ss.get_interrupted_tasks.return_value = []
    ss.get_recent_tasks.return_value = tasks or []
    return ss


def _make_alert_manager(rules: list | None = None, alert_log: list | None = None) -> MagicMock:
    am = MagicMock()
    am.rules = rules or ["rule_a"]
    am.alert_log = alert_log or [{"level": "warn", "msg": "low battery"}]
    return am


_DEFAULT_TIMELINE = [
    {"timestamp": 1_000.0, "event": "start"},
    {"timestamp": 1_001.5, "event": "end"},
]

def _make_replay(timeline: list | None = None) -> MagicMock:
    rp = MagicMock()
    rp.get_timeline.return_value = _DEFAULT_TIMELINE if timeline is None else timeline
    return rp


def _make_task(
    robot_id: str = "tb4",
    status: str = "completed",
    created_at: float = 1_000.0,
    updated_at: float = 1_001.0,
    skill_id: str = "pick",
) -> MagicMock:
    t = MagicMock()
    t.robot_id = robot_id
    t.status = status
    t.created_at = created_at
    t.updated_at = updated_at
    t.metadata = {"skill_id": skill_id}
    return t


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from apyrobo.dashboard import Dashboard, create_app, _render_dashboard_html  # noqa: E402


# ===========================================================================
# Dashboard.__init__
# ===========================================================================

class TestDashboardInit:
    def test_no_args(self) -> None:
        d = Dashboard()
        assert d.router is None
        assert d.metrics is None
        assert d.state_store is None
        assert d.replay is None
        assert d.alert_manager is None
        assert d.timeseries is None
        assert d._server_thread is None

    def test_all_args(self) -> None:
        router = _make_router()
        metrics = _make_metrics()
        ss = _make_state_store()
        rp = _make_replay()
        am = _make_alert_manager()
        ts = MagicMock()

        d = Dashboard(
            router=router,
            metrics=metrics,
            state_store=ss,
            replay=rp,
            alert_manager=am,
            timeseries=ts,
        )
        assert d.router is router
        assert d.metrics is metrics
        assert d.state_store is ss
        assert d.replay is rp
        assert d.alert_manager is am
        assert d.timeseries is ts


# ===========================================================================
# Dashboard.get_metrics_text
# ===========================================================================

class TestGetMetricsText:
    def test_without_metrics_returns_placeholder(self) -> None:
        d = Dashboard()
        text = d.get_metrics_text()
        assert "No metrics" in text

    def test_with_metrics_calls_prometheus_text(self) -> None:
        metrics = _make_metrics("# real metrics\nmy_counter 42\n")
        d = Dashboard(metrics=metrics)
        text = d.get_metrics_text()
        assert text == "# real metrics\nmy_counter 42\n"
        metrics.prometheus_text.assert_called_once()


# ===========================================================================
# Dashboard.get_health
# ===========================================================================

class TestGetHealth:
    def test_minimal_no_subsystems(self) -> None:
        d = Dashboard()
        health = d.get_health()
        assert health["status"] == "ok"
        assert "timestamp" in health
        assert "router" not in health
        assert "metrics" not in health
        assert "state_store" not in health
        assert "alerts" not in health

    def test_with_router(self) -> None:
        router = _make_router(
            health_report={"tiers": [{"name": "local", "circuit_state": "closed"}]},
            connectivity_check={"ok": True},
        )
        d = Dashboard(router=router)
        health = d.get_health()
        assert "router" in health
        assert "connectivity" in health
        router.health_report.assert_called_once()
        router.connectivity_check.assert_called_once()

    def test_with_metrics(self) -> None:
        metrics = _make_metrics()
        d = Dashboard(metrics=metrics)
        health = d.get_health()
        assert "metrics" in health
        metrics.summary.assert_called_once()

    def test_with_state_store(self) -> None:
        ss = _make_state_store(task_count=7)
        ss.get_interrupted_tasks.return_value = [MagicMock(), MagicMock()]
        d = Dashboard(state_store=ss)
        health = d.get_health()
        assert health["state_store"]["task_count"] == 7
        assert health["state_store"]["interrupted"] == 2
        assert "backend" in health["state_store"]

    def test_with_alert_manager(self) -> None:
        am = _make_alert_manager(
            rules=["rule1", "rule2"],
            alert_log=[{"level": "critical", "msg": "stuck"}] * 12,
        )
        d = Dashboard(alert_manager=am)
        health = d.get_health()
        assert "alerts" in health
        # Only the last 10 entries are returned
        assert len(health["alerts"]["recent_alerts"]) == 10
        assert health["alerts"]["rules"] == ["rule1", "rule2"]

    def test_all_subsystems_present(self) -> None:
        d = Dashboard(
            router=_make_router(),
            metrics=_make_metrics(),
            state_store=_make_state_store(),
            alert_manager=_make_alert_manager(),
        )
        health = d.get_health()
        for key in ("router", "connectivity", "metrics", "state_store", "alerts"):
            assert key in health


# ===========================================================================
# Dashboard.get_routes
# ===========================================================================

class TestGetRoutes:
    def test_without_router_returns_empty(self) -> None:
        d = Dashboard()
        assert d.get_routes() == []

    def test_with_router_returns_log(self) -> None:
        log = [{"skill": "pick", "adapter": "local"}] * 10
        d = Dashboard(router=_make_router(route_log=log))
        result = d.get_routes()
        assert result == log

    def test_limit_is_applied(self) -> None:
        log = [{"idx": i} for i in range(100)]
        d = Dashboard(router=_make_router(route_log=log))
        result = d.get_routes(limit=5)
        # Should return last 5
        assert len(result) == 5
        assert result[-1]["idx"] == 99

    def test_limit_larger_than_log(self) -> None:
        log = [{"idx": i} for i in range(3)]
        d = Dashboard(router=_make_router(route_log=log))
        result = d.get_routes(limit=500)
        assert len(result) == 3


# ===========================================================================
# Dashboard.get_fleet_utilization  (OB-09)
# ===========================================================================

class TestGetFleetUtilization:
    def test_without_state_store_returns_error(self) -> None:
        d = Dashboard()
        result = d.get_fleet_utilization()
        assert "error" in result

    def test_empty_tasks_returns_zero_totals(self) -> None:
        ss = _make_state_store(tasks=[])
        d = Dashboard(state_store=ss)
        result = d.get_fleet_utilization()
        assert result["totals"]["tasks"] == 0
        assert result["robots"] == {}

    def test_single_robot_completed_tasks(self) -> None:
        tasks = [
            _make_task("tb4", "completed", 1000.0, 1010.0, "pick"),
            _make_task("tb4", "completed", 1020.0, 1030.0, "place"),
            _make_task("tb4", "failed",    1040.0, 1050.0, "pick"),
        ]
        ss = _make_state_store(tasks=tasks)
        d = Dashboard(state_store=ss)
        result = d.get_fleet_utilization()
        robot = result["robots"]["tb4"]
        assert robot["tasks_total"] == 3
        assert robot["tasks_completed"] == 2
        assert robot["tasks_failed"] == 1
        assert "success_rate" in robot
        assert "tasks_per_hour" in robot
        assert "uptime_hours" in robot

    def test_multiple_robots(self) -> None:
        tasks = [
            _make_task("tb4",  "completed", 1000.0, 1010.0),
            _make_task("spot", "completed", 2000.0, 2010.0),
            _make_task("tb4",  "failed",    1020.0, 1030.0),
        ]
        ss = _make_state_store(tasks=tasks)
        d = Dashboard(state_store=ss)
        result = d.get_fleet_utilization()
        assert len(result["robots"]) == 2
        assert result["totals"]["robot_count"] == 2
        assert result["totals"]["tasks"] == 3

    def test_unknown_robot_id(self) -> None:
        t = _make_task(status="completed", created_at=1000.0, updated_at=1001.0)
        t.robot_id = None  # type: ignore[assignment]
        ss = _make_state_store(tasks=[t])
        d = Dashboard(state_store=ss)
        result = d.get_fleet_utilization()
        assert "unknown" in result["robots"]

    def test_skill_distribution_tracked(self) -> None:
        tasks = [
            _make_task("tb4", "completed", 1000.0, 1010.0, "pick"),
            _make_task("tb4", "completed", 1020.0, 1030.0, "pick"),
            _make_task("tb4", "completed", 1040.0, 1050.0, "place"),
        ]
        ss = _make_state_store(tasks=tasks)
        d = Dashboard(state_store=ss)
        result = d.get_fleet_utilization()
        skills = result["robots"]["tb4"]["skills_used"]
        assert skills["pick"] == 2
        assert skills["place"] == 1

    def test_success_rate_computation(self) -> None:
        tasks = [
            _make_task("tb4", "completed", 1000.0, 1060.0),
            _make_task("tb4", "completed", 1060.0, 1120.0),
        ]
        ss = _make_state_store(tasks=tasks)
        d = Dashboard(state_store=ss)
        result = d.get_fleet_utilization()
        robot = result["robots"]["tb4"]
        assert robot["success_rate"] == pytest.approx(1.0, abs=0.01)

    def test_totals_success_rate_with_no_tasks(self) -> None:
        """Edge: empty task list produces zero success_rate without ZeroDivisionError."""
        ss = _make_state_store(tasks=[])
        d = Dashboard(state_store=ss)
        result = d.get_fleet_utilization()
        assert result["totals"]["success_rate"] == 0.0


# ===========================================================================
# Dashboard.get_replay  (OB-10)
# ===========================================================================

class TestGetReplay:
    def test_without_replay_returns_error(self) -> None:
        d = Dashboard()
        result = d.get_replay("trace-abc")
        assert "error" in result

    def test_with_replay_two_events(self) -> None:
        timeline = [
            {"timestamp": 1_000.0, "event": "start"},
            {"timestamp": 1_002.5, "event": "end"},
        ]
        rp = _make_replay(timeline=timeline)
        d = Dashboard(replay=rp)
        result = d.get_replay("trace-xyz")
        assert result["trace_id"] == "trace-xyz"
        assert result["event_count"] == 2
        assert result["duration_ms"] == pytest.approx(2500.0, abs=1.0)
        rp.get_timeline.assert_called_once_with("trace-xyz")

    def test_with_replay_single_event_zero_duration(self) -> None:
        timeline = [{"timestamp": 1_000.0, "event": "only"}]
        rp = _make_replay(timeline=timeline)
        d = Dashboard(replay=rp)
        result = d.get_replay("trace-1")
        assert result["duration_ms"] == 0
        assert result["event_count"] == 1

    def test_with_replay_empty_timeline(self) -> None:
        rp = _make_replay(timeline=[])
        d = Dashboard(replay=rp)
        result = d.get_replay("trace-empty")
        assert result["duration_ms"] == 0
        assert result["event_count"] == 0


# ===========================================================================
# Dashboard.start  (background thread lifecycle)
# ===========================================================================

class TestDashboardStart:
    def test_start_sets_server_thread(self) -> None:
        """start() creates a daemon background thread."""
        d = Dashboard()
        # Patch uvicorn so the background thread exits immediately after being entered
        import sys
        fake_uvicorn = types.ModuleType("uvicorn")
        fake_uvicorn.run = MagicMock()  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"uvicorn": fake_uvicorn}):
            with patch("apyrobo.dashboard.create_app", return_value=MagicMock()):
                d.start(host="127.0.0.1", port=19999)
                time.sleep(0.1)

        assert d._server_thread is not None
        assert isinstance(d._server_thread, threading.Thread)
        assert d._server_thread.daemon is True

    def test_start_handles_uvicorn_import_error(self) -> None:
        """Dashboard.start logs a warning when uvicorn is not installed."""
        import sys
        d = Dashboard()
        # Remove uvicorn from sys.modules so 'import uvicorn' raises ImportError
        original = sys.modules.pop("uvicorn", None)
        try:
            with patch.dict(sys.modules, {"uvicorn": None}):  # type: ignore[dict-item]
                d.start(port=19998)
                time.sleep(0.1)
        finally:
            if original is not None:
                sys.modules["uvicorn"] = original

        assert d._server_thread is not None

    def test_start_handles_generic_exception(self) -> None:
        """Exceptions inside the thread are swallowed and logged."""
        import sys
        d = Dashboard()
        fake_uvicorn = types.ModuleType("uvicorn")
        fake_uvicorn.run = MagicMock(side_effect=RuntimeError("crash"))  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"uvicorn": fake_uvicorn}):
            with patch("apyrobo.dashboard.create_app", return_value=MagicMock()):
                d.start(port=19997)
                time.sleep(0.1)

        assert d._server_thread is not None


# ===========================================================================
# create_app()
# ===========================================================================

class TestCreateApp:
    def test_raises_when_fastapi_missing(self) -> None:
        """create_app raises RuntimeError when FastAPI is not installed."""
        with patch.dict(sys.modules, {"fastapi": None}):
            with pytest.raises((RuntimeError, ImportError)):
                create_app()

    def test_returns_fastapi_app(self) -> None:
        try:
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("FastAPI not installed")

        app = create_app()
        assert app is not None
        # FastAPI apps have a 'routes' attribute
        assert hasattr(app, "routes")

    def test_app_has_expected_routes(self) -> None:
        try:
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("FastAPI not installed")

        app = create_app(
            router=_make_router(),
            metrics=_make_metrics(),
            state_store=_make_state_store(),
            alert_manager=_make_alert_manager(),
        )
        route_paths = {r.path for r in app.routes}  # type: ignore[attr-defined]
        assert "/metrics" in route_paths
        assert "/health" in route_paths
        assert "/routes" in route_paths
        assert "/fleet" in route_paths
        assert "/alerts" in route_paths

    def test_app_metrics_endpoint(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        metrics = _make_metrics("# test\nfoo 1\n")
        app = create_app(metrics=metrics)
        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "foo" in resp.text

    def test_app_health_endpoint(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        app = create_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_app_index_returns_html(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        app = create_app()
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "APYROBO" in resp.text

    def test_app_routes_endpoint(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        log = [{"skill": "pick"}] * 5
        router = _make_router(route_log=log)
        app = create_app(router=router)
        client = TestClient(app)
        resp = client.get("/routes?limit=3")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) <= 3

    def test_app_fleet_endpoint(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        app = create_app()
        client = TestClient(app)
        resp = client.get("/fleet")
        assert resp.status_code == 200

    def test_app_alerts_endpoint_with_manager(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        am = _make_alert_manager(rules=["r1"], alert_log=[{"msg": "test"}])
        app = create_app(alert_manager=am)
        client = TestClient(app)
        resp = client.get("/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert "rules" in data

    def test_app_alerts_endpoint_without_manager(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        app = create_app()
        client = TestClient(app)
        resp = client.get("/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["rules"] == []
        assert data["log"] == []

    def test_app_replay_endpoint(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        rp = _make_replay()
        app = create_app(replay=rp)
        client = TestClient(app)
        resp = client.get("/replay/trace-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trace_id"] == "trace-001"


# ===========================================================================
# _render_dashboard_html helper
# ===========================================================================

class TestRenderDashboardHtml:
    def test_status_ok_produces_green(self) -> None:
        html = _render_dashboard_html({"status": "ok"})
        assert "#4CAF50" in html
        assert "ok" in html

    def test_status_error_produces_red(self) -> None:
        html = _render_dashboard_html({"status": "error"})
        assert "#f44336" in html

    def test_router_tiers_rendered(self) -> None:
        health = {
            "status": "ok",
            "router": {
                "tiers": [
                    {
                        "name": "local",
                        "circuit_state": "closed",
                        "avg_latency_ms": 5.2,
                        "error_rate": 0.01,
                        "total_calls": 100,
                    }
                ]
            },
        }
        html = _render_dashboard_html(health)
        assert "local" in html
        assert "closed" in html
        assert "#4CAF50" in html  # green for closed circuit

    def test_router_tier_open_circuit_is_red(self) -> None:
        health = {
            "status": "ok",
            "router": {
                "tiers": [
                    {
                        "name": "cloud",
                        "circuit_state": "open",
                        "avg_latency_ms": 0.0,
                        "error_rate": 0.5,
                        "total_calls": 10,
                    }
                ]
            },
        }
        html = _render_dashboard_html(health)
        assert "cloud" in html
        assert "#f44336" in html

    def test_router_tier_half_open_circuit_is_orange(self) -> None:
        health = {
            "status": "ok",
            "router": {
                "tiers": [
                    {
                        "name": "edge",
                        "circuit_state": "half_open",
                        "avg_latency_ms": 10.0,
                        "error_rate": 0.05,
                        "total_calls": 50,
                    }
                ]
            },
        }
        html = _render_dashboard_html(health)
        assert "#ff9800" in html  # orange for half-open

    def test_metrics_section_rendered(self) -> None:
        health = {
            "status": "ok",
            "metrics": {
                "total_skill_executions": 42,
                "skill_success_rate": 0.95,
                "total_graph_executions": 7,
                "graph_success_rate": 0.857,
            },
        }
        html = _render_dashboard_html(health)
        assert "42" in html
        assert "Skill executions" in html

    def test_no_router_produces_empty_tiers_table(self) -> None:
        html = _render_dashboard_html({"status": "ok"})
        # The table header should still be there
        assert "Inference Tiers" in html
        # But no tier rows
        assert "<tr><td>" not in html

    def test_links_always_present(self) -> None:
        html = _render_dashboard_html({"status": "ok"})
        for path in ("/metrics", "/routes", "/fleet", "/alerts"):
            assert path in html
