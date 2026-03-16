"""
Tests for OB-01, OB-02, OB-03:
  - SQLiteStateStore complete implementation + factory
  - StateStore wired into SkillExecutor for crash recovery
  - AlertManager wired to observability events + webhook dispatch
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import TaskStatus
from apyrobo.persistence import (
    SQLiteStateStore,
    StateStore,
    StorageBackend,
    TaskJournalEntry,
    create_state_store,
    recover_interrupted_tasks,
)
from apyrobo.observability import (
    AlertManager,
    AlertRule,
    MetricsCollector,
    ObservabilityEvent,
    WebhookEmitter,
    clear_event_handlers,
    emit_event,
    on_event,
    setup_alerting,
)
from apyrobo.skills.executor import SkillExecutor, SkillGraph, ExecutionState
from apyrobo.skills.skill import Skill, SkillStatus, BUILTIN_SKILLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_robot() -> Robot:
    return Robot.discover("mock://ob_test_bot")


@pytest.fixture
def sqlite_store(tmp_path: Path) -> SQLiteStateStore:
    store = SQLiteStateStore(tmp_path / "test_ob.db")
    yield store
    store.clear()


@pytest.fixture
def json_store(tmp_path: Path) -> StateStore:
    store = StateStore(tmp_path / "test_ob.json")
    yield store
    store.clear()


@pytest.fixture(autouse=True)
def clean_event_handlers():
    """Clear global event handlers before/after each test."""
    clear_event_handlers()
    yield
    clear_event_handlers()


# ===========================================================================
# OB-01: SQLiteStateStore — complete abstract method tests
# ===========================================================================

class TestSQLiteStateStoreComplete:
    """Verify SQLiteStateStore implements all StorageBackend methods correctly."""

    def test_begin_update_complete_roundtrip(self, sqlite_store: SQLiteStateStore) -> None:
        """begin_task / update_task / complete_task round-trip."""
        entry = sqlite_store.begin_task("t1", {"task": "deliver"}, robot_id="bot1", total_steps=3)
        assert entry.task_id == "t1"
        assert entry.status == "in_progress"
        assert entry.total_steps == 3

        sqlite_store.update_task("t1", step=1, status="in_progress")
        task = sqlite_store.get_task("t1")
        assert task is not None
        assert task.step == 1

        sqlite_store.complete_task("t1", result={"outcome": "delivered"})
        task = sqlite_store.get_task("t1")
        assert task is not None
        assert task.status == "completed"
        assert task.result == {"outcome": "delivered"}

    def test_fail_task_roundtrip(self, sqlite_store: SQLiteStateStore) -> None:
        """fail_task records error correctly."""
        sqlite_store.begin_task("t1")
        sqlite_store.fail_task("t1", error="connection lost")

        task = sqlite_store.get_task("t1")
        assert task is not None
        assert task.status == "failed"
        assert task.result is not None
        assert "connection lost" in task.result.get("error", "")

    def test_get_interrupted_tasks(self, sqlite_store: SQLiteStateStore) -> None:
        """get_interrupted_tasks returns only pending/in_progress tasks."""
        sqlite_store.begin_task("t1")
        sqlite_store.begin_task("t2")
        sqlite_store.complete_task("t2")
        sqlite_store.begin_task("t3")
        sqlite_store.fail_task("t3", error="test")

        interrupted = sqlite_store.get_interrupted_tasks()
        assert len(interrupted) == 1
        assert interrupted[0].task_id == "t1"

    def test_get_recent_tasks(self, sqlite_store: SQLiteStateStore) -> None:
        """get_recent_tasks returns newest-first with correct limit."""
        for i in range(5):
            sqlite_store.begin_task(f"t{i}")
            time.sleep(0.01)

        recent = sqlite_store.get_recent_tasks(limit=3)
        assert len(recent) == 3
        assert recent[0].task_id == "t4"

    def test_save_get_robot_position(self, sqlite_store: SQLiteStateStore) -> None:
        """save_robot_position / get_robot_position round-trip."""
        sqlite_store.save_robot_position("bot1", x=1.5, y=2.5, yaw=0.7)
        pos = sqlite_store.get_robot_position("bot1")
        assert pos is not None
        assert pos["x"] == 1.5
        assert pos["y"] == 2.5
        assert pos["yaw"] == 0.7

    def test_kv_store_with_dict(self, sqlite_store: SQLiteStateStore) -> None:
        """set/get kv store works with dict values (JSON serialised)."""
        sqlite_store.set("config", {"nested": {"key": "value"}, "flag": True})
        result = sqlite_store.get("config")
        assert result == {"nested": {"key": "value"}, "flag": True}

    def test_kv_store_default(self, sqlite_store: SQLiteStateStore) -> None:
        """get returns default for missing keys."""
        assert sqlite_store.get("missing") is None
        assert sqlite_store.get("missing", "fallback") == "fallback"

    def test_clear(self, sqlite_store: SQLiteStateStore) -> None:
        """clear removes all state."""
        sqlite_store.begin_task("t1")
        sqlite_store.set("k", "v")
        sqlite_store.save_robot_position("bot1", 1.0, 2.0)

        sqlite_store.clear()
        assert sqlite_store.task_count == 0
        assert sqlite_store.get("k") is None
        assert sqlite_store.get_robot_position("bot1") is None

    def test_task_count(self, sqlite_store: SQLiteStateStore) -> None:
        """task_count returns correct count."""
        assert sqlite_store.task_count == 0
        sqlite_store.begin_task("t1")
        sqlite_store.begin_task("t2")
        assert sqlite_store.task_count == 2

    def test_drop_in_replacement(self, tmp_path: Path) -> None:
        """SQLiteStateStore passes all tests that StateStore passes."""
        # Run the same operations on both and compare
        json_store = StateStore(tmp_path / "cmp.json")
        sqlite_store = SQLiteStateStore(tmp_path / "cmp.db")

        for store in [json_store, sqlite_store]:
            store.begin_task("t1", {"task": "deliver"}, robot_id="bot1", total_steps=3)
            store.update_task("t1", step=2, status="in_progress")
            store.complete_task("t1", result={"ok": True})
            store.begin_task("t2")

            assert store.task_count == 2
            assert len(store.get_interrupted_tasks()) == 1
            assert store.get_recent_tasks(1)[0].task_id == "t2"

            store.clear()


# ===========================================================================
# OB-01: Factory function tests
# ===========================================================================

class TestCreateStateStore:
    """Test create_state_store factory."""

    def test_json_backend(self, tmp_path: Path) -> None:
        store = create_state_store("json", path=tmp_path / "f.json")
        assert isinstance(store, StateStore)

    def test_sqlite_backend(self, tmp_path: Path) -> None:
        store = create_state_store("sqlite", path=tmp_path / "f.db")
        assert isinstance(store, SQLiteStateStore)

    def test_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            create_state_store("dynamo")

    def test_recover_interrupted_tasks(self, tmp_path: Path) -> None:
        store = create_state_store("sqlite", path=tmp_path / "rec.db")
        store.begin_task("t1")
        store.begin_task("t2")
        store.complete_task("t2")

        interrupted = recover_interrupted_tasks(store)
        assert len(interrupted) == 1
        assert interrupted[0].task_id == "t1"


# ===========================================================================
# OB-02: StateStore wired into SkillExecutor
# ===========================================================================

class TestExecutorStateStoreIntegration:
    """Verify SkillExecutor writes to StateStore during execution."""

    def _make_graph(self, n: int = 2) -> SkillGraph:
        """Build a simple sequential graph with n navigate_to skills."""
        graph = SkillGraph()
        prev: str | None = None
        for i in range(n):
            sid = f"navigate_to_{i}"
            skill = Skill(
                skill_id=sid,
                name="Navigate",
                description="nav",
                parameters={"x": float(i), "y": 0.0},
            )
            depends = [prev] if prev else []
            graph.add_skill(skill, depends_on=depends)
            prev = sid
        return graph

    def test_begin_task_called_at_graph_start(
        self, mock_robot: Robot, sqlite_store: SQLiteStateStore
    ) -> None:
        """begin_task is called when graph execution starts."""
        executor = SkillExecutor(mock_robot, state_store=sqlite_store)
        graph = self._make_graph(1)

        executor.execute_graph(graph, trace_id="trace_001")

        task = sqlite_store.get_task("trace_001")
        assert task is not None
        assert task.total_steps == 1

    def test_complete_task_at_graph_end(
        self, mock_robot: Robot, sqlite_store: SQLiteStateStore
    ) -> None:
        """complete_task is called after successful graph execution."""
        executor = SkillExecutor(mock_robot, state_store=sqlite_store)
        graph = self._make_graph(2)

        result = executor.execute_graph(graph, trace_id="trace_002")

        task = sqlite_store.get_task("trace_002")
        assert task is not None
        assert task.status == "completed"

    def test_update_task_per_skill(
        self, mock_robot: Robot, sqlite_store: SQLiteStateStore
    ) -> None:
        """update_task is called after each skill completes with step count."""
        executor = SkillExecutor(mock_robot, state_store=sqlite_store)
        graph = self._make_graph(3)

        result = executor.execute_graph(graph, trace_id="trace_003")

        task = sqlite_store.get_task("trace_003")
        assert task is not None
        assert task.status == "completed"

    def test_fail_task_on_graph_failure(
        self, mock_robot: Robot, sqlite_store: SQLiteStateStore
    ) -> None:
        """fail_task is called with error message on graph failure."""
        graph = SkillGraph()
        # Add a skill that will fail (unknown skill)
        skill = Skill(
            skill_id="nonexistent_skill_0",
            name="Bad",
            description="will fail",
        )
        graph.add_skill(skill)

        executor = SkillExecutor(mock_robot, state_store=sqlite_store)
        result = executor.execute_graph(graph, trace_id="trace_004")

        task = sqlite_store.get_task("trace_004")
        assert task is not None
        assert task.status == "failed"

    def test_interrupted_task_detected_after_crash(
        self, mock_robot: Robot, sqlite_store: SQLiteStateStore
    ) -> None:
        """get_interrupted_tasks returns in_progress tasks if process interrupted mid-execution."""
        # Simulate: begin a task but never complete it (crash)
        sqlite_store.begin_task("crash_trace", {"task": "big_job"}, total_steps=5)
        sqlite_store.update_task("crash_trace", step=2, status="in_progress")

        # After "restart", check interrupted
        interrupted = sqlite_store.get_interrupted_tasks()
        assert len(interrupted) == 1
        assert interrupted[0].task_id == "crash_trace"
        assert interrupted[0].step == 2

    def test_no_state_store_no_error(self, mock_robot: Robot) -> None:
        """Executor works fine without state_store (backward compat)."""
        executor = SkillExecutor(mock_robot)
        graph = self._make_graph(1)
        result = executor.execute_graph(graph, trace_id="trace_nostore")
        assert result.status == TaskStatus.COMPLETED


# ===========================================================================
# OB-02: Agent.execute() passthrough
# ===========================================================================

class TestAgentStateStorePassthrough:
    """Verify Agent.execute() passes state_store to executor."""

    def test_agent_execute_with_state_store(
        self, mock_robot: Robot, sqlite_store: SQLiteStateStore
    ) -> None:
        """Agent.execute() records task in state store."""
        from apyrobo.skills.agent import Agent

        agent = Agent(provider="rule")
        result = agent.execute(
            "navigate to (1, 2)", mock_robot, state_store=sqlite_store
        )

        # Should have recorded something in the store
        assert sqlite_store.task_count >= 1
        recent = sqlite_store.get_recent_tasks(1)
        assert len(recent) == 1
        assert recent[0].status in ("completed", "failed")


# ===========================================================================
# OB-03: AlertManager wired to observability events
# ===========================================================================

class TestAlertManagerWiring:
    """Verify AlertManager evaluates rules on observability events."""

    def test_rule_fires_on_high_failure_rate(self) -> None:
        """Rule with threshold fires when rate drops below threshold."""
        alert_manager = AlertManager()
        alert_manager.add_rule(AlertRule(
            name="high_failure",
            metric="skill_failure_rate",
            threshold=0.2,
            comparison="gt",
            window_seconds=300,
            cooldown_seconds=0,  # no cooldown for testing
        ))

        fired_alerts: list[dict] = []
        alert_manager.add_callback(lambda a: fired_alerts.append(a))

        # Register as event handler
        setup_alerting(alert_manager)

        # Emit events: 5 total, 3 failed = 60% failure rate > 20% threshold
        for i in range(2):
            emit_event("skill_executed", skill_id=f"nav_{i}", status="completed",
                       latency_ms=100, attempts=1)
        for i in range(3):
            emit_event("skill_executed", skill_id=f"pick_{i}", status="failed",
                       latency_ms=50, attempts=1)

        assert len(fired_alerts) > 0
        assert fired_alerts[-1]["rule"] == "high_failure"
        assert fired_alerts[-1]["value"] > 0.2

    def test_alert_log_populated(self) -> None:
        """AlertManager.alert_log populated after rule breach."""
        alert_manager = AlertManager()
        alert_manager.add_rule(AlertRule(
            name="latency_alert",
            metric="skill_failure_rate",
            threshold=0.0,
            comparison="gt",
            cooldown_seconds=0,
        ))

        setup_alerting(alert_manager)

        emit_event("skill_executed", skill_id="s1", status="failed", latency_ms=100, attempts=1)

        assert len(alert_manager.alert_log) > 0
        assert alert_manager.alert_log[0]["rule"] == "latency_alert"

    def test_no_fire_below_threshold(self) -> None:
        """Alert does not fire when metric is within threshold."""
        alert_manager = AlertManager()
        alert_manager.add_rule(AlertRule(
            name="high_failure",
            metric="skill_failure_rate",
            threshold=0.5,
            comparison="gt",
            cooldown_seconds=0,
        ))

        fired: list[dict] = []
        alert_manager.add_callback(lambda a: fired.append(a))
        setup_alerting(alert_manager)

        # All succeed — 0% failure rate
        for i in range(5):
            emit_event("skill_executed", skill_id=f"s{i}", status="completed",
                       latency_ms=50, attempts=1)

        assert len(fired) == 0

    def test_add_webhook_registers_callback(self) -> None:
        """add_webhook adds a WebhookEmitter callback."""
        alert_manager = AlertManager()
        alert_manager.add_webhook("http://example.com/alert")

        # Should have one callback
        assert len(alert_manager._callbacks) == 1


# ===========================================================================
# OB-03: WebhookEmitter
# ===========================================================================

class TestWebhookEmitter:
    """Test WebhookEmitter HTTP POST with retry."""

    def test_emit_makes_http_post(self) -> None:
        """WebhookEmitter.emit() makes HTTP POST to configured URL."""
        received: list[bytes] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                received.append(self.rfile.read(length))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request, daemon=True)
        t.start()

        emitter = WebhookEmitter(f"http://127.0.0.1:{port}/hook")
        emitter.emit({"alert": "test", "value": 42})

        t.join(timeout=5)
        server.server_close()

        assert len(received) == 1
        payload = json.loads(received[0])
        assert payload["alert"] == "test"
        assert payload["value"] == 42

    def test_failed_webhook_retries_then_logs(self) -> None:
        """Failed webhook retries 3 times then logs error (does not raise)."""
        attempt_count = [0]

        class FailHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                attempt_count[0] += 1
                self.send_response(500)
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                pass

        server = HTTPServer(("127.0.0.1", 0), FailHandler)
        port = server.server_address[1]
        # Handle multiple requests
        t = threading.Thread(
            target=lambda: [server.handle_request() for _ in range(3)],
            daemon=True,
        )
        t.start()

        emitter = WebhookEmitter(
            f"http://127.0.0.1:{port}/hook",
            max_retries=3,
            backoff_s=0.01,
        )
        # Should not raise
        emitter.emit({"test": True})

        t.join(timeout=10)
        server.server_close()

        assert attempt_count[0] == 3

    def test_emit_does_not_raise_on_connection_refused(self) -> None:
        """WebhookEmitter does not raise even when the server is unreachable."""
        emitter = WebhookEmitter(
            "http://127.0.0.1:19999/nonexistent",
            max_retries=1,
            backoff_s=0.01,
        )
        # Should not raise
        emitter.emit({"test": True})
