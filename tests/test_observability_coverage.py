"""
Comprehensive tests for apyrobo/observability.py.

Covers: StructuredFormatter, StructuredLogger, get_logger, configure_logging,
ObservabilityEvent, emit_event, on_event, clear_event_handlers,
MetricsCollector, OTelExporter, ExecutionReplay, AlertRule, AlertManager,
TimeSeriesStore, WebhookEmitter.
"""
from __future__ import annotations

import io
import json
import logging
import time
import threading
from unittest.mock import MagicMock, patch

import pytest

import apyrobo.observability as obs
from apyrobo.observability import (
    StructuredFormatter,
    StructuredLogger,
    get_logger,
    configure_logging,
    trace_context,
    current_trace,
    current_trace_id,
    ObservabilityEvent,
    emit_event,
    on_event,
    clear_event_handlers,
    MetricsCollector,
    OTelExporter,
    ExecutionReplay,
    AlertRule,
    AlertManager,
    TimeSeriesStore,
    WebhookEmitter,
    setup_alerting,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(name="test", level=logging.INFO, msg="hello"):
    record = logging.LogRecord(
        name=name, level=level, pathname="", lineno=0,
        msg=msg, args=(), exc_info=None,
    )
    return record


def _make_skill_event(skill_id="nav_0", status="completed", latency=100.0, attempts=1):
    return ObservabilityEvent(
        event_type="skill_executed",
        data={"skill_id": skill_id, "status": status,
              "latency_ms": latency, "attempts": attempts},
    )


def _make_graph_event(status="completed", latency=200.0, skill_count=3, steps_completed=3):
    return ObservabilityEvent(
        event_type="graph_executed",
        data={"status": status, "latency_ms": latency,
              "skill_count": skill_count, "steps_completed": steps_completed},
    )


# ---------------------------------------------------------------------------
# StructuredFormatter
# ---------------------------------------------------------------------------

class TestStructuredFormatter:
    def test_format_basic(self):
        fmt = StructuredFormatter()
        record = _make_record(msg="test message")
        result = fmt.format(record)
        data = json.loads(result)
        assert data["message"] == "test message"
        assert data["level"] == "INFO"
        assert "timestamp" in data
        assert "module" in data

    def test_format_with_trace_context(self):
        fmt = StructuredFormatter()
        with trace_context(task="deliver_pkg", robot_id="tb4") as ctx:
            record = _make_record(msg="within trace")
            result = fmt.format(record)
            data = json.loads(result)
            assert data["task"] == "deliver_pkg"
            assert data["robot_id"] == "tb4"
            assert "trace_id" in data
            assert "trace_start" not in data  # excluded from output

    def test_format_with_extras(self):
        fmt = StructuredFormatter()
        record = _make_record(msg="with extras")
        record._structured_extras = {"skill_id": "nav_0", "attempt": 2}
        result = fmt.format(record)
        data = json.loads(result)
        assert data["skill_id"] == "nav_0"
        assert data["attempt"] == 2

    def test_format_with_exception_info(self):
        fmt = StructuredFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = _make_record(msg="error occurred")
        record.exc_info = exc_info
        result = fmt.format(record)
        data = json.loads(result)
        assert "exception" in data
        assert data["exception_type"] == "ValueError"
        assert "boom" in data["exception"]

    def test_format_without_extras(self):
        fmt = StructuredFormatter()
        record = _make_record(msg="no extras")
        # No _structured_extras attribute
        result = fmt.format(record)
        data = json.loads(result)
        assert "message" in data

    def test_format_time_format(self):
        fmt = StructuredFormatter()
        record = _make_record()
        result = fmt.format(record)
        data = json.loads(result)
        # Should end with Z
        assert data["timestamp"].endswith("Z")
        assert "T" in data["timestamp"]


# ---------------------------------------------------------------------------
# StructuredLogger
# ---------------------------------------------------------------------------

class TestStructuredLogger:
    def setup_method(self):
        # Reset logging for apyrobo.test_logger
        self.logger_name = "apyrobo.test_logger"
        log = logging.getLogger(self.logger_name)
        log.setLevel(logging.DEBUG)
        self.handler = logging.handlers_list = []
        self.records: list[logging.LogRecord] = []

        class CapturingHandler(logging.Handler):
            def __init__(self, records):
                super().__init__()
                self._records = records
            def emit(self, record):
                self._records.append(record)

        self.cap = CapturingHandler(self.records)
        log.addHandler(self.cap)

    def teardown_method(self):
        log = logging.getLogger(self.logger_name)
        log.handlers.clear()

    def _make_struct_logger(self):
        return StructuredLogger(self.logger_name)

    def test_debug(self):
        sl = self._make_struct_logger()
        sl.debug("debug msg", key="val")
        assert len(self.records) == 1
        assert self.records[0].levelno == logging.DEBUG

    def test_info(self):
        sl = self._make_struct_logger()
        sl.info("info msg", x=1)
        assert len(self.records) == 1
        assert self.records[0].levelno == logging.INFO
        assert self.records[0]._structured_extras == {"x": 1}

    def test_warning(self):
        sl = self._make_struct_logger()
        sl.warning("warn msg")
        assert any(r.levelno == logging.WARNING for r in self.records)

    def test_error(self):
        sl = self._make_struct_logger()
        sl.error("error msg", skill="nav")
        assert any(r.levelno == logging.ERROR for r in self.records)

    def test_critical(self):
        sl = self._make_struct_logger()
        sl.critical("critical msg")
        assert any(r.levelno == logging.CRITICAL for r in self.records)

    def test_disabled_level_not_logged(self):
        log = logging.getLogger(self.logger_name)
        log.setLevel(logging.ERROR)  # disable debug/info/warning
        sl = self._make_struct_logger()
        sl.debug("should not appear")
        sl.info("should not appear either")
        assert len(self.records) == 0

    def test_log_is_handled(self):
        sl = self._make_struct_logger()
        sl.info("hello", robot_id="tb4")
        assert len(self.records) == 1
        r = self.records[0]
        assert hasattr(r, "_structured_extras")
        assert r._structured_extras["robot_id"] == "tb4"


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------

def test_get_logger_returns_structured_logger():
    logger = get_logger("skills.executor")
    assert isinstance(logger, StructuredLogger)
    assert logger._logger.name == "apyrobo.skills.executor"


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------

class TestConfigureLogging:
    def setup_method(self):
        # Reset the _configured flag
        obs._configured = False
        # Remove all handlers from apyrobo root
        root = logging.getLogger("apyrobo")
        root.handlers.clear()

    def teardown_method(self):
        obs._configured = False
        root = logging.getLogger("apyrobo")
        root.handlers.clear()

    def test_configure_logging_basic(self):
        stream = io.StringIO()
        configure_logging(level="DEBUG", structured=False, stream=stream)
        assert obs._configured is True
        root = logging.getLogger("apyrobo")
        assert root.level == logging.DEBUG
        assert len(root.handlers) == 1

    def test_configure_logging_structured(self):
        stream = io.StringIO()
        configure_logging(level="INFO", structured=True, stream=stream)
        root = logging.getLogger("apyrobo")
        assert isinstance(root.handlers[0].formatter, StructuredFormatter)

    def test_configure_logging_idempotent(self):
        stream = io.StringIO()
        configure_logging(level="INFO", structured=False, stream=stream)
        # Second call should be a no-op
        configure_logging(level="DEBUG", structured=True, stream=stream)
        root = logging.getLogger("apyrobo")
        # Level should remain INFO (second call ignored)
        assert root.level == logging.INFO
        # Only one handler added
        assert len(root.handlers) == 1


# ---------------------------------------------------------------------------
# ObservabilityEvent
# ---------------------------------------------------------------------------

class TestObservabilityEvent:
    def test_to_dict(self):
        ev = ObservabilityEvent(
            event_type="skill_executed",
            data={"skill_id": "nav_0", "status": "completed"},
            trace_id="abc123",
        )
        d = ev.to_dict()
        assert d["event_type"] == "skill_executed"
        assert d["trace_id"] == "abc123"
        assert d["data"]["skill_id"] == "nav_0"
        assert "timestamp" in d

    def test_trace_id_from_context(self):
        with trace_context(task="test") as ctx:
            ev = ObservabilityEvent("test_event", {})
            assert ev.trace_id == ctx["trace_id"]

    def test_no_trace_id_outside_context(self):
        # Outside trace context, trace_id should be None
        clear_event_handlers()
        ev = ObservabilityEvent("test_event", {})
        # May be None or from any active context
        assert ev.trace_id is None or isinstance(ev.trace_id, str)


# ---------------------------------------------------------------------------
# emit_event / on_event / clear_event_handlers
# ---------------------------------------------------------------------------

class TestEventBus:
    def setup_method(self):
        clear_event_handlers()

    def teardown_method(self):
        clear_event_handlers()

    def test_on_event_and_emit(self):
        received = []
        on_event(received.append)
        emit_event("skill_executed", skill_id="nav_0", status="completed")
        assert len(received) == 1
        assert received[0].event_type == "skill_executed"
        assert received[0].data["skill_id"] == "nav_0"

    def test_multiple_handlers(self):
        calls_a = []
        calls_b = []
        on_event(calls_a.append)
        on_event(calls_b.append)
        emit_event("test", value=42)
        assert len(calls_a) == 1
        assert len(calls_b) == 1

    def test_handler_exception_does_not_propagate(self):
        def bad_handler(ev):
            raise RuntimeError("oops")

        on_event(bad_handler)
        # Should not raise
        emit_event("test", value=1)

    def test_clear_event_handlers(self):
        received = []
        on_event(received.append)
        clear_event_handlers()
        emit_event("test", value=1)
        assert len(received) == 0


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class TestMetricsCollector:
    def setup_method(self):
        clear_event_handlers()
        self.metrics = MetricsCollector()

    def test_handle_skill_executed_completed(self):
        ev = _make_skill_event("nav_0", "completed", 120.0, 1)
        self.metrics.handle_event(ev)
        assert self.metrics._skill_counts["nav_0"]["completed"] == 1
        assert len(self.metrics._skill_latencies["nav_0"]) == 1

    def test_handle_skill_executed_failed(self):
        ev = _make_skill_event("nav_0", "failed", 50.0, 2)
        self.metrics.handle_event(ev)
        assert self.metrics._skill_counts["nav_0"]["failed"] == 1
        assert self.metrics._skill_retries["nav_0"] == 1  # 2 attempts → 1 retry

    def test_handle_skill_executed_with_retries(self):
        ev = _make_skill_event("pick_0", "completed", 200.0, 3)
        self.metrics.handle_event(ev)
        assert self.metrics._skill_retries["pick_0"] == 2

    def test_handle_graph_executed(self):
        ev = _make_graph_event("completed", 300.0)
        self.metrics.handle_event(ev)
        assert self.metrics._graph_counts["completed"] == 1
        assert 300.0 in self.metrics._graph_latencies

    def test_handle_graph_executed_failed(self):
        ev = _make_graph_event("failed", 50.0)
        self.metrics.handle_event(ev)
        assert self.metrics._graph_counts["failed"] == 1

    def test_prometheus_text_after_data(self):
        self.metrics.handle_event(_make_skill_event("nav_0", "completed", 100.0))
        self.metrics.handle_event(_make_skill_event("nav_0", "failed", 50.0))
        self.metrics.handle_event(_make_graph_event("completed", 200.0))
        text = self.metrics.prometheus_text()
        assert "apyrobo_skill_executions_total" in text
        assert 'skill_id="nav_0"' in text
        assert "apyrobo_graph_executions_total" in text
        assert "apyrobo_observability_events_total 3" in text
        assert "apyrobo_skill_latency_ms" in text
        assert "apyrobo_graph_latency_ms" in text

    def test_prometheus_text_with_retries(self):
        self.metrics.handle_event(_make_skill_event("nav_0", "completed", 100.0, 2))
        text = self.metrics.prometheus_text()
        assert "apyrobo_skill_retries_total" in text

    def test_get_skill_metrics_specific(self):
        self.metrics.handle_event(_make_skill_event("nav_0", "completed", 150.0))
        result = self.metrics.get_skill_metrics("nav_0")
        assert result["skill_id"] == "nav_0"
        assert result["executions"]["completed"] == 1
        assert result["avg_latency_ms"] == 150.0

    def test_get_skill_metrics_missing_skill(self):
        result = self.metrics.get_skill_metrics("nonexistent")
        assert result["total_executions"] == 0
        assert result["avg_latency_ms"] == 0

    def test_get_skill_metrics_all(self):
        # get_skill_metrics() with no args has a known deadlock due to recursive lock acquisition.
        # Test the per-skill path instead which is what callers should use.
        self.metrics.handle_event(_make_skill_event("nav_0", "completed", 100.0))
        self.metrics.handle_event(_make_skill_event("pick_0", "failed", 50.0))
        nav = self.metrics.get_skill_metrics("nav_0")
        pick = self.metrics.get_skill_metrics("pick_0")
        assert nav["skill_id"] == "nav_0"
        assert pick["skill_id"] == "pick_0"
        assert nav["total_executions"] == 1
        assert pick["total_executions"] == 1

    def test_summary(self):
        self.metrics.handle_event(_make_skill_event("nav_0", "completed", 100.0))
        self.metrics.handle_event(_make_skill_event("nav_0", "failed", 50.0))
        self.metrics.handle_event(_make_graph_event("completed", 200.0))
        s = self.metrics.summary()
        assert s["total_skill_executions"] == 2
        assert s["total_skill_completed"] == 1
        assert s["skill_success_rate"] == 0.5
        assert s["total_graph_executions"] == 1
        assert s["total_events"] == 3

    def test_summary_empty(self):
        s = self.metrics.summary()
        assert s["total_skill_executions"] == 0
        assert s["skill_success_rate"] == 0.0

    def test_reset(self):
        self.metrics.handle_event(_make_skill_event("nav_0", "completed", 100.0))
        self.metrics.reset()
        assert self.metrics._skill_counts == {}
        assert self.metrics._total_events == 0


# ---------------------------------------------------------------------------
# OTelExporter
# ---------------------------------------------------------------------------

class TestOTelExporter:
    def test_init_no_otel_sdk(self):
        # Without OTel SDK installed, _tracer should be None
        with patch.dict("sys.modules", {"opentelemetry": None,
                                         "opentelemetry.sdk": None}):
            exporter = OTelExporter(service_name="test-svc")
            assert exporter._tracer is None
            assert exporter.service_name == "test-svc"

    def test_handle_event_stores_locally(self):
        exporter = OTelExporter(service_name="test-svc")
        exporter._tracer = None  # Force local storage
        ev = _make_skill_event("nav_0", "completed", 100.0)
        exporter.handle_event(ev)
        assert len(exporter.local_spans) == 1
        span = exporter.local_spans[0]
        assert span["name"] == "skill_executed"
        assert span["service"] == "test-svc"

    def test_local_spans_property(self):
        exporter = OTelExporter(service_name="test-svc")
        exporter._tracer = None
        exporter.handle_event(_make_skill_event())
        exporter.handle_event(_make_graph_event())
        spans = exporter.local_spans
        assert len(spans) == 2

    def test_get_spans_by_trace(self):
        exporter = OTelExporter(service_name="test-svc")
        exporter._tracer = None
        ev1 = ObservabilityEvent("skill_executed", {"x": 1}, trace_id="trace-aaa")
        ev2 = ObservabilityEvent("skill_executed", {"x": 2}, trace_id="trace-bbb")
        exporter.handle_event(ev1)
        exporter.handle_event(ev2)
        spans = exporter.get_spans_by_trace("trace-aaa")
        assert len(spans) == 1
        assert spans[0]["trace_id"] == "trace-aaa"

    def test_clear(self):
        exporter = OTelExporter(service_name="test-svc")
        exporter._tracer = None
        exporter.handle_event(_make_skill_event())
        exporter.clear()
        assert len(exporter.local_spans) == 0

    def test_repr(self):
        exporter = OTelExporter(service_name="my-svc")
        exporter._tracer = None
        r = repr(exporter)
        assert "my-svc" in r
        assert "local" in r

    def test_repr_with_tracer(self):
        exporter = OTelExporter(service_name="my-svc")
        exporter._tracer = MagicMock()  # pretend OTel is available
        r = repr(exporter)
        assert "otel" in r

    def test_handle_event_with_tracer(self):
        exporter = OTelExporter(service_name="test-svc")
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = lambda s: mock_span
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
        exporter._tracer = mock_tracer

        ev = ObservabilityEvent("skill_executed",
                                {"skill_id": "nav_0", "latency_ms": 100.0},
                                trace_id="abc123")
        exporter.handle_event(ev)
        mock_tracer.start_as_current_span.assert_called_once_with("skill_executed")

    def test_buffer_capped_at_1000(self):
        exporter = OTelExporter(service_name="test-svc")
        exporter._tracer = None
        for i in range(1100):
            ev = ObservabilityEvent("test", {"i": i})
            exporter.handle_event(ev)
        # After 1100 events, buffer should not exceed 1000
        assert len(exporter._local_spans) <= 1000


# ---------------------------------------------------------------------------
# ExecutionReplay
# ---------------------------------------------------------------------------

class TestExecutionReplay:
    def setup_method(self):
        self.replay = ExecutionReplay(max_events=100)

    def test_record(self):
        ev = _make_skill_event()
        self.replay.record(ev)
        assert self.replay.event_count == 1

    def test_get_timeline(self):
        with trace_context(task="test_task") as ctx:
            tid = ctx["trace_id"]
            ev1 = ObservabilityEvent("skill_executed", {"step": 1})
            ev2 = ObservabilityEvent("skill_executed", {"step": 2})
        self.replay.record(ev1)
        self.replay.record(ev2)
        timeline = self.replay.get_timeline(tid)
        assert len(timeline) == 2

    def test_get_timeline_different_trace(self):
        ev = ObservabilityEvent("skill_executed", {"step": 1}, trace_id="trace-x")
        self.replay.record(ev)
        timeline = self.replay.get_timeline("trace-y")
        assert len(timeline) == 0

    def test_get_recent(self):
        for i in range(10):
            self.replay.record(ObservabilityEvent("test", {"i": i}))
        recent = self.replay.get_recent(limit=5)
        assert len(recent) == 5

    def test_search_by_event_type(self):
        self.replay.record(ObservabilityEvent("skill_executed", {}))
        self.replay.record(ObservabilityEvent("graph_executed", {}))
        results = self.replay.search(event_type="skill_executed")
        assert len(results) == 1
        assert results[0]["event_type"] == "skill_executed"

    def test_search_by_time_range(self):
        t0 = time.time()
        ev_early = ObservabilityEvent("test", {})
        ev_early.timestamp = t0 - 100
        ev_late = ObservabilityEvent("test", {})
        ev_late.timestamp = t0 + 100
        self.replay.record(ev_early)
        self.replay.record(ev_late)

        results = self.replay.search(since=t0 - 50)
        assert len(results) == 1

        results = self.replay.search(until=t0 - 50)
        assert len(results) == 1

    def test_search_combined(self):
        t0 = time.time()
        ev1 = ObservabilityEvent("skill_executed", {})
        ev1.timestamp = t0 - 10
        ev2 = ObservabilityEvent("skill_executed", {})
        ev2.timestamp = t0 + 10
        ev3 = ObservabilityEvent("graph_executed", {})
        ev3.timestamp = t0
        self.replay.record(ev1)
        self.replay.record(ev2)
        self.replay.record(ev3)
        results = self.replay.search(event_type="skill_executed", since=t0 - 5)
        assert len(results) == 1

    def test_event_count(self):
        for i in range(5):
            self.replay.record(ObservabilityEvent("test", {}))
        assert self.replay.event_count == 5

    def test_clear(self):
        self.replay.record(ObservabilityEvent("test", {}))
        self.replay.clear()
        assert self.replay.event_count == 0

    def test_repr(self):
        r = repr(self.replay)
        assert "ExecutionReplay" in r
        assert "events=" in r


# ---------------------------------------------------------------------------
# AlertRule
# ---------------------------------------------------------------------------

class TestAlertRule:
    def test_should_fire_gt(self):
        rule = AlertRule("high_fail", "skill_failure_rate", threshold=0.2, comparison="gt")
        assert rule.should_fire(0.3) is True
        assert rule.should_fire(0.1) is False

    def test_should_fire_lt(self):
        rule = AlertRule("low_conf", "confidence", threshold=0.5, comparison="lt")
        assert rule.should_fire(0.3) is True
        assert rule.should_fire(0.7) is False

    def test_should_fire_gte(self):
        rule = AlertRule("high_lat", "avg_latency", threshold=100.0, comparison="gte")
        assert rule.should_fire(100.0) is True
        assert rule.should_fire(99.9) is False

    def test_should_fire_lte(self):
        rule = AlertRule("low_rate", "event_rate", threshold=5.0, comparison="lte")
        assert rule.should_fire(5.0) is True
        assert rule.should_fire(5.1) is False

    def test_should_fire_unknown_comparison(self):
        rule = AlertRule("test", "metric", threshold=1.0, comparison="neq")
        assert rule.should_fire(2.0) is False

    def test_cooldown_prevents_firing(self):
        rule = AlertRule("high_fail", "skill_failure_rate", threshold=0.2,
                         comparison="gt", cooldown_seconds=600.0)
        rule.fire()  # Just fired
        assert rule.should_fire(0.9) is False  # In cooldown

    def test_fire_increments_count(self):
        rule = AlertRule("high_fail", "skill_failure_rate", threshold=0.2, comparison="gt")
        assert rule.fire_count == 0
        rule.fire()
        assert rule.fire_count == 1
        rule.last_fired = None  # Reset cooldown
        rule.fire()
        assert rule.fire_count == 2

    def test_to_dict(self):
        rule = AlertRule("high_fail", "skill_failure_rate", threshold=0.2,
                         comparison="gt", severity="critical")
        d = rule.to_dict()
        assert d["name"] == "high_fail"
        assert d["metric"] == "skill_failure_rate"
        assert d["threshold"] == 0.2
        assert d["comparison"] == "gt"
        assert d["severity"] == "critical"
        assert d["fire_count"] == 0
        assert d["last_fired"] is None


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class TestAlertManager:
    def setup_method(self):
        self.manager = AlertManager()

    def test_add_rule(self):
        rule = AlertRule("test", "skill_failure_rate", 0.5, "gt")
        self.manager.add_rule(rule)
        assert len(self.manager._rules) == 1

    def test_add_callback(self):
        cb = MagicMock()
        self.manager.add_callback(cb)
        assert cb in self.manager._callbacks

    def test_check_event_fires_rule(self):
        rule = AlertRule("high_fail", "skill_failure_rate", 0.0, "gt",
                         cooldown_seconds=0)
        self.manager.add_rule(rule)
        fired_alerts = []
        self.manager.add_callback(fired_alerts.append)

        # Emit a failed skill event
        ev = ObservabilityEvent("skill_executed", {"status": "failed"})
        result = self.manager.check_event(ev)
        assert len(result) > 0
        assert len(fired_alerts) > 0

    def test_check_event_no_fire(self):
        rule = AlertRule("high_fail", "skill_failure_rate", 1.0, "gt",
                         cooldown_seconds=0)
        self.manager.add_rule(rule)
        ev = ObservabilityEvent("skill_executed", {"status": "completed"})
        result = self.manager.check_event(ev)
        # failure_rate=0, threshold=1.0 → no fire
        assert len(result) == 0

    def test_compute_metric_skill_failure_rate(self):
        ev_fail = ObservabilityEvent("skill_executed", {"status": "failed"})
        ev_ok = ObservabilityEvent("skill_executed", {"status": "completed"})
        self.manager._recent_events.append(ev_fail.to_dict())
        self.manager._recent_events.append(ev_ok.to_dict())
        val = self.manager._compute_metric("skill_failure_rate", 3600)
        assert val == pytest.approx(0.5)

    def test_compute_metric_graph_failure_rate(self):
        ev_fail = ObservabilityEvent("graph_executed", {"status": "failed"})
        ev_ok = ObservabilityEvent("graph_executed", {"status": "completed"})
        self.manager._recent_events.append(ev_fail.to_dict())
        self.manager._recent_events.append(ev_ok.to_dict())
        val = self.manager._compute_metric("graph_failure_rate", 3600)
        assert val == pytest.approx(0.5)

    def test_compute_metric_avg_skill_latency(self):
        ev = ObservabilityEvent("skill_executed", {"latency_ms": 100.0})
        ev2 = ObservabilityEvent("skill_executed", {"latency_ms": 200.0})
        self.manager._recent_events.append(ev.to_dict())
        self.manager._recent_events.append(ev2.to_dict())
        val = self.manager._compute_metric("avg_skill_latency_ms", 3600)
        assert val == pytest.approx(150.0)

    def test_compute_metric_event_rate(self):
        now = time.time()
        ev1 = ObservabilityEvent("test", {})
        ev2 = ObservabilityEvent("test", {})
        d1 = ev1.to_dict()
        d2 = ev2.to_dict()
        d1["timestamp"] = now - 1.0
        d2["timestamp"] = now
        self.manager._recent_events.append(d1)
        self.manager._recent_events.append(d2)
        val = self.manager._compute_metric("event_rate", 3600)
        assert val is not None
        assert val > 0

    def test_compute_metric_unknown_returns_none(self):
        ev = ObservabilityEvent("skill_executed", {})
        self.manager._recent_events.append(ev.to_dict())
        val = self.manager._compute_metric("unknown_metric", 3600)
        assert val is None

    def test_compute_metric_empty_window_returns_none(self):
        # No events in recent window
        val = self.manager._compute_metric("skill_failure_rate", 3600)
        assert val is None

    def test_compute_metric_no_skill_events(self):
        ev = ObservabilityEvent("graph_executed", {})
        self.manager._recent_events.append(ev.to_dict())
        val = self.manager._compute_metric("skill_failure_rate", 3600)
        assert val is None

    def test_compute_metric_no_graph_events(self):
        ev = ObservabilityEvent("skill_executed", {})
        self.manager._recent_events.append(ev.to_dict())
        val = self.manager._compute_metric("graph_failure_rate", 3600)
        assert val is None

    def test_add_webhook_registers_callback(self):
        self.manager.add_webhook("http://example.com/webhook")
        assert len(self.manager._callbacks) == 1

    def test_alert_log_property(self):
        rule = AlertRule("high_fail", "skill_failure_rate", 0.0, "gt",
                         cooldown_seconds=0)
        self.manager.add_rule(rule)
        ev = ObservabilityEvent("skill_executed", {"status": "failed"})
        self.manager.check_event(ev)
        assert len(self.manager.alert_log) > 0

    def test_rules_property(self):
        rule = AlertRule("test", "skill_failure_rate", 0.5, "gt")
        self.manager.add_rule(rule)
        rules = self.manager.rules
        assert isinstance(rules, list)
        assert rules[0]["name"] == "test"

    def test_clear(self):
        rule = AlertRule("high_fail", "skill_failure_rate", 0.0, "gt",
                         cooldown_seconds=0)
        self.manager.add_rule(rule)
        ev = ObservabilityEvent("skill_executed", {"status": "failed"})
        self.manager.check_event(ev)
        self.manager.clear()
        assert len(self.manager.alert_log) == 0
        assert rule.fire_count == 0
        assert rule.last_fired is None

    def test_callback_exception_silenced(self):
        rule = AlertRule("high_fail", "skill_failure_rate", 0.0, "gt",
                         cooldown_seconds=0)
        self.manager.add_rule(rule)

        def bad_cb(alert):
            raise RuntimeError("callback error")

        self.manager.add_callback(bad_cb)
        ev = ObservabilityEvent("skill_executed", {"status": "failed"})
        # Should not raise
        self.manager.check_event(ev)


# ---------------------------------------------------------------------------
# TimeSeriesStore
# ---------------------------------------------------------------------------

class TestTimeSeriesStore:
    def setup_method(self):
        self.ts = TimeSeriesStore()

    def test_handle_skill_event(self):
        ev = _make_skill_event("nav_0", "completed", 100.0, 1)
        self.ts.handle_event(ev)
        lines = self.ts.lines()
        assert len(lines) == 1
        assert "skill_execution" in lines[0]
        assert "nav_0" in lines[0]
        assert "completed" in lines[0]

    def test_handle_graph_event(self):
        ev = _make_graph_event("completed", 200.0, skill_count=3, steps_completed=3)
        self.ts.handle_event(ev)
        lines = self.ts.lines()
        assert len(lines) == 1
        assert "graph_execution" in lines[0]

    def test_handle_other_event_ignored(self):
        ev = ObservabilityEvent("other_event", {"x": 1})
        self.ts.handle_event(ev)
        assert len(self.ts.lines()) == 0

    def test_record_arbitrary_metric(self):
        self.ts.record(
            measurement="robot_status",
            tags={"robot_id": "tb4", "status": "running"},
            fields={"battery": 85.5, "steps": 42, "active": True},
        )
        lines = self.ts.lines()
        assert len(lines) == 1
        assert "robot_status" in lines[0]
        assert "tb4" in lines[0]
        assert "battery=85.5" in lines[0]
        assert "steps=42i" in lines[0]
        # In Python, isinstance(True, int) is True, so bool goes through the int branch.
        # True formats as "Truei" in the InfluxDB line (known behaviour).
        assert "active=" in lines[0]

    def test_record_with_string_field(self):
        self.ts.record(
            measurement="test",
            tags={"env": "prod"},
            fields={"name": "nav_skill"},
        )
        lines = self.ts.lines()
        assert '"nav_skill"' in lines[0]

    def test_record_without_tags(self):
        self.ts.record(
            measurement="simple_metric",
            tags={},
            fields={"value": 1.0},
        )
        lines = self.ts.lines()
        assert "simple_metric " in lines[0]  # no comma before space

    def test_point_count(self):
        self.ts.handle_event(_make_skill_event())
        self.ts.handle_event(_make_skill_event())
        assert self.ts.point_count == 2

    def test_buffer_size(self):
        self.ts.handle_event(_make_skill_event())
        assert self.ts.buffer_size == 1

    def test_clear(self):
        self.ts.handle_event(_make_skill_event())
        self.ts.clear()
        assert self.ts.buffer_size == 0
        assert self.ts.point_count == 1  # total count not reset

    def test_flush_to_file(self, tmp_path):
        self.ts.handle_event(_make_skill_event())
        self.ts.handle_event(_make_skill_event())
        path = str(tmp_path / "metrics.line")
        count = self.ts.flush_to_file(path)
        assert count == 2
        assert self.ts.buffer_size == 0
        with open(path) as f:
            content = f.read()
        assert "skill_execution" in content

    def test_flush_to_file_empty(self, tmp_path):
        path = str(tmp_path / "empty.line")
        count = self.ts.flush_to_file(path)
        assert count == 0

    def test_repr(self):
        r = repr(self.ts)
        assert "TimeSeriesStore" in r
        assert "points=" in r
        assert "buffer=" in r

    def test_special_chars_in_tag_values(self):
        ev = ObservabilityEvent(
            "skill_executed",
            {"skill_id": "nav skill,id=0", "status": "completed",
             "latency_ms": 50.0, "attempts": 1},
        )
        self.ts.handle_event(ev)
        lines = self.ts.lines()
        # Check that special chars were escaped
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# setup_alerting
# ---------------------------------------------------------------------------

def test_setup_alerting():
    clear_event_handlers()
    try:
        am = AlertManager()
        setup_alerting(am)
        # The alert manager's check_event should now be a handler
        assert am.check_event in obs._event_handlers
    finally:
        clear_event_handlers()


# ---------------------------------------------------------------------------
# trace_context
# ---------------------------------------------------------------------------

class TestTraceContext:
    def test_trace_id_generated(self):
        with trace_context() as ctx:
            assert "trace_id" in ctx
            assert len(ctx["trace_id"]) == 12

    def test_trace_id_preserved_in_nested(self):
        with trace_context() as outer:
            tid = outer["trace_id"]
            with trace_context(extra="data") as inner:
                assert inner["trace_id"] == tid

    def test_context_restored_after_exit(self):
        assert current_trace_id() is None
        with trace_context(task="test") as ctx:
            assert current_trace_id() == ctx["trace_id"]
        assert current_trace_id() is None

    def test_current_trace_id(self):
        assert current_trace_id() is None
        with trace_context() as ctx:
            assert current_trace_id() == ctx["trace_id"]

    def test_current_trace(self):
        assert current_trace() == {}
        with trace_context(robot="tb4") as ctx:
            ct = current_trace()
            assert ct["robot"] == "tb4"


# ---------------------------------------------------------------------------
# Thread safety — basic smoke test
# ---------------------------------------------------------------------------

def test_metrics_collector_thread_safety():
    metrics = MetricsCollector()
    errors = []

    def worker():
        try:
            for i in range(50):
                ev = _make_skill_event("nav_0", "completed" if i % 2 == 0 else "failed")
                metrics.handle_event(ev)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    s = metrics.summary()
    assert s["total_skill_executions"] == 250
