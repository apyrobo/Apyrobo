"""
Structured Observability — JSON logging, tracing, metrics, and alerting.

Every operation in APYROBO gets a trace_id that follows it from agent plan
through executor, safety enforcer, and down to individual ROS 2 commands.
This makes it possible to answer "why did the robot stop at 3am?" by
searching logs for the trace_id.

Features:
    OB-01: Per-skill execution telemetry via emit_event()
    OB-03: Trace context propagated from Agent.execute() through every layer
    OB-08: OpenTelemetry export: ship traces to Jaeger/Zipkin/Datadog
    OB-11: Alerting: configurable thresholds → webhook/Slack on breach
    OB-12: Time-series storage: InfluxDB-compatible line protocol

Usage:
    from apyrobo.observability import get_logger, trace_context

    with trace_context(task="deliver_package") as ctx:
        log = get_logger("skills.executor")
        log.info("Starting execution", skill_id="nav_0", robot="tb4")
        # → {"timestamp": "...", "level": "INFO", "module": "skills.executor",
        #    "trace_id": "abc123", "task": "deliver_package",
        #    "message": "Starting execution", "skill_id": "nav_0", "robot": "tb4"}
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from typing import Any, Generator


# ---------------------------------------------------------------------------
# Trace context (thread-local)
# ---------------------------------------------------------------------------

_trace_local = threading.local()


def current_trace() -> dict[str, Any]:
    """Get the current trace context."""
    return getattr(_trace_local, "context", {})


def current_trace_id() -> str | None:
    """Get the current trace ID."""
    return current_trace().get("trace_id")


@contextmanager
def trace_context(**kwargs: Any) -> Generator[dict[str, Any], None, None]:
    """
    Set trace context for the current thread.

    OB-03: All structured log entries within this context will include
    the trace_id and any extra fields.

    Usage:
        with trace_context(task="deliver", robot_id="tb4") as ctx:
            print(ctx["trace_id"])
            # all logs now include trace_id, task, robot_id
    """
    parent = getattr(_trace_local, "context", {})
    ctx = dict(parent)
    if "trace_id" not in ctx:
        ctx["trace_id"] = uuid.uuid4().hex[:12]
    ctx.update(kwargs)
    ctx["trace_start"] = time.time()

    old = getattr(_trace_local, "context", None)
    _trace_local.context = ctx
    try:
        yield ctx
    finally:
        _trace_local.context = old if old is not None else {}


# ---------------------------------------------------------------------------
# Structured JSON formatter
# ---------------------------------------------------------------------------

class StructuredFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON with trace context.

    Output:
        {"timestamp": "...", "level": "INFO", "module": "skills.executor",
         "trace_id": "abc123", "message": "Skill completed", "skill_id": "nav_0"}
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": self._format_time(record),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }

        # Merge trace context
        ctx = current_trace()
        if ctx:
            entry.update({k: v for k, v in ctx.items() if k != "trace_start"})

        # Merge any extra fields passed via our custom StructuredLogger
        extras = getattr(record, "_structured_extras", {})
        if extras:
            entry.update(extras)

        # Exception info
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = str(record.exc_info[1])
            entry["exception_type"] = type(record.exc_info[1]).__name__

        return json.dumps(entry, default=str)

    @staticmethod
    def _format_time(record: logging.LogRecord) -> str:
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(record.msecs):03d}Z"


# ---------------------------------------------------------------------------
# Structured logger wrapper
# ---------------------------------------------------------------------------

class StructuredLogger:
    """
    Wrapper around stdlib logger that supports keyword arguments
    as structured fields.

    Usage:
        log = StructuredLogger("skills.executor")
        log.info("Skill started", skill_id="nav_0", attempt=1)
    """

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        if not self._logger.isEnabledFor(level):
            return
        record = self._logger.makeRecord(
            self._logger.name, level, "(structured)", 0,
            message, (), None,
        )
        record._structured_extras = kwargs  # type: ignore[attr-defined]
        self._logger.handle(record)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, message, **kwargs)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger for a module."""
    return StructuredLogger(f"apyrobo.{name}")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_configured = False


def configure_logging(
    level: str = "INFO",
    structured: bool = False,
    stream: Any = None,
) -> None:
    """
    Configure APYROBO logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        structured: If True, output JSON. If False, human-readable.
        stream: Output stream (default: stderr)
    """
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger("apyrobo")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(stream or sys.stderr)

    if structured:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))

    root.addHandler(handler)


# ---------------------------------------------------------------------------
# Event bus for observability consumers
# ---------------------------------------------------------------------------

class ObservabilityEvent:
    """A structured event for external consumers (dashboards, webhooks)."""

    def __init__(self, event_type: str, data: dict[str, Any],
                 trace_id: str | None = None) -> None:
        self.event_type = event_type
        self.data = data
        self.trace_id = trace_id or current_trace_id()
        self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }


_event_handlers: list[Any] = []


def on_event(handler: Any) -> None:
    """Register a global event handler for observability."""
    _event_handlers.append(handler)


def clear_event_handlers() -> None:
    """Remove all event handlers (for testing)."""
    _event_handlers.clear()


def emit_event(event_type: str, **data: Any) -> None:
    """Emit an observability event to all registered handlers."""
    event = ObservabilityEvent(event_type, data)
    for handler in _event_handlers:
        try:
            handler(event)
        except Exception:
            pass  # never let observability break the main flow


# ---------------------------------------------------------------------------
# OB-01: Metrics Collector — aggregates per-skill telemetry
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Aggregates per-skill telemetry for Prometheus-compatible metrics.

    OB-01: Collects latency, success/fail counts, and retry rates.
    OB-02: Exposes metrics in Prometheus text format.

    Usage:
        metrics = MetricsCollector()
        on_event(metrics.handle_event)

        # After some execution...
        print(metrics.prometheus_text())
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._skill_latencies: dict[str, list[float]] = {}
        self._skill_counts: dict[str, dict[str, int]] = {}  # skill_id → {completed, failed}
        self._skill_retries: dict[str, int] = {}
        self._graph_latencies: list[float] = []
        self._graph_counts: dict[str, int] = {"completed": 0, "failed": 0}
        self._total_events = 0

    def handle_event(self, event: ObservabilityEvent) -> None:
        """Handle observability events and update metrics."""
        with self._lock:
            self._total_events += 1

            if event.event_type == "skill_executed":
                sid = event.data.get("skill_id", "unknown")
                status = event.data.get("status", "unknown")
                latency = event.data.get("latency_ms", 0.0)
                attempts = event.data.get("attempts", 1)

                if sid not in self._skill_latencies:
                    self._skill_latencies[sid] = []
                self._skill_latencies[sid].append(latency)

                if sid not in self._skill_counts:
                    self._skill_counts[sid] = {"completed": 0, "failed": 0}
                if status in self._skill_counts[sid]:
                    self._skill_counts[sid][status] += 1

                self._skill_retries[sid] = self._skill_retries.get(sid, 0) + max(0, attempts - 1)

            elif event.event_type == "graph_executed":
                status = event.data.get("status", "unknown")
                latency = event.data.get("latency_ms", 0.0)
                self._graph_latencies.append(latency)
                if status in self._graph_counts:
                    self._graph_counts[status] += 1

    def prometheus_text(self) -> str:
        """
        OB-02: Generate Prometheus-compatible text exposition format.

        Returns metrics in the format expected by /metrics endpoint.
        """
        lines: list[str] = []

        with self._lock:
            # Skill execution counts
            lines.append("# HELP apyrobo_skill_executions_total Total skill executions by status")
            lines.append("# TYPE apyrobo_skill_executions_total counter")
            for sid, counts in self._skill_counts.items():
                for status, count in counts.items():
                    lines.append(f'apyrobo_skill_executions_total{{skill_id="{sid}",status="{status}"}} {count}')

            # Skill latency
            lines.append("# HELP apyrobo_skill_latency_ms Skill execution latency in milliseconds")
            lines.append("# TYPE apyrobo_skill_latency_ms summary")
            for sid, latencies in self._skill_latencies.items():
                if latencies:
                    avg = sum(latencies) / len(latencies)
                    lines.append(f'apyrobo_skill_latency_ms_sum{{skill_id="{sid}"}} {sum(latencies):.1f}')
                    lines.append(f'apyrobo_skill_latency_ms_count{{skill_id="{sid}"}} {len(latencies)}')
                    lines.append(f'apyrobo_skill_latency_ms_avg{{skill_id="{sid}"}} {avg:.1f}')

            # Skill retries
            lines.append("# HELP apyrobo_skill_retries_total Total skill retries")
            lines.append("# TYPE apyrobo_skill_retries_total counter")
            for sid, retries in self._skill_retries.items():
                lines.append(f'apyrobo_skill_retries_total{{skill_id="{sid}"}} {retries}')

            # Graph execution counts
            lines.append("# HELP apyrobo_graph_executions_total Total graph executions by status")
            lines.append("# TYPE apyrobo_graph_executions_total counter")
            for status, count in self._graph_counts.items():
                lines.append(f'apyrobo_graph_executions_total{{status="{status}"}} {count}')

            # Graph latency
            if self._graph_latencies:
                avg = sum(self._graph_latencies) / len(self._graph_latencies)
                lines.append("# HELP apyrobo_graph_latency_ms Graph execution latency")
                lines.append("# TYPE apyrobo_graph_latency_ms summary")
                lines.append(f"apyrobo_graph_latency_ms_sum {sum(self._graph_latencies):.1f}")
                lines.append(f"apyrobo_graph_latency_ms_count {len(self._graph_latencies)}")
                lines.append(f"apyrobo_graph_latency_ms_avg {avg:.1f}")

            # Total events
            lines.append("# HELP apyrobo_observability_events_total Total observability events")
            lines.append("# TYPE apyrobo_observability_events_total counter")
            lines.append(f"apyrobo_observability_events_total {self._total_events}")

        return "\n".join(lines) + "\n"

    def get_skill_metrics(self, skill_id: str | None = None) -> dict[str, Any]:
        """Get metrics for a specific skill or all skills."""
        with self._lock:
            if skill_id:
                latencies = self._skill_latencies.get(skill_id, [])
                counts = self._skill_counts.get(skill_id, {"completed": 0, "failed": 0})
                return {
                    "skill_id": skill_id,
                    "executions": counts,
                    "retries": self._skill_retries.get(skill_id, 0),
                    "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
                    "total_executions": sum(counts.values()),
                }
            # All skills
            result = {}
            for sid in self._skill_counts:
                result[sid] = self.get_skill_metrics(sid)
            return result

    def summary(self) -> dict[str, Any]:
        """High-level metrics summary."""
        with self._lock:
            total_skills = sum(sum(c.values()) for c in self._skill_counts.values())
            total_completed = sum(c.get("completed", 0) for c in self._skill_counts.values())
            return {
                "total_skill_executions": total_skills,
                "total_skill_completed": total_completed,
                "skill_success_rate": round(total_completed / total_skills, 3) if total_skills > 0 else 0.0,
                "total_graph_executions": sum(self._graph_counts.values()),
                "graph_success_rate": round(
                    self._graph_counts["completed"] / sum(self._graph_counts.values()), 3
                ) if sum(self._graph_counts.values()) > 0 else 0.0,
                "total_events": self._total_events,
                "skills_tracked": len(self._skill_counts),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._skill_latencies.clear()
            self._skill_counts.clear()
            self._skill_retries.clear()
            self._graph_latencies.clear()
            self._graph_counts = {"completed": 0, "failed": 0}
            self._total_events = 0


# ---------------------------------------------------------------------------
# OB-08: OpenTelemetry Exporter
# ---------------------------------------------------------------------------

class OTelExporter:
    """
    OB-08: Ships traces to Jaeger/Zipkin/Datadog via OpenTelemetry.

    When the opentelemetry SDK is available, creates real spans.
    Otherwise, stores spans locally for inspection.

    Usage:
        exporter = OTelExporter(service_name="apyrobo")
        on_event(exporter.handle_event)

        # With real OTel SDK:
        exporter = OTelExporter(service_name="apyrobo",
                                endpoint="http://jaeger:4317")
    """

    def __init__(self, service_name: str = "apyrobo",
                 endpoint: str | None = None) -> None:
        self.service_name = service_name
        self.endpoint = endpoint
        self._tracer = None
        self._local_spans: list[dict[str, Any]] = []
        self._lock = threading.Lock()

        # Try to set up real OTel
        self._setup_otel()

    def _setup_otel(self) -> None:
        """Attempt to set up OpenTelemetry SDK."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor

            provider = TracerProvider()

            if self.endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                    exporter = OTLPSpanExporter(endpoint=self.endpoint)
                    provider.add_span_processor(SimpleSpanProcessor(exporter))
                except ImportError:
                    pass  # OTLP exporter not available

            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(self.service_name)
        except ImportError:
            pass  # OTel SDK not installed — use local spans

    def handle_event(self, event: ObservabilityEvent) -> None:
        """Convert observability events to OTel spans or local records."""
        span_data = {
            "name": event.event_type,
            "trace_id": event.trace_id,
            "timestamp": event.timestamp,
            "attributes": event.data,
            "service": self.service_name,
        }

        if self._tracer:
            try:
                with self._tracer.start_as_current_span(event.event_type) as span:
                    for k, v in event.data.items():
                        if isinstance(v, (str, int, float, bool)):
                            span.set_attribute(k, v)
                    if event.trace_id:
                        span.set_attribute("apyrobo.trace_id", event.trace_id)
            except Exception:
                pass  # Never let OTel break the main flow
        else:
            # Store locally
            with self._lock:
                self._local_spans.append(span_data)
                if len(self._local_spans) > 1000:
                    self._local_spans = self._local_spans[-500:]

    @property
    def local_spans(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._local_spans)

    def get_spans_by_trace(self, trace_id: str) -> list[dict[str, Any]]:
        """Get all spans for a given trace ID."""
        with self._lock:
            return [s for s in self._local_spans if s.get("trace_id") == trace_id]

    def clear(self) -> None:
        with self._lock:
            self._local_spans.clear()

    def __repr__(self) -> str:
        backend = "otel" if self._tracer else "local"
        return f"<OTelExporter {self.service_name} backend={backend} spans={len(self._local_spans)}>"


# ---------------------------------------------------------------------------
# OB-10: Execution Replay
# ---------------------------------------------------------------------------

class ExecutionReplay:
    """
    OB-10: Reconstruct task execution from trace log for debugging.

    Records all events and allows replay/query by trace_id.
    Answers questions like "why did the robot stop at 3am?"

    Usage:
        replay = ExecutionReplay()
        on_event(replay.record)

        # Later: reconstruct what happened
        timeline = replay.get_timeline(trace_id="abc123")
    """

    def __init__(self, max_events: int = 10000) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def record(self, event: ObservabilityEvent) -> None:
        """Record an event for replay."""
        with self._lock:
            self._events.append(event.to_dict())

    def get_timeline(self, trace_id: str) -> list[dict[str, Any]]:
        """Reconstruct a timeline for a trace ID."""
        with self._lock:
            events = [e for e in self._events if e.get("trace_id") == trace_id]
        return sorted(events, key=lambda e: e.get("timestamp", 0))

    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get the most recent events."""
        with self._lock:
            events = list(self._events)
        return events[-limit:]

    def search(self, event_type: str | None = None,
               since: float | None = None,
               until: float | None = None) -> list[dict[str, Any]]:
        """Search events by type and/or time range."""
        with self._lock:
            results = list(self._events)

        if event_type:
            results = [e for e in results if e.get("event_type") == event_type]
        if since:
            results = [e for e in results if e.get("timestamp", 0) >= since]
        if until:
            results = [e for e in results if e.get("timestamp", 0) <= until]
        return results

    @property
    def event_count(self) -> int:
        return len(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def __repr__(self) -> str:
        return f"<ExecutionReplay events={len(self._events)}>"


# ---------------------------------------------------------------------------
# OB-11: Alerting — configurable thresholds → webhook/Slack on breach
# ---------------------------------------------------------------------------

class AlertRule:
    """A single alert rule with threshold and cooldown."""

    def __init__(
        self,
        name: str,
        metric: str,
        threshold: float,
        comparison: str = "gt",  # "gt", "lt", "gte", "lte"
        window_seconds: float = 300.0,
        cooldown_seconds: float = 600.0,
        severity: str = "warning",
    ) -> None:
        self.name = name
        self.metric = metric
        self.threshold = threshold
        self.comparison = comparison
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.severity = severity
        self.last_fired: float | None = None
        self.fire_count = 0

    def should_fire(self, value: float) -> bool:
        """Check if the rule should fire based on value and cooldown."""
        # Check cooldown
        if self.last_fired and time.time() - self.last_fired < self.cooldown_seconds:
            return False

        # Check threshold
        if self.comparison == "gt":
            return value > self.threshold
        elif self.comparison == "lt":
            return value < self.threshold
        elif self.comparison == "gte":
            return value >= self.threshold
        elif self.comparison == "lte":
            return value <= self.threshold
        return False

    def fire(self) -> None:
        self.last_fired = time.time()
        self.fire_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "severity": self.severity,
            "fire_count": self.fire_count,
            "last_fired": self.last_fired,
        }


class AlertManager:
    """
    OB-11: Monitors metrics and fires alerts when thresholds are breached.

    Usage:
        alerter = AlertManager()
        alerter.add_rule(AlertRule(
            name="high_failure_rate",
            metric="skill_failure_rate",
            threshold=0.2,
            comparison="gt",
            window_seconds=300,
        ))
        alerter.add_callback(lambda alert: print(f"ALERT: {alert}"))

        # Automatically check after each event:
        on_event(alerter.check_event)
    """

    def __init__(self) -> None:
        self._rules: list[AlertRule] = []
        self._callbacks: list[Any] = []
        self._alert_log: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        # Rolling window of events for metric computation
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=1000)

    def add_rule(self, rule: AlertRule) -> None:
        self._rules.append(rule)

    def add_callback(self, callback: Any) -> None:
        """Register callback: callback(alert_dict) called when rule fires."""
        self._callbacks.append(callback)

    def check_event(self, event: ObservabilityEvent) -> list[dict[str, Any]]:
        """
        Process an event and check all alert rules.

        Returns list of alerts that fired.
        """
        with self._lock:
            self._recent_events.append(event.to_dict())

        fired = []
        for rule in self._rules:
            value = self._compute_metric(rule.metric, rule.window_seconds)
            if value is not None and rule.should_fire(value):
                rule.fire()
                alert = {
                    "rule": rule.name,
                    "metric": rule.metric,
                    "value": round(value, 4),
                    "threshold": rule.threshold,
                    "severity": rule.severity,
                    "timestamp": time.time(),
                }
                fired.append(alert)
                self._alert_log.append(alert)

                # Notify callbacks
                for cb in self._callbacks:
                    try:
                        cb(alert)
                    except Exception:
                        pass

        return fired

    def _compute_metric(self, metric: str, window: float) -> float | None:
        """Compute a metric value from recent events."""
        cutoff = time.time() - window
        with self._lock:
            recent = [e for e in self._recent_events if e.get("timestamp", 0) >= cutoff]

        if not recent:
            return None

        if metric == "skill_failure_rate":
            skill_events = [e for e in recent if e.get("event_type") == "skill_executed"]
            if not skill_events:
                return None
            failed = sum(1 for e in skill_events if e.get("data", {}).get("status") == "failed")
            return failed / len(skill_events)

        elif metric == "graph_failure_rate":
            graph_events = [e for e in recent if e.get("event_type") == "graph_executed"]
            if not graph_events:
                return None
            failed = sum(1 for e in graph_events if e.get("data", {}).get("status") == "failed")
            return failed / len(graph_events)

        elif metric == "avg_skill_latency_ms":
            skill_events = [e for e in recent if e.get("event_type") == "skill_executed"]
            latencies = [e.get("data", {}).get("latency_ms", 0) for e in skill_events]
            return sum(latencies) / len(latencies) if latencies else None

        elif metric == "event_rate":
            if not recent:
                return 0.0
            time_span = max(0.001, recent[-1].get("timestamp", 0) - recent[0].get("timestamp", 0))
            return len(recent) / time_span

        return None

    def add_webhook(self, url: str, headers: dict[str, str] | None = None) -> None:
        """Register a WebhookEmitter as a built-in alert callback."""
        emitter = WebhookEmitter(url, headers=headers)
        self._callbacks.append(emitter.emit)

    @property
    def alert_log(self) -> list[dict[str, Any]]:
        return list(self._alert_log)

    @property
    def rules(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._rules]

    def clear(self) -> None:
        self._alert_log.clear()
        self._recent_events.clear()
        for rule in self._rules:
            rule.fire_count = 0
            rule.last_fired = None


# ---------------------------------------------------------------------------
# OB-11: WebhookEmitter — HTTP POST alerts with retry
# ---------------------------------------------------------------------------

class WebhookEmitter:
    """
    Sends alert payloads to a webhook URL via HTTP POST.

    Uses stdlib urllib only (no extra dependencies).
    Retries up to 3 times with exponential backoff on failure.
    """

    def __init__(self, url: str, headers: dict[str, str] | None = None,
                 max_retries: int = 3, backoff_s: float = 0.5) -> None:
        self._url = url
        self._headers = headers or {}
        self._max_retries = max_retries
        self._backoff_s = backoff_s

    def emit(self, payload: dict[str, Any]) -> None:
        """POST payload as JSON to the configured URL with retry."""
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", **self._headers}

        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                req = urllib.request.Request(
                    self._url, data=data, headers=headers, method="POST",
                )
                urllib.request.urlopen(req, timeout=5)
                return
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    time.sleep(self._backoff_s * attempt)

        logging.getLogger(__name__).error(
            "Webhook POST to %s failed after %d attempts: %s",
            self._url, self._max_retries, last_error,
        )


def setup_alerting(alert_manager: AlertManager) -> None:
    """
    Register an AlertManager as a global observability event handler.

    After calling this, every emit_event() will be checked against
    the alert manager's rules.
    """
    on_event(alert_manager.check_event)


# ---------------------------------------------------------------------------
# OB-12: Time-series storage (InfluxDB line protocol)
# ---------------------------------------------------------------------------

class TimeSeriesStore:
    """
    OB-12: Stores skill execution metrics in InfluxDB line protocol format.

    Enables trend analysis over weeks/months. Can write to:
    - Local file (for later import into InfluxDB/Telegraf)
    - In-memory buffer (for testing or direct API push)
    - InfluxDB HTTP API (when available)

    InfluxDB Line Protocol format:
        measurement,tag=value field=value timestamp_ns

    Usage:
        ts = TimeSeriesStore()
        on_event(ts.handle_event)

        # Export as line protocol
        for line in ts.lines():
            print(line)

        # Or write to file
        ts.flush_to_file("/var/log/apyrobo_metrics.line")
    """

    def __init__(self, max_buffer: int = 10000, influx_url: str | None = None,
                 influx_bucket: str = "apyrobo", influx_token: str | None = None) -> None:
        self._buffer: deque[str] = deque(maxlen=max_buffer)
        self._lock = threading.Lock()
        self._influx_url = influx_url
        self._influx_bucket = influx_bucket
        self._influx_token = influx_token
        self._total_points = 0

    def handle_event(self, event: ObservabilityEvent) -> None:
        """Convert an observability event to InfluxDB line protocol."""
        if event.event_type == "skill_executed":
            self._record_skill_metric(event)
        elif event.event_type == "graph_executed":
            self._record_graph_metric(event)

    def _record_skill_metric(self, event: ObservabilityEvent) -> None:
        """Record skill execution as a time-series point."""
        d = event.data
        sid = d.get("skill_id", "unknown")
        status = d.get("status", "unknown")
        latency = d.get("latency_ms", 0)
        attempts = d.get("attempts", 1)
        ts_ns = int(event.timestamp * 1e9)

        # Escape tag values (InfluxDB protocol)
        sid_escaped = sid.replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")
        status_escaped = status.replace(" ", "\\ ")

        line = (
            f"skill_execution,skill_id={sid_escaped},status={status_escaped} "
            f"latency_ms={latency},attempts={attempts}i "
            f"{ts_ns}"
        )
        self._append(line)

    def _record_graph_metric(self, event: ObservabilityEvent) -> None:
        """Record graph execution as a time-series point."""
        d = event.data
        status = d.get("status", "unknown")
        latency = d.get("latency_ms", 0)
        skills = d.get("skill_count", 0)
        completed = d.get("steps_completed", 0)
        ts_ns = int(event.timestamp * 1e9)

        line = (
            f"graph_execution,status={status} "
            f"latency_ms={latency},skill_count={skills}i,steps_completed={completed}i "
            f"{ts_ns}"
        )
        self._append(line)

    def record(self, measurement: str, tags: dict[str, str],
               fields: dict[str, Any], timestamp: float | None = None) -> None:
        """Record an arbitrary metric point."""
        ts_ns = int((timestamp or time.time()) * 1e9)
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        field_parts = []
        for k, v in sorted(fields.items()):
            if isinstance(v, int):
                field_parts.append(f"{k}={v}i")
            elif isinstance(v, float):
                field_parts.append(f"{k}={v}")
            elif isinstance(v, bool):
                field_parts.append(f"{k}={'t' if v else 'f'}")
            else:
                field_parts.append(f'{k}="{v}"')
        field_str = ",".join(field_parts)
        line = f"{measurement},{tag_str} {field_str} {ts_ns}" if tag_str else f"{measurement} {field_str} {ts_ns}"
        self._append(line)

    def _append(self, line: str) -> None:
        with self._lock:
            self._buffer.append(line)
            self._total_points += 1

        # Auto-flush to InfluxDB if configured
        if self._influx_url and len(self._buffer) >= 100:
            self._flush_to_influx()

    def lines(self) -> list[str]:
        """Get all buffered line protocol lines."""
        with self._lock:
            return list(self._buffer)

    def flush_to_file(self, path: str) -> int:
        """Write buffered lines to a file (append mode)."""
        with self._lock:
            lines = list(self._buffer)
            self._buffer.clear()

        if not lines:
            return 0

        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write("\n".join(lines) + "\n")
        return len(lines)

    def _flush_to_influx(self) -> None:
        """Send buffered lines to InfluxDB HTTP API."""
        if not self._influx_url:
            return
        with self._lock:
            lines = list(self._buffer)
            self._buffer.clear()

        if not lines:
            return

        try:
            import urllib.request
            body = "\n".join(lines).encode("utf-8")
            url = f"{self._influx_url}/api/v2/write?bucket={self._influx_bucket}&precision=ns"
            headers = {"Content-Type": "text/plain"}
            if self._influx_token:
                headers["Authorization"] = f"Token {self._influx_token}"
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception as e:
            logging.getLogger(__name__).warning("InfluxDB write failed: %s", e)
            # Re-buffer the lines
            with self._lock:
                for line in lines:
                    self._buffer.append(line)

    @property
    def point_count(self) -> int:
        return self._total_points

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def __repr__(self) -> str:
        return f"<TimeSeriesStore points={self._total_points} buffer={len(self._buffer)}>"
