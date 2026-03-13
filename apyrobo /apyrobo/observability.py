"""
Structured Observability — JSON logging with correlation IDs and trace context.

Every operation in APYROBO gets a trace_id that follows it from agent plan
through executor, safety enforcer, and down to individual ROS 2 commands.
This makes it possible to answer "why did the robot stop at 3am?" by
searching logs for the trace_id.

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

    All structured log entries within this context will include
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
        {"timestamp": "2026-03-10T12:00:00.123", "level": "INFO",
         "module": "skills.executor", "trace_id": "abc123",
         "message": "Skill completed", "skill_id": "nav_0", "duration_ms": 1234}
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

        # Merge any extra fields passed via log.info("msg", extra={"key": "val"})
        # or via our custom StructuredLogger
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


def emit_event(event_type: str, **data: Any) -> None:
    """Emit an observability event to all registered handlers."""
    event = ObservabilityEvent(event_type, data)
    for handler in _event_handlers:
        try:
            handler(event)
        except Exception:
            pass  # never let observability break the main flow
