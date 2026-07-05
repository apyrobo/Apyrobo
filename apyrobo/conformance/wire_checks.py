"""Wire-protocol conformance checks (spec/wire-protocol.md).

Exercises a live orchestration server — any implementation, any language —
over one of the two spec transports:

* WebSocket: ``apyrobo conformance ws://host:8765``
* stdio (NDJSON): ``apyrobo conformance "stdio:apyrobo serve"`` — the command
  after ``stdio:`` is spawned and spoken to over its stdin/stdout.

Responses are correlated by the echoed ``task`` text (the WebSocket
transport broadcasts every response to every client), so each probe uses a
unique task string.
"""
from __future__ import annotations

import contextlib
import json
import logging
import shlex
import subprocess
import threading
import uuid
from queue import Empty, Queue
from typing import Any

from apyrobo.conformance.report import ConformanceReport
from apyrobo.conformance.spec_schemas import JSONSCHEMA_AVAILABLE, validation_error

logger = logging.getLogger(__name__)

_WIRE = "wire-protocol.md"


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------

class WireClient:
    """Minimal transport interface the checks run against."""

    def send_text(self, text: str) -> None:
        raise NotImplementedError

    def receive(self, timeout: float) -> dict[str, Any] | None:
        """Next parsed JSON object from the server, or None on timeout."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class StdioWireClient(WireClient):
    """Spawns a server command and speaks NDJSON over its stdin/stdout."""

    def __init__(self, command: str) -> None:
        self._proc = subprocess.Popen(
            shlex.split(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._queue: Queue[dict[str, Any]] = Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._queue.put(json.loads(line))
            except json.JSONDecodeError:
                logger.debug("conformance: non-JSON server output: %r", line)

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def send_text(self, text: str) -> None:
        if not self.alive or self._proc.stdin is None:
            raise ConnectionError("server process has exited")
        self._proc.stdin.write(text + "\n")
        self._proc.stdin.flush()

    def receive(self, timeout: float) -> dict[str, Any] | None:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def close(self) -> None:
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()


class WebSocketWireClient(WireClient):
    """Connects to a running server over WebSocket (text frames)."""

    def __init__(self, uri: str, connect_timeout: float = 10.0) -> None:
        try:
            from websockets.sync.client import connect
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'websockets' package is required for ws:// targets.\n"
                "Install with: pip install 'apyrobo[websocket]'"
            ) from exc
        self._ws = connect(uri, open_timeout=connect_timeout)

    @property
    def alive(self) -> bool:
        return True  # a dropped connection surfaces as send/receive errors

    def send_text(self, text: str) -> None:
        self._ws.send(text)

    def receive(self, timeout: float) -> dict[str, Any] | None:
        try:
            raw = self._ws.recv(timeout=timeout)
        except TimeoutError:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("conformance: non-JSON frame: %r", raw)
            return None

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._ws.close()


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def run_wire_checks(
    client: WireClient,
    report: ConformanceReport,
    robot_uri: str = "mock://conformance-probe",
    timeout: float = 15.0,
) -> None:
    """Append wire-protocol check results to *report*."""
    probe = _Probe(client, timeout)

    response = probe.round_trip("plan a conformance probe task", robot_uri)
    if response is None:
        report.add(
            "WP-01", "server responds to a valid task message",
            "MUST", f"{_WIRE} §1", "fail",
            f"no response within {timeout}s — remaining checks skipped",
        )
        for check_id, title, ref in _REMAINING_AFTER_WP01:
            report.add(check_id, title, "MUST", ref, "skip",
                       "no response to the initial probe")
        return
    report.add("WP-01", "server responds to a valid task message",
               "MUST", f"{_WIRE} §1", "pass")

    _check_message_shape(response, report)
    _check_echo(response, report)
    _check_status(response, report)
    _check_error_path(probe, report)
    _check_malformed_input(probe, report, robot_uri)
    _check_unknown_keys(probe, report, robot_uri)
    _check_sequential(probe, report, robot_uri)


_REMAINING_AFTER_WP01 = [
    ("WP-02", "response is a spec-valid orchestration message", f"{_WIRE} §1"),
    ("WP-03", "response echoes task and robot_uri", f"{_WIRE} §3"),
    ("WP-04", "metadata.status is 'planned' or 'error'", f"{_WIRE} §3"),
    ("WP-05", "unknown robot scheme yields a status:error response", f"{_WIRE} §3"),
    ("WP-06", "malformed input does not terminate the connection", f"{_WIRE} §2.3"),
    ("WP-07", "unknown message keys are ignored", f"{_WIRE} §1"),
    ("WP-08", "messages are processed sequentially, one response each", f"{_WIRE} §4"),
]


class _Probe:
    """Sends uniquely-tagged tasks and collects the matching responses."""

    def __init__(self, client: WireClient, timeout: float) -> None:
        self.client = client
        self.timeout = timeout

    def unique_task(self, label: str) -> str:
        return f"{label} [conformance:{uuid.uuid4().hex[:8]}]"

    def send(self, message: dict[str, Any]) -> None:
        self.client.send_text(json.dumps(message))

    def wait_for(self, task: str) -> dict[str, Any] | None:
        """Drain responses until one echoes *task* (broadcast tolerance)."""
        import time

        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            response = self.client.receive(timeout=remaining)
            if response is None:
                return None
            if isinstance(response, dict) and response.get("task") == task:
                return response

    def round_trip(
        self, label: str, robot_uri: str, extra: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Send a tagged task and return its response — None on send failure
        or timeout (a dead connection counts as a failed check, not a crash)."""
        task = self.unique_task(label)
        message: dict[str, Any] = {"task": task, "robot_uri": robot_uri}
        if extra:
            message.update(extra)
        try:
            self.send(message)
        except Exception as exc:
            logger.debug("conformance: send failed: %s", exc)
            return None
        return self.wait_for(task)


def _check_message_shape(response: dict[str, Any], report: ConformanceReport) -> None:
    if not JSONSCHEMA_AVAILABLE:
        report.add(
            "WP-02", "response is a spec-valid orchestration message",
            "MUST", f"{_WIRE} §1", "skip",
            "jsonschema not installed — pip install 'apyrobo[conformance]'",
        )
        return
    error = validation_error(response, "orchestration-message")
    report.add(
        "WP-02", "response is a spec-valid orchestration message",
        "MUST", f"{_WIRE} §1", "fail" if error else "pass", error or "",
    )


def _check_echo(response: dict[str, Any], report: ConformanceReport) -> None:
    # wait_for() already matched on task; robot_uri echo is what's left.
    has_uri = isinstance(response.get("robot_uri"), str) and response["robot_uri"]
    report.add(
        "WP-03", "response echoes task and robot_uri",
        "MUST", f"{_WIRE} §3", "pass" if has_uri else "fail",
        "" if has_uri else f"robot_uri={response.get('robot_uri')!r}",
    )


def _check_status(response: dict[str, Any], report: ConformanceReport) -> None:
    metadata = response.get("metadata")
    status = metadata.get("status") if isinstance(metadata, dict) else None
    if status not in ("planned", "error"):
        report.add(
            "WP-04", "metadata.status is 'planned' or 'error'",
            "MUST", f"{_WIRE} §3", "fail",
            f"metadata.status={status!r}",
        )
        return
    details = ""
    ok = True
    if status == "planned":
        skills = metadata.get("skills")
        count = metadata.get("count")
        if not isinstance(skills, list) or count != len(skills):
            ok = False
            details = (
                f"planned response must carry skills[] with matching count "
                f"(skills={type(skills).__name__}, count={count!r})"
            )
    report.add(
        "WP-04", "metadata.status is 'planned' or 'error'",
        "MUST", f"{_WIRE} §3", "pass" if ok else "fail", details,
    )


def _check_error_path(probe: _Probe, report: ConformanceReport) -> None:
    response = probe.round_trip(
        "probe error handling", "conformance-bogus://nowhere"
    )
    if response is None:
        report.add(
            "WP-05", "unknown robot scheme yields a status:error response",
            "MUST", f"{_WIRE} §3", "fail",
            "no response to a task targeting an unknown robot scheme",
        )
        return
    metadata = response.get("metadata") or {}
    status = metadata.get("status") if isinstance(metadata, dict) else None
    error_text = metadata.get("error") if isinstance(metadata, dict) else None
    ok = status == "error" and isinstance(error_text, str) and error_text
    report.add(
        "WP-05", "unknown robot scheme yields a status:error response",
        "MUST", f"{_WIRE} §3", "pass" if ok else "fail",
        "" if ok else f"status={status!r} error={error_text!r}",
    )


def _check_malformed_input(
    probe: _Probe, report: ConformanceReport, robot_uri: str
) -> None:
    try:
        probe.client.send_text("this is not json {")
    except Exception as exc:
        report.add(
            "WP-06", "malformed input does not terminate the connection",
            "MUST", f"{_WIRE} §2.3", "fail",
            f"send failed: {type(exc).__name__}: {exc}",
        )
        return
    # The spec allows either an error response or treating the raw text as a
    # task; conformance is judged by the connection remaining usable.
    response = probe.round_trip("probe after malformed input", robot_uri)
    ok = response is not None
    report.add(
        "WP-06", "malformed input does not terminate the connection",
        "MUST", f"{_WIRE} §2.3", "pass" if ok else "fail",
        "" if ok else "no response to a valid message sent after malformed input",
    )


def _check_unknown_keys(
    probe: _Probe, report: ConformanceReport, robot_uri: str
) -> None:
    response = probe.round_trip(
        "probe unknown keys", robot_uri,
        extra={
            "x-conformance-extension": {"nested": True},
            "metadata": {"x-correlation": "abc123"},
        },
    )
    ok = response is not None
    report.add(
        "WP-07", "unknown message keys are ignored",
        "MUST", f"{_WIRE} §1", "pass" if ok else "fail",
        "" if ok else "no response to a message carrying unknown keys",
    )


def _check_sequential(
    probe: _Probe, report: ConformanceReport, robot_uri: str
) -> None:
    first = probe.unique_task("probe sequential first")
    second = probe.unique_task("probe sequential second")
    try:
        probe.send({"task": first, "robot_uri": robot_uri})
        probe.send({"task": second, "robot_uri": robot_uri})
    except Exception as exc:
        report.add(
            "WP-08", "messages are processed sequentially, one response each",
            "MUST", f"{_WIRE} §4", "fail",
            f"send failed: {type(exc).__name__}: {exc}",
        )
        return
    got_first = probe.wait_for(first) is not None
    got_second = probe.wait_for(second) is not None
    ok = got_first and got_second
    report.add(
        "WP-08", "messages are processed sequentially, one response each",
        "MUST", f"{_WIRE} §4", "pass" if ok else "fail",
        "" if ok else f"responses received: first={got_first} second={got_second}",
    )
