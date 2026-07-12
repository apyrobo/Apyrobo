"""Orchestration adapter base classes.

An *orchestration adapter* decouples the apyrobo planner from whatever
front-end is sending it tasks: a REST API, a CLI, a Slack bot, a ROS 2 topic,
or a raw stdin/stdout pipe.

The adapter contract is intentionally minimal:

* ``receive() -> OrchestrationMessage | None`` — block until the next task
  arrives, return ``None`` to signal shutdown.
* ``send(msg)`` — push a response back to the caller.
* ``startup()`` / ``shutdown()`` — lifecycle hooks (optional override).

``OrchestrationServer`` wires an adapter to an ``Agent`` and runs the
receive→plan→send loop.  ``StdioOrchestrationAdapter`` implements the contract
over stdin/stdout (one JSON object per line), making it easy to pipe tasks
from a shell script.  ``MockOrchestrationAdapter`` is pre-loaded with a list
of tasks and is intended for unit tests.  ``WebSocketOrchestrationAdapter``
exposes the same contract over a WebSocket server (requires the ``websockets``
library: ``pip install 'apyrobo[websocket]'``).

Crash recovery: pass a ``CheckpointStore`` to ``OrchestrationServer`` to persist
in-flight tasks. On restart, call ``server.resume_incomplete()`` to recover tasks
that were in-flight when the server last crashed.
"""
from __future__ import annotations

import abc
import contextlib
import json
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message schema
# ---------------------------------------------------------------------------

@dataclass
class OrchestrationMessage:
    """A single task sent to, or response sent from, an orchestration adapter.

    An empty ``robot_uri`` means "target the server's default robot"
    (wire-protocol.md §1). It is omitted from the serialized form so the
    wire message stays schema-valid; the server always resolves it before
    responding.
    """

    task: str
    robot_uri: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "task": self.task,
            "metadata": self.metadata,
            "source": self.source,
        }
        if self.robot_uri:
            d["robot_uri"] = self.robot_uri
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrchestrationMessage":
        """Parse a wire dict, normalizing out-of-spec field types.

        Hostile or buggy clients may send any JSON type in any field; the
        parsed message always has spec-shaped fields so downstream code
        (and the echoed response) stays schema-valid. A non-string ``task``
        is kept as its JSON text — the same "treat raw input as the task"
        rule the transports apply to malformed frames (wire-protocol §2.3).
        """
        task = data.get("task", "")
        if not isinstance(task, str):
            try:
                task = json.dumps(task)
            except (TypeError, ValueError):
                task = str(task)
        robot_uri = data.get("robot_uri", "")
        metadata = data.get("metadata", {})
        source = data.get("source", "")
        return cls(
            task=task,
            robot_uri=robot_uri if isinstance(robot_uri, str) else "",
            metadata=metadata if isinstance(metadata, dict) else {},
            source=source if isinstance(source, str) else "",
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class OrchestrationAdapter(abc.ABC):
    """Abstract base for all orchestration adapters."""

    @abc.abstractmethod
    def receive(self) -> OrchestrationMessage | None:
        """Block until the next task is available.

        Returns ``None`` to signal that the adapter is done (shutdown).
        """

    @abc.abstractmethod
    def send(self, msg: OrchestrationMessage) -> None:
        """Send a response back to the caller."""

    def startup(self) -> None:
        """Called once before the server loop starts. Override as needed."""

    def shutdown(self) -> None:
        """Called once after the server loop exits. Override as needed."""


# ---------------------------------------------------------------------------
# Orchestration server
# ---------------------------------------------------------------------------

class OrchestrationServer:
    """Runs the receive→plan→send loop for a given adapter and agent.

    Crash recovery: pass ``checkpoint_store`` to persist in-flight tasks to
    SQLite. On restart, call ``resume_incomplete()`` to recover tasks that
    were in-flight when the server last crashed.

    Robot ownership: robots the server discovers from task ``robot_uri``
    fields are owned by the server — they are disconnected when ``run()``
    exits. A ``default_robot`` passed to the constructor is owned by the
    caller and is never disconnected by the server.

    Usage::

        from apyrobo.core.robot import Robot
        from apyrobo.skills.agent import Agent
        from apyrobo.orchestration import OrchestrationServer, StdioOrchestrationAdapter
        from apyrobo.skills.checkpoint import CheckpointStore

        robot = Robot.discover("mock://turtlebot4")
        agent = Agent(provider="rule")
        adapter = StdioOrchestrationAdapter()
        store = CheckpointStore("/var/lib/apyrobo/orchestration.db")
        server = OrchestrationServer(adapter, agent, default_robot=robot,
                                     checkpoint_store=store)
        # Recover tasks from previous crash
        recovered = server.resume_incomplete()
        server.run()
    """

    def __init__(
        self,
        adapter: OrchestrationAdapter,
        agent: Any,
        default_robot: Any = None,
        max_iterations: int | None = None,
        checkpoint_store: Any = None,
        default_robot_uri: str = "mock://turtlebot4",
    ) -> None:
        self.adapter = adapter
        self.agent = agent
        self.default_robot = default_robot
        self.default_robot_uri = default_robot_uri
        self.max_iterations = max_iterations
        self._checkpoint_store = checkpoint_store
        self._iterations = 0
        # Discovered robots per URI, so repeated tasks against the same
        # target reuse one adapter (and its connection state).
        self._robot_cache: dict[str, Any] = {}
        if default_robot is not None:
            self._robot_cache[default_robot_uri] = default_robot

    def resume_incomplete(self) -> list[OrchestrationMessage]:
        """Return tasks that were in-flight when the server last crashed.

        Each returned message was being processed when the server died.
        Re-queue them by calling adapter.send() or prepending to the task list.
        Clears the recovered checkpoints.
        """
        if self._checkpoint_store is None:
            return []
        recovered: list[OrchestrationMessage] = []
        try:
            task_ids = self._checkpoint_store.list_tasks()
            for task_id in task_ids:
                if not task_id.startswith("orch:"):
                    continue
                entry = self._checkpoint_store.load(task_id)
                if entry is None:
                    continue
                msg = OrchestrationMessage.from_dict(entry.state)
                recovered.append(msg)
                self._checkpoint_store.delete(task_id)
                logger.info("OrchestrationServer: recovered in-flight task %r", msg.task)
        except Exception as exc:
            logger.warning("OrchestrationServer.resume_incomplete error: %s", exc)
        return recovered

    def run(self) -> None:
        """Start the receive→plan→send loop."""
        self.adapter.startup()
        logger.info("OrchestrationServer started")
        try:
            while True:
                if self.max_iterations is not None and self._iterations >= self.max_iterations:
                    logger.info("OrchestrationServer: max_iterations reached")
                    break

                msg = self.adapter.receive()
                if msg is None:
                    logger.info("OrchestrationServer: adapter signalled shutdown")
                    break

                self._iterations += 1
                task_id = f"orch:{self._iterations}:{time.time()}"
                self._checkpoint_begin(task_id, msg)
                response = self._handle(msg)
                self._checkpoint_complete(task_id)
                self.adapter.send(response)
        finally:
            try:
                self.adapter.shutdown()
            finally:
                self._release_robots()
            logger.info("OrchestrationServer stopped")

    def _release_robots(self) -> None:
        """Disconnect and evict robots the server discovered itself.

        The caller-supplied ``default_robot`` is left connected and cached —
        the server does not own it. Best-effort: a failing disconnect is
        logged and does not block the others.
        """
        for uri, robot in list(self._robot_cache.items()):
            if robot is self.default_robot:
                continue
            del self._robot_cache[uri]
            try:
                robot.disconnect()
            except Exception as exc:
                logger.warning(
                    "OrchestrationServer: disconnect of %s failed: %s", uri, exc
                )

    def _checkpoint_begin(self, task_id: str, msg: OrchestrationMessage) -> None:
        """Persist in-flight task so it can be recovered after a crash."""
        if self._checkpoint_store is None:
            return
        try:
            from apyrobo.skills.checkpoint import CheckpointEntry
            entry = CheckpointEntry(
                task_id=task_id,
                skill_name="orchestration",
                step_index=0,
                total_steps=1,
                state=msg.to_dict(),
                completed_steps=[],
            )
            self._checkpoint_store.save(entry)
        except Exception as exc:
            logger.warning("OrchestrationServer._checkpoint_begin error: %s", exc)

    def _checkpoint_complete(self, task_id: str) -> None:
        """Remove in-flight checkpoint after successful handling."""
        if self._checkpoint_store is None:
            return
        try:
            self._checkpoint_store.delete(task_id)
        except Exception as exc:
            logger.warning("OrchestrationServer._checkpoint_complete error: %s", exc)

    def _handle(self, msg: OrchestrationMessage) -> OrchestrationMessage:
        """Plan a task from *msg* and return a response message.

        The message's ``robot_uri`` selects the target robot; only when it
        is absent does the configured default robot apply (wire-protocol.md
        §1). The response always carries the resolved URI.
        """
        robot_uri = msg.robot_uri or self.default_robot_uri
        try:
            robot = self._resolve_robot(robot_uri)
            graph = self.agent.plan(msg.task, robot)
            order = graph.get_execution_order()
            skills = [
                {"skill_id": s.skill_id, "name": s.name}
                for s in order
            ]
            return OrchestrationMessage(
                task=msg.task,
                robot_uri=robot_uri,
                metadata={"status": "planned", "skills": skills, "count": len(skills)},
                source="orchestration_server",
            )
        except Exception as exc:
            logger.warning("OrchestrationServer._handle error: %s", exc)
            return OrchestrationMessage(
                task=msg.task,
                robot_uri=robot_uri,
                metadata={"status": "error", "error": str(exc)},
                source="orchestration_server",
            )

    def _resolve_robot(self, robot_uri: str) -> Any:
        """Return the robot for *robot_uri*, discovering and caching it."""
        robot = self._robot_cache.get(robot_uri)
        if robot is None:
            from apyrobo.core.robot import Robot  # lazy import

            robot = Robot.discover(robot_uri)
            self._robot_cache[robot_uri] = robot
        return robot


# ---------------------------------------------------------------------------
# Stdio adapter
# ---------------------------------------------------------------------------

class StdioOrchestrationAdapter(OrchestrationAdapter):
    """JSON-over-stdio adapter.

    Reads one JSON object per line from stdin, writes one JSON object per
    line to stdout.  Exits cleanly on EOF.

    Input format::

        {"task": "navigate to dock", "robot_uri": "mock://turtlebot4"}

    Output format::

        {"task": "...", "robot_uri": "...", "metadata": {...}, "source": "..."}
    """

    def __init__(
        self,
        infile: Any = None,
        outfile: Any = None,
    ) -> None:
        self._in = infile or sys.stdin
        self._out = outfile or sys.stdout

    def receive(self) -> OrchestrationMessage | None:
        try:
            line = self._in.readline()
        except (EOFError, KeyboardInterrupt):
            return None
        if not line:
            return None
        line = line.strip()
        if not line:
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("StdioOrchestrationAdapter: bad JSON (%s): %r", exc, line)
            return OrchestrationMessage(task=line)  # treat raw text as task
        if not isinstance(data, dict):
            # Valid JSON but not a message object ("[1, 2]", "42", …) —
            # same malformed-input rule per wire-protocol §2.3.
            logger.warning("StdioOrchestrationAdapter: non-object JSON: %r", line)
            return OrchestrationMessage(task=line)
        return OrchestrationMessage.from_dict(data)

    def send(self, msg: OrchestrationMessage) -> None:
        print(json.dumps(msg.to_dict()), file=self._out, flush=True)

    def startup(self) -> None:
        logger.info("StdioOrchestrationAdapter ready")

    def shutdown(self) -> None:
        logger.info("StdioOrchestrationAdapter closed")


# ---------------------------------------------------------------------------
# Mock adapter — for tests
# ---------------------------------------------------------------------------

class MockOrchestrationAdapter(OrchestrationAdapter):
    """Pre-loaded adapter for unit tests.

    Pass a list of task strings (or ``OrchestrationMessage`` objects).
    ``receive()`` returns them one by one, then returns ``None``.

    All responses from ``send()`` are accumulated in ``sent``.
    """

    def __init__(
        self,
        tasks: list[str | OrchestrationMessage] | None = None,
    ) -> None:
        self._queue: list[OrchestrationMessage] = []
        for t in (tasks or []):
            if isinstance(t, str):
                self._queue.append(OrchestrationMessage(task=t))
            else:
                self._queue.append(t)
        self.sent: list[OrchestrationMessage] = []
        self.startup_called = False
        self.shutdown_called = False

    def receive(self) -> OrchestrationMessage | None:
        if not self._queue:
            return None
        return self._queue.pop(0)

    def send(self, msg: OrchestrationMessage) -> None:
        self.sent.append(msg)

    def startup(self) -> None:
        self.startup_called = True

    def shutdown(self) -> None:
        self.shutdown_called = True


# ---------------------------------------------------------------------------
# WebSocket adapter
# ---------------------------------------------------------------------------

try:
    from websockets.asyncio.server import serve as _ws_serve  # type: ignore
    _WEBSOCKETS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _WEBSOCKETS_AVAILABLE = False


class WebSocketOrchestrationAdapter(OrchestrationAdapter):
    """WebSocket-based orchestration adapter.

    Starts a WebSocket server on ``host:port``.  Each connected client can
    send JSON task messages; responses are broadcast to **all** currently
    connected clients.

    The adapter presents the same *synchronous* interface as the other adapters
    (``receive()`` blocks, ``send()`` is immediate) while running the async
    WebSocket server in a dedicated background thread.

    Requires the ``websockets`` library::

        pip install 'apyrobo[websocket]'

    Protocol — identical to ``StdioOrchestrationAdapter``:

    * Client → server: ``{"task": "navigate to dock", "robot_uri": "mock://..."}``
    * Server → client: ``{"task": "...", "robot_uri": "...", "metadata": {...}, "source": "..."}``
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        if not _WEBSOCKETS_AVAILABLE:
            raise ImportError(
                "The 'websockets' package is required for WebSocketOrchestrationAdapter.\n"
                "Install with: pip install 'apyrobo[websocket]'"
            )
        self._host = host
        self._port = port
        self._recv_queue: queue.Queue[OrchestrationMessage | None] = queue.Queue()
        self._clients: set[Any] = set()
        self._clients_lock = threading.Lock()
        self._loop: Any = None          # asyncio event loop running in background
        self._server: Any = None        # websockets Server
        self._stop_event: Any = None    # asyncio.Event created inside the loop
        self._thread: threading.Thread | None = None
        self._started = threading.Event()

    # ------------------------------------------------------------------
    # Async internals (run inside background thread)
    # ------------------------------------------------------------------

    async def _handler(self, websocket: Any) -> None:
        """Handle a single WebSocket client connection."""
        with self._clients_lock:
            self._clients.add(websocket)
        logger.info("WebSocketOrchestrationAdapter: client connected (%s)", websocket.remote_address)
        try:
            async for raw in websocket:
                raw = raw.strip() if isinstance(raw, str) else raw.decode().strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                    msg = OrchestrationMessage.from_dict(data)
                except (json.JSONDecodeError, Exception) as exc:
                    logger.warning(
                        "WebSocketOrchestrationAdapter: bad JSON (%s): %r", exc, raw
                    )
                    msg = OrchestrationMessage(task=raw)
                self._recv_queue.put(msg)
        except Exception as exc:
            logger.debug("WebSocketOrchestrationAdapter: client error: %s", exc)
        finally:
            with self._clients_lock:
                self._clients.discard(websocket)
            logger.info("WebSocketOrchestrationAdapter: client disconnected")

    async def _serve(self) -> None:
        """Start the WebSocket server and run until shutdown() is called.

        Leaving the ``async with`` block closes the server and waits for it,
        so the loop only ends after a clean close — no teardown races.
        """
        import asyncio

        self._stop_event = asyncio.Event()
        async with _ws_serve(self._handler, self._host, self._port) as srv:
            self._server = srv
            self._started.set()
            logger.info(
                "WebSocketOrchestrationAdapter: listening on ws://%s:%d",
                self._host, self._port,
            )
            await self._stop_event.wait()

    def _run_loop(self) -> None:
        """Entry point for the background thread."""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
        except Exception:
            logger.exception("WebSocketOrchestrationAdapter: server loop failed")
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    # ------------------------------------------------------------------
    # OrchestrationAdapter interface
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the WebSocket server in a background thread."""
        self._thread = threading.Thread(
            target=self._run_loop, name="ws-orchestration", daemon=True
        )
        self._thread.start()
        # Wait until the server is actually listening (up to 10 s).
        if not self._started.wait(timeout=10):
            raise RuntimeError(
                f"WebSocketOrchestrationAdapter: server did not start within 10 s "
                f"(host={self._host!r}, port={self._port})"
            )
        logger.info("WebSocketOrchestrationAdapter ready")

    def shutdown(self) -> None:
        """Stop the WebSocket server and background thread.

        Signals the server coroutine to exit; it closes the server and
        drains connections before the loop ends. Safe to call repeatedly.
        """
        if (
            self._loop is not None
            and not self._loop.is_closed()
            and self._stop_event is not None
        ):
            # The loop may close between the check and the call.
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread is not None:
            self._thread.join(timeout=5)
        # Unblock any waiting receive() call.
        self._recv_queue.put(None)
        logger.info("WebSocketOrchestrationAdapter closed")

    def receive(self) -> OrchestrationMessage | None:
        """Block until a message arrives from any connected client.

        Returns ``None`` when the adapter is shut down.
        """
        return self._recv_queue.get()

    def send(self, msg: OrchestrationMessage) -> None:
        """Broadcast *msg* as JSON to all currently connected clients."""
        import asyncio

        payload = json.dumps(msg.to_dict())
        with self._clients_lock:
            clients = set(self._clients)

        if not clients:
            logger.debug("WebSocketOrchestrationAdapter.send: no connected clients")
            return

        async def _broadcast() -> None:
            import asyncio
            coros = [c.send(payload) for c in clients]
            results = await asyncio.gather(*coros, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    logger.debug("WebSocketOrchestrationAdapter.send error: %s", res)

        if self._loop is not None and not self._loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(_broadcast(), self._loop)
            try:
                future.result(timeout=5)
            except Exception as exc:
                logger.warning("WebSocketOrchestrationAdapter.send failed: %s", exc)
