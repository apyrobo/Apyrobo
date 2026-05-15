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
of tasks and is intended for unit tests.
"""
from __future__ import annotations

import abc
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message schema
# ---------------------------------------------------------------------------

@dataclass
class OrchestrationMessage:
    """A single task sent to, or response sent from, an orchestration adapter."""

    task: str
    robot_uri: str = "mock://turtlebot4"
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "robot_uri": self.robot_uri,
            "metadata": self.metadata,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrchestrationMessage":
        return cls(
            task=data.get("task", ""),
            robot_uri=data.get("robot_uri", "mock://turtlebot4"),
            metadata=data.get("metadata", {}),
            source=data.get("source", ""),
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

    Usage::

        from apyrobo.core.robot import Robot
        from apyrobo.skills.agent import Agent
        from apyrobo.orchestration import OrchestrationServer, StdioOrchestrationAdapter

        robot = Robot.discover("mock://turtlebot4")
        agent = Agent(provider="rule")
        adapter = StdioOrchestrationAdapter()
        server = OrchestrationServer(adapter, agent, default_robot=robot)
        server.run()
    """

    def __init__(
        self,
        adapter: OrchestrationAdapter,
        agent: Any,
        default_robot: Any = None,
        max_iterations: int | None = None,
    ) -> None:
        self.adapter = adapter
        self.agent = agent
        self.default_robot = default_robot
        self.max_iterations = max_iterations
        self._iterations = 0

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
                response = self._handle(msg)
                self.adapter.send(response)
        finally:
            self.adapter.shutdown()
            logger.info("OrchestrationServer stopped")

    def _handle(self, msg: OrchestrationMessage) -> OrchestrationMessage:
        """Plan a task from *msg* and return a response message."""
        try:
            from apyrobo.core.robot import Robot  # lazy import
            robot = self.default_robot or Robot.discover(msg.robot_uri)
            graph = self.agent.plan(msg.task, robot)
            order = graph.get_execution_order()
            skills = [
                {"skill_id": s.skill_id, "name": s.name}
                for s in order
            ]
            return OrchestrationMessage(
                task=msg.task,
                robot_uri=msg.robot_uri,
                metadata={"status": "planned", "skills": skills, "count": len(skills)},
                source="orchestration_server",
            )
        except Exception as exc:
            logger.warning("OrchestrationServer._handle error: %s", exc)
            return OrchestrationMessage(
                task=msg.task,
                robot_uri=msg.robot_uri,
                metadata={"status": "error", "error": str(exc)},
                source="orchestration_server",
            )


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
            return OrchestrationMessage.from_dict(data)
        except json.JSONDecodeError as exc:
            logger.warning("StdioOrchestrationAdapter: bad JSON (%s): %r", exc, line)
            return OrchestrationMessage(task=line)  # treat raw text as task

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
