"""Multi-agent coordination bus for APYROBO.

Implements a ``TaskBus`` that allows multiple ``Agent`` instances (each
attached to a different robot) to negotiate tasks without explicit
point-to-point wiring.  An agent can sub-contract a skill to whichever
peer agent has the required capability.

Architecture::

    Agent A ──┐                           ┌── Agent B (arm-bot: PICK/PLACE)
              ▼                           ▼
         MultiAgentCoordinator A    MultiAgentCoordinator B
              │                           │
              └──────► TaskBus ◄──────────┘
                        │
                    dispatch("pick up cup", required_capability="PICK")
                        │
                   Routes to B → plans → returns TaskResult

v7.0.0 — APYROBO Category Ownership
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TaskRequest:
    """A task sent over the bus from one agent to another (or broadcast)."""

    task: str
    robot_uri: str = ""
    required_capability: str = ""
    requester_id: str = ""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "robot_uri": self.robot_uri,
            "required_capability": self.required_capability,
            "requester_id": self.requester_id,
            "request_id": self.request_id,
            "metadata": self.metadata,
        }


@dataclass
class TaskResult:
    """Result returned by the agent that handled a TaskRequest."""

    request_id: str
    agent_id: str
    robot_uri: str
    success: bool
    skills_planned: list[str] = field(default_factory=list)
    error: str = ""
    elapsed_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "agent_id": self.agent_id,
            "robot_uri": self.robot_uri,
            "success": self.success,
            "skills_planned": self.skills_planned,
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# ---------------------------------------------------------------------------
# TaskBus
# ---------------------------------------------------------------------------

class TaskBus:
    """Shared coordination bus for multi-agent skill sub-contracting.

    The bus maintains a registry of active ``MultiAgentCoordinator``
    instances and routes ``TaskRequest`` objects to the best available
    agent.

    **Routing logic** (in priority order):

    1. If ``required_capability`` is set, pick an agent that advertises it.
    2. Among matching agents, prefer the one with the shortest pending queue.
    3. If no agents match, try any agent (fallback).
    4. If no agents are registered, return a failure ``TaskResult``.

    All operations are thread-safe — coordinators may register/unregister
    from any thread.

    Parameters
    ----------
    timeout:
        Maximum seconds to wait for an agent to return a result (default 30).

    Examples
    --------
    ::

        bus = TaskBus()
        result = bus.dispatch("pick up cup", required_capability="PICK")
        if result.success:
            print("Planned skills:", result.skills_planned)
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout
        self._coordinators: dict[str, "MultiAgentCoordinator"] = {}
        self._lock = threading.Lock()
        self._result_queues: dict[str, queue.Queue[TaskResult]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, coordinator: "MultiAgentCoordinator") -> None:
        """Register a coordinator on this bus.

        Called automatically by ``MultiAgentCoordinator.start()``.
        """
        with self._lock:
            self._coordinators[coordinator.agent_id] = coordinator
        logger.info("TaskBus: registered agent %r (caps=%s)",
                    coordinator.agent_id, coordinator.capabilities)

    def unregister(self, agent_id: str) -> None:
        """Remove a coordinator from the bus.

        Called automatically by ``MultiAgentCoordinator.stop()``.
        """
        with self._lock:
            self._coordinators.pop(agent_id, None)
        logger.info("TaskBus: unregistered agent %r", agent_id)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(
        self,
        task: str,
        *,
        required_capability: str = "",
        robot_uri: str = "",
        requester_id: str = "bus",
        metadata: dict[str, Any] | None = None,
    ) -> TaskResult:
        """Dispatch *task* to the best available agent and wait for a result.

        Parameters
        ----------
        task:
            Natural language task description (same as ``Agent.plan()``).
        required_capability:
            Optional capability string (e.g. ``"PICK"``, ``"NAVIGATE"``).
            If set, only agents that advertise this capability are eligible.
        robot_uri:
            Override the target robot URI.  If empty, the selected agent's
            default robot URI is used.
        requester_id:
            Identifier of the requesting agent (for logging).
        metadata:
            Arbitrary key-value data forwarded to the handler.

        Returns
        -------
        TaskResult
            Result from the handling agent; ``success=False`` if no agent
            was available or the task timed out.
        """
        request = TaskRequest(
            task=task,
            robot_uri=robot_uri,
            required_capability=required_capability,
            requester_id=requester_id,
            metadata=metadata or {},
        )

        agent = self._select_agent(required_capability)
        if agent is None:
            logger.warning(
                "TaskBus.dispatch: no agent available for capability %r",
                required_capability or "any",
            )
            return TaskResult(
                request_id=request.request_id,
                agent_id="",
                robot_uri=robot_uri,
                success=False,
                error=f"No agent available for capability {required_capability!r}",
            )

        result_q: queue.Queue[TaskResult] = queue.Queue()
        with self._lock:
            self._result_queues[request.request_id] = result_q

        try:
            agent._enqueue(request)
            logger.info(
                "TaskBus.dispatch: routed %r to agent %r (request_id=%s)",
                task, agent.agent_id, request.request_id,
            )
            result = result_q.get(timeout=self._timeout)
            return result
        except queue.Empty:
            return TaskResult(
                request_id=request.request_id,
                agent_id=agent.agent_id,
                robot_uri=robot_uri or agent.robot_uri,
                success=False,
                error=f"Timed out after {self._timeout}s waiting for agent {agent.agent_id!r}",
            )
        finally:
            with self._lock:
                self._result_queues.pop(request.request_id, None)

    def _deliver_result(self, result: TaskResult) -> None:
        """Called by a coordinator to deliver a result back to dispatch()."""
        with self._lock:
            q = self._result_queues.get(result.request_id)
        if q is not None:
            q.put(result)

    def _select_agent(self, required_capability: str) -> "MultiAgentCoordinator | None":
        """Pick the best registered agent for *required_capability*."""
        with self._lock:
            candidates = list(self._coordinators.values())

        if not candidates:
            return None

        if required_capability:
            cap_upper = required_capability.upper()
            matching = [
                c for c in candidates
                if cap_upper in (cap.upper() for cap in c.capabilities)
            ]
            if matching:
                return min(matching, key=lambda c: c.queue_depth)

        # Fall back to least-loaded agent
        return min(candidates, key=lambda c: c.queue_depth)

    # ------------------------------------------------------------------
    # Broadcast
    # ------------------------------------------------------------------

    def broadcast(
        self,
        task: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[TaskResult]:
        """Dispatch *task* to ALL registered agents and collect results.

        Returns a list of ``TaskResult`` objects (one per agent) in the
        order they complete.  Results are collected in parallel with a
        per-agent timeout of ``self._timeout`` seconds.

        Parameters
        ----------
        task:
            Task description forwarded to each agent.
        metadata:
            Optional metadata forwarded to every agent.

        Returns
        -------
        list[TaskResult]
            Results from all registered agents.
        """
        with self._lock:
            agents = list(self._coordinators.values())

        if not agents:
            return []

        results: list[TaskResult] = []
        results_lock = threading.Lock()

        def _dispatch_one(agent: "MultiAgentCoordinator") -> None:
            r = self.dispatch(task, requester_id="broadcast", metadata=metadata)
            with results_lock:
                results.append(r)

        threads = [
            threading.Thread(target=_dispatch_one, args=(a,), daemon=True)
            for a in agents
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=self._timeout + 1)

        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def agent_count(self) -> int:
        """Number of currently registered agents."""
        with self._lock:
            return len(self._coordinators)

    def agent_ids(self) -> list[str]:
        """Return a snapshot list of registered agent IDs."""
        with self._lock:
            return list(self._coordinators.keys())

    def agent_capabilities(self) -> dict[str, list[str]]:
        """Return {agent_id: [capability, ...]} for all registered agents."""
        with self._lock:
            return {
                aid: list(c.capabilities)
                for aid, c in self._coordinators.items()
            }


# ---------------------------------------------------------------------------
# MultiAgentCoordinator
# ---------------------------------------------------------------------------

class MultiAgentCoordinator:
    """Connects a single Agent+Robot pair to a shared TaskBus.

    Each coordinator runs a background thread that dequeues
    ``TaskRequest`` objects from the bus, calls ``agent.plan(task, robot)``,
    and returns a ``TaskResult`` via the bus's result queue.

    Parameters
    ----------
    agent:
        An ``apyrobo.skills.agent.Agent`` instance.
    robot:
        An ``apyrobo.core.robot.Robot`` instance (or any object with
        ``capabilities()`` returning a ``RobotCapability``).
    bus:
        The shared ``TaskBus``.
    agent_id:
        Unique identifier for this coordinator (auto-generated if omitted).
    capabilities:
        List of capability strings this agent advertises (e.g.
        ``["NAVIGATE", "PICK"]``).  If empty, auto-populated from the
        robot's capabilities.
    robot_uri:
        Robot URI for logging and metadata; defaults to ``repr(robot)``.

    Examples
    --------
    ::

        bus = TaskBus()
        coord = MultiAgentCoordinator(agent, robot, bus, agent_id="arm-bot",
                                      capabilities=["MANIPULATE", "PICK"])
        coord.start()
        # The agent is now available via bus.dispatch()
        coord.stop()
    """

    def __init__(
        self,
        agent: Any,
        robot: Any,
        bus: TaskBus,
        *,
        agent_id: str | None = None,
        capabilities: list[str] | None = None,
        robot_uri: str = "",
    ) -> None:
        self._agent = agent
        self._robot = robot
        self._bus = bus
        self.agent_id: str = agent_id or f"agent-{str(uuid.uuid4())[:6]}"
        self.robot_uri: str = robot_uri or repr(robot)
        self._task_queue: queue.Queue[TaskRequest | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False

        # Auto-discover capabilities from robot if not provided
        if capabilities is not None:
            self.capabilities: list[str] = list(capabilities)
        else:
            self.capabilities = self._discover_capabilities()

    def _discover_capabilities(self) -> list[str]:
        """Query the robot's capabilities and return a list of strings."""
        try:
            caps = self._robot.capabilities()
            return [c.capability_type.value for c in caps.capabilities]
        except Exception as exc:
            logger.debug(
                "MultiAgentCoordinator %r: could not discover capabilities: %s",
                self.agent_id, exc,
            )
            return []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Register on the bus and start the background worker thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._work_loop,
            name=f"coord-{self.agent_id}",
            daemon=True,
        )
        self._thread.start()
        self._bus.register(self)
        logger.info("MultiAgentCoordinator %r: started", self.agent_id)

    def stop(self) -> None:
        """Unregister from the bus and stop the background thread."""
        self._bus.unregister(self.agent_id)
        self._running = False
        self._task_queue.put(None)  # sentinel
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("MultiAgentCoordinator %r: stopped", self.agent_id)

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _work_loop(self) -> None:
        """Dequeue TaskRequests and process them sequentially."""
        while self._running:
            try:
                request = self._task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if request is None:
                break  # sentinel

            self._handle(request)

    def _handle(self, request: TaskRequest) -> None:
        """Plan *request.task* and deliver a TaskResult to the bus."""
        t0 = time.monotonic()
        robot = self._robot
        robot_uri = request.robot_uri or self.robot_uri

        try:
            graph = self._agent.plan(request.task, robot)
            order = graph.get_execution_order()
            skills_planned = [
                getattr(s, "name", None) or getattr(s, "skill_id", str(s))
                for s in order
            ]
            elapsed_ms = (time.monotonic() - t0) * 1000
            result = TaskResult(
                request_id=request.request_id,
                agent_id=self.agent_id,
                robot_uri=robot_uri,
                success=True,
                skills_planned=skills_planned,
                elapsed_ms=elapsed_ms,
            )
            logger.info(
                "MultiAgentCoordinator %r: planned %d skills for %r in %.0fms",
                self.agent_id, len(skills_planned), request.task, elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            result = TaskResult(
                request_id=request.request_id,
                agent_id=self.agent_id,
                robot_uri=robot_uri,
                success=False,
                error=str(exc),
                elapsed_ms=elapsed_ms,
            )
            logger.warning(
                "MultiAgentCoordinator %r: planning failed for %r: %s",
                self.agent_id, request.task, exc,
            )

        self._bus._deliver_result(result)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enqueue(self, request: TaskRequest) -> None:
        """Put a task request into this coordinator's work queue."""
        self._task_queue.put(request)

    @property
    def queue_depth(self) -> int:
        """Number of pending tasks in this coordinator's queue."""
        return self._task_queue.qsize()
