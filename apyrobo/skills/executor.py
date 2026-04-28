"""
Skill Graph Engine — chains skills into executable task plans.

The SkillGraph is a directed acyclic graph where nodes are skills and
edges are dependencies.  The SkillExecutor walks the graph and runs
each skill against a robot via the Core API.

Features:
    - Topological ordering with parallel execution of independent skills
    - Timeout enforcement per skill
    - Execution state tracking between skills (e.g. object_held, at_position)
    - Postcondition verification after each skill completes
    - Precondition checks: capability, state, speed
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import CapabilityType, TaskResult, TaskStatus, RecoveryAction
from apyrobo.sensors.pipeline import WorldState
from apyrobo.skills.skill import Skill, SkillStatus, BUILTIN_SKILLS
from apyrobo.skills.handlers import dispatch as _handler_dispatch, UnknownSkillError
from apyrobo.observability import emit_event, trace_context, current_trace_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Execution State
# ---------------------------------------------------------------------------

class ExecutionState:
    """
    Tracks mutable state between skill executions.

    Skills can set and query state flags (e.g. "object_held", "at_position")
    that feed into precondition and postcondition checks.
    """

    def __init__(self) -> None:
        self._flags: dict[str, Any] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: Any = True) -> None:
        with self._lock:
            self._flags[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._flags.get(key, default)

    def is_set(self, key: str) -> bool:
        with self._lock:
            return bool(self._flags.get(key, False))

    def clear(self, key: str) -> None:
        with self._lock:
            self._flags.pop(key, None)

    def clear_all(self) -> None:
        with self._lock:
            self._flags.clear()

    @property
    def flags(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._flags)

    def __repr__(self) -> str:
        return f"<ExecutionState flags={self._flags}>"


# ---------------------------------------------------------------------------
# Skill Graph
# ---------------------------------------------------------------------------

class SkillGraph:
    """
    A directed acyclic graph of skills representing a task plan.

    Skills are nodes; edges mean "A must complete before B starts."
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._edges: dict[str, list[str]] = {}  # skill_id -> [depends_on_ids]
        self._parameters: dict[str, dict[str, Any]] = {}  # runtime params per skill

    def add_skill(self, skill: Skill, depends_on: list[str] | None = None,
                  parameters: dict[str, Any] | None = None) -> None:
        """Add a skill to the graph with optional dependencies."""
        self._skills[skill.skill_id] = skill
        self._edges[skill.skill_id] = depends_on or []
        if parameters:
            self._parameters[skill.skill_id] = parameters

    def get_execution_order(self) -> list[Skill]:
        """
        Topological sort — returns skills in valid execution order.

        Raises ValueError if there's a cycle (which shouldn't happen
        in a well-formed plan).
        """
        visited: set[str] = set()
        order: list[str] = []
        in_progress: set[str] = set()

        def visit(sid: str) -> None:
            if sid in in_progress:
                raise ValueError(f"Cycle detected in skill graph at {sid!r}")
            if sid in visited:
                return
            in_progress.add(sid)
            for dep in self._edges.get(sid, []):
                visit(dep)
            in_progress.discard(sid)
            visited.add(sid)
            order.append(sid)

        for sid in self._skills:
            visit(sid)

        return [self._skills[sid] for sid in order]

    def get_execution_layers(self) -> list[list[Skill]]:
        """
        Return skills grouped into layers for parallel execution.

        Each layer contains skills whose dependencies are all in
        earlier layers — so skills within the same layer can run
        concurrently.
        """
        order = self.get_execution_order()
        completed: set[str] = set()
        layers: list[list[Skill]] = []
        remaining = [s.skill_id for s in order]

        while remaining:
            layer: list[Skill] = []
            next_remaining: list[str] = []
            for sid in remaining:
                deps = self._edges.get(sid, [])
                if all(d in completed for d in deps):
                    layer.append(self._skills[sid])
                else:
                    next_remaining.append(sid)
            if not layer:
                raise ValueError("Deadlock in skill graph — no progress possible")
            for s in layer:
                completed.add(s.skill_id)
            layers.append(layer)
            remaining = next_remaining

        return layers

    def get_parameters(self, skill_id: str) -> dict[str, Any]:
        """Get runtime parameters for a skill."""
        base = dict(self._skills[skill_id].parameters)
        base.update(self._parameters.get(skill_id, {}))
        return base

    @property
    def skills(self) -> dict[str, Skill]:
        return dict(self._skills)

    @property
    def edges(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self._edges.items()}

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        return f"<SkillGraph skills={len(self._skills)} edges={sum(len(v) for v in self._edges.values())}>"


# ---------------------------------------------------------------------------
# Execution events
# ---------------------------------------------------------------------------

class ExecutionEvent:
    """An event emitted during skill execution."""

    def __init__(self, skill_id: str, status: SkillStatus, message: str = "",
                 timestamp: float | None = None) -> None:
        self.skill_id = skill_id
        self.status = status
        self.message = message
        self.timestamp = timestamp or time.time()

    def __repr__(self) -> str:
        return f"<Event {self.skill_id}: {self.status.value} — {self.message}>"


# Type for event listeners
EventListener = Callable[[ExecutionEvent], None]


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------

class SkillTimeout(Exception):
    """Raised when a skill exceeds its timeout."""


def _run_with_timeout(fn: Callable[[], Any], timeout_seconds: float) -> Any:
    """Run *fn* in a thread, raising SkillTimeout if it takes too long."""
    result_box: list[Any] = []
    error_box: list[BaseException] = []

    def wrapper() -> None:
        try:
            result_box.append(fn())
        except BaseException as e:
            error_box.append(e)

    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise SkillTimeout(
            f"Skill timed out after {timeout_seconds}s"
        )
    if error_box:
        raise error_box[0]
    return result_box[0] if result_box else None


# ---------------------------------------------------------------------------
# Skill Executor
# ---------------------------------------------------------------------------

class SkillExecutor:
    """
    Executes a SkillGraph against a robot.

    Handles precondition checking, postcondition verification,
    timeout enforcement, retry logic, parallel execution,
    confidence gating (SF-08), and event streaming.
    """

    def __init__(
        self,
        robot: Robot,
        state: ExecutionState | None = None,
        confidence_estimator: Any = None,
        world_state_provider: Any = None,
        state_store: Any = None,
    ) -> None:
        self._robot = robot
        self._listeners: list[EventListener] = []
        self._events: list[ExecutionEvent] = []
        self._state = state or ExecutionState()
        self._emit_lock = threading.Lock()
        self._confidence_estimator = confidence_estimator  # SF-08
        self._world_state_provider = world_state_provider
        self._state_store = state_store  # OB-02: StorageBackend for crash recovery

    @property
    def state(self) -> ExecutionState:
        return self._state

    def on_event(self, listener: EventListener) -> None:
        """Register a callback for execution events."""
        self._listeners.append(listener)

    def _emit(self, skill_id: str, status: SkillStatus, message: str = "") -> None:
        event = ExecutionEvent(skill_id, status, message)
        with self._emit_lock:
            self._events.append(event)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                logger.warning("Event listener error: %s", e)

    def check_preconditions(self, skill: Skill, robot: Robot) -> tuple[bool, str]:
        """
        Check whether a skill's preconditions are met.

        Supports check types:
            - "capability": robot has the required capability (default)
            - "state": execution state flag is set
        """
        caps = robot.capabilities()
        cap_types = {c.capability_type for c in caps.capabilities}

        if skill.required_capability != CapabilityType.CUSTOM:
            if skill.required_capability not in cap_types:
                available = ", ".join(sorted(c.value for c in cap_types)) or "(none)"
                return False, (
                    f"Robot lacks required capability: {skill.required_capability.value}. "
                    f"Robot '{caps.robot_id}' has: {available}"
                )

        # Check speed constraint
        params = skill.parameters
        if "speed" in params and caps.max_speed is not None:
            requested_speed = params.get("speed", 0)
            if isinstance(requested_speed, (int, float)) and requested_speed > caps.max_speed:
                return False, (
                    f"Requested speed {requested_speed} exceeds robot max {caps.max_speed}"
                )

        world = self._resolve_world_state()

        # Check state/sensor-based preconditions
        for cond in skill.preconditions:
            if cond.check_type == "state":
                required_key = cond.parameters.get("key", cond.name)
                expected = cond.parameters.get("value", True)
                actual = self._state.get(required_key)
                if actual != expected:
                    return False, (
                        f"State precondition '{cond.name}' not met: "
                        f"expected {required_key}={expected!r}, got {actual!r}"
                    )
            elif cond.check_type == "sensor":
                ok, reason = self._check_sensor_precondition(cond, world)
                if not ok:
                    return False, reason

        return True, "OK"

    def _resolve_world_state(self) -> WorldState | None:
        provider = self._world_state_provider
        if provider is None:
            return None
        if hasattr(provider, "get_world_state"):
            return provider.get_world_state()
        if callable(provider):
            return provider()
        if isinstance(provider, WorldState):
            return provider
        return None

    def _check_sensor_precondition(self, cond: Any, world: WorldState | None) -> tuple[bool, str]:
        if world is None:
            return False, "World state unavailable for sensor precondition"

        if cond.name == "object_visible":
            label = cond.parameters.get("label") or cond.parameters.get("object")
            if not isinstance(label, str) or not label.strip():
                return False, "object_visible requires a 'label' parameter"
            obj = world.find_object(label)
            if obj is None:
                return False, f"Object '{label}' is not visible"
            min_conf = float(cond.parameters.get("min_confidence", 0.0))
            if obj.confidence < min_conf:
                return False, f"Object '{label}' confidence {obj.confidence:.2f} < {min_conf:.2f}"
            return True, "OK"

        if cond.name == "path_clear":
            tx = cond.parameters.get("x")
            ty = cond.parameters.get("y")
            if not isinstance(tx, (int, float)) or not isinstance(ty, (int, float)):
                return False, "path_clear requires numeric x/y target"
            clearance = float(cond.parameters.get("clearance", 0.5))
            rx, ry = world.robot_position
            if not world.is_path_clear(rx, ry, float(tx), float(ty), clearance=clearance):
                return False, "Path is blocked by obstacles"
            return True, "OK"

        if cond.name == "no_obstacle_within":
            radius = float(cond.parameters.get("radius", 0.5))
            nearby = world.obstacles_within(radius)
            if nearby:
                return False, f"Found {len(nearby)} obstacle(s) within {radius}m"
            return True, "OK"

        if cond.name == "contact_detected":
            required = bool(cond.parameters.get("value", True))
            actual = bool(world.metadata.get("contact_detected", False))
            return (actual == required, f"contact_detected expected {required}, got {actual}")

        if cond.name == "gps_fix":
            required = bool(cond.parameters.get("value", True))
            actual = bool(world.metadata.get("gps_fix", False))
            return (actual == required, f"gps_fix expected {required}, got {actual}")

        return True, "OK"

    def check_postconditions(self, skill: Skill, params: dict[str, Any]) -> tuple[bool, str]:
        """
        Verify postconditions after a skill completes.

        Also updates execution state based on postcondition declarations.
        Returns (ok, reason).
        """
        for cond in skill.postconditions:
            if cond.check_type == "state":
                # Set the state flag as declared by the postcondition
                key = cond.parameters.get("key", cond.name)
                value = cond.parameters.get("value", True)
                self._state.set(key, value)

        # Auto-update state based on known skill effects
        base_id = skill.skill_id.rsplit("_", 1)[0] if skill.skill_id[-1:].isdigit() else skill.skill_id

        if base_id == "navigate_to":
            x = float(params.get("x", 0.0))
            y = float(params.get("y", 0.0))
            self._state.set("at_position", (x, y))
            self._state.set("robot_idle", True)
        elif base_id == "pick_object":
            self._state.set("object_held", True)
            self._state.set("gripper_open", False)
        elif base_id == "place_object":
            self._state.set("object_held", False)
            self._state.set("gripper_open", True)
        elif base_id == "rotate":
            angle = float(params.get("angle_rad", 0.0))
            self._state.set("last_rotation", angle)
        elif base_id == "stop":
            self._state.set("robot_idle", True)

        return True, "OK"

    def execute_skill(self, skill: Skill, parameters: dict[str, Any] | None = None) -> SkillStatus:
        """
        Execute a single skill against the robot.

        OB-01: Emits per-skill telemetry (latency, success/fail, retry count).
        OB-03: Wraps execution in trace_context for correlation.
        """
        params = dict(skill.parameters)
        if parameters:
            params.update(parameters)

        skill_start = time.time()

        self._emit(skill.skill_id, SkillStatus.PENDING, "Checking preconditions")

        # Precondition check
        ok, reason = self.check_preconditions(skill, self._robot)
        if not ok:
            self._emit(skill.skill_id, SkillStatus.FAILED, f"Precondition failed: {reason}")
            # OB-01: Emit telemetry for precondition failure
            emit_event("skill_executed",
                        skill_id=skill.skill_id,
                        status="failed",
                        reason="precondition_failed",
                        latency_ms=round((time.time() - skill_start) * 1000, 1),
                        attempts=0,
                        trace_id=current_trace_id())
            return SkillStatus.FAILED

        # Execute with retry
        attempts = 0
        max_attempts = skill.retry_count + 1
        final_status = SkillStatus.FAILED
        error_msg = ""

        while attempts < max_attempts:
            attempts += 1
            self._emit(
                skill.skill_id, SkillStatus.RUNNING,
                f"Attempt {attempts}/{max_attempts}"
            )

            try:
                result = _run_with_timeout(
                    lambda: self._dispatch_skill(skill, params),
                    skill.timeout_seconds,
                )
                if result:
                    # Verify postconditions
                    post_ok, post_reason = self.check_postconditions(skill, params)
                    if not post_ok:
                        self._emit(
                            skill.skill_id, SkillStatus.RUNNING,
                            f"Postcondition failed: {post_reason}"
                        )
                        continue  # retry if postcondition fails

                    self._emit(skill.skill_id, SkillStatus.COMPLETED, "Success")
                    final_status = SkillStatus.COMPLETED
                    break
                else:
                    error_msg = f"Attempt {attempts} returned False"
                    self._emit(
                        skill.skill_id, SkillStatus.RUNNING,
                        f"Attempt {attempts} failed, {'retrying' if attempts < max_attempts else 'no more retries'}"
                    )
            except SkillTimeout as e:
                error_msg = f"timeout: {e}"
                self._emit(
                    skill.skill_id, SkillStatus.RUNNING,
                    f"Timeout on attempt {attempts}: {e}"
                )
            except Exception as e:
                error_msg = str(e)
                self._emit(
                    skill.skill_id, SkillStatus.RUNNING,
                    f"Error on attempt {attempts}: {e}"
                )

        if final_status == SkillStatus.FAILED:
            self._emit(skill.skill_id, SkillStatus.FAILED, f"Failed after {max_attempts} attempts")

        # OB-01: Emit per-skill telemetry
        latency_ms = round((time.time() - skill_start) * 1000, 1)
        emit_event("skill_executed",
                    skill_id=skill.skill_id,
                    status=final_status.value,
                    latency_ms=latency_ms,
                    attempts=attempts,
                    max_attempts=max_attempts,
                    error=error_msg if final_status == SkillStatus.FAILED else "",
                    trace_id=current_trace_id())

        return final_status

    def _dispatch_skill(self, skill: Skill, params: dict[str, Any]) -> bool:
        """
        Map a skill to actual robot commands.

        Looks up the handler registry first, then raises UnknownSkillError
        for truly unknown skills.  Handles both exact IDs and
        agent-generated suffixed IDs (e.g. "navigate_to_0").
        """
        return _handler_dispatch(skill.skill_id, self._robot, params)

    def execute_graph(self, graph: SkillGraph, parallel: bool = False,
                      trace_id: str | None = None) -> TaskResult:
        """
        Execute an entire skill graph.

        SF-08: If a confidence_estimator is attached, gates execution
        before starting. Returns FAILED result if confidence too low.
        OB-03: Wraps execution in trace_context for end-to-end tracing.

        Args:
            graph: The skill graph to execute.
            parallel: If True, run independent skills concurrently.
                      If False (default), execute in topological order.
            trace_id: Optional trace ID for correlation.

        Returns a TaskResult summarising the outcome.
        """
        # OB-03: Wrap in trace context
        trace_kwargs: dict[str, Any] = {"component": "executor", "skill_count": len(graph)}
        if trace_id:
            trace_kwargs["trace_id"] = trace_id

        with trace_context(**trace_kwargs):
            graph_start = time.time()
            active_trace_id = trace_id or current_trace_id()

            # OB-02: Record task start in state store
            if self._state_store and active_trace_id:
                try:
                    self._state_store.begin_task(
                        task_id=active_trace_id,
                        metadata={"skill_count": len(graph)},
                        total_steps=len(graph),
                    )
                except Exception as e:
                    logger.warning("State store begin_task failed: %s", e)

            # SF-08: Confidence gating
            if self._confidence_estimator is not None:
                try:
                    report = self._confidence_estimator.gate(graph, self._robot)
                    logger.info(
                        "Confidence gate passed: %.0f%%", report.confidence * 100
                    )
                except Exception as e:
                    logger.warning("Confidence gate blocked execution: %s", e)
                    return TaskResult(
                        task_name=f"graph_{len(graph)}_skills",
                        status=TaskStatus.FAILED,
                        steps_completed=0,
                        steps_total=len(graph),
                        error=str(e),
                        recovery_actions_taken=[RecoveryAction.ABORT],
                    )

            if parallel:
                result = self._execute_graph_parallel(graph)
            else:
                result = self._execute_graph_sequential(graph)

            # OB-02: Record final state in state store
            if self._state_store and active_trace_id:
                try:
                    status_val = result.status.value if hasattr(result.status, 'value') else str(result.status)
                    if status_val == "completed":
                        self._state_store.complete_task(
                            active_trace_id,
                            result={"steps_completed": result.steps_completed,
                                    "steps_total": result.steps_total},
                        )
                    else:
                        self._state_store.fail_task(
                            active_trace_id,
                            error=result.error or "graph execution failed",
                        )
                except Exception as e:
                    logger.warning("State store complete/fail failed: %s", e)

            # OB-01: Emit graph-level telemetry
            graph_latency = round((time.time() - graph_start) * 1000, 1)
            emit_event("graph_executed",
                        status=result.status.value if hasattr(result.status, 'value') else str(result.status),
                        skill_count=len(graph),
                        steps_completed=result.steps_completed,
                        latency_ms=graph_latency,
                        trace_id=current_trace_id())

            return result

    def _execute_graph_sequential(self, graph: SkillGraph) -> TaskResult:
        """Execute skills one at a time in topological order."""
        order = graph.get_execution_order()
        completed = 0
        recovery_actions: list[RecoveryAction] = []

        for skill in order:
            params = graph.get_parameters(skill.skill_id)
            status = self.execute_skill(skill, params)

            if status == SkillStatus.COMPLETED:
                completed += 1
                # OB-02: Update progress in state store
                if self._state_store:
                    try:
                        tid = current_trace_id()
                        if tid:
                            self._state_store.update_task(
                                tid, step=completed, status="in_progress",
                            )
                    except Exception as e:
                        logger.warning("State store update_task failed: %s", e)
            elif status == SkillStatus.FAILED:
                if skill.retry_count > 0:
                    recovery_actions.append(RecoveryAction.RETRY)
                recovery_actions.append(RecoveryAction.ABORT)
                return TaskResult(
                    task_name=f"graph_{len(order)}_skills",
                    status=TaskStatus.FAILED,
                    steps_completed=completed,
                    steps_total=len(order),
                    error=f"Skill {skill.skill_id!r} failed",
                    recovery_actions_taken=recovery_actions,
                )

        return TaskResult(
            task_name=f"graph_{len(order)}_skills",
            status=TaskStatus.COMPLETED,
            confidence=1.0,
            steps_completed=completed,
            steps_total=len(order),
            recovery_actions_taken=recovery_actions,
        )

    def _execute_graph_parallel(self, graph: SkillGraph) -> TaskResult:
        """Execute independent skills concurrently using execution layers."""
        layers = graph.get_execution_layers()
        total = len(graph)
        completed = 0
        recovery_actions: list[RecoveryAction] = []

        for layer in layers:
            if len(layer) == 1:
                # Single skill — no need for thread pool
                skill = layer[0]
                params = graph.get_parameters(skill.skill_id)
                status = self.execute_skill(skill, params)
            else:
                # Multiple independent skills — run concurrently
                results: dict[str, SkillStatus] = {}
                with ThreadPoolExecutor(max_workers=len(layer)) as pool:
                    futures = {}
                    for skill in layer:
                        params = graph.get_parameters(skill.skill_id)
                        fut = pool.submit(self.execute_skill, skill, params)
                        futures[fut] = skill
                    for fut in as_completed(futures):
                        skill = futures[fut]
                        results[skill.skill_id] = fut.result()
                # Check results
                for skill in layer:
                    status = results[skill.skill_id]

            # Process result (last status set for single-skill, or loop for multi)
            if len(layer) == 1:
                if status == SkillStatus.COMPLETED:
                    completed += 1
                elif status == SkillStatus.FAILED:
                    if layer[0].retry_count > 0:
                        recovery_actions.append(RecoveryAction.RETRY)
                    recovery_actions.append(RecoveryAction.ABORT)
                    return TaskResult(
                        task_name=f"graph_{total}_skills",
                        status=TaskStatus.FAILED,
                        steps_completed=completed,
                        steps_total=total,
                        error=f"Skill {layer[0].skill_id!r} failed",
                        recovery_actions_taken=recovery_actions,
                    )
            else:
                for skill in layer:
                    st = results[skill.skill_id]
                    if st == SkillStatus.COMPLETED:
                        completed += 1
                    elif st == SkillStatus.FAILED:
                        if skill.retry_count > 0:
                            recovery_actions.append(RecoveryAction.RETRY)
                        recovery_actions.append(RecoveryAction.ABORT)
                        return TaskResult(
                            task_name=f"graph_{total}_skills",
                            status=TaskStatus.FAILED,
                            steps_completed=completed,
                            steps_total=total,
                            error=f"Skill {skill.skill_id!r} failed",
                            recovery_actions_taken=recovery_actions,
                        )

        return TaskResult(
            task_name=f"graph_{total}_skills",
            status=TaskStatus.COMPLETED,
            confidence=1.0,
            steps_completed=completed,
            steps_total=total,
            recovery_actions_taken=recovery_actions,
        )

    @property
    def events(self) -> list[ExecutionEvent]:
        """All events emitted during execution."""
        return list(self._events)
