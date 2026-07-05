"""Adapter-contract conformance checks (spec/adapter-contract.md).

Runs a live :class:`~apyrobo.core.adapters.CapabilityAdapter` instance
against the behavioral contract: capability declaration, required and
optional operations, lifecycle, safety-stop, and failure semantics.

.. warning::
   These checks issue real commands — ``move``, ``rotate``, ``stop``,
   gripper open/close, ``disconnect``/``connect``. Motion is minimized
   (the move target is the robot's current position) but you should run
   against a simulator or mock, never a robot near people.
"""
from __future__ import annotations

import contextlib
import logging
import numbers
from typing import Any

from apyrobo.conformance.report import ConformanceReport
from apyrobo.conformance.spec_schemas import JSONSCHEMA_AVAILABLE, validation_error
from apyrobo.core.schemas import AdapterState, RobotCapability

logger = logging.getLogger(__name__)

_CONTRACT = "adapter-contract.md"
_CAPMODEL = "capability-model.md"


def run_adapter_checks(adapter: Any, report: ConformanceReport) -> None:
    """Append adapter-contract check results to *report*."""
    capability = _check_capability_declaration(adapter, report)
    _check_required_operations(adapter, report, capability)
    _check_optional_operations(adapter, report)
    _check_lifecycle(adapter, report)
    _check_failure_semantics(adapter, report)


# ---------------------------------------------------------------------------
# Capability declaration (§2 of adapter-contract, capability-model §2)
# ---------------------------------------------------------------------------

def _check_capability_declaration(
    adapter: Any, report: ConformanceReport
) -> RobotCapability | None:
    try:
        capability = adapter.get_capabilities()
    except Exception as exc:
        report.add(
            "CAP-01", "get_capabilities() returns a capability profile",
            "MUST", f"{_CONTRACT} §2", "fail",
            f"get_capabilities() raised {type(exc).__name__}: {exc}",
        )
        return None

    if not isinstance(capability, RobotCapability):
        report.add(
            "CAP-01", "get_capabilities() returns a capability profile",
            "MUST", f"{_CONTRACT} §2", "fail",
            f"expected RobotCapability, got {type(capability).__name__}",
        )
        return None
    report.add(
        "CAP-01", "get_capabilities() returns a capability profile",
        "MUST", f"{_CONTRACT} §2", "pass",
    )

    robot_id = getattr(capability, "robot_id", "") or ""
    name = getattr(capability, "name", "") or ""
    if robot_id and name:
        report.add(
            "CAP-02", "robot_id and name are non-empty",
            "MUST", f"{_CAPMODEL} §2", "pass",
        )
    else:
        report.add(
            "CAP-02", "robot_id and name are non-empty",
            "MUST", f"{_CAPMODEL} §2", "fail",
            f"robot_id={robot_id!r} name={name!r}",
        )

    max_speed = getattr(capability, "max_speed", None)
    if max_speed is None or (
        isinstance(max_speed, numbers.Real) and max_speed >= 0
    ):
        report.add(
            "CAP-03", "max_speed is absent or a non-negative number",
            "MUST", f"{_CAPMODEL} §2", "pass",
        )
    else:
        report.add(
            "CAP-03", "max_speed is absent or a non-negative number",
            "MUST", f"{_CAPMODEL} §2", "fail",
            f"max_speed={max_speed!r}",
        )

    if not JSONSCHEMA_AVAILABLE:
        report.add(
            "CAP-04", "serialized profile validates against robot-capability schema",
            "MUST", f"{_CAPMODEL} §2", "skip",
            "jsonschema not installed — pip install 'apyrobo[conformance]'",
        )
    else:
        try:
            serialized = _to_json_dict(capability)
            error = validation_error(serialized, "robot-capability")
        except Exception as exc:
            error = f"could not serialize/validate: {type(exc).__name__}: {exc}"
        report.add(
            "CAP-04", "serialized profile validates against robot-capability schema",
            "MUST", f"{_CAPMODEL} §2", "fail" if error else "pass",
            error or "",
        )
    return capability


def _to_json_dict(model: Any) -> dict[str, Any]:
    """Serialize a pydantic model (or the no-pydantic fallback) to JSON types."""
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    import json

    return json.loads(
        json.dumps(vars(model), default=lambda o: getattr(o, "value", vars(o)))
    )


# ---------------------------------------------------------------------------
# Required operations (§2): move, stop
# ---------------------------------------------------------------------------

def _check_required_operations(
    adapter: Any, report: ConformanceReport, capability: RobotCapability | None
) -> None:
    # Minimize physical motion: command a move to the current position.
    try:
        x, y = adapter.get_position()
    except Exception:
        x, y = 0.0, 0.0

    try:
        adapter.move(x, y)
        adapter.stop()
        report.add(
            "OPS-01", "move(x, y) accepted and stop() halts it",
            "MUST", f"{_CONTRACT} §2", "pass",
        )
    except Exception as exc:
        report.add(
            "OPS-01", "move(x, y) accepted and stop() halts it",
            "MUST", f"{_CONTRACT} §2", "fail",
            f"{type(exc).__name__}: {exc}",
        )

    max_speed = getattr(capability, "max_speed", None) if capability else None
    speed = min(0.1, max_speed) if max_speed is not None else 0.1
    try:
        adapter.move(x, y, speed)
        adapter.stop()
        report.add(
            "OPS-02", "move() accepts an explicit speed argument",
            "MUST", f"{_CONTRACT} §2", "pass",
        )
    except Exception as exc:
        report.add(
            "OPS-02", "move() accepts an explicit speed argument",
            "MUST", f"{_CONTRACT} §2", "fail",
            f"{type(exc).__name__}: {exc}",
        )

    try:
        adapter.stop()
        adapter.stop()
        report.add(
            "OPS-03", "stop() is safe to call repeatedly",
            "MUST", f"{_CONTRACT} §2", "pass",
        )
    except Exception as exc:
        report.add(
            "OPS-03", "stop() is safe to call repeatedly",
            "MUST", f"{_CONTRACT} §2", "fail",
            f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Optional operations (§3): specified defaults, never raise
# ---------------------------------------------------------------------------

def _check_optional_operations(adapter: Any, report: ConformanceReport) -> None:
    try:
        adapter.rotate(0.0)
        report.add(
            "OPT-01", "rotate() does not raise",
            "MUST", f"{_CONTRACT} §3", "pass",
        )
    except Exception as exc:
        report.add(
            "OPT-01", "rotate() does not raise",
            "MUST", f"{_CONTRACT} §3", "fail",
            f"{type(exc).__name__}: {exc}",
        )

    for check_id, op in (("OPT-02", "gripper_open"), ("OPT-03", "gripper_close")):
        title = f"{op}() returns a bool and does not raise"
        try:
            result = getattr(adapter, op)()
            status = "pass" if isinstance(result, bool) else "fail"
            details = "" if status == "pass" else f"returned {result!r}"
        except Exception as exc:
            status, details = "fail", f"{type(exc).__name__}: {exc}"
        report.add(check_id, title, "MUST", f"{_CONTRACT} §3", status, details)

    try:
        adapter.cancel()
        report.add(
            "OPT-04", "cancel() does not raise",
            "MUST", f"{_CONTRACT} §3", "pass",
        )
    except Exception as exc:
        report.add(
            "OPT-04", "cancel() does not raise",
            "MUST", f"{_CONTRACT} §3", "fail",
            f"{type(exc).__name__}: {exc}",
        )

    try:
        position = adapter.get_position()
        ok = (
            isinstance(position, (tuple, list))
            and len(position) == 2
            and all(isinstance(v, numbers.Real) for v in position)
        )
        report.add(
            "OPT-05", "get_position() returns an (x, y) pair of numbers",
            "MUST", f"{_CONTRACT} §3", "pass" if ok else "fail",
            "" if ok else f"returned {position!r}",
        )
    except Exception as exc:
        report.add(
            "OPT-05", "get_position() returns an (x, y) pair of numbers",
            "MUST", f"{_CONTRACT} §3", "fail",
            f"{type(exc).__name__}: {exc}",
        )

    try:
        heading = adapter.get_orientation()
        ok = isinstance(heading, numbers.Real)
        report.add(
            "OPT-06", "get_orientation() returns a number (radians)",
            "MUST", f"{_CONTRACT} §3", "pass" if ok else "fail",
            "" if ok else f"returned {heading!r}",
        )
    except Exception as exc:
        report.add(
            "OPT-06", "get_orientation() returns a number (radians)",
            "MUST", f"{_CONTRACT} §3", "fail",
            f"{type(exc).__name__}: {exc}",
        )

    _check_health(adapter, report, "OPT-07", f"{_CONTRACT} §3")


def _check_health(
    adapter: Any, report: ConformanceReport, check_id: str, spec_ref: str
) -> bool:
    title = "get_health() returns a mapping with state/adapter/robot"
    try:
        health = adapter.get_health()
    except Exception as exc:
        report.add(check_id, title, "MUST", spec_ref, "fail",
                   f"{type(exc).__name__}: {exc}")
        return False
    if not isinstance(health, dict):
        report.add(check_id, title, "MUST", spec_ref, "fail",
                   f"returned {type(health).__name__}")
        return False
    missing = [k for k in ("state", "adapter", "robot") if k not in health]
    if missing:
        report.add(check_id, title, "MUST", spec_ref, "fail",
                   f"missing keys: {missing}")
        return False
    report.add(check_id, title, "MUST", spec_ref, "pass")
    return True


# ---------------------------------------------------------------------------
# Lifecycle and state (§4)
# ---------------------------------------------------------------------------

def _check_lifecycle(adapter: Any, report: ConformanceReport) -> None:
    state = getattr(adapter, "state", None)
    if isinstance(state, AdapterState):
        report.add(
            "LIF-01", "adapter exposes a valid AdapterState",
            "MUST", f"{_CONTRACT} §4", "pass",
        )
    else:
        report.add(
            "LIF-01", "adapter exposes a valid AdapterState",
            "MUST", f"{_CONTRACT} §4", "fail",
            f"state={state!r}",
        )

    disconnect_seen: list[bool] = []
    try:
        adapter.on_disconnect(lambda: disconnect_seen.append(True))
        # Make sure the connection is established before dropping it, so the
        # disconnect callback contract applies.
        if not adapter.is_connected:
            adapter.connect()
        adapter.disconnect()
        was_disconnected = not adapter.is_connected
        callback_fired = bool(disconnect_seen)
        adapter.connect()
        reconnected = adapter.is_connected

        if was_disconnected and reconnected:
            report.add(
                "LIF-02", "connect()/disconnect() transition is_connected",
                "MUST", f"{_CONTRACT} §4", "pass",
            )
        else:
            report.add(
                "LIF-02", "connect()/disconnect() transition is_connected",
                "MUST", f"{_CONTRACT} §4", "fail",
                f"after disconnect is_connected={not was_disconnected}, "
                f"after connect is_connected={reconnected}",
            )
        report.add(
            "LIF-03", "disconnect callbacks fire when the connection is lost",
            "MUST", f"{_CONTRACT} §4", "pass" if callback_fired else "fail",
            "" if callback_fired else
            "on_disconnect handler was not invoked by disconnect()",
        )
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        report.add("LIF-02", "connect()/disconnect() transition is_connected",
                   "MUST", f"{_CONTRACT} §4", "fail", detail)
        report.add("LIF-03", "disconnect callbacks fire when the connection is lost",
                   "MUST", f"{_CONTRACT} §4", "fail", detail)


# ---------------------------------------------------------------------------
# Failure semantics (§5) + safety-stop in every state (§2)
# ---------------------------------------------------------------------------

def _check_failure_semantics(adapter: Any, report: ConformanceReport) -> None:
    try:
        x, y = adapter.get_position()
    except Exception:
        x, y = 0.0, 0.0

    try:
        adapter.disconnect()
    except Exception as exc:
        report.add(
            "SAF-01", "stop() works while disconnected",
            "MUST", f"{_CONTRACT} §2", "fail",
            f"disconnect() raised {type(exc).__name__}: {exc}",
        )
        return

    try:
        adapter.stop()
        report.add(
            "SAF-01", "stop() works while disconnected",
            "MUST", f"{_CONTRACT} §2", "pass",
        )
    except Exception as exc:
        report.add(
            "SAF-01", "stop() works while disconnected",
            "MUST", f"{_CONTRACT} §2", "fail",
            f"{type(exc).__name__}: {exc}",
        )

    # SHOULD: commands while not connected fail fast rather than queue.
    try:
        adapter.move(x, y)
        report.add(
            "FAIL-01", "commands while disconnected fail fast",
            "SHOULD", f"{_CONTRACT} §4", "warn",
            "move() while disconnected returned without error — if the "
            "command was queued for replay on reconnect, that is a safety "
            "hazard (a fail-fast exception is the recommended behavior)",
        )
        # Ensure any accepted motion command is halted before we reconnect.
        with contextlib.suppress(Exception):
            adapter.stop()
    except Exception:
        report.add(
            "FAIL-01", "commands while disconnected fail fast",
            "SHOULD", f"{_CONTRACT} §4", "pass",
        )

    # MUST: after any command failure the adapter stays queryable.
    _check_health(adapter, report, "FAIL-02", f"{_CONTRACT} §5")
    try:
        adapter.stop()
        report.add(
            "FAIL-03", "stop() still works after a command failure",
            "MUST", f"{_CONTRACT} §5", "pass",
        )
    except Exception as exc:
        report.add(
            "FAIL-03", "stop() still works after a command failure",
            "MUST", f"{_CONTRACT} §5", "fail",
            f"{type(exc).__name__}: {exc}",
        )

    try:
        adapter.connect()
    except Exception:
        logger.warning("conformance: could not reconnect adapter after checks")
