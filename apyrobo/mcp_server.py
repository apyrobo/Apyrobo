"""MCP server: let an MCP client (Claude Desktop, Claude Code, Cursor, …)
drive a robot through APYROBO's safety enforcer.

Run it with ``apyrobo mcp``. The client talks to the server over stdio using
the Model Context Protocol (JSON-RPC 2.0, one message per line). Every motion
command goes through :class:`~apyrobo.safety.enforcer.SafetyEnforcer`, so the
model can ask for anything but the robot only does what the policy allows:
speeds above the limit are clamped, targets inside a no-go zone are refused,
and the refusal comes back as a tool error the model can read and act on.

The server needs no extra dependencies. It defaults to ``mock://turtlebot4``,
so it can be tried on a laptop with no ROS install.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import hmac
import importlib.util
import json
import logging
import math
import os
import queue
import re
import secrets
import select
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from apyrobo.__version__ import __version__

logger = logging.getLogger(__name__)

# Legacy (``initialize`` handshake) versions, newest first. The server answers
# ``initialize`` with the client's requested version when it is one of these,
# otherwise with the newest.
SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")

# Modern (stateless) versions: no handshake, every request carries its version
# and client capabilities in ``params._meta``. The server is dual-era: a
# request with that metadata is served statelessly, anything else as before.
MODERN_PROTOCOL_VERSIONS = ("2026-07-28",)

META_PROTOCOL_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"

# How long a modern client may cache tools/list and server/discover. The
# tool set never changes while the server runs.
LIST_TTL_MS = 3_600_000

# JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
# MCP error codes (2026-07-28)
UNSUPPORTED_PROTOCOL_VERSION = -32022

# --confirm-moves: tools that make the robot move ask the user first.
MOTION_TOOLS = ("move_to", "rotate", "execute_task")
CONFIRM_INPUT_KEY = "confirm_move"
# How long a confirmation request stays valid (seconds).
CONFIRM_TTL_S = 600
# End of the client's input stream (a JSON ``null`` line is a message).
_EOF = object()
# Tools that can make the robot move; refused once the client is gone.
MOTION_TOOLS = frozenset({"move_to", "rotate", "execute_task"})
# Task wording the rule planner's coordinate parser does not read.
_NUMBER = r"-?\d+(?:\.\d+)?"
_SPEED_IN_TASK = re.compile(
    rf"\b(?:at\s+)?(?:a\s+)?speed\s*(?:of\s+|[:=]\s*)?{_NUMBER}(?:\s*m/s)?"
    rf"|\bat\s+{_NUMBER}\s*m/s",
    re.IGNORECASE,
)
_XY_IN_TASK = re.compile(
    rf"\bx\s*=\s*({_NUMBER})\s*,?\s*y\s*=\s*({_NUMBER})", re.IGNORECASE
)
_BARE_PAIR_IN_TASK = re.compile(
    rf"\b(to|at)\s+({_NUMBER})\s+({_NUMBER})(?![\w.])", re.IGNORECASE
)
_NO_TARGET = (
    "No target: this task gives no coordinates for where to drive, and the robot "
    "will not guess one. Give them, like 'go to (3, 3)' or 'deliver from (1, 1) "
    "to (5, 1)'."
)
_NOT_UNDERSTOOD = (
    "Not planned: the rule planner did not recognise this task, so the robot "
    "will not do something else and call it done. It understands: 'go to (x, y)', "
    "'pick up the box at (x, y)', 'deliver from (x1, y1) to (x2, y2)', "
    "'rotate 90 degrees', 'stop' and 'report status'. For other tasks, use "
    "move_to and rotate directly, or start the server with --provider llm."
)
_MANY_TARGETS = (
    "Not planned: this task names more than one place to drive, and the rule "
    "planner would drive to only some of them and drop the rest, so the robot will "
    "not move. Send one target per call, like 'go to (1, 1)', then 'go to (3, 3)'. "
    "A delivery may name two: 'deliver from (1, 1) to (5, 1)'."
)
_MANY_MOTIONS = (
    "Not planned: this task asks for a turn together with a drive, more than one "
    "turn, or a turn repeated, and the rule planner would make only one of them "
    "and drop the rest, so the robot will not move. Send one motion per call, like "
    "'rotate 90 degrees', then 'go to (1, 1)', or call rotate and move_to directly."
)
_TWO_DIRECTIONS = (
    "Not planned: this turn names both directions (left and right, or clockwise "
    "and anticlockwise), and the rule planner would turn one way only, so the "
    "robot will not move. Send one turn per call with one direction, like 'turn "
    "right 90 degrees', or call rotate with angle_rad (positive turns left)."
)
_REPEAT_WORD = re.compile(r"\b(?:twice|thrice|again|times)\b")
_NO_ANGLE = (
    "Not planned: this task asks to turn but gives no angle with a unit, and the "
    "robot will not guess one. Give it, like 'rotate 90 degrees' or 'turn right "
    "45 degrees', or call rotate with angle_rad."
)
_SPEED_NO_DRIVE = (
    "Not planned: this task names a speed, but the plan has no drive to apply it "
    "to, and the robot will not drop it with no word. A speed in a task sets the "
    "drive speed, like 'go to (1, 2) at 0.2 m/s'; to turn at a set speed, call "
    "rotate with speed."
)
# How often the stop is repeated while a motion call is still unwinding.
HALT_REPEAT_S = 0.2
# --audit-log: longest "detail" kept on a line.
AUDIT_DETAIL_CHARS = 500
# How long a motion call may wait for the stdin thread to reach pending input.
STDIN_SETTLE_S = 0.25
STDIN_SETTLE_POLL_S = 0.001
# How long exit waits for the stop after a client loss to be logged.
HALT_JOIN_S = 1.0
# How long exit waits for a status read that a move left running.
STATUS_JOIN_S = 1.0
# How often a motion call whose request carries a progressToken reports
# progress. Clients reset their idle timeout on each notification.
PROGRESS_INTERVAL_S = 5.0

INSTRUCTIONS = (
    "These tools control a robot through APYROBO's safety enforcer. "
    "Call robot_status first to see the robot, its position and the active "
    "safety policy. Motion requests may be changed or refused by the policy: "
    "read each result, and when a command is refused, pick a different "
    "target rather than retrying the same one. stop is always allowed."
)


class ToolError(Exception):
    """A tool ran but could not do what was asked (reported as isError).

    *structured*, when given, goes out as the result's ``structuredContent``
    beside the text, for clients that negotiated 2025-06-18 or later.
    """

    def __init__(self, message: str, structured: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.structured = structured


# structuredContent arrived in MCP 2025-06-18; older clients get text only.
STRUCTURED_CONTENT_SINCE = "2025-06-18"


def _refusal_content(policy: str, violation: dict[str, Any] | None) -> dict[str, Any]:
    """The machine-readable side of a safety refusal.

    ``code`` is the enforcer's violation type (``collision_zone``,
    ``outside_allowed_area``, ``battery_below_minimum``, ...), or
    ``safety_violation`` when the enforcer recorded none; ``violation`` holds
    the rest of what the enforcer recorded (the zone, the allowed areas, the
    limit that tripped).
    """
    record = dict(violation or {})
    code = record.pop("type", None)
    record.pop("message", None)
    return {
        "status": "refused",
        "code": code if isinstance(code, str) and code else "safety_violation",
        "policy": policy,
        "violation": _jsonable(record),
    }


class AuditLog:
    """``--audit-log``: one JSON line per motion decision, on disk at once.

    Each line is synced before :meth:`record` returns, so what was written
    survives a crash, and the file is opened for each line, so it can be
    rotated while the server runs. :meth:`record` raises ``OSError`` when the
    line could not be written.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = path
        with open(path, "ab"):  # Fail at start, not at the first move.
            pass

    def record(self, entry: dict[str, Any]) -> None:
        data = (json.dumps(_finite(entry), separators=(",", ":")) + "\n").encode()
        with open(self.path, "ab", buffering=0) as log:
            if log.write(data) != len(data):
                raise OSError(errno.EIO, "short write")
            try:
                os.fsync(log.fileno())
            except OSError as exc:
                # Pipes and character devices have nothing to sync.
                if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
                    raise


def _finite(value: Any) -> Any:
    """*value* with NaN and infinities as strings: the log stays strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def _num(args: dict[str, Any], key: str, *, required: bool = True) -> float | None:
    value = args.get(key)
    if value is None:
        if required:
            raise ToolError(f"Missing required argument '{key}'.")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"Argument '{key}' must be a number, got {value!r}.")
    try:
        number = float(value)
    except OverflowError:
        number = math.inf
    if not math.isfinite(number):
        # JSON parsers accept NaN/Infinity literals and overflow 1e400 to inf;
        # none of them is a position or a speed. Refuse before the robot moves.
        raise ToolError(f"Argument '{key}' must be a finite number, got {value!r}.")
    return number


def _move_speed(args: dict[str, Any]) -> float | None:
    """The optional linear ``speed`` of a move: finite and above 0."""
    speed = _num(args, "speed", required=False)
    if speed is not None and speed <= 0:
        # 0 used to run the move at the default speed and report it moved.
        raise ToolError(
            f"Argument 'speed' must be more than 0 m/s, got {speed!r}. "
            "Leave it out for the default speed; call stop to stop."
        )
    return speed


def _turn_speed(args: dict[str, Any]) -> float | None:
    """The optional angular ``speed`` of a rotate: finite and not 0."""
    speed = _num(args, "speed", required=False)
    if speed == 0:
        raise ToolError(
            f"Argument 'speed' must be a turn rate other than 0 rad/s, got {speed!r}. "
            "Leave it out for the default speed; call stop to stop."
        )
    return speed


def _str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"Argument '{key}' must be a non-empty string.")
    return value.strip()


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of enums, tuples and models to plain JSON."""
    return json.loads(json.dumps(value, default=lambda o: getattr(o, "value", str(o))))


def _rule_coordinates_text(task: str) -> str:
    """Spell a task's coordinates the way the rule planner parses them.

    "move to 3 3 at speed 2" becomes "move to (3, 3)": a bare pair after
    to/at and x=3 y=3 turn into (3, 3), and a speed is dropped so that it is
    not read as a coordinate.
    """
    text = _SPEED_IN_TASK.sub(" ", task)
    text = _XY_IN_TASK.sub(r"(\1, \2)", text)
    return _BARE_PAIR_IN_TASK.sub(r"\1 (\2, \3)", text)


#: Said in a reply when a drive went through Nav2, which takes no speed.
SPEED_NOT_APPLIED = (
    "The speed limit was not applied to this drive: on ros2:// a Nav2 goal carries "
    "no speed, so the robot drove at whatever speed Nav2's controller is configured "
    "for. Set the limit in Nav2's controller parameters."
)

#: Said in a reply when a Nav2 drive got the limit but the controller's own
#: top speed could not be read to check it against.
SPEED_NOT_CHECKED = (
    "The speed limit was sent to Nav2 but not checked against the controller's own "
    "top speed, which could not be read (unknown controller, several controllers, or "
    "no answer from controller_server). On a Regulated Pure Pursuit or MPPI "
    "controller a limit above that speed raises the robot's speed: keep --max-speed "
    "at or below it."
)

#: Said in a reply when another node also publishes Nav2 speed limits.
SPEED_SHARED = (
    "Another node also publishes on Nav2's speed_limit topic (a costmap speed "
    "filter, for one). Its next message replaces the limit sent for this drive: "
    "entering a speed zone sets that zone's limit, even above the policy's, and "
    "leaving one lifts the limit for the rest of the goal. Keep zone limits and "
    "the controller's own top speed at or below what you would allow."
)


class RobotToolbox:
    """The tools the MCP server exposes, bound to one robot and one policy."""

    def __init__(
        self,
        robot: Any,
        enforcer: Any,
        agent: Any,
        robot_uri: str,
    ) -> None:
        self.robot = robot
        self.enforcer = enforcer
        self.agent = agent
        self.robot_uri = robot_uri
        # Set while a motion call is being stopped: why execute_task ends early.
        self.stop_reason: str | None = None
        # --confirm-moves on execute_task: the plan the last question showed,
        # and the plan the user said yes to, which execute_task then runs
        # instead of planning again.
        self.last_plan: tuple[str, Any, dict[str, Any]] | None = None
        self.approved_plan: tuple[str, Any, dict[str, Any]] | None = None
        self.last_refusal: str | None = None
        self._tools: dict[str, tuple[dict[str, Any], Callable[[dict[str, Any]], Any]]] = {}
        self._register()

    # -- registry ------------------------------------------------------

    def _add(
        self,
        name: str,
        description: str,
        properties: dict[str, Any],
        required: list[str],
        handler: Callable[[dict[str, Any]], Any],
        *,
        read_only: bool,
    ) -> None:
        spec = {
            "name": name,
            "description": description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            "annotations": {"readOnlyHint": read_only},
        }
        self._tools[name] = (spec, handler)

    def _register(self) -> None:
        speed = {
            "type": "number",
            "description": "Requested speed. Values above the policy limit are clamped.",
        }
        self._add(
            "robot_status",
            "Robot id, position (m), heading (rad), health, and the active safety "
            "policy (speed limit, no-go zones).",
            {},
            [],
            self._robot_status,
            read_only=True,
        )
        self._add(
            "list_capabilities",
            "What this robot can do (navigate, rotate, pick, place, …) and its sensors.",
            {},
            [],
            self._list_capabilities,
            read_only=True,
        )
        self._add(
            "move_to",
            "Drive to (x, y) in metres. Checked by the safety policy first: speed "
            "may be clamped, and targets inside a no-go zone or outside the "
            "allowed areas are refused.",
            {
                "x": {"type": "number", "description": "Target x in metres."},
                "y": {"type": "number", "description": "Target y in metres."},
                "speed": {**speed, "description": speed["description"] + " (m/s)"},
            },
            ["x", "y"],
            self._move_to,
            read_only=False,
        )
        self._add(
            "rotate",
            "Rotate in place by angle_rad radians (positive = counter-clockwise).",
            {
                "angle_rad": {"type": "number", "description": "Angle in radians."},
                "speed": {**speed, "description": speed["description"] + " (rad/s)"},
            },
            ["angle_rad"],
            self._rotate,
            read_only=False,
        )
        self._add(
            "stop",
            "Stop all motion now. Always allowed.",
            {},
            [],
            self._stop,
            read_only=False,
        )
        self._add(
            "plan_task",
            "Turn a natural-language task into a skill plan without moving the "
            "robot. Returns the ordered skills, their parameters and a confidence "
            "estimate.",
            {"task": {"type": "string", "description": "e.g. 'deliver the package to (5, 1)'"}},
            ["task"],
            self._plan_task,
            read_only=True,
        )
        self._add(
            "execute_task",
            "Plan a natural-language task and run it. Every step goes through the "
            "safety policy; the run stops at the first refused step and says why.",
            {"task": {"type": "string", "description": "e.g. 'deliver the package to (5, 1)'"}},
            ["task"],
            self._execute_task,
            read_only=False,
        )
        self._add(
            "safety_report",
            "Every safety intervention (e.g. clamped speed) and violation (e.g. "
            "refused target) since the server started.",
            {},
            [],
            self._safety_report,
            read_only=True,
        )

    def list_tools(self) -> list[dict[str, Any]]:
        return [spec for spec, _ in self._tools.values()]

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def check_arguments(self, name: str, arguments: dict[str, Any]) -> None:
        """Refuse argument names a motion tool's schema does not list.

        A caller that writes ``velocity`` or ``max_speed`` for ``speed`` would
        otherwise get the default speed and a success reply. Only motion tools
        are checked, so a host that adds its own arguments to read-only calls
        keeps working.
        """
        if name not in MOTION_TOOLS:
            return
        spec, _ = self._tools[name]
        known = spec["inputSchema"]["properties"]
        unknown = sorted(str(key) for key in arguments if key not in known)
        if unknown:
            names = ", ".join(f"'{key}'" for key in unknown)
            allowed = ", ".join(f"'{key}'" for key in known)
            plural = "s" if len(unknown) > 1 else ""
            raise ToolError(
                f"Unknown argument{plural} {names} for {name}; it takes {allowed}. "
                "The robot did not move."
            )

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        _, handler = self._tools[name]
        self.check_arguments(name, arguments)
        return handler(arguments)

    def confirmation_prompt(self, name: str, arguments: dict[str, Any]) -> str | None:
        """What the user is asked to confirm under --confirm-moves.

        Shows the target and the speed the policy will actually allow; for
        execute_task, each step of the plan that will run. Returns None for
        a move or a plan the policy refuses anyway (a target in a no-go zone
        or outside the allowed areas, or a route that crosses a zone or
        leaves the areas) and for a task that cannot be planned, so the user
        is not asked to approve something that will be refused.
        For execute_task the plan shown is kept in ``last_plan``: the server
        hands it to ``approved_plan`` on a yes, and that plan is the one run.
        Raises ToolError for invalid arguments.
        """
        self.check_arguments(name, arguments)
        p = self.enforcer.policy
        if name == "move_to":
            x, y = _num(arguments, "x"), _num(arguments, "y")
            speed = _move_speed(arguments)
            from apyrobo.safety.enforcer import SafetyEnforcer, SafetyViolation

            if any(SafetyEnforcer._point_in_zone(x, y, z) for z in p.collision_zones):
                return None
            if self.enforcer.outside_allowed_areas(x, y):
                return None
            try:
                if self.enforcer.route_fault(x, y) is not None:
                    return None
            except SafetyViolation:
                return None
            return f"Allow the robot to move to x={x}, y={y} at {self._drive_speed_text(speed)}?"
        if name == "rotate":
            angle = _num(arguments, "angle_rad")
            speed = _turn_speed(arguments)
            return f"Allow the robot to rotate by {angle} rad at {self._turn_speed_text(speed)}?"
        task = _str(arguments, "task")
        self.last_plan = self.last_refusal = None
        graph, plan = self._plan(task)
        if plan.get("reason") or not plan["steps"]:
            return None  # execute_task refuses it without moving
        self.last_refusal = self._plan_refusal(plan["steps"], "The robot did not move.")
        if self.last_refusal is not None:
            return None
        self.last_plan = (task, graph, plan)
        lines = [f"Allow the robot to carry out the task {task!r}? It will run this plan:"]
        for number, step in enumerate(plan["steps"], start=1):
            lines.append(f"{number}. {self._step_text(step)}")
        return "\n".join(lines)

    def _drive_speed_text(self, speed: float | None) -> str:
        limit = self.enforcer.policy.max_speed
        if not self.speed_limit_applies():
            return (
                "the speed Nav2's controller is configured for (on this robot the "
                f"{limit} m/s policy limit does not reach the drive)"
            )
        if speed is None:
            return f"default speed (policy limit {limit} m/s)"
        if speed > limit:
            return f"{limit} m/s (requested {speed}, clamped to the policy limit)"
        return f"{speed} m/s"

    def _turn_speed_text(self, speed: float | None) -> str:
        limit = self.enforcer.policy.max_angular_speed
        if speed is None:
            return f"default speed (policy limit {limit} rad/s)"
        if abs(speed) > limit:
            return f"{limit} rad/s (requested {speed}, clamped to the policy limit)"
        return f"{speed} rad/s"

    def _step_text(self, step: dict[str, Any]) -> str:
        """One plan step as the user reads it in a confirmation question."""
        params = step.get("parameters") or {}
        if step["skill_id"].startswith("navigate_to"):
            x = float(params.get("x", 0.0))
            y = float(params.get("y", 0.0))
            return f"Drive to x={x}, y={y} at {self._drive_speed_text(params.get('speed'))}"
        if step["skill_id"].startswith("rotate"):
            return (
                f"Turn by {params.get('angle_rad')} rad at "
                f"{self._turn_speed_text(params.get('speed'))}"
            )
        shown = ", ".join(f"{k}={v}" for k, v in params.items() if v is not None)
        return f"{step['name']} ({shown})" if shown else str(step["name"])

    # -- helpers -------------------------------------------------------

    def _policy_summary(self) -> dict[str, Any]:
        p = self.enforcer.policy
        summary: dict[str, Any] = {
            "name": p.name,
            "max_speed_m_s": p.max_speed,
            "max_angular_speed_rad_s": p.max_angular_speed,
            "no_go_zones": [dict(z) for z in p.collision_zones],
        }
        if p.allowed_areas:
            summary["allowed_areas"] = [dict(a) for a in p.allowed_areas]
        return summary

    def speed_limit_applies(self) -> bool:
        """False when the robot ignores the speed it is sent (ros2:// through Nav2)."""
        adapter = getattr(self.robot, "_adapter", None)
        return getattr(adapter, "speed_limit_applies", True) is not False

    def speed_limit_checked(self) -> bool:
        """False when a Nav2 limit could not be checked against the controller's top speed."""
        adapter = getattr(self.robot, "_adapter", None)
        return getattr(adapter, "speed_limit_checked", True) is not False

    def speed_limit_shared(self) -> bool:
        """True when another node also publishes Nav2 speed limits (ros2://)."""
        adapter = getattr(self.robot, "_adapter", None)
        return getattr(adapter, "speed_limit_shared", False) is True

    def _drove_shared(self, moves_before: int) -> bool:
        """True when a move ran since *moves_before* with its limit open to replacement."""
        moves = getattr(self.enforcer, "moves_sent", 0)
        return (
            isinstance(moves, int) and moves > moves_before
            and self.speed_limit_applies() and self.speed_limit_shared()
        )

    def speed_notes(self, moves_before: int) -> dict[str, Any]:
        """The speed_* reply fields for the moves run since *moves_before*."""
        if self._drove_unlimited(moves_before):
            return {"speed_limit_applied": False, "speed_note": SPEED_NOT_APPLIED}
        fields: dict[str, Any] = {}
        notes = []
        if self._drove_unchecked(moves_before):
            fields["speed_limit_checked"] = False
            notes.append(SPEED_NOT_CHECKED)
        if self._drove_shared(moves_before):
            fields["speed_limit_shared"] = True
            notes.append(SPEED_SHARED)
        if notes:
            fields["speed_note"] = " ".join(notes)
        return fields

    def _drove_unchecked(self, moves_before: int) -> bool:
        """True when a move ran since *moves_before* with a limit Nav2 got unchecked."""
        moves = getattr(self.enforcer, "moves_sent", 0)
        return (
            isinstance(moves, int) and moves > moves_before
            and self.speed_limit_applies() and not self.speed_limit_checked()
        )

    def _drove_unlimited(self, moves_before: int) -> bool:
        """True when a move ran since *moves_before* and its speed limit did not apply."""
        moves = getattr(self.enforcer, "moves_sent", 0)
        return isinstance(moves, int) and moves > moves_before and not self.speed_limit_applies()

    def _position(self) -> dict[str, float]:
        x, y = self.enforcer.get_position()
        return {"x": float(x), "y": float(y)}

    def _guarded(self, action: Callable[[], Any]) -> list[dict[str, Any]]:
        """Run *action*; turn a safety refusal into a readable ToolError.

        Returns the interventions (e.g. clamped speed) the action caused.
        """
        from apyrobo.safety.enforcer import SafetyViolation

        before = len(self.enforcer.interventions)
        violations_before = len(self.enforcer.violations)
        try:
            action()
        except SafetyViolation as exc:
            recorded = self.enforcer.violations[violations_before:]
            policy = self.enforcer.policy.name
            raise ToolError(
                f"REFUSED by safety policy '{policy}': {exc}. "
                "The robot did not move. Choose a different target.",
                structured=_refusal_content(policy, recorded[-1] if recorded else None),
            ) from exc
        return self.enforcer.interventions[before:]

    # -- tools ---------------------------------------------------------

    def _robot_status(self, args: dict[str, Any]) -> dict[str, Any]:
        status = {
            "robot_uri": self.robot_uri,
            "robot_id": getattr(self.robot, "robot_id", None),
            "position": self._position(),
            "heading_rad": float(self.enforcer.get_orientation()),
            "health": self.enforcer.get_health(),
            "safety_policy": self._policy_summary(),
        }
        # ros2:// falls back to /odom when there is no map-to-base transform;
        # a caller reading zones in map needs to know which one this is.
        frame = getattr(getattr(self.robot, "_adapter", None), "position_frame", None)
        if isinstance(frame, str):
            status["position_frame"] = frame
        return status

    def _list_capabilities(self, args: dict[str, Any]) -> dict[str, Any]:
        caps = self.robot.capabilities()
        return {
            "robot_id": caps.robot_id,
            "name": caps.name,
            "capabilities": [
                {
                    "name": c.name,
                    "type": getattr(c.capability_type, "value", str(c.capability_type)),
                    "parameters": c.parameters,
                    "description": c.description,
                }
                for c in caps.capabilities
            ],
            "sensors": [
                {"id": s.sensor_id, "type": getattr(s.sensor_type, "value", str(s.sensor_type))}
                for s in caps.sensors
            ],
        }

    def _move_to(self, args: dict[str, Any]) -> dict[str, Any]:
        x = _num(args, "x")
        y = _num(args, "y")
        speed = _move_speed(args)
        moves_before = getattr(self.enforcer, "moves_sent", 0)
        interventions = self._guarded(lambda: self.enforcer.move(x=x, y=y, speed=speed))
        result: dict[str, Any] = {
            "status": "moved",
            "requested": {"x": x, "y": y, "speed": speed},
            "position": self._position(),
            "safety_interventions": interventions,
        }
        result.update(self.speed_notes(moves_before))
        return result

    def rotate_available(self) -> bool:
        """False when the adapter keeps the base no-op rotate (ros2://, for one)."""
        from apyrobo.core.adapters import CapabilityAdapter

        adapter = getattr(self.robot, "_adapter", None)
        if not isinstance(adapter, CapabilityAdapter):
            return True
        return getattr(type(adapter), "rotate", None) is not CapabilityAdapter.rotate

    def _rotate(self, args: dict[str, Any]) -> dict[str, Any]:
        angle = _num(args, "angle_rad")
        speed = _turn_speed(args)
        if not self.rotate_available():
            adapter = type(getattr(self.robot, "_adapter", None)).__name__
            raise ToolError(
                f"rotate is not available on this robot: {adapter} cannot turn in "
                "place, so nothing was sent and the robot did not move. Use move_to; "
                "the robot turns as it drives."
            )
        interventions = self._guarded(lambda: self.enforcer.rotate(angle_rad=angle, speed=speed))
        return {
            "status": "rotated",
            "heading_rad": float(self.enforcer.get_orientation()),
            "safety_interventions": interventions,
        }

    def _stop(self, args: dict[str, Any]) -> dict[str, Any]:
        self.enforcer.stop()
        return {"status": "stopped", "position": self._position()}

    def _rule_planner_text(self, task: str) -> str:
        """Rewrite a task so the rule planner reads the coordinates it holds."""
        from apyrobo.skills.agent import RuleBasedProvider

        text = _rule_coordinates_text(task)
        # "deliver to (5, 1)" names one place: the pickup is where the robot is.
        delivery_words = RuleBasedProvider.TASK_PATTERNS[0][0]
        if len(RuleBasedProvider._extract_coordinates(text)) == 1 and any(
            word in text.lower() for word in delivery_words
        ):
            here = self._position()
            text = f"from ({here['x']:.6f}, {here['y']:.6f}) {text}"
        # The coordinates text drops a speed so it is not read as a target;
        # hand each one back to the planner in the form it reads as a speed.
        for found in RuleBasedProvider._SPEED.finditer(task.lower()):
            number = found.group(1) if found.group(1) is not None else found.group(2)
            text = f"{text} at speed {number}"
        return text

    def _plan(self, task: str) -> tuple[Any, dict[str, Any]]:
        from apyrobo.safety.confidence import ConfidenceEstimator
        from apyrobo.skills.agent import RuleBasedProvider

        text = task
        if isinstance(getattr(self.agent, "provider", None), RuleBasedProvider):
            text = self._rule_planner_text(task)
        graph = self.agent.plan(text, self.robot)
        steps = [
            {
                "skill_id": s.skill_id,
                "name": s.name,
                "parameters": graph.get_parameters(s.skill_id),
            }
            for s in graph.get_execution_order()
        ]
        report = ConfidenceEstimator().assess(graph, self.robot)
        plan = {
            "task": task,
            "steps": steps,
            "confidence": round(float(report.confidence), 3),
            "risk_level": getattr(report.risk_level, "value", str(report.risk_level)),
            "can_proceed": bool(report.can_proceed),
        }
        if isinstance(getattr(self.agent, "planned_by", None), RuleBasedProvider):
            # The task as sent first: rewriting it drops a speed, and with it
            # the sign that the speed was not a number it could read.
            unreadable = RuleBasedProvider.cannot_read(task) or RuleBasedProvider.cannot_read(text)
            if unreadable is not None:
                plan.update(steps=[], confidence=0.0, can_proceed=False, reason=unreadable)
                return graph, plan
            # It drives to one place per step and keeps one coordinate for each:
            # every place the task names must be a step's target, or the rest
            # would be dropped with no word.
            named = RuleBasedProvider._extract_coordinates(_rule_coordinates_text(text))
            distinct = {(c["x"], c["y"]) for c in named}
            if len(distinct) > 1:
                driven = {
                    (
                        float(s["parameters"].get("x", 0.0)),
                        float(s["parameters"].get("y", 0.0)),
                    )
                    for s in steps
                    if s["skill_id"].startswith("navigate_to")
                }
                if not distinct <= driven:
                    plan.update(
                        steps=[], confidence=0.0, can_proceed=False, reason=_MANY_TARGETS
                    )
                    return graph, plan
            # It plans one turn, or drives, never both: a second turn, a repeat
            # or a turn next to a drive would be dropped with no word.
            lowered = text.lower()
            if RuleBasedProvider._TURN_WORD.search(lowered):
                ids = [step["skill_id"].rsplit("_", 1)[0] for step in steps]
                if (
                    "navigate_to" in ids
                    or len(RuleBasedProvider._ANGLE.findall(lowered)) > 1
                    or len(RuleBasedProvider._TURN_WORD.findall(lowered)) > 1
                    or _REPEAT_WORD.search(lowered)
                ):
                    plan.update(
                        steps=[], confidence=0.0, can_proceed=False, reason=_MANY_MOTIONS
                    )
                    return graph, plan
                # One turn, one way: with both directions named it would
                # pick one and drop the other with no word.
                if len(RuleBasedProvider.turn_directions(lowered)) > 1:
                    plan.update(
                        steps=[], confidence=0.0, can_proceed=False, reason=_TWO_DIRECTIONS
                    )
                    return graph, plan
            # Its last resort is a status report: for a task that did not ask
            # for one, that would end in "completed" with nothing done.
            ids = [step["skill_id"].rsplit("_", 1)[0] for step in steps]
            if ids == ["report_status"] and not RuleBasedProvider.is_status_ask(text):
                turn = RuleBasedProvider._TURN_WORD.search(text.lower())
                reason = _NO_ANGLE if turn else _NOT_UNDERSTOOD
                plan.update(steps=[], confidence=0.0, can_proceed=False, reason=reason)
                return graph, plan
            # A turn must be the one the task stated, never the 0 rad default.
            if "rotate" in ids and RuleBasedProvider.extract_angle(text) is None:
                plan.update(steps=[], confidence=0.0, can_proceed=False, reason=_NO_ANGLE)
                return graph, plan
            if "rotate" in ids and not self.rotate_available():
                plan.update(
                    steps=[], confidence=0.0, can_proceed=False,
                    reason=(
                        f"Not planned: {self.robot_uri} cannot turn in place, so the "
                        "robot will not report a turn it did not make."
                    ),
                )
                return graph, plan
            # A speed applies to drives only: a turn or a status report would
            # drop it with no word.
            if RuleBasedProvider.extract_speed(text) is not None and not any(
                step["skill_id"].startswith("navigate_to") for step in steps
            ):
                plan.update(steps=[], confidence=0.0, can_proceed=False, reason=_SPEED_NO_DRIVE)
                return graph, plan
            # A plan from the rule planner: its templates hold placeholder
            # targets, so every drive must go to a point the task named.
            for step in steps:
                if step["skill_id"].startswith("navigate_to"):
                    target = {
                        "x": float(step["parameters"].get("x", 0.0)),
                        "y": float(step["parameters"].get("y", 0.0)),
                    }
                    if target not in named:
                        plan.update(
                            steps=[], confidence=0.0, can_proceed=False, reason=_NO_TARGET
                        )
                        break
        return graph, plan

    def _plan_task(self, args: dict[str, Any]) -> dict[str, Any]:
        _, plan = self._plan(_str(args, "task"))
        if plan.get("can_proceed") and not plan.get("reason"):
            refusal = self._plan_refusal(plan["steps"])
            if refusal is not None:
                plan.update(can_proceed=False, reason=refusal)
        notes = self._speed_notes(plan["steps"])
        if notes:
            plan["speed_notes"] = notes
        return plan

    def _speed_notes(self, steps: list[dict[str, Any]]) -> list[str]:
        """Say which drives will not run at the speed the plan shows."""
        notes = []
        for number, step in enumerate(steps, start=1):
            if not step["skill_id"].startswith("navigate_to"):
                continue
            speed = step["parameters"].get("speed")
            speed = float(speed) if speed is not None else None
            if not self.speed_limit_applies() or (
                speed is not None and speed > self.enforcer.policy.max_speed
            ):
                notes.append(f"Step {number} drives at {self._drive_speed_text(speed)}.")
        return notes

    def _plan_refusal(
        self,
        steps: list[dict[str, Any]],
        tail: str = "Nothing was moved; execute_task would be refused at this step.",
    ) -> str | None:
        """The refusal the first drive in *steps* would get, checked without moving."""
        check = getattr(self.enforcer, "move_problem", None)
        if check is None:
            return None
        start: tuple[float, float] | None = None
        for number, step in enumerate(steps, start=1):
            if not step["skill_id"].startswith("navigate_to"):
                continue
            x = float(step["parameters"].get("x", 0.0))
            y = float(step["parameters"].get("y", 0.0))
            problem = check(x, y, start)
            if problem is not None:
                return (
                    f"REFUSED by safety policy '{self.enforcer.policy.name}' at step "
                    f"{number}: {problem}. {tail} Choose a different target."
                )
            start = (x, y)
        return None

    def _execute_task(self, args: dict[str, Any]) -> dict[str, Any]:
        from apyrobo.skills.executor import SkillExecutor

        task = _str(args, "task")
        approved, self.approved_plan = self.approved_plan, None
        if approved is not None and approved[0] == task:
            # The user said yes to this plan: run it, never a new one.
            graph, plan = approved[1], approved[2]
        else:
            graph, plan = self._plan(task)
        if plan.get("reason"):
            raise ToolError(plan["reason"])
        if not plan["steps"]:
            raise ToolError("The planner produced no steps for this task.")
        v_before = len(self.enforcer.violations)
        i_before = len(self.enforcer.interventions)
        moves_before = getattr(self.enforcer, "moves_sent", 0)
        executor = SkillExecutor(self.enforcer, abort_reason=lambda: self.stop_reason)
        result = executor.execute_graph(graph)
        status = getattr(result.status, "value", str(result.status))
        refusals = self.enforcer.violations[v_before:]
        outcome = {
            "status": status,
            "steps_completed": result.steps_completed,
            "steps_total": len(plan["steps"]),
            "plan": plan["steps"],
            "position": self._position(),
            "safety_interventions": self.enforcer.interventions[i_before:],
        }
        outcome.update(self.speed_notes(moves_before))
        if status != "completed":
            reason = getattr(result, "error", None) or "execution failed"
            if refusals:
                refusal = refusals[-1]
                still = "The robot did not move. " if not result.steps_completed else ""
                reason = (
                    f"REFUSED by safety policy '{self.enforcer.policy.name}' at step "
                    f"{result.steps_completed + 1}: "
                    f"{refusal.get('message', refusal.get('type'))}. "
                    f"{still}Choose a different target."
                )
                outcome["refusal"] = refusal
            outcome["error"] = reason
            raise ToolError(f"{reason}\n\n{json.dumps(_jsonable(outcome), indent=2)}")
        return outcome

    def _safety_report(self, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "safety_policy": self._policy_summary(),
            "interventions": self.enforcer.interventions,
            "violations": self.enforcer.violations,
        }


class MCPServer:
    """A minimal MCP server over newline-delimited JSON-RPC on stdio.

    Implements the parts of the protocol a tools-only server needs. Legacy
    clients (2025-11-25 and earlier) use ``initialize``, ``ping``,
    ``tools/list`` and ``tools/call``. Modern clients (2026-07-28) send no
    handshake: each request carries its protocol version in ``params._meta``,
    and the server answers ``server/discover``, ``tools/list`` and
    ``tools/call`` statelessly.
    """

    def __init__(
        self,
        toolbox: RobotToolbox,
        *,
        confirm_moves: bool = False,
        audit_log: AuditLog | None = None,
    ) -> None:
        self.toolbox = toolbox
        self.audit_log = audit_log
        self.protocol_version = SUPPORTED_PROTOCOL_VERSIONS[0]
        self.confirm_moves = confirm_moves
        # Seals the requestState of a confirmation round trip, so a client
        # cannot confirm one move and then send different arguments.
        self._state_key = secrets.token_bytes(32)
        # requestStates already answered, with the time they were issued: a
        # yes runs its move once, and a resend of the same state is refused
        # until it would have expired anyway.
        self._spent_states: dict[str, int] = {}
        # execute_task plans shown in a question, by its requestState, so a
        # yes runs the plan the user saw and the planner runs once.
        self._asked_plans: dict[str, tuple[str, Any, dict[str, Any]]] = {}
        # Capabilities the client declared in initialize (legacy sessions).
        self._client_caps: dict[str, Any] = {}
        # Set by serve(): the stdio channel used to send a legacy client an
        # elicitation/create request in the middle of a tools/call.
        self._lines: Iterator[str] | None = None
        self._out: IO[str] | None = None
        self._backlog: deque[Any] = deque()
        self._elicit_seq = 0
        # Plain flags, not locks: client_lost() also runs in a signal handler.
        self.client_gone = False
        self._motion_running = False
        self._halting = False
        self._halter: threading.Thread | None = None
        self._reader: threading.Thread | None = None
        self._client_stdin: IO[str] | None = None
        # A stop or cancel during a move. The stdin thread reads them while
        # the serving loop is blocked in the move, so it starts the stop; the
        # line is queued as usual, so a stop still gets its reply in turn.
        self._motion_seq = 0
        self._running_id: Any = None
        self._interrupted: str | None = None
        self._queue_lock = threading.Lock()
        # Stop calls read from stdin but not served yet. A motion call reached
        # while one is waiting was sent before that stop, so it must not start.
        self._stops_waiting = 0
        # Motion calls read but not served yet, and which of them were cancelled.
        self._unserved_motions: set[Any] = set()
        self._cancelled_unserved: set[Any] = set()
        # A robot_status read during a move runs on its own thread, one at a
        # time: an adapter that blocks must not hold up a stop behind it.
        self._status_thread: threading.Thread | None = None
        self._status_running = False
        self._write_lock = threading.Lock()
        # notifications/progress during a motion call, for a request that
        # carries a progressToken. Its own thread: it never touches the stop.
        self.progress_interval = PROGRESS_INTERVAL_S

    # -- dispatch ------------------------------------------------------

    def handle(self, message: Any) -> dict[str, Any] | None:
        """Handle one decoded JSON-RPC message; return the response or None."""
        if self._cancelled_while_queued(message):
            return None
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(None, INVALID_REQUEST, "Invalid JSON-RPC 2.0 message")
        method = message.get("method")
        msg_id = message.get("id")
        is_notification = "id" not in message
        if not isinstance(method, str):
            # A response from the client (we never send requests) — ignore.
            return (
                None
                if "result" in message or "error" in message
                else _error(msg_id, INVALID_REQUEST, "Missing method")
            )
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return (
                None
                if is_notification
                else _error(msg_id, INVALID_PARAMS, "params must be an object")
            )

        if is_notification:
            # notifications/initialized, notifications/cancelled, … need no reply.
            return None

        meta = params.get("_meta")
        if method != "initialize" and (
            method == "server/discover"
            or (isinstance(meta, dict) and META_PROTOCOL_VERSION in meta)
        ):
            return self._handle_modern(msg_id, method, params)

        try:
            if method == "initialize":
                result = self._initialize(params)
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": self.toolbox.list_tools()}
            elif method == "tools/call":
                gate = self._legacy_confirm(msg_id, params)
                if gate is not None:
                    self._audit_gate(params, gate)
                    return gate
                return self._tools_call(msg_id, params)
            else:
                return _error(msg_id, METHOD_NOT_FOUND, f"Method not found: {method}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("MCP: internal error in %s", method)
            return _error(msg_id, INTERNAL_ERROR, str(exc))
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _handle_modern(self, msg_id: Any, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Serve one stateless (2026-07-28) request."""
        meta = params.get("_meta")
        if not isinstance(meta, dict) or META_PROTOCOL_VERSION not in meta:
            return _error(msg_id, INVALID_PARAMS, f"Missing _meta[{META_PROTOCOL_VERSION!r}]")
        requested = meta[META_PROTOCOL_VERSION]
        if requested not in MODERN_PROTOCOL_VERSIONS:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": UNSUPPORTED_PROTOCOL_VERSION,
                    "message": "Unsupported protocol version",
                    "data": {"supported": list(MODERN_PROTOCOL_VERSIONS), "requested": requested},
                },
            }
        if not isinstance(meta.get(META_CLIENT_CAPABILITIES), dict):
            return _error(msg_id, INVALID_PARAMS, f"Missing _meta[{META_CLIENT_CAPABILITIES!r}]")

        try:
            if method == "server/discover":
                result: dict[str, Any] = {
                    "supportedVersions": list(MODERN_PROTOCOL_VERSIONS),
                    "capabilities": {"tools": {"listChanged": False}},
                    "instructions": INSTRUCTIONS,
                    "ttlMs": LIST_TTL_MS,
                    "cacheScope": "public",
                }
            elif method == "tools/list":
                result = {
                    "tools": self.toolbox.list_tools(),
                    "ttlMs": LIST_TTL_MS,
                    "cacheScope": "public",
                }
            elif method == "tools/call":
                gate = self._confirm_gate(msg_id, params, meta[META_CLIENT_CAPABILITIES])
                if gate is not None:
                    self._audit_gate(params, gate)
                if gate is not None and "error" in gate:
                    return gate
                if gate is not None:
                    result = gate["result"]
                    if result.get("resultType") == "input_required":
                        result["_meta"] = {META_SERVER_INFO: _server_info()}
                        return {"jsonrpc": "2.0", "id": msg_id, "result": result}
                else:
                    response = self._tools_call(msg_id, params)
                    if response is None or "error" in response:
                        return response
                    result = response["result"]
            else:
                return _error(msg_id, METHOD_NOT_FOUND, f"Method not found: {method}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("MCP: internal error in %s", method)
            return _error(msg_id, INTERNAL_ERROR, str(exc))
        result["resultType"] = "complete"
        result["_meta"] = {META_SERVER_INFO: _server_info()}
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    # -- --audit-log -------------------------------------------------------

    def _audit(
        self,
        name: Any,
        arguments: dict[str, Any],
        decision: str,
        *,
        enforced_speed: float | None = None,
        detail: str | None = None,
    ) -> None:
        """Write one audit line; raises when it cannot be written."""
        if self.audit_log is None:
            return
        entry: dict[str, Any] = {
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "tool": name,
            "arguments": arguments,
            "decision": decision,
            "enforced_speed": enforced_speed,
            "policy": self.toolbox.enforcer.policy.name,
        }
        if detail:
            entry["detail"] = detail[:AUDIT_DETAIL_CHARS]
        self.audit_log.record(entry)

    def _audit_after(self, *args: Any, **kwargs: Any) -> None:
        """Write an audit line when nothing is left to refuse.

        The call has already run or never will, so a failure can only be
        reported. The next motion command fails to write too, and is refused.
        """
        try:
            self._audit(*args, **kwargs)
        except OSError as exc:
            logger.error("MCP: could not write the audit log: %s", exc)

    def _audit_gate(self, params: dict[str, Any], gate: dict[str, Any]) -> None:
        """Log a motion command that --confirm-moves stopped from running."""
        result = gate.get("result") or {}
        if self.audit_log is None or result.get("resultType") == "input_required":
            return
        arguments = params.get("arguments")
        self._audit_after(
            params.get("name"),
            arguments if isinstance(arguments, dict) else {},
            "declined" if result.get("isError") is False else "refused",
        )

    def _audit_result(
        self,
        name: str,
        arguments: dict[str, Any],
        text: str,
        is_error: bool,
        before: tuple[int, ...],
    ) -> None:
        """Log how a tool call ended.

        *before*: violations, interventions and moves sent at its start.
        """
        enforcer = self.toolbox.enforcer
        clamped = any("enforced" in i for i in enforcer.interventions[before[1]:])
        if text.startswith("REFUSED") or len(enforcer.violations) > before[0]:
            self._audit_after(name, arguments, "refused", detail=text)
        elif is_error:
            self._audit_after(name, arguments, "failed", detail=text)
        elif self.toolbox._drove_unlimited(before[2]):
            # Through Nav2 no speed reaches the robot, so nothing was clamped.
            self._audit_after(name, arguments, "not_limited", detail=SPEED_NOT_APPLIED)
        else:
            note = self.toolbox.speed_notes(before[2]).get("speed_note")
            self._audit_after(
                name, arguments, "clamped" if clamped else "ran",
                enforced_speed=enforcer.last_speed,
                detail=note,
            )

    # -- --confirm-moves ---------------------------------------------------

    def _seal(
        self, name: str, arguments: dict[str, Any], issued: int, nonce: str | None = None
    ) -> str:
        # The nonce makes each question's state unique, so two questions about
        # the same move in the same second can each be answered once.
        if nonce is None:
            nonce = secrets.token_hex(8)
        body = json.dumps(
            {"tool": name, "arguments": arguments, "issued": issued, "nonce": nonce},
            sort_keys=True, separators=(",", ":"),
        )
        mac = hmac.new(self._state_key, body.encode(), hashlib.sha256).hexdigest()
        return f"{issued}.{nonce}.{mac}"

    def _state_ok(self, state: Any, name: str, arguments: dict[str, Any]) -> bool:
        if not isinstance(state, str):
            return False
        parts = state.split(".")
        if len(parts) != 3 or not parts[0].isdigit():
            return False
        issued = int(parts[0])
        if not 0 <= time.time() - issued <= CONFIRM_TTL_S:
            return False
        if state in self._spent_states:
            return False
        return hmac.compare_digest(state, self._seal(name, arguments, issued, parts[1]))

    def _spend_state(self, state: str) -> None:
        """Mark an answered requestState used; forget ones past their TTL."""
        now = time.time()
        for old, issued in list(self._spent_states.items()):
            if now - issued > CONFIRM_TTL_S:
                del self._spent_states[old]
        for old in list(self._asked_plans):
            if now - int(old.split(".", 1)[0]) > CONFIRM_TTL_S:
                del self._asked_plans[old]
        self._spent_states[state] = int(state.split(".", 1)[0])

    def _confirm_gate(
        self, msg_id: Any, params: dict[str, Any], client_caps: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Ask the user to confirm a motion command (2026-07-28 clients).

        Returns None when the call may run, or a response to send instead:
        an ``input_required`` result carrying the confirmation request, a
        declined or refused tool result, or a JSON-RPC error.
        """
        name = params.get("name")
        if not self.confirm_moves or name not in MOTION_TOOLS:
            return None
        self.toolbox.approved_plan = None  # only a yes in this call sets it
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return None  # _tools_call reports the bad arguments
        elicitation = client_caps.get("elicitation")
        if not isinstance(elicitation, dict) or (elicitation and "form" not in elicitation):
            return _tool_result(
                msg_id,
                "REFUSED: this server runs with --confirm-moves, so every motion command "
                "needs the user's confirmation, and this client does not support form "
                "elicitation. The robot did not move.",
                is_error=True,
            )
        responses = params.get("inputResponses")
        if responses is not None and name == "execute_task":
            # The answer to a question about a plan: run that plan, not a new one.
            state = params.get("requestState")
            planned = self._asked_plans.pop(state, None) if isinstance(state, str) else None
            if planned is None or not self._state_ok(state, name, arguments):
                return _error(
                    msg_id, INVALID_PARAMS,
                    "Invalid or expired requestState: call the tool again without "
                    "inputResponses to ask for a new confirmation",
                )
            self._spend_state(state)
            answer = responses.get(CONFIRM_INPUT_KEY) if isinstance(responses, dict) else None
            if isinstance(answer, dict) and answer.get("action") == "accept":
                self.toolbox.approved_plan = planned
                return None
            action = answer.get("action") if isinstance(answer, dict) else None
            return _declined(msg_id, name, action)
        try:
            prompt = self.toolbox.confirmation_prompt(name, arguments)
        except ToolError as exc:
            return _tool_result(msg_id, str(exc), is_error=True)
        if prompt is None:
            return self._refused_unasked(msg_id, name)

        if responses is None:
            state = self._seal(name, arguments, int(time.time()))
            if name == "execute_task" and self.toolbox.last_plan is not None:
                self._asked_plans[state] = self.toolbox.last_plan
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "resultType": "input_required",
                    "inputRequests": {
                        CONFIRM_INPUT_KEY: {
                            "method": "elicitation/create",
                            "params": {
                                "mode": "form",
                                "message": prompt,
                                "requestedSchema": {"type": "object", "properties": {}},
                            },
                        }
                    },
                    "requestState": state,
                },
            }
        if not self._state_ok(params.get("requestState"), name, arguments):
            return _error(
                msg_id, INVALID_PARAMS,
                "Invalid or expired requestState: call the tool again without "
                "inputResponses to ask for a new confirmation",
            )
        # Any answer uses the state up: a yes runs once, and a no cannot be
        # turned into a yes by resending the same state.
        self._spend_state(params["requestState"])
        answer = responses.get(CONFIRM_INPUT_KEY) if isinstance(responses, dict) else None
        if isinstance(answer, dict) and answer.get("action") == "accept":
            return None
        action = answer.get("action") if isinstance(answer, dict) else None
        return _declined(msg_id, name, action)

    def _refused_unasked(self, msg_id: Any, name: str) -> dict[str, Any] | None:
        """No question for a call the policy refuses anyway.

        A move or a task that cannot be planned is refused when it runs,
        before anything moves. A task whose plan the policy refuses is
        refused here, so its earlier steps never run without a yes.
        """
        refusal = self.toolbox.last_refusal if name == "execute_task" else None
        if refusal is None:
            return None
        return _tool_result(msg_id, refusal, is_error=True)

    def _legacy_confirm(self, msg_id: Any, params: dict[str, Any]) -> dict[str, Any] | None:
        """Ask the user to confirm a motion command (initialize-era sessions).

        The server sends the client an ``elicitation/create`` request on the
        same stdio stream and waits for its answer before the tool runs.
        Returns None when the call may run, or the tool result to send instead.
        """
        name = params.get("name")
        if not self.confirm_moves or name not in MOTION_TOOLS:
            return None
        self.toolbox.approved_plan = None  # only a yes in this call sets it
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return None  # _tools_call reports the bad arguments
        elicitation = self._client_caps.get("elicitation")
        if not isinstance(elicitation, dict) or (elicitation and "form" not in elicitation):
            return _tool_result(
                msg_id,
                "REFUSED: this server runs with --confirm-moves, so every motion command "
                "needs the user's confirmation, and this client did not declare form "
                "elicitation support. The robot did not move.",
                is_error=True,
            )
        if self._lines is None or self._out is None:
            return _tool_result(
                msg_id,
                "REFUSED: this server runs with --confirm-moves and has no channel to ask "
                "the user for confirmation. The robot did not move.",
                is_error=True,
            )
        try:
            prompt = self.toolbox.confirmation_prompt(name, arguments)
        except ToolError as exc:
            return _tool_result(msg_id, str(exc), is_error=True)
        if prompt is None:
            return self._refused_unasked(msg_id, name)
        planned = self.toolbox.last_plan if name == "execute_task" else None
        answer = self._elicit(msg_id, prompt)
        action = answer.get("action") if isinstance(answer, dict) else None
        if action == "accept":
            self.toolbox.approved_plan = planned
            return None
        return _declined(msg_id, name, action)

    def _elicit(self, call_id: Any, prompt: str) -> dict[str, Any] | None:
        """Send ``elicitation/create`` and wait for the client's answer.

        Requests that arrive meanwhile are kept and handled after this call
        (``ping`` is answered at once). Returns the elicitation result, a
        ``cancel`` action when the client cancels the tool call, or None when
        the client errors or closes the stream.
        """
        assert self._out is not None
        self._elicit_seq += 1
        request_id = f"apyrobo-confirm-{self._elicit_seq}"
        request_params: dict[str, Any] = {
            "message": prompt,
            "requestedSchema": {"type": "object", "properties": {}},
        }
        if self.protocol_version >= "2025-11-25":
            request_params["mode"] = "form"
        self._write(self._out, {
            "jsonrpc": "2.0", "id": request_id,
            "method": "elicitation/create", "params": request_params,
        })
        while True:
            message = self._next_message()
            if message is _EOF:
                return None
            if not isinstance(message, dict):
                self._backlog.append(message)
                continue
            method = message.get("method")
            if method is None and message.get("id") == request_id:
                result = message.get("result")
                return result if isinstance(result, dict) else None
            if method == "notifications/cancelled":
                cancelled = (message.get("params") or {}).get("requestId")
                if cancelled in (call_id, request_id):
                    return {"action": "cancel"}
                continue
            if method == "ping" and "id" in message:
                self._write(self._out, {"jsonrpc": "2.0", "id": message["id"], "result": {}})
                continue
            self._backlog.append(message)

    def _next_message(self) -> Any:
        """Next decoded message from stdin, or ``_EOF`` at end of input."""
        assert self._lines is not None and self._out is not None
        for line in self._lines:
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError as exc:
                self._write(self._out, _error(None, PARSE_ERROR, f"Parse error: {exc}"))
        return _EOF

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        caps = params.get("capabilities")
        self._client_caps = caps if isinstance(caps, dict) else {}
        requested = params.get("protocolVersion")
        if requested in SUPPORTED_PROTOCOL_VERSIONS:
            self.protocol_version = requested
        else:
            self.protocol_version = SUPPORTED_PROTOCOL_VERSIONS[0]
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": _server_info(),
            "instructions": INSTRUCTIONS,
        }

    def _tools_call(self, msg_id: Any, params: dict[str, Any]) -> dict[str, Any] | None:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not self.toolbox.has_tool(name):
            return _error(msg_id, INVALID_PARAMS, f"Unknown tool: {name!r}")
        if not isinstance(arguments, dict):
            return _error(msg_id, INVALID_PARAMS, "arguments must be an object")
        moves = name in MOTION_TOOLS
        if moves:
            self._settle_client()
        if moves:
            self._motion_seq += 1
            self._running_id = msg_id
            self._interrupted = None
            self.toolbox.stop_reason = None
        # In this order: _halt() stops repeating only when no motion is
        # running, and it starts after client_gone is set, so a move either
        # sees client_gone here or is stopped by _halt(). The same holds for
        # _stops_waiting and _read_stop(). Only motion calls touch the flag:
        # a status read can run here on its own thread during a move.
        if moves:
            self._motion_running = True
        if moves and (self.client_gone or self._stops_waiting):
            self._motion_running = False
            text = (
                f"REFUSED: the MCP client disconnected, so {name} did not run."
                if self.client_gone else
                f"REFUSED: a stop came in after this request, so {name} did not run."
            )
            self._audit_after(name, arguments, "refused", detail=text)
            return _tool_result(msg_id, text, is_error=True)
        audited = self.audit_log is not None and (moves or name == "stop")
        if audited:
            enforcer = self.toolbox.enforcer
            enforcer.last_speed = None
            before = (
                len(enforcer.violations),
                len(enforcer.interventions),
                getattr(enforcer, "moves_sent", 0),
            )
        if audited and moves:
            # Before the robot moves: no record, no motion, and a crash
            # during the move leaves a "started" line with no outcome.
            try:
                self._audit(name, arguments, "started")
            except OSError as exc:
                self._motion_running = False
                logger.error("MCP: could not write the audit log: %s", exc)
                return _tool_result(
                    msg_id,
                    f"REFUSED: could not write the audit log, so {name} did not run: {exc}",
                    is_error=True,
                )
        done = threading.Event()
        structured: dict[str, Any] | None = None
        if moves:
            self._start_progress(params, name, arguments, done)
        try:
            payload = self.toolbox.call(name, arguments)
            text = json.dumps(_jsonable(payload), indent=2)
            is_error = False
        except ToolError as exc:
            text, is_error = str(exc), True
            structured = exc.structured
        except Exception as exc:
            logger.warning("MCP: tool %s failed: %s", name, exc)
            text, is_error = f"{type(exc).__name__}: {exc}", True
        finally:
            if moves:
                self._motion_running = False
                # Under the write lock: a progress notification already being
                # written goes out first, and none follows the reply.
                with self._write_lock:
                    done.set()
        interrupted = self._interrupted if moves else None
        if interrupted is not None:
            text = self._interrupted_text(name, interrupted, text if is_error else None)
            self._audit_after(name, arguments, interrupted, detail=text)
            # The spec: a receiver should not answer a cancelled request.
            return None if interrupted == "cancelled" else _tool_result(
                msg_id, text, is_error=True
            )
        if audited:
            self._audit_result(name, arguments, text, is_error, before)
        result: dict[str, Any] = {"content": [{"type": "text", "text": text}], "isError": is_error}
        if structured is not None and self.protocol_version >= STRUCTURED_CONTENT_SINCE:
            result["structuredContent"] = structured
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _start_progress(
        self,
        params: dict[str, Any],
        name: str,
        arguments: dict[str, Any],
        done: threading.Event,
    ) -> None:
        """Report progress on this motion call if the client asked for it."""
        meta = params.get("_meta")
        token = meta.get("progressToken") if isinstance(meta, dict) else None
        if isinstance(token, bool) or not isinstance(token, (str, int)):
            return
        out = self._out
        if out is None:
            return
        threading.Thread(
            target=self._report_progress, args=(out, token, name, arguments, done),
            name="apyrobo-mcp-progress", daemon=True,
        ).start()

    def _report_progress(
        self,
        out: IO[str],
        token: str | int,
        name: str,
        arguments: dict[str, Any],
        done: threading.Event,
    ) -> None:
        """Send notifications/progress until the motion call returns.

        Reads the position only; a failed read or write ends the reports and
        nothing else. The client losing stdout is noticed by the stdin side.
        """
        started = time.monotonic()
        count = 0
        while not done.wait(self.progress_interval):
            count += 1
            message = self._progress_message(name, arguments, time.monotonic() - started)
            notification = {
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {"progressToken": token, "progress": count, "message": message},
            }
            line = json.dumps(notification, separators=(",", ":")) + "\n"
            try:
                with self._write_lock:
                    if done.is_set():
                        return
                    out.write(line)
                    out.flush()
            except (OSError, ValueError) as exc:
                logger.warning("MCP: could not send a progress notification: %s", exc)
                return

    def _progress_message(self, name: str, arguments: dict[str, Any], elapsed: float) -> str:
        text = f"{name} running for {elapsed:.0f} s"
        try:
            here = self.toolbox._position()
        except Exception:
            return text
        text += f", at ({here['x']:.2f}, {here['y']:.2f})"
        if name == "move_to":
            x, y = _finite(arguments.get("x")), _finite(arguments.get("y"))
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                left = math.hypot(float(x) - here["x"], float(y) - here["y"])
                text += f", {left:.2f} m from the target"
        return text

    def _interrupted_text(self, name: str, interrupted: str, error: str | None) -> str:
        """Say how a motion call ended after a stop or a cancel reached it."""
        what = "a stop request" if interrupted == "stopped" else "the client's cancel"
        text = f"STOPPED: {what} arrived during {name}, so it did not finish."
        try:
            here = self.toolbox._position()
            text += f" Last known position: ({here['x']:.3f}, {here['y']:.3f})."
        except Exception as exc:
            logger.warning("MCP: could not read the position after a stop: %s", exc)
        return f"{text}\n\n{error}" if error else text

    # -- transport -----------------------------------------------------

    def serve(
        self,
        stdin: IO[str] | None = None,
        stdout: IO[str] | None = None,
        *,
        watch_client: bool = False,
    ) -> None:
        """Read messages from *stdin* until EOF, write responses to *stdout*.

        While a request is handled, ``sys.stdout`` points at stderr, so a stray
        ``print`` in an adapter can never corrupt the protocol stream.

        With *watch_client*, stdin is read on a background thread, so the end
        of input is seen during a blocking move: the robot is stopped at once
        and any motion request not yet started is refused. A ``stop`` call or
        a ``notifications/cancelled`` for the running motion call acts the
        same way on that call, and the session goes on. A ``robot_status``
        call read during a move is answered at once, before the move's reply.
        """
        stdin = stdin or sys.stdin
        out = stdout or sys.stdout
        self._lines, self._out = iter(stdin), out
        self._backlog.clear()
        if watch_client:
            self._client_stdin = stdin
            inbox: queue.SimpleQueue[Any] = queue.SimpleQueue()
            self._lines = iter(inbox.get, _EOF)
            self._reader = threading.Thread(
                target=self._read_client, args=(stdin, inbox),
                name="apyrobo-mcp-stdin", daemon=True,
            )
            self._reader.start()
        try:
            self._serve_loop(out)
        except BrokenPipeError:
            self.client_lost("stdout closed")
        finally:
            status = self._status_thread
            if status is not None:
                status.join(STATUS_JOIN_S)
            self._lines = self._out = None

    def _read_client(self, stdin: IO[str], inbox: queue.SimpleQueue[Any]) -> None:
        try:
            for line in stdin:
                # Before the line is queued, so the serving loop cannot reach
                # it first and take a stop nobody has counted.
                if not self._read_ahead(line):
                    inbox.put(line)
        except Exception as exc:
            logger.warning("MCP: reading stdin failed: %s", exc)
        finally:
            self.client_lost("stdin closed")
            inbox.put(_EOF)

    def _read_ahead(self, line: str) -> bool:
        """Act on a stop or cancel as it is read, ahead of the blocked loop.

        Returns True when the line has been dealt with and must not be queued.
        """
        try:
            decoded = json.loads(line)
        except (ValueError, RecursionError):
            return False  # The serving loop answers it with a parse error.
        if self._status_during_motion(decoded):
            return True
        for message in decoded if isinstance(decoded, list) else [decoded]:
            if _is_stop_call(message):
                self._read_stop()
            elif _is_motion_call(message):
                with self._queue_lock:
                    self._unserved_motions.add(message["id"])
            elif isinstance(message, dict) and message.get("method") == "notifications/cancelled":
                params = message.get("params")
                if isinstance(params, dict):
                    self._read_cancel(params.get("requestId"))
        return False

    def _status_during_motion(self, message: Any) -> bool:
        """Start answering a status read now if a move is holding the loop up.

        Not on this thread: it has to go on reading, or a stop sent after a
        status read that blocks in the adapter would never be seen. While one
        read is out, the next waits in the queue like any other request. So
        does a batch, and a read sent before the move started.
        """
        if not self._motion_running or not _is_status_call(message):
            return False
        out = self._out
        if out is None or self._status_running:
            return False
        self._status_running = True
        self._status_thread = threading.Thread(
            target=self._answer_status, args=(message, out),
            name="apyrobo-mcp-status", daemon=True,
        )
        self._status_thread.start()
        return True

    def _answer_status(self, message: dict[str, Any], out: IO[str]) -> None:
        try:
            try:
                response = self.handle(message)
            finally:
                # Before the reply goes out: a client may send the next read
                # the moment it has this one.
                self._status_running = False
            if response is not None:
                self._write(out, response)
        except BrokenPipeError:
            self.client_lost("stdout closed")
        except ValueError as exc:  # The stream closed as the session ended.
            logger.warning("MCP: could not answer a status read: %s", exc)

    def _read_stop(self) -> None:
        with self._queue_lock:
            self._stops_waiting += 1
        self._interrupt("stopped")

    def _read_cancel(self, request_id: Any) -> None:
        """Cancel the running motion call or a queued one; anything else is ignored."""
        if self._motion_running and self._running_id == request_id:
            self._interrupt("cancelled")
        elif isinstance(request_id, (str, int)):
            with self._queue_lock:
                if request_id in self._unserved_motions:
                    self._cancelled_unserved.add(request_id)

    def _cancelled_while_queued(self, message: Any) -> bool:
        """Settle what _read_ahead() counted; True for a call cancelled unserved."""
        with self._queue_lock:
            if _is_stop_call(message):
                self._stops_waiting = max(0, self._stops_waiting - 1)
            if not _is_motion_call(message):
                return False
            msg_id = message["id"]
            self._unserved_motions.discard(msg_id)
            cancelled = msg_id in self._cancelled_unserved
            self._cancelled_unserved.discard(msg_id)
        if cancelled:
            params = message["params"]
            arguments = params.get("arguments")
            self._audit_after(
                params["name"], arguments if isinstance(arguments, dict) else {}, "cancelled",
                detail="Cancelled by the client before it started.",
            )
        return cancelled

    def _interrupt(self, how: str) -> None:
        """Stop the motion call running now, if any. It may start another one later."""
        seq = self._motion_seq
        if not self._motion_running:
            return
        self._interrupted = how
        self.toolbox.stop_reason = (
            "A stop request arrived, so the remaining steps did not run."
            if how == "stopped" else
            "The client cancelled the call, so the remaining steps did not run."
        )
        logger.warning("MCP: %s during a motion call, stopping the robot", how)
        threading.Thread(
            target=self._stop_motion, args=(seq,), name="apyrobo-mcp-stop", daemon=True
        ).start()

    def _stop_motion(self, seq: int) -> None:
        """Repeat the stop until motion call *seq* returns, as _halt() does.

        A later motion call is not touched: a stop ends the call it reached,
        and the client may move the robot again afterwards.
        """
        while self._motion_running and self._motion_seq == seq:
            try:
                self.toolbox.enforcer.stop()
            except Exception as exc:
                logger.error("MCP: stop during a motion call failed: %s", exc)
            time.sleep(HALT_REPEAT_S)

    def _settle_client(self) -> None:
        """Let the stdin thread catch up before a motion call.

        A stdin that has already closed must refuse every time, but the thread
        may not have reached the end of input yet. While the descriptor is
        readable it has work to do, so wait for it; a client closing after this
        check is the real race, and client_lost() stops that move.
        """
        if self._client_stdin is None:
            return
        deadline = time.monotonic() + STDIN_SETTLE_S
        try:
            fd = self._client_stdin.fileno()
        except (OSError, ValueError):
            return  # No descriptor (StringIO): nothing to wait for.
        try:
            while (
                not self.client_gone
                and select.select([fd], [], [], 0)[0]
                and time.monotonic() < deadline
            ):
                time.sleep(STDIN_SETTLE_POLL_S)
        except (OSError, ValueError):
            self._settle_without_select(deadline)

    def _settle_without_select(self, deadline: float) -> None:
        """Give the stdin thread its time when select cannot watch stdin.

        Windows select takes only sockets, and asking about a pipe directly can
        block while the stdin thread is reading it. So every motion call waits
        out the settle time, a small cost for not moving after the client left.
        """
        while (
            not self.client_gone
            and self.reader_running()
            and time.monotonic() < deadline
        ):
            time.sleep(STDIN_SETTLE_POLL_S)

    def reader_running(self) -> bool:
        """True while the stdin thread is still blocked on a read."""
        return self._reader is not None and self._reader.is_alive()

    def client_lost(self, reason: str) -> None:
        """Nobody is supervising any more: stop the robot from another thread.

        Safe to call from a signal handler, and more than once.
        """
        self.client_gone = True
        self.toolbox.stop_reason = (
            "The MCP client disconnected, so the remaining steps did not run."
        )
        if self._halting:
            return
        self._halting = True
        logger.warning("MCP: client lost (%s), stopping the robot", reason)
        self._halter = threading.Thread(
            target=self._halt, args=(reason,), name="apyrobo-mcp-halt", daemon=True
        )
        self._halter.start()

    def _halt(self, reason: str) -> None:
        logged = False
        while True:
            detail = reason
            try:
                self.toolbox.enforcer.stop()
            except Exception as exc:
                logger.error("MCP: stop after client loss failed: %s", exc)
                detail = f"{reason}; stop failed: {exc}"
            if not logged:
                # Here, not in client_lost(): that also runs in a signal handler.
                self._audit_after("stop", {}, "client_lost", detail=detail)
                logged = True
            if not self._motion_running:
                return
            time.sleep(HALT_REPEAT_S)

    def join_halt(self, timeout: float = HALT_JOIN_S) -> None:
        """Give the stop after a client loss time to finish and be logged."""
        if self._halter is not None:
            self._halter.join(timeout)

    def _serve_loop(self, out: IO[str]) -> None:
        while True:
            message = self._backlog.popleft() if self._backlog else self._next_message()
            if message is _EOF:
                return
            with contextlib.redirect_stdout(sys.stderr):
                if isinstance(message, list) and not message:
                    # JSON-RPC 2.0: an empty batch is a single Invalid Request.
                    response: Any = _error(None, INVALID_REQUEST, "Empty batch")
                elif isinstance(message, list):
                    replies = [r for r in (self.handle(m) for m in message) if r is not None]
                    response = replies or None
                else:
                    response = self.handle(message)
            if response is not None:
                self._write(out, response)

    def _write(self, out: IO[str], response: Any) -> None:
        try:
            with self._write_lock:
                out.write(json.dumps(response, separators=(",", ":")) + "\n")
                out.flush()
        except OSError as exc:
            # Windows reports a closed pipe as EINVAL, not BrokenPipeError.
            raise BrokenPipeError(f"could not write to the client: {exc}") from exc


def _is_tool_call(message: Any) -> bool:
    return (
        isinstance(message, dict)
        and message.get("method") == "tools/call"
        and isinstance(message.get("id"), (str, int))
        and isinstance(message.get("params"), dict)
    )


def _is_stop_call(message: Any) -> bool:
    return _is_tool_call(message) and message["params"].get("name") == "stop"


def _is_status_call(message: Any) -> bool:
    return _is_tool_call(message) and message["params"].get("name") == "robot_status"


def _is_motion_call(message: Any) -> bool:
    if not _is_tool_call(message):
        return False
    name = message["params"].get("name")
    return isinstance(name, str) and name in MOTION_TOOLS


def _declined(msg_id: Any, name: Any, action: Any) -> dict[str, Any]:
    return _tool_result(
        msg_id,
        json.dumps({
            "status": "declined",
            "tool": name,
            "user_action": action or "none",
            "detail": "The user did not confirm this motion command. The robot did not move.",
        }, indent=2),
        is_error=False,
    )


def _server_info() -> dict[str, str]:
    return {"name": "apyrobo", "version": __version__}


def _tool_result(msg_id: Any, text: str, *, is_error: bool) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {"content": [{"type": "text", "text": text}], "isError": is_error},
    }


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def parse_zone(spec: str, what: str = "no-go zone") -> dict[str, float]:
    """Parse ``x_min,y_min,x_max,y_max`` into a rectangle dict."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(f"{what} must be x_min,y_min,x_max,y_max — got {spec!r}")
    x_min, y_min, x_max, y_max = (float(p) for p in parts)
    if not all(math.isfinite(v) for v in (x_min, y_min, x_max, y_max)):
        raise ValueError(f"{what} must use finite numbers: {spec!r}")
    if x_min > x_max or y_min > y_max:
        raise ValueError(f"{what} has min > max: {spec!r}")
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}


_POLICY_FILE_KEYS = {
    "name", "base", "max_speed", "max_angular_speed", "no_go_zones", "allowed_areas",
}
_ZONE_KEYS = ("x_min", "y_min", "x_max", "y_max")


def _positive(value: Any, where: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        with contextlib.suppress(OverflowError):  # an int too big for a float
            number = float(value)
            # NaN compares false with everything, so it must not get here as
            # a limit: every "speed > limit" check would then pass.
            if math.isfinite(number) and number > 0:
                return number
    raise ValueError(f"{where} must be a positive number, got {value!r}")


def _file_zone(
    raw: Any, index: int, where: str, key: str = "no_go_zones"
) -> dict[str, Any]:
    label = f"{where}: {key}[{index}]"
    if isinstance(raw, str):
        try:
            return parse_zone(raw, "allowed area" if key == "allowed_areas" else "no-go zone")
        except ValueError as exc:
            raise ValueError(f"{label}: {exc}") from None
    if not isinstance(raw, dict):
        raise ValueError(
            f"{label} must be a mapping with x_min, y_min, x_max, y_max "
            f"(and an optional name) or an 'x_min,y_min,x_max,y_max' string"
        )
    unknown = sorted(set(raw) - set(_ZONE_KEYS) - {"name"})
    if unknown:
        raise ValueError(f"{label} has unknown keys {unknown}")
    missing = [k for k in _ZONE_KEYS if k not in raw]
    if missing:
        raise ValueError(f"{label} is missing {missing}")
    zone: dict[str, Any] = {}
    for field in _ZONE_KEYS:
        value = raw[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{label}.{field} must be a finite number, got {value!r}")
        zone[field] = float(value)
    if zone["x_min"] > zone["x_max"] or zone["y_min"] > zone["y_max"]:
        raise ValueError(f"{label} has min > max")
    if "name" in raw:
        if not isinstance(raw["name"], str) or not raw["name"].strip():
            raise ValueError(f"{label}.name must be a non-empty string")
        zone = {"name": raw["name"].strip(), **zone}
    return zone


def load_policy_file(path: str | Path) -> dict[str, Any]:
    """Read a JSON or YAML safety policy file for ``apyrobo mcp --policy-file``.

    Accepted keys (all optional): ``name``, ``base`` (``default`` or
    ``strict``), ``max_speed`` (m/s), ``max_angular_speed`` (rad/s) and
    ``no_go_zones`` and ``allowed_areas`` — lists of
    ``{name?, x_min, y_min, x_max, y_max}`` mappings or
    ``"x_min,y_min,x_max,y_max"`` strings, in metres. ``allowed_areas``, when
    present, must list at least one area: an empty list would quietly mean
    "anywhere".

    Any problem (missing file, bad syntax, unknown key, wrong type) raises
    ``ValueError`` with a readable message; nothing falls back silently.
    """
    import yaml

    p = Path(path)
    where = str(p)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read policy file {where}: {exc.strerror or exc}") from None
    try:
        # YAML is a superset of JSON, so .yaml/.yml and any other suffix use it.
        data = json.loads(text) if p.suffix.lower() == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        fmt = "JSON" if p.suffix.lower() == ".json" else "YAML"
        raise ValueError(f"policy file {where} is not valid {fmt}: {exc}") from None
    if not isinstance(data, dict):
        raise ValueError(f"policy file {where} must contain a mapping of settings at the top level")
    unknown = sorted(set(data) - _POLICY_FILE_KEYS)
    if unknown:
        raise ValueError(
            f"policy file {where} has unknown keys {unknown}; "
            f"allowed: {sorted(_POLICY_FILE_KEYS)}"
        )
    out: dict[str, Any] = {}
    if "name" in data:
        if not isinstance(data["name"], str) or not data["name"].strip():
            raise ValueError(f"{where}: name must be a non-empty string")
        out["name"] = data["name"].strip()
    if "base" in data:
        if not isinstance(data["base"], str):
            raise ValueError(f"{where}: base must be a policy name such as 'default' or 'strict'")
        out["base"] = data["base"]
    for key in ("max_speed", "max_angular_speed"):
        if key in data:
            out[key] = _positive(data[key], f"{where}: {key}")
    zones = data.get("no_go_zones", [])
    if zones is None:
        zones = []
    if not isinstance(zones, list):
        raise ValueError(f"{where}: no_go_zones must be a list")
    out["no_go_zones"] = [_file_zone(z, i, where) for i, z in enumerate(zones)]
    if "allowed_areas" in data:
        areas = data["allowed_areas"]
        if not isinstance(areas, list) or not areas:
            raise ValueError(
                f"{where}: allowed_areas must list at least one area; "
                "remove the key to allow anywhere outside the no-go zones"
            )
        out["allowed_areas"] = [
            _file_zone(a, i, where, "allowed_areas") for i, a in enumerate(areas)
        ]
    return out


# Providers that plan through litellm. Without it they would fall back to
# the rule planner on every call, and only a stderr line would say so.
_LITELLM_PROVIDERS = ("llm", "tool_calling")


def _check_provider_installed(provider: str) -> None:
    """Refuse to start with an LLM planner that cannot be imported."""
    if provider in _LITELLM_PROVIDERS and importlib.util.find_spec("litellm") is None:
        raise ImportError(
            f"--provider {provider} needs litellm, which is not installed. "
            "Install it with: pip install 'apyrobo[llm]', "
            "or use --provider rule (the default)"
        )


def build_server(
    robot_uri: str = "mock://turtlebot4",
    policy: str | None = None,
    max_speed: float | None = None,
    no_go_zones: list[dict[str, float]] | None = None,
    provider: str = "rule",
    agent: Any = None,
    policy_file: str | Path | None = None,
    confirm_moves: bool = False,
    allowed_areas: list[dict[str, float]] | None = None,
    audit_log: str | Path | None = None,
) -> MCPServer:
    """Discover the robot, wrap it in a safety enforcer and return the server.

    Precedence: ``max_speed`` and ``policy`` given here override the policy
    file; ``no_go_zones`` and ``allowed_areas`` given here are added to the
    file's; the
    file overrides the base policy (``policy`` or the file's ``base``,
    else ``default``). ``audit_log`` is the path of the ``--audit-log`` file,
    opened here so that a path that cannot be written stops the start.
    """
    from apyrobo.core.robot import Robot
    from apyrobo.safety.enforcer import POLICY_REGISTRY, SafetyEnforcer, SafetyPolicy

    if max_speed is not None:
        max_speed = _positive(max_speed, "--max-speed")
    if agent is None:
        _check_provider_installed(provider)
    log = AuditLog(audit_log) if audit_log is not None else None
    file_cfg = load_policy_file(policy_file) if policy_file is not None else {}
    base_name = policy or file_cfg.get("base") or "default"
    base = POLICY_REGISTRY.get(base_name)
    if base is None:
        raise ValueError(f"unknown policy {base_name!r}; choose from {sorted(POLICY_REGISTRY)}")
    name = file_cfg.get("name", base.name)
    if max_speed is not None or no_go_zones or allowed_areas:
        name = f"{name}+cli"
    speed = max_speed if max_speed is not None else file_cfg.get("max_speed", base.max_speed)
    safety = SafetyPolicy(
        name=name,
        max_speed=speed,
        max_angular_speed=file_cfg.get("max_angular_speed", base.max_angular_speed),
        collision_zones=(
            list(base.collision_zones)
            + list(file_cfg.get("no_go_zones", []))
            + list(no_go_zones or [])
        ),
        allowed_areas=(
            list(base.allowed_areas)
            + list(file_cfg.get("allowed_areas", []))
            + list(allowed_areas or [])
        ),
        human_proximity_limit=base.human_proximity_limit,
        move_timeout=base.move_timeout,
        speed_profile=base.speed_profile,
    )
    with contextlib.redirect_stdout(sys.stderr):
        robot = Robot.discover(robot_uri)
        if agent is None:
            from apyrobo.skills.agent import Agent

            agent = Agent(provider=provider)
    enforcer = SafetyEnforcer(robot, policy=safety)
    return MCPServer(
        RobotToolbox(robot, enforcer, agent, robot_uri),
        confirm_moves=confirm_moves,
        audit_log=log,
    )
