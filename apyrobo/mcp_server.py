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
import hashlib
import hmac
import json
import logging
import secrets
import sys
import time
from collections.abc import Callable
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

INSTRUCTIONS = (
    "These tools control a robot through APYROBO's safety enforcer. "
    "Call robot_status first to see the robot, its position and the active "
    "safety policy. Motion requests may be changed or refused by the policy: "
    "read each result, and when a command is refused, pick a different "
    "target rather than retrying the same one. stop is always allowed."
)


class ToolError(Exception):
    """A tool ran but could not do what was asked (reported as isError)."""


def _num(args: dict[str, Any], key: str, *, required: bool = True) -> float | None:
    value = args.get(key)
    if value is None:
        if required:
            raise ToolError(f"Missing required argument '{key}'.")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"Argument '{key}' must be a number, got {value!r}.")
    return float(value)


def _str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"Argument '{key}' must be a non-empty string.")
    return value.strip()


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of enums, tuples and models to plain JSON."""
    return json.loads(json.dumps(value, default=lambda o: getattr(o, "value", str(o))))


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
            "may be clamped, and targets inside a no-go zone are refused.",
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

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        _, handler = self._tools[name]
        return handler(arguments)

    def confirmation_prompt(self, name: str, arguments: dict[str, Any]) -> str | None:
        """What the user is asked to confirm under --confirm-moves.

        Shows the target and the speed the policy will actually allow.
        Returns None for a move the policy refuses anyway (target in a
        no-go zone), so the user is not asked to approve a refused move.
        Raises ToolError for invalid arguments.
        """
        p = self.enforcer.policy
        if name == "move_to":
            x, y = _num(arguments, "x"), _num(arguments, "y")
            speed = _num(arguments, "speed", required=False)
            from apyrobo.safety.enforcer import SafetyEnforcer

            if any(SafetyEnforcer._point_in_zone(x, y, z) for z in p.collision_zones):
                return None
            if speed is None:
                speed_text = f"default speed (policy limit {p.max_speed} m/s)"
            elif speed > p.max_speed:
                speed_text = f"{p.max_speed} m/s (requested {speed}, clamped to the policy limit)"
            else:
                speed_text = f"{speed} m/s"
            return f"Allow the robot to move to x={x}, y={y} at {speed_text}?"
        if name == "rotate":
            angle = _num(arguments, "angle_rad")
            speed = _num(arguments, "speed", required=False)
            limit = p.max_angular_speed
            if speed is None:
                speed_text = f"default speed (policy limit {limit} rad/s)"
            elif abs(speed) > limit:
                speed_text = f"{limit} rad/s (requested {speed}, clamped to the policy limit)"
            else:
                speed_text = f"{speed} rad/s"
            return f"Allow the robot to rotate by {angle} rad at {speed_text}?"
        task = _str(arguments, "task")
        return (
            f"Allow the robot to carry out the task {task!r}? Each step stays within "
            f"the safety policy '{p.name}' (max {p.max_speed} m/s, "
            f"{len(p.collision_zones)} no-go zone(s))."
        )

    # -- helpers -------------------------------------------------------

    def _policy_summary(self) -> dict[str, Any]:
        p = self.enforcer.policy
        return {
            "name": p.name,
            "max_speed_m_s": p.max_speed,
            "max_angular_speed_rad_s": p.max_angular_speed,
            "no_go_zones": [dict(z) for z in p.collision_zones],
        }

    def _position(self) -> dict[str, float]:
        x, y = self.enforcer.get_position()
        return {"x": float(x), "y": float(y)}

    def _guarded(self, action: Callable[[], Any]) -> list[dict[str, Any]]:
        """Run *action*; turn a safety refusal into a readable ToolError.

        Returns the interventions (e.g. clamped speed) the action caused.
        """
        from apyrobo.safety.enforcer import SafetyViolation

        before = len(self.enforcer.interventions)
        try:
            action()
        except SafetyViolation as exc:
            raise ToolError(
                f"REFUSED by safety policy '{self.enforcer.policy.name}': {exc}. "
                "The robot did not move. Choose a different target."
            ) from exc
        return self.enforcer.interventions[before:]

    # -- tools ---------------------------------------------------------

    def _robot_status(self, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "robot_uri": self.robot_uri,
            "robot_id": getattr(self.robot, "robot_id", None),
            "position": self._position(),
            "heading_rad": float(self.enforcer.get_orientation()),
            "health": self.enforcer.get_health(),
            "safety_policy": self._policy_summary(),
        }

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
        speed = _num(args, "speed", required=False)
        interventions = self._guarded(lambda: self.enforcer.move(x=x, y=y, speed=speed))
        return {
            "status": "moved",
            "requested": {"x": x, "y": y, "speed": speed},
            "position": self._position(),
            "safety_interventions": interventions,
        }

    def _rotate(self, args: dict[str, Any]) -> dict[str, Any]:
        angle = _num(args, "angle_rad")
        speed = _num(args, "speed", required=False)
        interventions = self._guarded(lambda: self.enforcer.rotate(angle_rad=angle, speed=speed))
        return {
            "status": "rotated",
            "heading_rad": float(self.enforcer.get_orientation()),
            "safety_interventions": interventions,
        }

    def _stop(self, args: dict[str, Any]) -> dict[str, Any]:
        self.enforcer.stop()
        return {"status": "stopped", "position": self._position()}

    def _plan(self, task: str) -> tuple[Any, dict[str, Any]]:
        from apyrobo.safety.confidence import ConfidenceEstimator

        graph = self.agent.plan(task, self.robot)
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
        return graph, plan

    def _plan_task(self, args: dict[str, Any]) -> dict[str, Any]:
        _, plan = self._plan(_str(args, "task"))
        return plan

    def _execute_task(self, args: dict[str, Any]) -> dict[str, Any]:
        from apyrobo.skills.executor import SkillExecutor

        graph, plan = self._plan(_str(args, "task"))
        if not plan["steps"]:
            raise ToolError("The planner produced no steps for this task.")
        v_before = len(self.enforcer.violations)
        i_before = len(self.enforcer.interventions)
        result = SkillExecutor(self.enforcer).execute_graph(graph)
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
        if status != "completed":
            reason = getattr(result, "error", None) or "execution failed"
            if refusals:
                reason = (
                    f"REFUSED by safety policy '{self.enforcer.policy.name}' at step "
                    f"{result.steps_completed + 1}: {refusals[-1]}"
                )
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

    def __init__(self, toolbox: RobotToolbox, *, confirm_moves: bool = False) -> None:
        self.toolbox = toolbox
        self.protocol_version = SUPPORTED_PROTOCOL_VERSIONS[0]
        self.confirm_moves = confirm_moves
        # Seals the requestState of a confirmation round trip, so a client
        # cannot confirm one move and then send different arguments.
        self._state_key = secrets.token_bytes(32)

    # -- dispatch ------------------------------------------------------

    def handle(self, message: Any) -> dict[str, Any] | None:
        """Handle one decoded JSON-RPC message; return the response or None."""
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
                if self.confirm_moves and params.get("name") in MOTION_TOOLS:
                    return _tool_result(
                        msg_id,
                        "REFUSED: this server runs with --confirm-moves, so every motion "
                        "command needs the user's confirmation, and confirmation currently "
                        "needs an MCP client on protocol 2026-07-28 with elicitation "
                        "support. The robot did not move.",
                        is_error=True,
                    )
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
                if gate is not None and "error" in gate:
                    return gate
                if gate is not None:
                    result = gate["result"]
                    if result.get("resultType") == "input_required":
                        result["_meta"] = {META_SERVER_INFO: _server_info()}
                        return {"jsonrpc": "2.0", "id": msg_id, "result": result}
                else:
                    response = self._tools_call(msg_id, params)
                    if "error" in response:
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

    # -- --confirm-moves ---------------------------------------------------

    def _seal(self, name: str, arguments: dict[str, Any], issued: int) -> str:
        body = json.dumps(
            {"tool": name, "arguments": arguments, "issued": issued},
            sort_keys=True, separators=(",", ":"),
        )
        mac = hmac.new(self._state_key, body.encode(), hashlib.sha256).hexdigest()
        return f"{issued}.{mac}"

    def _state_ok(self, state: Any, name: str, arguments: dict[str, Any]) -> bool:
        if not isinstance(state, str) or "." not in state:
            return False
        issued_s, _ = state.split(".", 1)
        if not issued_s.isdigit():
            return False
        issued = int(issued_s)
        if not 0 <= time.time() - issued <= CONFIRM_TTL_S:
            return False
        return hmac.compare_digest(state, self._seal(name, arguments, issued))

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
        try:
            prompt = self.toolbox.confirmation_prompt(name, arguments)
        except ToolError as exc:
            return _tool_result(msg_id, str(exc), is_error=True)
        if prompt is None:
            return None  # the policy refuses it anyway; no need to ask

        responses = params.get("inputResponses")
        if responses is None:
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
                    "requestState": self._seal(name, arguments, int(time.time())),
                },
            }
        if not self._state_ok(params.get("requestState"), name, arguments):
            return _error(
                msg_id, INVALID_PARAMS,
                "Invalid or expired requestState: call the tool again without "
                "inputResponses to ask for a new confirmation",
            )
        answer = responses.get(CONFIRM_INPUT_KEY) if isinstance(responses, dict) else None
        if isinstance(answer, dict) and answer.get("action") == "accept":
            return None
        action = answer.get("action") if isinstance(answer, dict) else None
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

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
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

    def _tools_call(self, msg_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not self.toolbox.has_tool(name):
            return _error(msg_id, INVALID_PARAMS, f"Unknown tool: {name!r}")
        if not isinstance(arguments, dict):
            return _error(msg_id, INVALID_PARAMS, "arguments must be an object")
        try:
            payload = self.toolbox.call(name, arguments)
            text = json.dumps(_jsonable(payload), indent=2)
            is_error = False
        except ToolError as exc:
            text, is_error = str(exc), True
        except Exception as exc:
            logger.warning("MCP: tool %s failed: %s", name, exc)
            text, is_error = f"{type(exc).__name__}: {exc}", True
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"content": [{"type": "text", "text": text}], "isError": is_error},
        }

    # -- transport -----------------------------------------------------

    def serve(self, stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> None:
        """Read messages from *stdin* until EOF, write responses to *stdout*.

        While a request is handled, ``sys.stdout`` points at stderr, so a stray
        ``print`` in an adapter can never corrupt the protocol stream.
        """
        stdin = stdin or sys.stdin
        out = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                self._write(out, _error(None, PARSE_ERROR, f"Parse error: {exc}"))
                continue
            with contextlib.redirect_stdout(sys.stderr):
                if isinstance(message, list):
                    replies = [r for r in (self.handle(m) for m in message) if r is not None]
                    response: Any = replies or None
                else:
                    response = self.handle(message)
            if response is not None:
                self._write(out, response)

    @staticmethod
    def _write(out: IO[str], response: Any) -> None:
        out.write(json.dumps(response, separators=(",", ":")) + "\n")
        out.flush()


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


def parse_zone(spec: str) -> dict[str, float]:
    """Parse ``x_min,y_min,x_max,y_max`` into a collision-zone dict."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(f"no-go zone must be x_min,y_min,x_max,y_max — got {spec!r}")
    x_min, y_min, x_max, y_max = (float(p) for p in parts)
    if x_min > x_max or y_min > y_max:
        raise ValueError(f"no-go zone has min > max: {spec!r}")
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}


_POLICY_FILE_KEYS = {"name", "base", "max_speed", "max_angular_speed", "no_go_zones"}
_ZONE_KEYS = ("x_min", "y_min", "x_max", "y_max")


def _positive(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{where} must be a positive number, got {value!r}")
    return float(value)


def _file_zone(raw: Any, index: int, where: str) -> dict[str, Any]:
    label = f"{where}: no_go_zones[{index}]"
    if isinstance(raw, str):
        try:
            return parse_zone(raw)
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
    for key in _ZONE_KEYS:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}.{key} must be a number, got {value!r}")
        zone[key] = float(value)
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
    ``no_go_zones`` — a list of ``{name?, x_min, y_min, x_max, y_max}``
    mappings or ``"x_min,y_min,x_max,y_max"`` strings, in metres.

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
    return out


def build_server(
    robot_uri: str = "mock://turtlebot4",
    policy: str | None = None,
    max_speed: float | None = None,
    no_go_zones: list[dict[str, float]] | None = None,
    provider: str = "rule",
    agent: Any = None,
    policy_file: str | Path | None = None,
    confirm_moves: bool = False,
) -> MCPServer:
    """Discover the robot, wrap it in a safety enforcer and return the server.

    Precedence: ``max_speed`` and ``policy`` given here override the policy
    file; ``no_go_zones`` given here are added to the file's zones; the
    file overrides the base policy (``policy`` or the file's ``base``,
    else ``default``).
    """
    from apyrobo.core.robot import Robot
    from apyrobo.safety.enforcer import POLICY_REGISTRY, SafetyEnforcer, SafetyPolicy

    file_cfg = load_policy_file(policy_file) if policy_file is not None else {}
    base_name = policy or file_cfg.get("base") or "default"
    base = POLICY_REGISTRY.get(base_name)
    if base is None:
        raise ValueError(f"unknown policy {base_name!r}; choose from {sorted(POLICY_REGISTRY)}")
    name = file_cfg.get("name", base.name)
    if max_speed is not None or no_go_zones:
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
        RobotToolbox(robot, enforcer, agent, robot_uri), confirm_moves=confirm_moves
    )
