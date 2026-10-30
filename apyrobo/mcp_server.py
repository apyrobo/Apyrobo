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
import json
import logging
import sys
from collections.abc import Callable
from typing import IO, Any

from apyrobo.__version__ import __version__

logger = logging.getLogger(__name__)

# Newest first. The server answers with the client's requested version when
# it is one of these, otherwise with the newest.
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

# JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

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

    Implements the parts of the protocol a tools-only server needs:
    ``initialize``, ``ping``, ``tools/list`` and ``tools/call``.
    """

    def __init__(self, toolbox: RobotToolbox) -> None:
        self.toolbox = toolbox
        self.protocol_version = SUPPORTED_PROTOCOL_VERSIONS[0]

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

        try:
            if method == "initialize":
                result = self._initialize(params)
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": self.toolbox.list_tools()}
            elif method == "tools/call":
                return self._tools_call(msg_id, params)
            else:
                return _error(msg_id, METHOD_NOT_FOUND, f"Method not found: {method}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("MCP: internal error in %s", method)
            return _error(msg_id, INTERNAL_ERROR, str(exc))
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        if requested in SUPPORTED_PROTOCOL_VERSIONS:
            self.protocol_version = requested
        else:
            self.protocol_version = SUPPORTED_PROTOCOL_VERSIONS[0]
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "apyrobo", "version": __version__},
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


def build_server(
    robot_uri: str = "mock://turtlebot4",
    policy: str = "default",
    max_speed: float | None = None,
    no_go_zones: list[dict[str, float]] | None = None,
    provider: str = "rule",
    agent: Any = None,
) -> MCPServer:
    """Discover the robot, wrap it in a safety enforcer and return the server."""
    from apyrobo.core.robot import Robot
    from apyrobo.safety.enforcer import POLICY_REGISTRY, SafetyEnforcer, SafetyPolicy

    base = POLICY_REGISTRY.get(policy)
    if base is None:
        raise ValueError(f"unknown policy {policy!r}; choose from {sorted(POLICY_REGISTRY)}")
    safety = SafetyPolicy(
        name=base.name if max_speed is None and not no_go_zones else f"{base.name}+cli",
        max_speed=max_speed if max_speed is not None else base.max_speed,
        max_angular_speed=base.max_angular_speed,
        collision_zones=list(base.collision_zones) + list(no_go_zones or []),
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
    return MCPServer(RobotToolbox(robot, enforcer, agent, robot_uri))
