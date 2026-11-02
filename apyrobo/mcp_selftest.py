"""``apyrobo mcp selftest``: check an install and a safety policy on ``mock://``.

Builds the same server ``apyrobo mcp`` serves, with the same policy flags, and
talks to it in-process through the JSON-RPC handler a client reaches, so a new
user can see their limits hold before touching any client config:

* the tools are listed,
* a move within the limits runs,
* an over-limit move is clamped to the policy's speed limit,
* a move into each no-go zone is refused and the robot stays put.

It only ever drives ``mock://``. Any other robot is refused before anything
is built, so it cannot move hardware.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TextIO

#: Tools a working server must offer for the checks below to mean anything.
REQUIRED_TOOLS = ("robot_status", "move_to", "stop", "safety_report")
#: Fraction of the limit used for the "within the limits" move.
NORMAL_SPEED_FRACTION = 0.5
#: Multiple of the limit requested by the over-limit move.
OVER_LIMIT_FACTOR = 2.0


@dataclass(frozen=True)
class Check:
    """One line of the report."""

    ok: bool
    name: str
    detail: str


class _Client:
    """Calls the server the way a client does, one JSON-RPC message at a time."""

    def __init__(self, server: Any) -> None:
        self._server = server
        self._next_id = 0

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        message = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
            "params": params or {},
        }
        return self._server.handle(json.loads(json.dumps(message))) or {}

    def tool(self, name: str, **arguments: Any) -> tuple[bool, str]:
        """Return ``(is_error, text)`` for a ``tools/call``."""
        response = self.request("tools/call", {"name": name, "arguments": arguments})
        if "error" in response:
            return True, str(response["error"].get("message", response["error"]))
        result = response.get("result", {})
        content = result.get("content") or [{}]
        return bool(result.get("isError")), str(content[0].get("text", ""))

    def position(self) -> tuple[float, float] | None:
        is_error, text = self.tool("robot_status")
        if is_error:
            return None
        position = json.loads(text).get("position") or {}
        if "x" not in position or "y" not in position:
            return None
        return float(position["x"]), float(position["y"])


def _in_zone(x: float, y: float, zone: dict[str, Any]) -> bool:
    return zone["x_min"] <= x <= zone["x_max"] and zone["y_min"] <= y <= zone["y_max"]


def free_targets(zones: Sequence[dict[str, Any]]) -> list[tuple[float, float]]:
    """Two distinct targets outside every zone, close to the origin if possible."""
    near = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 1.0), (-1.0, -1.0)]
    points = [p for p in near if not any(_in_zone(*p, z) for z in zones)]
    # Past the right edge of every zone is always free.
    edge = max((float(z["x_max"]) for z in zones), default=0.0)
    points += [(edge + 1.0, 0.0), (edge + 2.0, 0.0)]
    return points[:2]


def _zone_label(zone: dict[str, Any], index: int) -> str:
    return str(zone.get("name") or f"#{index + 1}")


def _check_tools(client: _Client) -> Check:
    tools = client.request("tools/list").get("result", {}).get("tools", [])
    names = [t.get("name") for t in tools]
    missing = [t for t in REQUIRED_TOOLS if t not in names]
    if missing:
        return Check(False, "tools listed", f"missing {', '.join(missing)}")
    return Check(True, "tools listed", f"{len(names)} ({', '.join(str(n) for n in names)})")


def _check_normal(
    client: _Client, target: tuple[float, float], speed: float
) -> Check:
    name = "normal move"
    is_error, text = client.tool("move_to", x=target[0], y=target[1], speed=speed)
    if is_error:
        return Check(False, name, text)
    if json.loads(text).get("safety_interventions"):
        return Check(False, name, f"expected no intervention at {speed:g} m/s: {text}")
    if client.position() != target:
        return Check(False, name, f"robot did not reach {target}")
    return Check(True, name, f"reached {target} at {speed:g} m/s")


def _check_over_limit(
    client: _Client, target: tuple[float, float], max_speed: float
) -> Check:
    name = "over-limit move"
    requested = max_speed * OVER_LIMIT_FACTOR
    is_error, text = client.tool("move_to", x=target[0], y=target[1], speed=requested)
    if is_error:
        return Check(False, name, text)
    clamped = [
        i for i in json.loads(text).get("safety_interventions", [])
        if i.get("type") == "speed_clamped"
    ]
    if not clamped or float(clamped[0]["enforced"]) > max_speed:
        return Check(False, name, f"{requested:g} m/s was not clamped to {max_speed:g}: {text}")
    return Check(
        True, name, f"requested {requested:g} m/s, clamped to {clamped[0]['enforced']:g} m/s"
    )


def _check_zone(client: _Client, zone: dict[str, Any], index: int, speed: float) -> Check:
    name = f"no-go zone {_zone_label(zone, index)}"
    x = (float(zone["x_min"]) + float(zone["x_max"])) / 2
    y = (float(zone["y_min"]) + float(zone["y_max"])) / 2
    before = client.position()
    is_error, text = client.tool("move_to", x=x, y=y, speed=speed)
    after = client.position()
    if not is_error or "REFUSED" not in text:
        return Check(False, name, f"move to ({x:g}, {y:g}) was not refused: {text}")
    if before is None or before != after:
        return Check(False, name, f"robot moved from {before} to {after}")
    return Check(True, name, f"move to ({x:g}, {y:g}) refused, robot stayed at {after}")


def run_checks(server: Any) -> list[Check]:
    """Run every check against a server built on ``mock://``."""
    policy = server.toolbox.enforcer.policy
    zones = list(policy.collision_zones)
    max_speed = float(policy.max_speed)
    speed = max_speed * NORMAL_SPEED_FRACTION
    client = _Client(server)

    response = client.request(
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "apyrobo-selftest", "version": "1"},
        },
    )
    if "result" not in response:
        return [Check(False, "initialize", str(response.get("error", response)))]

    first, second = free_targets(zones)
    checks = [
        _check_tools(client),
        _check_normal(client, first, speed),
        _check_over_limit(client, second, max_speed),
    ]
    checks += [_check_zone(client, zone, i, speed) for i, zone in enumerate(zones)]
    return checks


def _write_report(
    checks: Sequence[Check], robot: str, policy: Any, confirm_moves: bool, out: TextIO
) -> None:
    print(
        f"apyrobo mcp selftest: robot={robot} policy={policy.name} "
        f"max_speed={policy.max_speed:g} no_go_zones={len(policy.collision_zones)}",
        file=out,
    )
    for check in checks:
        print(f"{'PASS' if check.ok else 'FAIL'}  {check.name}: {check.detail}", file=out)
    if not policy.collision_zones:
        print("note  no no-go zones set; add --no-go or --policy-file to test yours", file=out)
    if confirm_moves:
        print("note  --confirm-moves needs a client that can ask you; not tested here", file=out)
    print(
        "note  zones are checked at the move target only, not along the path; "
        "this ran on mock://, so no robot moved",
        file=out,
    )
    failed = sum(not c.ok for c in checks)
    print(f"{len(checks) - failed}/{len(checks)} checks passed", file=out)


def main(
    args: argparse.Namespace,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Entry point for ``apyrobo mcp selftest``. Returns the exit code."""
    from apyrobo.mcp_server import build_server, parse_zone

    out = stdout or sys.stdout
    err = stderr or sys.stderr

    # Flags work on either side of "selftest": ``apyrobo mcp --max-speed 1 selftest``.
    robot = getattr(args, "st_robot", None) or args.robot
    if not robot.startswith("mock://"):
        print(
            f"Error: selftest only runs on mock://, got {robot!r}. "
            "It exists to check your policy without moving a robot.",
            file=err,
        )
        return 1
    no_go = list(getattr(args, "no_go", None) or []) + list(getattr(args, "st_no_go", None) or [])
    max_speed = getattr(args, "st_max_speed", None)
    if max_speed is None:
        max_speed = getattr(args, "max_speed", None)

    try:
        server = build_server(
            robot_uri=robot,
            policy=getattr(args, "st_policy", None) or getattr(args, "policy", None),
            max_speed=max_speed,
            no_go_zones=[parse_zone(z) for z in no_go],
            policy_file=(
                getattr(args, "st_policy_file", None) or getattr(args, "policy_file", None)
            ),
        )
    except (ValueError, ImportError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=err)
        return 1

    enforcer = server.toolbox.enforcer
    # The clamp and refusals below are expected; their warnings are not news.
    safety_logger = logging.getLogger("apyrobo.safety.enforcer")
    level = safety_logger.level
    safety_logger.setLevel(logging.CRITICAL)
    try:
        checks = run_checks(server)
    finally:
        safety_logger.setLevel(level)
        with contextlib.suppress(Exception):
            enforcer.stop()

    confirm_moves = bool(getattr(args, "confirm_moves", False))
    _write_report(checks, robot, enforcer.policy, confirm_moves, out)
    return 0 if all(c.ok for c in checks) else 1
