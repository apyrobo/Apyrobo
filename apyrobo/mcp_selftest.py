"""``apyrobo mcp selftest``: check an install and a safety policy on ``mock://``.

Builds the same server ``apyrobo mcp`` serves, with the same policy flags, and
talks to it in-process through the JSON-RPC handler a client reaches, so a new
user can see their limits hold before touching any client config:

* the tools are listed,
* a move within the limits runs,
* an over-limit move is clamped to the policy's speed limit,
* a move into each no-go zone is refused and the robot stays put,
* a move across each no-go zone, to a free target, is refused too,
* with allowed areas set, a move to a target outside them is refused,
* with more than one allowed area, a move from one to another through space
  outside them is refused, when the areas leave such a move,
* with ``--audit-log PATH``, the server could open that file for append.
  Nothing is written to it.

It then warns about what the policy leaves unset, without changing the exit
code: a passing run only says the limits you wrote down hold.

It only ever drives ``mock://``. Any other robot is refused before anything
is built, so it cannot move hardware.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
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


def _allowed(
    x: float, y: float, zones: Sequence[dict[str, Any]], areas: Sequence[dict[str, Any]]
) -> bool:
    """True if the policy accepts (x, y) as a target."""
    if any(_in_zone(x, y, z) for z in zones):
        return False
    return not areas or any(_in_zone(x, y, a) for a in areas)


def _clear(
    start: tuple[float, float],
    end: tuple[float, float],
    zones: Sequence[dict[str, Any]],
    areas: Sequence[dict[str, Any]] = (),
) -> bool:
    """True if ``end`` is a free target and the straight line to it is too."""
    from apyrobo.safety.enforcer import route_leaves_areas

    return (
        _allowed(*end, zones, areas)
        and not _enters_a_zone(start, end, zones)
        and route_leaves_areas([start, end], areas) is None
    )


def _enters_a_zone(
    start: tuple[float, float],
    end: tuple[float, float],
    zones: Sequence[dict[str, Any]],
) -> bool:
    from apyrobo.safety.enforcer import segment_enters_zone

    return any(segment_enters_zone(start, end, z) for z in zones)


def _area_spots(areas: Sequence[dict[str, Any]]) -> list[tuple[float, float]]:
    """A 5 by 5 grid of points across each allowed area."""
    spots = []
    for area in areas:
        width = float(area["x_max"]) - float(area["x_min"])
        height = float(area["y_max"]) - float(area["y_min"])
        spots += [
            (float(area["x_min"]) + width * i / 4, float(area["y_min"]) + height * j / 4)
            for i in range(5) for j in range(5)
        ]
    return spots


def free_targets(
    zones: Sequence[dict[str, Any]],
    start: tuple[float, float] = (0.0, 0.0),
    areas: Sequence[dict[str, Any]] = (),
) -> list[tuple[float, float]]:
    """Two targets the robot can drive to in turn from ``start``, near it if possible."""
    near = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 1.0), (-1.0, -1.0)]
    # Past the right edge of every zone is free, though zones may block the way.
    edge = max((float(z["x_max"]) for z in zones), default=0.0)
    points = [*near, (edge + 1.0, 0.0), (edge + 2.0, 0.0)]
    points += [(x * r, y * r) for r in (0.25, 0.5, 2.0, 4.0, 8.0) for x, y in near]
    # Spots across each allowed area, which may be nowhere near the start.
    points += _area_spots(areas)
    for first in points:
        if _clear(start, first, zones, areas):
            for second in [*points, start]:
                if second != first and _clear(first, second, zones, areas):
                    return [first, second]
    # Walled in: return free targets anyway; the checks report the refusal.
    free = [p for p in points if _allowed(*p, zones, areas)]
    return free[:2] or points[:2]


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


def _across(
    start: tuple[float, float],
    zone: dict[str, Any],
    zones: Sequence[dict[str, Any]],
    areas: Sequence[dict[str, Any]] = (),
) -> tuple[float, float] | None:
    """A free target past ``zone``, on the line from ``start`` through its centre.

    None when allowed areas leave no target past the zone: a refusal there
    would come from the area and prove nothing about the zone.
    """
    cx = (float(zone["x_min"]) + float(zone["x_max"])) / 2
    cy = (float(zone["y_min"]) + float(zone["y_max"])) / 2
    dx, dy = cx - start[0], cy - start[1]
    stretches = [2.0 ** i for i in range(64)]
    if areas:
        # Stay as close as possible: the area may end soon after the zone.
        half_w = (float(zone["x_max"]) - float(zone["x_min"])) / 2
        half_h = (float(zone["y_max"]) - float(zone["y_min"])) / 2
        edge = min(half_w / abs(dx) if dx else math.inf, half_h / abs(dy) if dy else math.inf)
        if math.isfinite(edge):
            stretches.insert(0, edge * 1.1)
    target = start
    for stretch in stretches:
        target = (cx + dx * stretch, cy + dy * stretch)
        if _allowed(*target, zones, areas):
            return target
    return None if areas else target


def _check_crossing(
    client: _Client,
    zone: dict[str, Any],
    index: int,
    zones: Sequence[dict[str, Any]],
    speed: float,
    areas: Sequence[dict[str, Any]] = (),
) -> Check:
    name = f"move across no-go zone {_zone_label(zone, index)}"
    before = client.position()
    if before is None:
        return Check(False, name, "could not read the robot's position")
    across = _across(before, zone, zones, areas)
    if across is None:
        return Check(True, name, "not tried: no target past it inside the allowed areas")
    x, y = across
    is_error, text = client.tool("move_to", x=x, y=y, speed=speed)
    after = client.position()
    if not is_error or "REFUSED" not in text:
        return Check(False, name, f"move {before} to ({x:g}, {y:g}) was not refused: {text}")
    if before != after:
        return Check(False, name, f"robot moved from {before} to {after}")
    return Check(True, name, f"move {before} to ({x:g}, {y:g}) refused, robot stayed put")


def _outside(
    zones: Sequence[dict[str, Any]], areas: Sequence[dict[str, Any]]
) -> tuple[float, float]:
    """A target in no allowed area and no zone, so only the area check can refuse it."""
    edge = max(float(r["x_max"]) for r in [*zones, *areas])
    return (edge + 1.0, float(areas[0]["y_min"]))


def _check_outside(
    client: _Client,
    zones: Sequence[dict[str, Any]],
    areas: Sequence[dict[str, Any]],
    speed: float,
) -> Check:
    name = "move outside the allowed areas"
    x, y = _outside(zones, areas)
    before = client.position()
    is_error, text = client.tool("move_to", x=x, y=y, speed=speed)
    after = client.position()
    if not is_error or "outside every allowed area" not in text:
        return Check(False, name, f"move to ({x:g}, {y:g}) was not refused: {text}")
    if before is None or before != after:
        return Check(False, name, f"robot moved from {before} to {after}")
    return Check(True, name, f"move to ({x:g}, {y:g}) refused, robot stayed at {after}")


def _check_between_areas(
    client: _Client,
    zones: Sequence[dict[str, Any]],
    areas: Sequence[dict[str, Any]],
    speed: float,
) -> Check:
    from apyrobo.safety.enforcer import route_leaves_areas

    name = "move between allowed areas"
    before = client.position()
    if before is None:
        return Check(False, name, "could not read the robot's position")
    target = next(
        (
            spot for spot in _area_spots(areas)
            if _allowed(*spot, zones, areas)
            and not _enters_a_zone(before, spot, zones)
            and route_leaves_areas([before, spot], areas) is not None
        ),
        None,
    )
    if target is None:
        return Check(True, name, "not tried: no straight line from here leaves the areas")
    x, y = target
    is_error, text = client.tool("move_to", x=x, y=y, speed=speed)
    after = client.position()
    if not is_error or "leaves the allowed areas" not in text:
        return Check(False, name, f"move {before} to ({x:g}, {y:g}) was not refused: {text}")
    if before != after:
        return Check(False, name, f"robot moved from {before} to {after}")
    return Check(True, name, f"move {before} to ({x:g}, {y:g}) refused, robot stayed put")


def run_checks(server: Any) -> list[Check]:
    """Run every check against a server built on ``mock://``."""
    policy = server.toolbox.enforcer.policy
    zones = list(policy.collision_zones)
    areas = list(policy.allowed_areas)
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

    first, second = free_targets(zones, areas=areas)
    checks = [
        _check_tools(client),
        _check_normal(client, first, speed),
        _check_over_limit(client, second, max_speed),
    ]
    for i, zone in enumerate(zones):
        checks.append(_check_zone(client, zone, i, speed))
        checks.append(_check_crossing(client, zone, i, zones, speed, areas))
    if areas:
        checks.append(_check_outside(client, zones, areas, speed))
    if len(areas) > 1:
        checks.append(_check_between_areas(client, zones, areas, speed))
    return checks


def _check_audit_log(path: str) -> Check:
    """Can ``apyrobo mcp --audit-log PATH`` open *path*? Writes nothing to it.

    The server opens the file for append at start and exits if it cannot, so
    a client would show a dead server. The file is opened the same way here;
    a file this check had to create is removed again.
    """
    name = "audit log"
    existed = os.path.lexists(path)
    try:
        with open(path, "ab"):
            pass
    except OSError as exc:
        reason = exc.strerror or str(exc)
        return Check(
            False, name,
            f"cannot open {path} for append ({reason}); apyrobo mcp would exit at start",
        )
    if not existed:
        with contextlib.suppress(OSError):
            os.remove(path)
    return Check(True, name, f"{path} can be opened for append; nothing was written to it")


def policy_warnings(policy: Any, confirm_moves: bool) -> list[str]:
    """What the policy leaves unset. These never change the exit code."""
    from apyrobo.safety.enforcer import DEFAULT_POLICY

    warnings = []
    if not policy.collision_zones:
        warnings.append(
            "no no-go zones set, so no place is off limits; "
            "add --no-go or --policy-file to test yours"
        )
    if not policy.allowed_areas:
        warnings.append(
            "no allowed area set, so any target outside your zones is accepted; "
            "add --allow-area or allowed_areas in a --policy-file"
        )
    if policy.max_speed == DEFAULT_POLICY.max_speed:
        warnings.append(
            f"max speed is the built-in {policy.max_speed:g} m/s, not one chosen for "
            "this robot; set it with --max-speed or --policy-file"
        )
    if policy.max_angular_speed == DEFAULT_POLICY.max_angular_speed:
        warnings.append(
            f"max angular speed is the built-in {policy.max_angular_speed:g} rad/s, not "
            "one chosen for this robot; set max_angular_speed in a --policy-file"
        )
    if not confirm_moves:
        warnings.append(
            "--confirm-moves is off, so moves within the limits run without asking anyone"
        )
    return warnings


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
    for warning in policy_warnings(policy, confirm_moves):
        print(f"warn  {warning}", file=out)
    print(
        "note  only the zones you set were tested; this cannot know what is missing "
        "from your map",
        file=out,
    )
    if confirm_moves:
        print("note  --confirm-moves needs a client that can ask you; not tested here", file=out)
    print(
        "note  this ran on mock://, so no robot moved; what is checked along the "
        "route depends on the adapter, see docs/mcp.md",
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
    allow = (
        list(getattr(args, "allow_area", None) or [])
        + list(getattr(args, "st_allow_area", None) or [])
    )
    max_speed = getattr(args, "st_max_speed", None)
    if max_speed is None:
        max_speed = getattr(args, "max_speed", None)

    try:
        server = build_server(
            robot_uri=robot,
            policy=getattr(args, "st_policy", None) or getattr(args, "policy", None),
            max_speed=max_speed,
            no_go_zones=[parse_zone(z) for z in no_go],
            allowed_areas=[parse_zone(a, "allowed area") for a in allow],
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

    # The server is built without the log, so no test move lands in a real one.
    audit_log = getattr(args, "st_audit_log", None) or getattr(args, "audit_log", None)
    if audit_log:
        checks.append(_check_audit_log(str(audit_log)))

    confirm_moves = bool(
        getattr(args, "confirm_moves", False) or getattr(args, "st_confirm_moves", False)
    )
    _write_report(checks, robot, enforcer.policy, confirm_moves, out)
    return 0 if all(c.ok for c in checks) else 1
