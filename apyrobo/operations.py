"""
Operations — battery management, map management, teleoperation, and webhooks.

These are the "glue" systems that real deployments need but frameworks
usually omit.

Modules:
    - BatteryMonitor: tracks charge, estimates range, triggers return-to-dock
    - MapManager: load/switch/SLAM maps, multi-floor support
    - TeleoperationBridge: human manual override via velocity commands
    - WebhookEmitter: notify external systems (Slack, dashboards, elevators)
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib import request as urllib_request

logger = logging.getLogger(__name__)

try:
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import BatteryState
    from geometry_msgs.msg import Twist
    _HAS_ROS2 = True
except Exception:
    _HAS_ROS2 = False


# =====================================================================
# Battery / Resource Management
# =====================================================================

class BatteryMonitor:
    """
    Tracks robot battery state and makes energy-aware decisions.

    Subscribes to battery topics (or accepts manual updates) and
    estimates remaining range. Triggers automatic return-to-dock
    when battery drops below threshold.

    Usage:
        battery = BatteryMonitor(robot_id="tb4", dock_position=(0, 0))
        battery.update(voltage=12.1, percentage=45.0)
        if not battery.can_complete_trip(distance_m=50.0):
            # don't assign this task
    """

    # Defaults for TurtleBot4
    LOW_BATTERY_PCT = 20.0
    CRITICAL_BATTERY_PCT = 10.0
    METERS_PER_PERCENT = 2.0  # rough estimate: 200m on full charge

    def __init__(
        self,
        robot_id: str,
        dock_position: tuple[float, float] = (0.0, 0.0),
        low_threshold: float = 20.0,
        critical_threshold: float = 10.0,
        meters_per_percent: float = 2.0,
    ) -> None:
        self.robot_id = robot_id
        self.dock_position = dock_position
        self.low_threshold = low_threshold
        self.critical_threshold = critical_threshold
        self.meters_per_percent = meters_per_percent

        self.percentage: float = 100.0
        self.voltage: float | None = None
        self.is_charging: bool = False
        self.is_docked: bool = False
        self._last_update: float = time.time()
        self._callbacks: list[Callable[[str, float], None]] = []
        self._return_to_dock_cb: Callable[[], None] | None = None
        self._ros_battery_sub: Any = None
        self._ros_node: Any = None

    def update(self, percentage: float | None = None, voltage: float | None = None,
               is_charging: bool | None = None) -> None:
        """Update battery state (from ROS 2 topic or manual)."""
        if percentage is not None:
            self.percentage = max(0.0, min(100.0, percentage))
        if voltage is not None:
            self.voltage = voltage
        if is_charging is not None:
            self.is_charging = is_charging
        self._last_update = time.time()

        # Check thresholds
        if self.percentage <= self.critical_threshold and not self.is_charging:
            self._notify("critical", self.percentage)
        elif self.percentage <= self.low_threshold and not self.is_charging:
            self._notify("low", self.percentage)

    @property
    def estimated_range_m(self) -> float:
        """Estimated remaining range in metres."""
        return self.percentage * self.meters_per_percent

    def can_complete_trip(self, distance_m: float, robot_position: tuple[float, float] | None = None) -> bool:
        """Can the robot complete a trip AND return to dock?"""
        # Distance to dock from task endpoint (approximate as same as outbound)
        if robot_position:
            dock_dist = math.sqrt(
                (robot_position[0] - self.dock_position[0])**2 +
                (robot_position[1] - self.dock_position[1])**2
            )
        else:
            dock_dist = distance_m  # worst case: same distance back

        total_needed = distance_m + dock_dist
        safety_margin = total_needed * 0.2  # 20% safety margin
        return self.estimated_range_m >= total_needed + safety_margin

    @property
    def status(self) -> str:
        if self.is_charging:
            return "charging"
        if self.percentage <= self.critical_threshold:
            return "critical"
        if self.percentage <= self.low_threshold:
            return "low"
        return "ok"

    def on_threshold(self, callback: Callable[[str, float], None]) -> None:
        """Register callback for battery threshold events."""
        self._callbacks.append(callback)

    def _notify(self, level: str, pct: float) -> None:
        logger.warning("Battery %s: %s (%.1f%%)", self.robot_id, level, pct)
        for cb in self._callbacks:
            try:
                cb(level, pct)
            except Exception:
                pass

    def set_return_to_dock_callback(self, callback: Callable[[], None]) -> None:
        """Set callback fired when battery reaches critical threshold (OP-04)."""
        self._return_to_dock_cb = callback

    def evaluate_return_to_dock(self) -> bool:
        """Trigger return-to-dock action if battery is critical and not charging."""
        if self.percentage <= self.critical_threshold and not self.is_charging:
            if self._return_to_dock_cb is not None:
                self._return_to_dock_cb()
                return True
        return False

    def attach_ros2(self, node: Any, topic: str = "/battery_state", qos_depth: int = 10,
                    reliability: str = "best_effort") -> bool:
        """OP-01: subscribe to ROS2 battery topic when rclpy is available."""
        if not _HAS_ROS2:
            logger.warning("ROS2 unavailable; cannot attach battery subscriber")
            return False
        rel = ReliabilityPolicy.RELIABLE if reliability == "reliable" else ReliabilityPolicy.BEST_EFFORT
        qos = QoSProfile(reliability=rel, durability=DurabilityPolicy.VOLATILE, depth=qos_depth)

        def _cb(msg: Any) -> None:
            pct = float(getattr(msg, "percentage", 0.0)) * (100.0 if getattr(msg, "percentage", 0.0) <= 1.0 else 1.0)
            voltage = float(getattr(msg, "voltage", 0.0)) if hasattr(msg, "voltage") else None
            is_charging = bool(getattr(msg, "power_supply_status", 0) == 1) if hasattr(msg, "power_supply_status") else None
            self.update(percentage=pct, voltage=voltage, is_charging=is_charging)
            self.evaluate_return_to_dock()

        self._ros_battery_sub = node.create_subscription(BatteryState, topic, _cb, qos)
        self._ros_node = node
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "percentage": self.percentage,
            "voltage": self.voltage,
            "status": self.status,
            "estimated_range_m": round(self.estimated_range_m, 1),
            "is_charging": self.is_charging,
            "dock_position": list(self.dock_position),
        }

    def __repr__(self) -> str:
        return f"<Battery {self.robot_id}: {self.percentage:.0f}% ({self.status}) range={self.estimated_range_m:.0f}m>"


# =====================================================================
# Map Management
# =====================================================================

class MapManager:
    """
    Manages maps for navigation — loading, switching, multi-floor.

    In production, interfaces with Nav2 map_server to load/swap maps
    and with slam_toolbox for live mapping.

    Usage:
        maps = MapManager(maps_dir="/workspace/maps")
        maps.register("warehouse_floor1", "/workspace/maps/floor1.yaml")
        maps.set_active("warehouse_floor1")
        current = maps.active_map
    """

    def __init__(self, maps_dir: str | Path | None = None) -> None:
        self._maps: dict[str, dict[str, Any]] = {}
        self._active: str | None = None
        self._maps_dir = Path(maps_dir) if maps_dir else None

        # Auto-discover maps
        if self._maps_dir and self._maps_dir.exists():
            self._discover_maps()

    def register(self, name: str, yaml_path: str | Path, floor: int = 0,
                 metadata: dict[str, Any] | None = None) -> None:
        """Register a map file."""
        self._maps[name] = {
            "name": name,
            "yaml_path": str(yaml_path),
            "floor": floor,
            "metadata": metadata or {},
            "registered_at": time.time(),
        }
        logger.info("MapManager: registered map %s (floor %d)", name, floor)

    def set_active(self, name: str) -> None:
        """Switch the active map. In production, tells Nav2 map_server to load it."""
        if name not in self._maps:
            raise ValueError(f"Unknown map: {name!r}. Available: {list(self._maps)}")
        self._active = name
        logger.info("MapManager: active map → %s", name)

    def load_map_ros2(self, map_yaml: str) -> bool:
        """OP-05: load map through Nav2 map_server service CLI."""
        cmd = ["ros2", "service", "call", "/map_server/load_map", "nav2_msgs/srv/LoadMap", f"{{map_url: '{map_yaml}'}}"]
        try:
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            logger.warning("Map load failed: %s", e)
            return False

    def save_map_ros2(self, map_name: str, output_dir: str | None = None) -> bool:
        """OP-05: save map through map_saver_cli."""
        target = Path(output_dir) / map_name if output_dir else Path(map_name)
        cmd = ["ros2", "run", "nav2_map_server", "map_saver_cli", "-f", str(target)]
        try:
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            logger.warning("Map save failed: %s", e)
            return False

    @property
    def active_map(self) -> dict[str, Any] | None:
        if self._active:
            return self._maps.get(self._active)
        return None

    @property
    def active_map_name(self) -> str | None:
        return self._active

    def get_floor_map(self, floor: int) -> dict[str, Any] | None:
        """Get the map for a specific floor."""
        for m in self._maps.values():
            if m["floor"] == floor:
                return m
        return None

    @property
    def available_maps(self) -> list[str]:
        return list(self._maps.keys())

    def _discover_maps(self) -> None:
        """Auto-discover .yaml map files in the maps directory."""
        if not self._maps_dir:
            return
        for yaml_file in self._maps_dir.glob("*.yaml"):
            name = yaml_file.stem
            if name not in self._maps:
                self.register(name, yaml_file)

    def __repr__(self) -> str:
        return f"<MapManager maps={len(self._maps)} active={self._active!r}>"


# =====================================================================
# Teleoperation Bridge
# =====================================================================

class TeleoperationMode(str):
    AUTONOMOUS = "autonomous"
    TELEOP = "teleop"
    SHARED = "shared"  # human + autonomy together


class TeleoperationBridge:
    """
    Enables human manual override of robot control.

    When teleoperation is active, the autonomy stack is paused and
    a human operator sends velocity commands directly. Used for:
    - Initial map building (teleoperate to SLAM)
    - Recovery when the robot is stuck
    - Situations requiring human judgment

    Safety: the SafetyEnforcer still wraps teleop commands.

    Usage:
        teleop = TeleoperationBridge(robot_id="tb4")
        teleop.enable()
        teleop.send_velocity(linear=0.3, angular=0.0)
        teleop.disable()  # returns control to autonomy
    """

    def __init__(self, robot_id: str) -> None:
        self.robot_id = robot_id
        self.mode = TeleoperationMode.AUTONOMOUS
        self._enabled = False
        self._operator_id: str | None = None
        self._command_log: list[dict[str, Any]] = []
        self._velocity_callback: Callable[[float, float], None] | None = None
        self._lock = threading.Lock()

    def set_velocity_callback(self, callback: Callable[[float, float], None]) -> None:
        """Set the function that sends velocity commands to the robot."""
        self._velocity_callback = callback

    def attach_ros2_publisher(self, node: Any, topic: str = "/cmd_vel", qos_depth: int = 10) -> bool:
        """OP-02: wire teleop to ROS2 cmd_vel publishing."""
        if not _HAS_ROS2:
            logger.warning("ROS2 unavailable; cannot attach teleop publisher")
            return False
        pub = node.create_publisher(Twist, topic, qos_depth)

        def _send(linear: float, angular: float) -> None:
            msg = Twist()
            msg.linear.x = float(linear)
            msg.angular.z = float(angular)
            pub.publish(msg)

        self._velocity_callback = _send
        return True

    def enable(self, operator_id: str = "operator") -> None:
        """Switch to teleoperation mode."""
        with self._lock:
            self._enabled = True
            self.mode = TeleoperationMode.TELEOP
            self._operator_id = operator_id
        logger.warning("TELEOP ENABLED for %s by %s", self.robot_id, operator_id)

    def disable(self) -> None:
        """Return to autonomous mode."""
        with self._lock:
            # Send zero velocity first
            if self._velocity_callback:
                self._velocity_callback(0.0, 0.0)
            self._enabled = False
            self.mode = TeleoperationMode.AUTONOMOUS
            self._operator_id = None
        logger.info("TELEOP DISABLED for %s — returning to autonomous", self.robot_id)

    def send_velocity(self, linear: float, angular: float) -> bool:
        """
        Send a velocity command during teleoperation.

        Returns False if teleop is not enabled.
        """
        if not self._enabled:
            logger.warning("Teleop command rejected — not in teleop mode")
            return False

        self._command_log.append({
            "linear": linear, "angular": angular,
            "operator": self._operator_id, "timestamp": time.time(),
        })

        if self._velocity_callback:
            self._velocity_callback(linear, angular)
            return True
        else:
            logger.warning("No velocity callback set — command not sent")
            return False

    @property
    def is_active(self) -> bool:
        return self._enabled

    @property
    def operator(self) -> str | None:
        return self._operator_id

    @property
    def command_count(self) -> int:
        return len(self._command_log)

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "mode": self.mode,
            "is_active": self._enabled,
            "operator": self._operator_id,
            "commands_sent": len(self._command_log),
        }

    def __repr__(self) -> str:
        return f"<Teleop {self.robot_id}: {self.mode} operator={self._operator_id}>"


# =====================================================================
# Webhook / External Event System
# =====================================================================

class WebhookTarget:
    """A registered webhook endpoint."""

    def __init__(self, name: str, url: str, events: list[str] | None = None,
                 headers: dict[str, str] | None = None) -> None:
        self.name = name
        self.url = url
        self.events = set(events) if events else None  # None = all events
        self.headers = headers or {}
        self.enabled = True
        self.failure_count = 0
        self.last_success: float | None = None

    def should_receive(self, event_type: str) -> bool:
        if not self.enabled:
            return False
        if self.events is None:
            return True
        return event_type in self.events

    def __repr__(self) -> str:
        return f"<Webhook {self.name} → {self.url} events={self.events or 'all'}>"


class WebhookEmitter:
    """
    Sends events to external systems via webhooks.

    Supports: HTTP POST endpoints, Slack incoming webhooks,
    and custom callback functions.

    Events: task_completed, task_failed, safety_violation,
    battery_low, robot_stuck, swarm_update, teleop_started, etc.

    Usage:
        webhooks = WebhookEmitter()
        webhooks.add_target("slack", "https://hooks.slack.com/...",
                           events=["task_completed", "safety_violation"])
        webhooks.add_callback("logger", lambda e: print(e))

        # Emit from anywhere in the system
        webhooks.emit("task_completed", task_id="t1", robot="tb4", duration=12.5)
    """

    def __init__(self, retry_count: int = 3, retry_backoff_s: float = 0.5) -> None:
        self._targets: dict[str, WebhookTarget] = {}
        self._callbacks: dict[str, Callable[[dict[str, Any]], None]] = {}
        self._event_log: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._retry_count = retry_count
        self._retry_backoff_s = retry_backoff_s

    def add_target(self, name: str, url: str, events: list[str] | None = None,
                   headers: dict[str, str] | None = None) -> None:
        """Register a webhook HTTP endpoint."""
        self._targets[name] = WebhookTarget(name, url, events, headers)
        logger.info("Webhook: registered target %s → %s", name, url)

    def add_callback(self, name: str, callback: Callable[[dict[str, Any]], None],
                     events: list[str] | None = None) -> None:
        """Register a local callback (for in-process consumers)."""
        self._callbacks[name] = callback

    def remove_target(self, name: str) -> None:
        self._targets.pop(name, None)
        self._callbacks.pop(name, None)

    def emit(self, event_type: str, **data: Any) -> None:
        """
        Emit an event to all matching targets.

        HTTP targets are called in a background thread so emit() never blocks.
        Callbacks are called synchronously.
        """
        payload = {
            "event_type": event_type,
            "timestamp": time.time(),
            "data": data,
        }

        with self._lock:
            self._event_log.append(payload)

        # Local callbacks (synchronous)
        for name, cb in self._callbacks.items():
            try:
                cb(payload)
            except Exception as e:
                logger.warning("Webhook callback %s error: %s", name, e)

        # HTTP targets (async via thread)
        for target in self._targets.values():
            if target.should_receive(event_type):
                threading.Thread(
                    target=self._send_http, args=(target, payload), daemon=True
                ).start()

    def _send_http(self, target: WebhookTarget, payload: dict[str, Any]) -> None:
        """Send an HTTP POST to a webhook target with retry (OP-03)."""
        for attempt in range(1, self._retry_count + 1):
            try:
                req = urllib_request.Request(
                    target.url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", **target.headers},
                    method="POST",
                )
                with urllib_request.urlopen(req, timeout=10):
                    target.last_success = time.time()
                    target.failure_count = 0
                    return
            except Exception as e:
                target.failure_count += 1
                logger.warning(
                    "Webhook %s failed attempt %d/%d: %s",
                    target.name, attempt, self._retry_count, e,
                )
                time.sleep(self._retry_backoff_s * attempt)

        if target.failure_count >= 5:
            target.enabled = False
            logger.error("Webhook %s disabled after %d failures", target.name, target.failure_count)

    def format_slack(self, event_type: str, **data: Any) -> dict[str, Any]:
        """Format a payload for Slack incoming webhooks."""
        icon = {
            "task_completed": ":white_check_mark:",
            "task_failed": ":x:",
            "safety_violation": ":rotating_light:",
            "battery_low": ":battery:",
            "teleop_started": ":joystick:",
        }.get(event_type, ":robot_face:")

        text = f"{icon} *{event_type}*"
        for k, v in data.items():
            text += f"\n• {k}: {v}"

        return {"text": text}

    def add_slack_target(self, name: str, webhook_url: str,
                         events: list[str] | None = None) -> None:
        """OP-07: register Slack incoming webhook target."""
        self.add_target(name=name, url=webhook_url, events=events)

    def add_teams_target(self, name: str, webhook_url: str,
                         events: list[str] | None = None) -> None:
        """OP-07: register Microsoft Teams webhook target."""
        self.add_target(name=name, url=webhook_url, events=events)

    @property
    def event_log(self) -> list[dict[str, Any]]:
        return list(self._event_log)

    @property
    def target_count(self) -> int:
        return len(self._targets) + len(self._callbacks)

    def __repr__(self) -> str:
        return f"<WebhookEmitter targets={self.target_count} events={len(self._event_log)}>"


# =====================================================================
# Scheduling / API / Dashboard
# =====================================================================


def _parse_cron_to_seconds(expr: str) -> float:
    """Parse a simple cron expression and return the interval in seconds.

    Supported patterns:
        '*/N * * * *'  -> every N minutes
        '0 N * * *'    -> every 24 hours (daily at hour N)
        '0 */N * * *'  -> every N hours
        '* * * * *'    -> every 60 seconds (every minute)
    Falls back to 3600 (1 hour) for unrecognised expressions.
    """
    parts = expr.strip().split()
    if len(parts) != 5:
        return 3600.0

    minute, hour = parts[0], parts[1]

    # */N * * * * -> every N minutes
    if minute.startswith("*/") and hour == "*":
        try:
            return float(minute[2:]) * 60
        except ValueError:
            return 3600.0

    # 0 */N * * * -> every N hours
    if minute == "0" and hour.startswith("*/"):
        try:
            return float(hour[2:]) * 3600
        except ValueError:
            return 3600.0

    # 0 N * * * -> daily (every 24h)
    if minute == "0" and hour.isdigit():
        return 86400.0

    # * * * * * -> every minute
    if minute == "*" and hour == "*":
        return 60.0

    return 3600.0


class ScheduledTaskRunner:
    """OP-06: cron-like periodic task runner for skill/task callbacks.

    Supports two registration modes:
        - ``add_interval_job(name, interval_s, fn)`` — run a plain callable
          on a fixed interval (simple mode, no agent needed).
        - ``add_task(name, cron_expr, task_description, robot, agent)`` —
          run ``agent.execute(task_description, robot)`` on a cron schedule,
          storing results in an optional StateStore.
    """

    def __init__(self, state_store: Any | None = None) -> None:
        self._jobs: list[dict[str, Any]] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._state_store = state_store

    # -- simple interval jobs (backwards-compatible) -------------------------

    def add_interval_job(self, name: str, interval_s: float, fn: Callable[[], None]) -> None:
        self._jobs.append({
            "name": name, "interval_s": interval_s, "fn": fn,
            "next": time.time() + interval_s, "mode": "fn",
        })

    # -- agent-based cron tasks (OP-02) --------------------------------------

    def add_task(
        self,
        name: str,
        cron_expr: str,
        task_description: str,
        robot: Any,
        agent: Any,
    ) -> None:
        """Register a periodic task that runs ``agent.execute(task_description, robot)``."""
        interval_s = _parse_cron_to_seconds(cron_expr)
        self._jobs.append({
            "name": name,
            "interval_s": interval_s,
            "task": task_description,
            "robot": robot,
            "agent": agent,
            "next": time.time() + interval_s,
            "mode": "agent",
        })

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.is_set():
                now = time.time()
                for job in self._jobs:
                    if now >= job["next"]:
                        self._execute(job, now)
                        job["next"] = now + float(job["interval_s"])
                self._stop.wait(0.1)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    # -- internal execution --------------------------------------------------

    def _execute(self, job: dict[str, Any], now: float) -> None:
        from apyrobo.observability import emit_event

        name = job["name"]
        if job["mode"] == "fn":
            try:
                job["fn"]()
                emit_event("scheduled_task_run", task_name=name, status="success")
            except Exception as e:
                logger.warning("Scheduled job %s failed: %s", name, e)
                emit_event("scheduled_task_run", task_name=name, status="error", error=str(e))
        else:
            # agent mode
            try:
                result = job["agent"].execute(job["task"], job["robot"])
                if self._state_store and hasattr(self._state_store, "set"):
                    self._state_store.set(f"scheduled:{name}:last_result", {
                        "status": getattr(result, "status", "completed"),
                        "timestamp": now,
                    })
                emit_event(
                    "scheduled_task_run",
                    task_name=name,
                    status="success",
                    result_status=getattr(result, "status", "completed"),
                )
            except Exception as e:
                logger.warning("Scheduled task %s failed: %s", name, e)
                emit_event("scheduled_task_run", task_name=name, status="error", error=str(e))


class OperationsApiServer:
    """OP-08: REST API server for fleet operations.

    Endpoints:
        GET  /health        → {"status": "ok"}
        GET  /robots        → list of robots with capabilities
        POST /tasks         → submit a task (returns 202 with task_id)
        GET  /tasks/{id}    → task status
        DELETE /tasks/{id}  → cancel a task

    Supports optional API key authentication via ``auth_manager``,
    SwarmBus integration for robot discovery, and StateStore for
    persisting task state.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8081,
        auth_manager: Any | None = None,
        swarm_bus: Any | None = None,
        state_store: Any | None = None,
        agent: Any | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self._auth = auth_manager
        self._bus = swarm_bus
        self._store = state_store
        self._agent = agent
        self._tasks: dict[str, dict[str, Any]] = {}
        self._robots: list[dict[str, Any]] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def set_robots(self, robots: list[dict[str, Any]]) -> None:
        self._robots = robots

    def _get_robots_list(self) -> list[dict[str, Any]]:
        """Return robots from SwarmBus if available, else static list."""
        if self._bus:
            result = []
            for rid in self._bus.robot_ids:
                entry: dict[str, Any] = {"id": rid}
                try:
                    cap = self._bus.get_capabilities(rid)
                    entry["capabilities"] = cap.model_dump() if hasattr(cap, "model_dump") else cap.dict()
                except Exception:
                    entry["capabilities"] = {}
                result.append(entry)
            return result
        return list(self._robots)

    def _check_auth(self, handler: Any) -> bool:
        """Return True if request is authorised (or no auth configured)."""
        if not self._auth:
            return True
        api_key = handler.headers.get("X-API-Key", "")
        if not api_key:
            return False
        user = self._auth.authenticate(api_key)
        return user is not None

    def _get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """Look up task in memory, falling back to StateStore."""
        if task_id in self._tasks:
            return self._tasks[task_id]
        if self._store and hasattr(self._store, "get_task"):
            entry = self._store.get_task(task_id)
            if entry:
                return entry.to_dict() if hasattr(entry, "to_dict") else {"task_id": task_id}
        return None

    def _run_task_background(self, task_id: str, task_desc: str, robot_id: str | None) -> None:
        """Execute a task via the agent in a background thread."""
        self._tasks[task_id]["status"] = "running"
        if self._store and hasattr(self._store, "begin_task"):
            self._store.begin_task(task_id, {"task": task_desc, "robot_id": robot_id})

        try:
            robot = None
            if robot_id and self._bus:
                try:
                    robot = self._bus.get_robot(robot_id)
                except KeyError:
                    pass

            if self._agent and robot:
                result = self._agent.execute(task_desc, robot, state_store=self._store)
                self._tasks[task_id]["status"] = getattr(result, "status", "completed")
                self._tasks[task_id]["result"] = getattr(result, "to_dict", lambda: {})()
            else:
                # No agent or robot — just mark accepted (manual processing)
                self._tasks[task_id]["status"] = "completed"

            if self._store and hasattr(self._store, "complete_task"):
                self._store.complete_task(task_id, self._tasks[task_id].get("result"))
        except Exception as e:
            logger.warning("Background task %s failed: %s", task_id, e)
            self._tasks[task_id]["status"] = "failed"
            self._tasks[task_id]["error"] = str(e)
            if self._store and hasattr(self._store, "fail_task"):
                self._store.fail_task(task_id, str(e))

    def start(self) -> None:
        import uuid

        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_body(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    return json.loads(raw.decode("utf-8"))
                except Exception:
                    return {}

            def do_GET(self):  # noqa: N802
                if not outer._check_auth(self):
                    self._send(401, {"error": "unauthorized"})
                    return

                if self.path == "/health":
                    self._send(200, {"status": "ok"})
                elif self.path == "/robots":
                    self._send(200, {"robots": outer._get_robots_list()})
                elif self.path.startswith("/tasks/"):
                    task_id = self.path[len("/tasks/"):]
                    info = outer._get_task_status(task_id)
                    if info:
                        self._send(200, info)
                    else:
                        self._send(404, {"error": "task_not_found"})
                else:
                    self._send(404, {"error": "not_found"})

            def do_POST(self):  # noqa: N802
                if not outer._check_auth(self):
                    self._send(401, {"error": "unauthorized"})
                    return

                if self.path != "/tasks":
                    self._send(404, {"error": "not_found"})
                    return

                payload = self._read_body()
                task_id = uuid.uuid4().hex[:10]
                task_desc = payload.get("task", "")
                robot_id = payload.get("robot_id")

                outer._tasks[task_id] = {
                    "task_id": task_id,
                    "task": task_desc,
                    "robot_id": robot_id,
                    "status": "queued",
                    "received_at": time.time(),
                }

                # Run in background thread
                t = threading.Thread(
                    target=outer._run_task_background,
                    args=(task_id, task_desc, robot_id),
                    daemon=True,
                )
                t.start()

                self._send(202, {"task_id": task_id, "status": "queued"})

            def do_DELETE(self):  # noqa: N802
                if not outer._check_auth(self):
                    self._send(401, {"error": "unauthorized"})
                    return

                if not self.path.startswith("/tasks/"):
                    self._send(404, {"error": "not_found"})
                    return

                task_id = self.path[len("/tasks/"):]
                if task_id in outer._tasks:
                    outer._tasks[task_id]["status"] = "cancelled"
                    self._send(200, {"task_id": task_id, "status": "cancelled"})
                else:
                    self._send(404, {"error": "task_not_found"})

            def log_message(self, fmt: str, *args: Any) -> None:  # silence std logging
                return

        self._server = ThreadingHTTPServer((self.host, self.port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()


class FleetDashboard:
    """OP-09: simple web dashboard payload provider for task/robot status."""

    def __init__(self) -> None:
        self._robot_status: dict[str, Any] = {}
        self._task_status: dict[str, Any] = {}
        self._events: list[dict[str, Any]] = []

    def update_robot(self, robot_id: str, status: dict[str, Any]) -> None:
        self._robot_status[robot_id] = status
        self._events.append({"type": "robot", "robot_id": robot_id, "status": status, "t": time.time()})

    def update_task(self, task_id: str, status: dict[str, Any]) -> None:
        self._task_status[task_id] = status
        self._events.append({"type": "task", "task_id": task_id, "status": status, "t": time.time()})

    def snapshot(self) -> dict[str, Any]:
        return {
            "robots": dict(self._robot_status),
            "tasks": dict(self._task_status),
            "events": list(self._events[-200:]),
        }
