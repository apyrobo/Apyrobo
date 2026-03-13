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
import time
import threading
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


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
        # TODO: In Docker, call map_server load_map service
        # ros2 service call /map_server/load_map nav2_msgs/srv/LoadMap "{map_url: path}"

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

    def __init__(self) -> None:
        self._targets: dict[str, WebhookTarget] = {}
        self._callbacks: dict[str, Callable[[dict[str, Any]], None]] = {}
        self._event_log: list[dict[str, Any]] = []
        self._lock = threading.Lock()

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
        """Send an HTTP POST to a webhook target."""
        try:
            import urllib.request
            req = urllib.request.Request(
                target.url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", **target.headers},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                target.last_success = time.time()
                target.failure_count = 0
        except Exception as e:
            target.failure_count += 1
            logger.warning("Webhook %s failed (%d): %s", target.name, target.failure_count, e)
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

    @property
    def event_log(self) -> list[dict[str, Any]]:
        return list(self._event_log)

    @property
    def target_count(self) -> int:
        return len(self._targets) + len(self._callbacks)

    def __repr__(self) -> str:
        return f"<WebhookEmitter targets={self.target_count} events={len(self._event_log)}>"
