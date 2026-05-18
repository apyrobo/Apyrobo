"""Live telemetry context injection for LLM-based planning.

``TelemetryContextProvider`` samples a connected robot's current state
(position, battery, velocity, active errors) and formats it as a compact
context string that is prepended to the LLM's planning prompt on every call.

This means the planner always knows where the robot is, how much power is
left, and whether any sensors are reporting faults — without the LLM needing
to make explicit tool calls.

Usage::

    from apyrobo.skills.telemetry_context import TelemetryContextProvider
    from apyrobo.skills.agent import Agent
    from apyrobo.core.robot import Robot

    robot = Robot.discover("ros2://turtlebot4")
    telemetry = TelemetryContextProvider(robot, refresh_interval=5.0)

    agent = Agent(provider="llm", model="claude-haiku-4-5-20251001",
                  telemetry_provider=telemetry)

    graph = agent.plan("navigate to the charging dock", robot)
    # The LLM prompt now includes:
    # [Robot State — sampled 0.1s ago]
    # Position: x=1.23, y=-0.45  Heading: 90°
    # Battery:  78%  (discharging)
    # Velocity: linear=0.0 m/s, angular=0.0 rad/s
    # Sensors:  camera OK, lidar OK
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class TelemetrySnapshot:
    """One point-in-time sample of robot sensor data."""

    __slots__ = (
        "timestamp",
        "position",
        "heading_deg",
        "battery_pct",
        "linear_velocity",
        "angular_velocity",
        "sensor_status",
        "active_errors",
    )

    def __init__(
        self,
        timestamp: float,
        position: tuple[float, float] | None = None,
        heading_deg: float | None = None,
        battery_pct: float | None = None,
        linear_velocity: float | None = None,
        angular_velocity: float | None = None,
        sensor_status: dict[str, str] | None = None,
        active_errors: list[str] | None = None,
    ) -> None:
        self.timestamp = timestamp
        self.position = position
        self.heading_deg = heading_deg
        self.battery_pct = battery_pct
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        self.sensor_status = sensor_status or {}
        self.active_errors = active_errors or []

    def age_seconds(self) -> float:
        return time.monotonic() - self.timestamp

    def to_context_string(self) -> str:
        """Format as a compact LLM-readable context block."""
        age = self.age_seconds()
        lines = [f"[Robot State — sampled {age:.1f}s ago]"]

        if self.position is not None:
            pos_str = f"x={self.position[0]:.2f}, y={self.position[1]:.2f}"
            heading_str = f"  Heading: {self.heading_deg:.0f}°" if self.heading_deg is not None else ""
            lines.append(f"Position: {pos_str}{heading_str}")

        if self.battery_pct is not None:
            lines.append(f"Battery:  {self.battery_pct:.0f}%")

        if self.linear_velocity is not None or self.angular_velocity is not None:
            lin = f"linear={self.linear_velocity:.2f} m/s" if self.linear_velocity is not None else ""
            ang = f"angular={self.angular_velocity:.2f} rad/s" if self.angular_velocity is not None else ""
            vel_parts = [p for p in (lin, ang) if p]
            if vel_parts:
                lines.append(f"Velocity: {', '.join(vel_parts)}")

        if self.sensor_status:
            sensor_str = ", ".join(
                f"{name} {'OK' if status == 'ok' else status}"
                for name, status in self.sensor_status.items()
            )
            lines.append(f"Sensors:  {sensor_str}")

        if self.active_errors:
            lines.append(f"Errors:   {'; '.join(self.active_errors)}")

        return "\n".join(lines)


class TelemetryContextProvider:
    """Samples live robot state and injects it into LLM planning prompts.

    Spawns a background thread that polls the robot at ``refresh_interval``
    seconds. The most recent snapshot is thread-safe to read.

    Parameters
    ----------
    robot:
        A connected ``Robot`` instance. Any robot that exposes ``get_position()``,
        ``get_battery_level()``, etc. is supported; missing methods are skipped.
    refresh_interval:
        How often to poll the robot in seconds (default: 5.0).
    max_snapshot_age:
        If the snapshot is older than this, return a stale-data warning in the
        context string instead of silently using old data (default: 30.0).
    """

    def __init__(
        self,
        robot: Any,
        refresh_interval: float = 5.0,
        max_snapshot_age: float = 30.0,
    ) -> None:
        self._robot = robot
        self._refresh_interval = refresh_interval
        self._max_snapshot_age = max_snapshot_age
        self._snapshot: TelemetrySnapshot | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_background_poll()

    # ------------------------------------------------------------------
    # Background polling
    # ------------------------------------------------------------------

    def _start_background_poll(self) -> None:
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                snap = self._sample()
                with self._lock:
                    self._snapshot = snap
            except Exception as exc:
                logger.debug("TelemetryContextProvider poll error: %s", exc)
            self._stop_event.wait(self._refresh_interval)

    def _sample(self) -> TelemetrySnapshot:
        """Pull current state from the robot. Missing methods are skipped."""
        pos: tuple[float, float] | None = None
        heading: float | None = None
        battery: float | None = None
        lin_vel: float | None = None
        ang_vel: float | None = None
        sensors: dict[str, str] = {}
        errors: list[str] = []

        try:
            raw_pos = self._robot.get_position()
            if raw_pos is not None:
                pos = (float(raw_pos[0]), float(raw_pos[1]))
                if len(raw_pos) > 2:
                    import math
                    heading = math.degrees(float(raw_pos[2])) % 360
        except Exception:
            pass

        try:
            batt = self._robot.get_battery_level()
            if batt is not None:
                battery = float(batt) * 100 if float(batt) <= 1.0 else float(batt)
        except Exception:
            pass

        try:
            vel = self._robot.get_velocity()
            if vel is not None:
                lin_vel = float(vel[0]) if hasattr(vel, "__getitem__") else float(vel)
                ang_vel = float(vel[1]) if hasattr(vel, "__getitem__") and len(vel) > 1 else None
        except Exception:
            pass

        try:
            sensor_data = self._robot.get_sensor_status()
            if isinstance(sensor_data, dict):
                sensors = {k: str(v) for k, v in sensor_data.items()}
        except Exception:
            pass

        try:
            errs = self._robot.get_active_errors()
            if errs:
                errors = [str(e) for e in errs]
        except Exception:
            pass

        return TelemetrySnapshot(
            timestamp=time.monotonic(),
            position=pos,
            heading_deg=heading,
            battery_pct=battery,
            linear_velocity=lin_vel,
            angular_velocity=ang_vel,
            sensor_status=sensors,
            active_errors=errors,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_snapshot(self) -> TelemetrySnapshot | None:
        with self._lock:
            return self._snapshot

    def get_context_string(self) -> str:
        """Return the current telemetry as an LLM context string.

        Returns an empty string if no data has been sampled yet.
        Returns a stale-data warning if the snapshot is older than
        ``max_snapshot_age`` seconds.
        """
        with self._lock:
            snap = self._snapshot

        if snap is None:
            return ""

        if snap.age_seconds() > self._max_snapshot_age:
            return (
                f"[Robot State — WARNING: data is {snap.age_seconds():.0f}s old, "
                "may not reflect current state]"
            )

        return snap.to_context_string()

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
