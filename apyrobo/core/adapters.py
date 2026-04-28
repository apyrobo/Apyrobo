"""
Capability adapters — the bridge between APYROBO's semantic API and hardware.

Each adapter knows how to translate APYROBO calls (move, rotate, stop,
gripper, etc.) into the correct interface for a specific robot platform.

To add support for a new robot:
    1. Subclass CapabilityAdapter
    2. Register it with @register_adapter("scheme")

Adapter contract (AD-01 through AD-05):
    - Navigation: move(), rotate(), stop(), cancel()
    - Gripper: gripper_open(), gripper_close()
    - State: get_position(), get_orientation(), get_health()
    - Lifecycle: connect(), disconnect(), is_connected
"""

from __future__ import annotations

import abc
import json
import logging
import math
import time
from enum import Enum
from typing import Any, Callable

from apyrobo.core.schemas import (
    Capability,
    CapabilityType,
    RobotCapability,
    SensorInfo,
    SensorType,
    AdapterState,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: dict[str, type["CapabilityAdapter"]] = {}


def register_adapter(scheme: str):  # noqa: ANN201
    """Class decorator that registers an adapter for a URI scheme."""

    def decorator(cls: type[CapabilityAdapter]) -> type[CapabilityAdapter]:
        _ADAPTER_REGISTRY[scheme] = cls
        return cls

    return decorator


def get_adapter(scheme: str, robot_name: str, **kwargs: Any) -> "CapabilityAdapter":
    """Instantiate the correct adapter for the given URI scheme."""
    cls = _ADAPTER_REGISTRY.get(scheme)
    if cls is None:
        available = ", ".join(sorted(_ADAPTER_REGISTRY)) or "(none)"
        raise ValueError(
            f"No adapter registered for scheme {scheme!r}. "
            f"Available: {available}"
        )
    return cls(robot_name=robot_name, **kwargs)


def register_adapter_class(scheme: str, cls: type["CapabilityAdapter"]) -> None:
    """Register an adapter class for a URI scheme (imperative, non-decorator form).

    Equivalent to decorating the class with @register_adapter(scheme).
    Useful when the adapter class is defined in third-party code.

        from apyrobo.core.adapters import register_adapter_class
        register_adapter_class("myrobot", MyRobotAdapter)
    """
    _ADAPTER_REGISTRY[scheme] = cls


def list_adapters() -> list[str]:
    """Return all registered adapter scheme names."""
    return sorted(_ADAPTER_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Abstract base — the full adapter contract
# ---------------------------------------------------------------------------

class CapabilityAdapter(abc.ABC):
    """
    Abstract base class for robot capability adapters.

    Subclasses translate between APYROBO's semantic commands and the
    underlying interface (ROS 2, MQTT, HTTP, simulator, etc.).

    Required (abstract):
        get_capabilities, move, stop

    Optional (with sensible defaults):
        rotate, gripper_open, gripper_close, cancel,
        get_position, get_orientation, get_health,
        connect, disconnect, is_connected
    """

    def __init__(self, robot_name: str, **kwargs: Any) -> None:
        self.robot_name = robot_name
        self._state = AdapterState.DISCONNECTED
        self._reconnect_attempts: int = 0
        self._last_disconnect_time: float | None = None
        self._disconnect_handlers: list[Callable[[], None]] = []
        self._reconnect_handlers: list[Callable[[], None]] = []

    # ------------------------------------------------------------------
    # Required — every adapter must implement these
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_capabilities(self) -> RobotCapability:
        """Query the robot and return its full capability profile."""
        ...

    @abc.abstractmethod
    def move(self, x: float, y: float, speed: float | None = None) -> None:
        """Send a movement command to the robot."""
        ...

    @abc.abstractmethod
    def stop(self) -> None:
        """Immediately halt all motion."""
        ...

    # ------------------------------------------------------------------
    # Rotation (AD-01)
    # ------------------------------------------------------------------

    def rotate(self, angle_rad: float, speed: float | None = None) -> None:
        """
        Rotate the robot in place by *angle_rad* radians.

        Positive = counter-clockwise (REP-103 convention).
        Default implementation logs a warning and does nothing.
        """
        logger.warning(
            "%s: rotate() not implemented — ignoring %.2f rad",
            type(self).__name__, angle_rad,
        )

    # ------------------------------------------------------------------
    # Gripper / manipulation (AD-02)
    # ------------------------------------------------------------------

    def gripper_open(self) -> bool:
        """
        Open the gripper/end-effector.

        Returns True if successful, False if not supported or failed.
        Default implementation returns True (mock success).
        """
        logger.info("%s: gripper_open (default no-op)", type(self).__name__)
        return True

    def gripper_close(self) -> bool:
        """
        Close the gripper/end-effector.

        Returns True if successful, False if not supported or failed.
        """
        logger.info("%s: gripper_close (default no-op)", type(self).__name__)
        return True

    # ------------------------------------------------------------------
    # Navigation control (AD-03)
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """
        Cancel the current navigation goal.

        Default implementation calls stop().
        """
        self.stop()

    # ------------------------------------------------------------------
    # State queries (AD-04)
    # ------------------------------------------------------------------

    def get_position(self) -> tuple[float, float]:
        """
        Return the robot's current (x, y) position.

        Default returns (0.0, 0.0).
        """
        return (0.0, 0.0)

    def get_orientation(self) -> float:
        """
        Return the robot's current heading in radians.

        Default returns 0.0.
        """
        return 0.0

    def get_health(self) -> dict[str, Any]:
        """
        Return a health/status dict for the adapter.

        Standard keys: state, battery_pct, uptime_s, errors.
        Default returns connected state.
        """
        return {
            "state": self._state.value,
            "adapter": type(self).__name__,
            "robot": self.robot_name,
        }

    # ------------------------------------------------------------------
    # Lifecycle (AD-05)
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Establish connection to the robot/platform.

        Default sets state to CONNECTED and notifies reconnect handlers.
        """
        self._state = AdapterState.CONNECTED
        logger.info("%s: connected to %s", type(self).__name__, self.robot_name)

    def disconnect(self) -> None:
        """
        Cleanly disconnect from the robot/platform.

        Default sets state to DISCONNECTED and notifies disconnect handlers.
        """
        was_connected = self._state == AdapterState.CONNECTED
        self._state = AdapterState.DISCONNECTED
        if was_connected:
            self._last_disconnect_time = time.time()
            self._notify_disconnect()
        logger.info("%s: disconnected from %s", type(self).__name__, self.robot_name)

    @property
    def is_connected(self) -> bool:
        """Whether the adapter has an active connection."""
        return self._state == AdapterState.CONNECTED

    @property
    def state(self) -> AdapterState:
        return self._state

    # ------------------------------------------------------------------
    # Connection resilience (AD-06)
    # ------------------------------------------------------------------

    def on_disconnect(self, handler: Callable[[], None]) -> None:
        """Register a callback invoked when the adapter disconnects."""
        self._disconnect_handlers.append(handler)

    def on_reconnect(self, handler: Callable[[], None]) -> None:
        """Register a callback invoked when the adapter successfully reconnects."""
        self._reconnect_handlers.append(handler)

    def _notify_disconnect(self) -> None:
        """Emit disconnect observability event and call registered handlers."""
        try:
            from apyrobo.observability import emit_event
            emit_event(
                "adapter.disconnect",
                robot=self.robot_name,
                adapter=type(self).__name__,
                timestamp=time.time(),
            )
        except Exception:
            pass
        for handler in self._disconnect_handlers:
            try:
                handler()
            except Exception:
                logger.exception("%s: disconnect handler raised", type(self).__name__)

    def _notify_reconnect(self) -> None:
        """Emit reconnect observability event and call registered handlers."""
        try:
            from apyrobo.observability import emit_event
            emit_event(
                "adapter.reconnect",
                robot=self.robot_name,
                adapter=type(self).__name__,
                attempts=self._reconnect_attempts,
                timestamp=time.time(),
            )
        except Exception:
            pass
        for handler in self._reconnect_handlers:
            try:
                handler()
            except Exception:
                logger.exception("%s: reconnect handler raised", type(self).__name__)

    def reconnect_with_backoff(
        self,
        max_attempts: int = 5,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
    ) -> bool:
        """
        Attempt to reconnect with exponential backoff.

        Calls ``connect()`` up to *max_attempts* times, sleeping between
        attempts with delays: ``initial_delay * backoff_factor^attempt``,
        capped at *max_delay*.

        Emits ``adapter.disconnect`` / ``adapter.reconnect`` observability
        events on transitions.

        Returns:
            True if connection was re-established, False if all attempts failed.
        """
        if self.is_connected:
            return True

        self._state = AdapterState.CONNECTING
        delay = initial_delay

        for attempt in range(1, max_attempts + 1):
            self._reconnect_attempts += 1
            logger.info(
                "%s: reconnect attempt %d/%d to %s (delay=%.1fs)",
                type(self).__name__, attempt, max_attempts, self.robot_name, delay,
            )
            try:
                self.connect()
                if self.is_connected:
                    logger.info(
                        "%s: reconnected to %s after %d attempt(s)",
                        type(self).__name__, self.robot_name, attempt,
                    )
                    self._notify_reconnect()
                    return True
            except Exception as exc:
                logger.warning(
                    "%s: reconnect attempt %d failed: %s",
                    type(self).__name__, attempt, exc,
                )

            if attempt < max_attempts:
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)

        self._state = AdapterState.ERROR
        logger.error(
            "%s: failed to reconnect to %s after %d attempts",
            type(self).__name__, self.robot_name, max_attempts,
        )
        return False


# ---------------------------------------------------------------------------
# Mock adapter — for testing without ROS 2 or hardware
# ---------------------------------------------------------------------------

@register_adapter("mock")
class MockAdapter(CapabilityAdapter):
    """
    In-memory mock adapter for unit testing.

    Tracks all commands and state so tests can assert on behaviour.

    Usage:
        robot = Robot.discover("mock://test_bot")
    """

    def __init__(self, robot_name: str, **kwargs: Any) -> None:
        super().__init__(robot_name, **kwargs)
        self._position = (0.0, 0.0)
        self._orientation = 0.0  # radians
        self._moving = False
        self._gripper_open = True
        self._move_history: list[dict[str, Any]] = []
        self._rotate_history: list[dict[str, Any]] = []
        self._state = AdapterState.CONNECTED
        logger.info("MockAdapter created for %s", robot_name)

    # --- Required ---

    def get_capabilities(self) -> RobotCapability:
        return RobotCapability(
            robot_id=self.robot_name,
            name=f"Mock-{self.robot_name}",
            capabilities=[
                Capability(
                    capability_type=CapabilityType.NAVIGATE,
                    name="navigate_to",
                    description="Move to a 2D position",
                    parameters={"x": "float", "y": "float", "speed": "float (optional)"},
                ),
                Capability(
                    capability_type=CapabilityType.ROTATE,
                    name="rotate",
                    description="Rotate in place",
                    parameters={"angle_rad": "float", "speed": "float (optional)"},
                ),
                Capability(
                    capability_type=CapabilityType.PICK,
                    name="pick_object",
                    description="Pick up an object at current location",
                ),
                Capability(
                    capability_type=CapabilityType.PLACE,
                    name="place_object",
                    description="Place held object at current location",
                ),
            ],
            sensors=[
                SensorInfo(
                    sensor_id="cam0",
                    sensor_type=SensorType.CAMERA,
                    topic="/mock/camera/image_raw",
                    hz=30.0,
                ),
                SensorInfo(
                    sensor_id="lidar0",
                    sensor_type=SensorType.LIDAR,
                    topic="/mock/scan",
                    hz=10.0,
                ),
            ],
            max_speed=1.5,
        )

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        self._moving = True
        self._position = (x, y)
        self._move_history.append({"x": x, "y": y, "speed": speed})
        logger.info("MockAdapter: moving to (%.2f, %.2f) speed=%s", x, y, speed)

    def stop(self) -> None:
        self._moving = False
        logger.info("MockAdapter: stopped at (%.2f, %.2f)", *self._position)

    # --- Optional overrides ---

    def rotate(self, angle_rad: float, speed: float | None = None) -> None:
        self._orientation = (self._orientation + angle_rad) % (2 * math.pi)
        self._rotate_history.append({"angle_rad": angle_rad, "speed": speed})
        logger.info("MockAdapter: rotated by %.2f rad → heading %.2f rad",
                     angle_rad, self._orientation)

    def gripper_open(self) -> bool:
        self._gripper_open = True
        logger.info("MockAdapter: gripper opened")
        return True

    def gripper_close(self) -> bool:
        self._gripper_open = False
        logger.info("MockAdapter: gripper closed")
        return True

    def cancel(self) -> None:
        self._moving = False
        logger.info("MockAdapter: navigation cancelled")

    def get_position(self) -> tuple[float, float]:
        return self._position

    def get_orientation(self) -> float:
        return self._orientation

    def get_health(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "adapter": "MockAdapter",
            "robot": self.robot_name,
            "battery_pct": 100.0,
            "uptime_s": 0.0,
            "errors": 0,
        }

    # --- Test helpers ---

    @property
    def position(self) -> tuple[float, float]:
        return self._position

    @property
    def orientation(self) -> float:
        return self._orientation

    @property
    def is_moving(self) -> bool:
        return self._moving

    @property
    def gripper_is_open(self) -> bool:
        return self._gripper_open

    @property
    def move_history(self) -> list[dict[str, Any]]:
        return list(self._move_history)

    @property
    def rotate_history(self) -> list[dict[str, Any]]:
        return list(self._rotate_history)


# ---------------------------------------------------------------------------
# Gazebo adapter — simulation with physics-like behaviour
# ---------------------------------------------------------------------------

@register_adapter("gazebo")
class GazeboAdapter(CapabilityAdapter):
    """
    Adapter for Gazebo-simulated robots.

    Without a live Gazebo instance, behaves like an enhanced mock with
    simulated delays and physics constraints. When Gazebo is available,
    subclass and override _send_gazebo_cmd().

    Usage:
        robot = Robot.discover("gazebo://turtlebot4")
    """

    def __init__(self, robot_name: str, **kwargs: Any) -> None:
        super().__init__(robot_name, **kwargs)
        self._position = (0.0, 0.0)
        self._orientation = 0.0
        self._moving = False
        self._gripper_open = True
        self._sim_speed_factor = kwargs.get("sim_speed_factor", 1.0)
        self._max_speed = kwargs.get("max_speed", 1.0)
        self._max_angular_speed = kwargs.get("max_angular_speed", 1.0)
        self._state = AdapterState.CONNECTED
        logger.info("GazeboAdapter created for %s (sim_speed=%.1fx)",
                     robot_name, self._sim_speed_factor)

    def get_capabilities(self) -> RobotCapability:
        return RobotCapability(
            robot_id=self.robot_name,
            name=f"Gazebo-{self.robot_name}",
            capabilities=[
                Capability(capability_type=CapabilityType.NAVIGATE, name="navigate_to",
                           description="Navigate to 2D pose in simulation"),
                Capability(capability_type=CapabilityType.ROTATE, name="rotate",
                           description="Rotate in place"),
                Capability(capability_type=CapabilityType.PICK, name="pick_object",
                           description="Simulated object pick"),
                Capability(capability_type=CapabilityType.PLACE, name="place_object",
                           description="Simulated object place"),
                Capability(capability_type=CapabilityType.SCAN, name="scan_area",
                           description="360-degree sensor scan"),
            ],
            sensors=[
                SensorInfo(sensor_id="cam0", sensor_type=SensorType.CAMERA,
                           topic=f"/{self.robot_name}/camera/image_raw", hz=30.0),
                SensorInfo(sensor_id="lidar0", sensor_type=SensorType.LIDAR,
                           topic=f"/{self.robot_name}/scan", hz=10.0),
                SensorInfo(sensor_id="imu0", sensor_type=SensorType.IMU,
                           topic=f"/{self.robot_name}/imu", hz=100.0),
                SensorInfo(sensor_id="depth0", sensor_type=SensorType.DEPTH,
                           topic=f"/{self.robot_name}/depth/image_rect_raw", hz=15.0),
            ],
            max_speed=self._max_speed,
            metadata={"sim": True, "engine": "gazebo", "speed_factor": self._sim_speed_factor},
        )

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        effective_speed = min(speed or self._max_speed, self._max_speed)
        dx = x - self._position[0]
        dy = y - self._position[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 0:
            self._orientation = math.atan2(dy, dx)
        self._moving = True
        self._position = (x, y)
        logger.info("GazeboAdapter: nav to (%.2f, %.2f) speed=%.2f dist=%.2f",
                     x, y, effective_speed, dist)

    def stop(self) -> None:
        self._moving = False
        logger.info("GazeboAdapter: stopped at (%.2f, %.2f)", *self._position)

    def rotate(self, angle_rad: float, speed: float | None = None) -> None:
        self._orientation = (self._orientation + angle_rad) % (2 * math.pi)
        logger.info("GazeboAdapter: rotated %.2f rad → heading %.2f",
                     angle_rad, self._orientation)

    def gripper_open(self) -> bool:
        self._gripper_open = True
        return True

    def gripper_close(self) -> bool:
        self._gripper_open = False
        return True

    def cancel(self) -> None:
        self._moving = False

    def get_position(self) -> tuple[float, float]:
        return self._position

    def get_orientation(self) -> float:
        return self._orientation

    def get_health(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "adapter": "GazeboAdapter",
            "robot": self.robot_name,
            "sim": True,
            "sim_speed_factor": self._sim_speed_factor,
        }

    @property
    def position(self) -> tuple[float, float]:
        return self._position

    @property
    def is_moving(self) -> bool:
        return self._moving


# ---------------------------------------------------------------------------
# MQTT adapter — for IoT / remote robots
# ---------------------------------------------------------------------------

@register_adapter("mqtt")
class MQTTAdapter(CapabilityAdapter):
    """
    Adapter that communicates with robots over MQTT.

    Publishes commands to topics and subscribes to state topics.
    Without an actual MQTT broker, operates in offline/buffered mode.

    URI: mqtt://robot_name?broker=host:port

    Topic conventions:
        apyrobo/{robot}/cmd/move     ← {"x": float, "y": float, "speed": float}
        apyrobo/{robot}/cmd/rotate   ← {"angle_rad": float, "speed": float}
        apyrobo/{robot}/cmd/stop     ← {}
        apyrobo/{robot}/cmd/gripper  ← {"action": "open"|"close"}
        apyrobo/{robot}/state/pose   → {"x": float, "y": float, "yaw": float}
        apyrobo/{robot}/state/health → {"battery_pct": float, ...}
    """

    def __init__(self, robot_name: str, **kwargs: Any) -> None:
        super().__init__(robot_name, **kwargs)
        self._broker = kwargs.get("broker", "localhost:1883")
        self._topic_prefix = f"apyrobo/{robot_name}"
        self._position = (0.0, 0.0)
        self._orientation = 0.0
        self._cmd_buffer: list[dict[str, Any]] = []
        self._connected = False
        self._gripper_open = True
        logger.info("MQTTAdapter created for %s (broker=%s)", robot_name, self._broker)

    def get_capabilities(self) -> RobotCapability:
        return RobotCapability(
            robot_id=self.robot_name,
            name=f"MQTT-{self.robot_name}",
            capabilities=[
                Capability(capability_type=CapabilityType.NAVIGATE, name="navigate_to",
                           description="Navigate via MQTT command"),
                Capability(capability_type=CapabilityType.ROTATE, name="rotate",
                           description="Rotate via MQTT command"),
                Capability(capability_type=CapabilityType.PICK, name="pick_object",
                           description="Gripper close via MQTT"),
                Capability(capability_type=CapabilityType.PLACE, name="place_object",
                           description="Gripper open via MQTT"),
            ],
            sensors=[],
            max_speed=1.0,
            metadata={"transport": "mqtt", "broker": self._broker},
        )

    def _publish(self, subtopic: str, payload: dict[str, Any]) -> None:
        """Publish a command. Buffers if not connected to broker."""
        topic = f"{self._topic_prefix}/{subtopic}"
        msg = {"topic": topic, "payload": payload, "timestamp": time.time()}
        self._cmd_buffer.append(msg)
        logger.info("MQTT publish → %s: %s", topic, json.dumps(payload))

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        self._publish("cmd/move", {"x": x, "y": y, "speed": speed})
        self._position = (x, y)

    def stop(self) -> None:
        self._publish("cmd/stop", {})

    def rotate(self, angle_rad: float, speed: float | None = None) -> None:
        self._publish("cmd/rotate", {"angle_rad": angle_rad, "speed": speed})
        self._orientation = (self._orientation + angle_rad) % (2 * math.pi)

    def gripper_open(self) -> bool:
        self._publish("cmd/gripper", {"action": "open"})
        self._gripper_open = True
        return True

    def gripper_close(self) -> bool:
        self._publish("cmd/gripper", {"action": "close"})
        self._gripper_open = False
        return True

    def cancel(self) -> None:
        self._publish("cmd/cancel", {})

    def get_position(self) -> tuple[float, float]:
        return self._position

    def get_orientation(self) -> float:
        return self._orientation

    def get_health(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "adapter": "MQTTAdapter",
            "robot": self.robot_name,
            "broker": self._broker,
            "buffered_commands": len(self._cmd_buffer),
        }

    def connect(self) -> None:
        self._state = AdapterState.CONNECTED
        self._connected = True
        logger.info("MQTTAdapter: connected to broker %s", self._broker)

    def disconnect(self) -> None:
        self._state = AdapterState.DISCONNECTED
        self._connected = False
        logger.info("MQTTAdapter: disconnected from broker %s", self._broker)

    @property
    def cmd_buffer(self) -> list[dict[str, Any]]:
        """All published commands (for testing)."""
        return list(self._cmd_buffer)


# ---------------------------------------------------------------------------
# HTTP adapter — for REST-based robot APIs
# ---------------------------------------------------------------------------

@register_adapter("http")
class HTTPAdapter(CapabilityAdapter):
    """
    Adapter for robots with HTTP/REST APIs.

    Sends commands as POST requests and queries state as GET requests.
    Without a live server, operates in offline/buffered mode.

    URI: http://robot_name?base_url=http://192.168.1.100:8080

    Endpoint conventions:
        POST /api/move     {"x": float, "y": float, "speed": float}
        POST /api/rotate   {"angle_rad": float, "speed": float}
        POST /api/stop     {}
        POST /api/gripper  {"action": "open"|"close"}
        GET  /api/position → {"x": float, "y": float}
        GET  /api/health   → {"battery_pct": float, ...}
    """

    def __init__(self, robot_name: str, **kwargs: Any) -> None:
        super().__init__(robot_name, **kwargs)
        self._base_url = kwargs.get("base_url", f"http://localhost:8080")
        self._position = (0.0, 0.0)
        self._orientation = 0.0
        self._gripper_open = True
        self._request_log: list[dict[str, Any]] = []
        self._state = AdapterState.CONNECTED
        logger.info("HTTPAdapter created for %s (url=%s)", robot_name, self._base_url)

    def _request(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Log a request. In a real implementation, this would use httpx/requests."""
        entry = {
            "method": method,
            "url": f"{self._base_url}{endpoint}",
            "payload": payload,
            "timestamp": time.time(),
        }
        self._request_log.append(entry)
        logger.info("HTTP %s %s%s %s", method, self._base_url, endpoint,
                     json.dumps(payload) if payload else "")
        return {"status": "ok"}

    def get_capabilities(self) -> RobotCapability:
        return RobotCapability(
            robot_id=self.robot_name,
            name=f"HTTP-{self.robot_name}",
            capabilities=[
                Capability(capability_type=CapabilityType.NAVIGATE, name="navigate_to",
                           description="Navigate via HTTP API"),
                Capability(capability_type=CapabilityType.ROTATE, name="rotate",
                           description="Rotate via HTTP API"),
            ],
            sensors=[],
            max_speed=1.0,
            metadata={"transport": "http", "base_url": self._base_url},
        )

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        self._request("POST", "/api/move", {"x": x, "y": y, "speed": speed})
        self._position = (x, y)

    def stop(self) -> None:
        self._request("POST", "/api/stop", {})

    def rotate(self, angle_rad: float, speed: float | None = None) -> None:
        self._request("POST", "/api/rotate", {"angle_rad": angle_rad, "speed": speed})
        self._orientation = (self._orientation + angle_rad) % (2 * math.pi)

    def gripper_open(self) -> bool:
        self._request("POST", "/api/gripper", {"action": "open"})
        self._gripper_open = True
        return True

    def gripper_close(self) -> bool:
        self._request("POST", "/api/gripper", {"action": "close"})
        self._gripper_open = False
        return True

    def cancel(self) -> None:
        self._request("POST", "/api/cancel", {})

    def get_position(self) -> tuple[float, float]:
        return self._position

    def get_orientation(self) -> float:
        return self._orientation

    def get_health(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "adapter": "HTTPAdapter",
            "robot": self.robot_name,
            "base_url": self._base_url,
            "requests_sent": len(self._request_log),
        }

    @property
    def request_log(self) -> list[dict[str, Any]]:
        """All HTTP requests made (for testing)."""
        return list(self._request_log)
