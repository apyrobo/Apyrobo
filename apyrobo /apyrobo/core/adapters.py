"""
Capability adapters — the bridge between APYROBO's semantic API and ROS 2.

Each adapter knows how to translate APYROBO calls (move, stop, capabilities)
into the correct ROS 2 topics, services, and actions for a specific robot
platform.  The rest of APYROBO never touches ROS 2 directly.

To add support for a new robot:
    1. Subclass CapabilityAdapter
    2. Register it with @register_adapter("scheme")
"""

from __future__ import annotations

import abc
import logging
from typing import Any

from apyrobo.core.schemas import (
    Capability,
    CapabilityType,
    RobotCapability,
    SensorInfo,
    SensorType,
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


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CapabilityAdapter(abc.ABC):
    """
    Abstract base class for robot capability adapters.

    Subclasses translate between APYROBO's semantic commands and the
    underlying ROS 2 (or simulator) interface for a specific robot.
    """

    def __init__(self, robot_name: str, **kwargs: Any) -> None:
        self.robot_name = robot_name

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


# ---------------------------------------------------------------------------
# Mock adapter — for testing without ROS 2 or hardware
# ---------------------------------------------------------------------------

@register_adapter("mock")
class MockAdapter(CapabilityAdapter):
    """
    In-memory mock adapter for unit testing.

    Usage:
        robot = Robot.discover("mock://test_bot")
    """

    def __init__(self, robot_name: str, **kwargs: Any) -> None:
        super().__init__(robot_name, **kwargs)
        self._position = (0.0, 0.0)
        self._moving = False
        self._move_history: list[dict[str, Any]] = []
        logger.info("MockAdapter created for %s", robot_name)

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

    # --- Test helpers ---

    @property
    def position(self) -> tuple[float, float]:
        return self._position

    @property
    def is_moving(self) -> bool:
        return self._moving

    @property
    def move_history(self) -> list[dict[str, Any]]:
        return list(self._move_history)
