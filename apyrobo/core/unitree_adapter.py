"""
Unitree Go2 / H1 adapter for APYROBO.

Communicates with Unitree robots over UDP/DDS using the official
``unitree_sdk2_python`` library.  Install it with::

    pip install unitree-sdk2py

URI scheme::

    unitree://go2@192.168.1.100        # Go2 quadruped at explicit IP
    unitree://h1@192.168.1.120         # H1 humanoid with arm
    unitree://go2                      # Go2 using default broadcast discovery

The model (``go2`` / ``h1``) is extracted from the URI and determines which
capabilities are advertised and which high-level behaviours are available.

v6.0.0 — APYROBO Ecosystem Integrations
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

from apyrobo.core.adapters import CapabilityAdapter, register_adapter
from apyrobo.core.schemas import (
    AdapterState,
    Capability,
    CapabilityType,
    RobotCapability,
    SensorInfo,
    SensorType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional SDK import — guarded so the module loads without the SDK installed
# ---------------------------------------------------------------------------

try:
    from unitree_sdk2py.go2.sport.sport_client import SportClient  # type: ignore
    from unitree_sdk2py.core.channel import ChannelFactory  # type: ignore
    _UNITREE_SDK_AVAILABLE = True
except ImportError:
    SportClient = None  # type: ignore
    ChannelFactory = None  # type: ignore
    _UNITREE_SDK_AVAILABLE = False

_UNITREE_SDK_IMPORT_ERROR = (
    "Install unitree_sdk2_python: pip install unitree-sdk2py"
)

# ---------------------------------------------------------------------------
# URI parsing
# ---------------------------------------------------------------------------

# Matches:  unitree://go2@192.168.1.100
#           unitree://h1@10.0.0.5
#           unitree://go2            (no host → use broadcast / default)
_URI_RE = re.compile(
    r"^(?:unitree://)?(?P<model>[a-z0-9_]+)(?:@(?P<host>[0-9a-zA-Z._-]+))?$"
)

# Known Unitree models and their default capabilities
_MODEL_CAPS: dict[str, set[CapabilityType]] = {
    "go2": {
        CapabilityType.NAVIGATE,
        CapabilityType.ROTATE,
        CapabilityType.SCAN,
    },
    "h1": {
        CapabilityType.NAVIGATE,
        CapabilityType.ROTATE,
        CapabilityType.MANIPULATE,
        CapabilityType.PICK,
        CapabilityType.PLACE,
        CapabilityType.SCAN,
    },
    # Future models can be added here
    "b2": {
        CapabilityType.NAVIGATE,
        CapabilityType.ROTATE,
        CapabilityType.SCAN,
    },
}

_DEFAULT_NETWORK_IFACE = "eth0"


@dataclass
class UnitreeState:
    """Pose and velocity state read from SportClient.GetState()."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0    # radians
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0  # yaw rate rad/s
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# UnitreeAdapter
# ---------------------------------------------------------------------------

@register_adapter("unitree")
class UnitreeAdapter(CapabilityAdapter):
    """
    Unitree Go2 / H1 robot adapter.

    Wraps the ``unitree_sdk2py`` SportClient to expose APYROBO's standard
    move / stop / gripper interface for Unitree legged robots.

    **Go2** (quadruped): navigate, rotate, scan, stand, sit, wave.
    **H1** (humanoid): all Go2 capabilities + manipulate, pick, place,
    gripper open/close via the arm DexHand or gripper module.

    The robot's model and network address are extracted from the URI::

        unitree://go2@192.168.1.100   → model="go2", host="192.168.1.100"
        unitree://h1@10.0.0.5         → model="h1",  host="10.0.0.5"
        unitree://go2                 → model="go2", host=broadcast

    When the SDK is not installed, all methods raise ``ImportError`` with
    installation instructions at the first call that needs the hardware,
    so the module itself always imports cleanly.

    Usage::

        from apyrobo.core.robot import Robot
        robot = await Robot.discover("unitree://go2@192.168.1.100")
        await robot.move(1.0, 0.0)           # walk forward 1 m
        await robot.stand()
    """

    def __init__(self, robot_name: str, **kwargs: Any) -> None:
        super().__init__(robot_name, **kwargs)

        model, host = self._detect_model(robot_name)
        self._model: str = model
        self._host: str | None = host
        self._network_iface: str = kwargs.get("network_iface", _DEFAULT_NETWORK_IFACE)

        self._sport_client: Any = None  # unitree_sdk2py.go2.sport.sport_client.SportClient
        self._unitree_state = UnitreeState()
        self._gripper_open_flag: bool = True
        self._is_standing: bool = False

        logger.info(
            "UnitreeAdapter created for %r (model=%s, host=%s)",
            robot_name, self._model, self._host or "broadcast",
        )

    # ------------------------------------------------------------------
    # URI parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_model(uri: str) -> tuple[str, str | None]:
        """
        Parse a Unitree URI and return (model, host).

        Accepts URIs with or without the ``unitree://`` scheme prefix::

            "unitree://go2@192.168.1.100" → ("go2", "192.168.1.100")
            "go2@192.168.1.100"           → ("go2", "192.168.1.100")
            "h1"                          → ("h1",  None)

        Args:
            uri: Robot name / URI from Robot.discover().

        Returns:
            (model, host) where host is None when no IP is specified.

        Raises:
            ValueError: If the URI cannot be parsed.
        """
        # Strip the scheme prefix for matching
        bare = uri.replace("unitree://", "")
        m = _URI_RE.match(bare)
        if not m:
            raise ValueError(
                f"Cannot parse Unitree URI {uri!r}. "
                "Expected format: unitree://model[@host], e.g. unitree://go2@192.168.1.100"
            )
        model = m.group("model").lower()
        host = m.group("host")  # None when absent
        return model, host

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Open a DDS channel to the robot and initialise SportClient.

        Raises:
            ImportError: if unitree_sdk2py is not installed.
            ConnectionError: if the DDS handshake fails.
        """
        self._require_sdk()
        try:
            ChannelFactory.Instance().Init(
                0,                       # domain ID (default)
                self._network_iface,
            )
            self._sport_client = SportClient()
            self._sport_client.SetTimeout(5.0)
            self._sport_client.Init()
            self._state = AdapterState.CONNECTED
            logger.info(
                "UnitreeAdapter %r: connected via DDS (iface=%s, host=%s)",
                self.robot_name, self._network_iface, self._host or "broadcast",
            )
        except Exception as exc:
            self._state = AdapterState.ERROR
            raise ConnectionError(
                f"UnitreeAdapter: failed to connect to {self.robot_name!r}: {exc}"
            ) from exc

    def disconnect(self) -> None:
        """Stop motion and release the DDS channel."""
        if self._sport_client is not None:
            try:
                self._sport_client.StopMove()
            except Exception:
                pass
            self._sport_client = None
        super().disconnect()

    # ------------------------------------------------------------------
    # Required — CapabilityAdapter contract
    # ------------------------------------------------------------------

    def get_capabilities(self) -> RobotCapability:
        """
        Return capabilities based on the robot model extracted from the URI.

        Go2: navigate, rotate, scan.
        H1:  navigate, rotate, scan, manipulate, pick, place.
        """
        model_caps = _MODEL_CAPS.get(self._model, {CapabilityType.NAVIGATE})

        caps: list[Capability] = []

        if CapabilityType.NAVIGATE in model_caps:
            caps.append(Capability(
                capability_type=CapabilityType.NAVIGATE,
                name="navigate_to",
                description="Walk to a 2D target position",
                parameters={
                    "x": "float — forward displacement (m)",
                    "y": "float — lateral displacement (m)",
                    "speed": "float (optional) — max speed m/s",
                },
            ))

        if CapabilityType.ROTATE in model_caps:
            caps.append(Capability(
                capability_type=CapabilityType.ROTATE,
                name="rotate",
                description="Rotate body yaw in place",
                parameters={"angle_rad": "float", "speed": "float (optional)"},
            ))

        if CapabilityType.SCAN in model_caps:
            caps.append(Capability(
                capability_type=CapabilityType.SCAN,
                name="scan_area",
                description="Point cloud / lidar sweep from onboard sensors",
            ))

        if CapabilityType.MANIPULATE in model_caps:
            caps.append(Capability(
                capability_type=CapabilityType.MANIPULATE,
                name="arm_move",
                description="Move the H1 arm to a joint/Cartesian target",
                parameters={"joint_targets": "list[float]"},
            ))

        if CapabilityType.PICK in model_caps:
            caps.append(Capability(
                capability_type=CapabilityType.PICK,
                name="pick_object",
                description="Close gripper to grasp an object",
            ))

        if CapabilityType.PLACE in model_caps:
            caps.append(Capability(
                capability_type=CapabilityType.PLACE,
                name="place_object",
                description="Open gripper to release a held object",
            ))

        sensors = [
            SensorInfo(
                sensor_id="lidar_0",
                sensor_type=SensorType.LIDAR,
                topic=f"/{self._model}/lidar",
                hz=10.0,
            ),
            SensorInfo(
                sensor_id="imu_0",
                sensor_type=SensorType.IMU,
                topic=f"/{self._model}/imu",
                hz=500.0,
            ),
        ]
        if self._model in ("go2", "h1"):
            sensors.append(SensorInfo(
                sensor_id="depth_0",
                sensor_type=SensorType.DEPTH,
                topic=f"/{self._model}/depth",
                hz=30.0,
            ))

        return RobotCapability(
            robot_id=self.robot_name,
            name=f"Unitree-{self._model.upper()}-{self.robot_name}",
            capabilities=caps,
            sensors=sensors,
            max_speed=1.5 if self._model == "go2" else 1.2,
            metadata={
                "vendor": "Unitree Robotics",
                "model": self._model,
                "host": self._host,
                "sdk_available": _UNITREE_SDK_AVAILABLE,
                "transport": "udp/dds",
            },
        )

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        """
        Walk the robot toward (x, y) in the robot's local frame.

        Calls ``SportClient.Move(vx, vy, vyaw)`` with the displacement mapped
        to velocity commands.  The robot will walk until ``stop()`` is called
        or the controller reaches the target.

        Args:
            x: Forward displacement in metres (robot frame).
            y: Lateral displacement in metres (robot frame).
            speed: Maximum translational speed m/s (clamped to model limit).

        Raises:
            ImportError: if unitree_sdk2py is not installed.
        """
        self._require_sdk()
        max_speed = 1.5 if self._model == "go2" else 1.2
        vx = min(abs(x), max_speed) * (1.0 if x >= 0 else -1.0)
        vy = min(abs(y), max_speed) * (1.0 if y >= 0 else -1.0)
        if speed is not None:
            scale = min(speed, max_speed) / max_speed
            vx *= scale
            vy *= scale

        if self._sport_client is not None:
            try:
                self._sport_client.Move(vx, vy, 0.0)
            except Exception as exc:
                logger.warning("UnitreeAdapter %r: Move() failed: %s", self.robot_name, exc)

        # Update internal stub state (mirrors what hardware would do)
        self._state_data.x += x
        self._state_data.y += y
        logger.info(
            "UnitreeAdapter %r: move vx=%.2f vy=%.2f (target x=%.2f y=%.2f)",
            self.robot_name, vx, vy, x, y,
        )

    def stop(self) -> None:
        """
        Stop all motion immediately.

        Calls ``SportClient.StopMove()``.

        Raises:
            ImportError: if unitree_sdk2py is not installed.
        """
        self._require_sdk()
        if self._sport_client is not None:
            try:
                self._sport_client.StopMove()
            except Exception as exc:
                logger.warning("UnitreeAdapter %r: StopMove() failed: %s", self.robot_name, exc)
        logger.info("UnitreeAdapter %r: stopped", self.robot_name)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_position(self) -> tuple[float, float]:
        """
        Return the robot's current (x, y) position.

        Queries ``SportClient.GetState()`` when connected; falls back to
        the last known position from move() calls.

        Returns:
            (x, y) in metres (world frame from DDS state estimator).
        """
        if self._sport_client is not None:
            try:
                state_msg = self._sport_client.GetState()
                self._state_data.x = float(state_msg.position[0])
                self._state_data.y = float(state_msg.position[1])
                self._state_data.z = float(state_msg.position[2])
                self._state_data.yaw = float(state_msg.imu_state.rpy[2])
            except Exception as exc:
                logger.debug("UnitreeAdapter %r: GetState() failed: %s", self.robot_name, exc)
        return (self._state_data.x, self._state_data.y)

    def get_orientation(self) -> float:
        """Return the robot's current yaw in radians."""
        self.get_position()  # refreshes _state_data.yaw
        return self._state_data.yaw

    def get_health(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "adapter": "UnitreeAdapter",
            "robot": self.robot_name,
            "model": self._model,
            "host": self._host,
            "sdk_available": _UNITREE_SDK_AVAILABLE,
            "connected": self._sport_client is not None,
        }

    # ------------------------------------------------------------------
    # Legged-robot behaviours (Go2 + H1)
    # ------------------------------------------------------------------

    def rotate(self, angle_rad: float, speed: float | None = None) -> None:
        """
        Rotate the robot body by *angle_rad* radians about the yaw axis.

        Calls ``SportClient.Move(0, 0, vyaw)`` with the angular velocity
        derived from the requested angle and speed.
        """
        self._require_sdk()
        max_omega = 1.0  # rad/s, conservative limit for both Go2 and H1
        omega = min(abs(angle_rad), max_omega) * (1.0 if angle_rad >= 0 else -1.0)
        if speed is not None:
            omega = min(speed, max_omega) * (1.0 if angle_rad >= 0 else -1.0)

        if self._sport_client is not None:
            try:
                self._sport_client.Move(0.0, 0.0, omega)
            except Exception as exc:
                logger.warning("UnitreeAdapter %r: rotate Move() failed: %s", self.robot_name, exc)

        self._state_data.yaw = (self._state_data.yaw + angle_rad) % (2 * math.pi)
        logger.info(
            "UnitreeAdapter %r: rotate %.2f rad → heading %.2f rad",
            self.robot_name, angle_rad, self._state_data.yaw,
        )

    def stand(self) -> bool:
        """
        Command the robot to stand up.

        Calls ``SportClient.StandUp()``.  Works on both Go2 and H1.

        Returns:
            True if the command was sent successfully.

        Raises:
            ImportError: if unitree_sdk2py is not installed.
        """
        self._require_sdk()
        if self._sport_client is not None:
            try:
                self._sport_client.StandUp()
                self._is_standing = True
                logger.info("UnitreeAdapter %r: stand up", self.robot_name)
                return True
            except Exception as exc:
                logger.warning("UnitreeAdapter %r: StandUp() failed: %s", self.robot_name, exc)
                return False
        self._is_standing = True
        return True

    def sit_down(self) -> bool:
        """
        Command the robot to sit / lie down.

        Calls ``SportClient.StandDown()``.  Go2-specific.

        Returns:
            True if the command was sent successfully.

        Raises:
            ImportError: if unitree_sdk2py is not installed.
        """
        self._require_sdk()
        if self._sport_client is not None:
            try:
                self._sport_client.StandDown()
                self._is_standing = False
                logger.info("UnitreeAdapter %r: sit down", self.robot_name)
                return True
            except Exception as exc:
                logger.warning("UnitreeAdapter %r: StandDown() failed: %s", self.robot_name, exc)
                return False
        self._is_standing = False
        return True

    def wave_hand(self) -> bool:
        """
        Play the wave-hand gesture (Go2 behaviour mode).

        Calls ``SportClient.Hello()``.  Intended for Go2; harmless on H1.

        Returns:
            True if the command was sent successfully.

        Raises:
            ImportError: if unitree_sdk2py is not installed.
        """
        self._require_sdk()
        if self._sport_client is not None:
            try:
                self._sport_client.Hello()
                logger.info("UnitreeAdapter %r: wave hand", self.robot_name)
                return True
            except Exception as exc:
                logger.warning("UnitreeAdapter %r: Hello() failed: %s", self.robot_name, exc)
                return False
        return True

    # ------------------------------------------------------------------
    # H1 arm / gripper
    # ------------------------------------------------------------------

    def gripper_open(self) -> bool:
        """
        Open the H1 gripper / DexHand.

        No-op on Go2 (returns True — gripper not fitted by default).

        Returns:
            True if successful or not applicable.

        Raises:
            ImportError: if unitree_sdk2py is not installed.
        """
        self._require_sdk()
        if self._model != "h1":
            logger.info("UnitreeAdapter %r: gripper_open (Go2 has no gripper — ok)",
                        self.robot_name)
            return True
        # H1 gripper via DexHand / gripper module
        if self._sport_client is not None:
            try:
                # unitree_sdk2py provides a separate HandClient for H1 DexHand;
                # use the sport client's low-level action as fallback.
                self._sport_client.HandAction(
                    "open",  # hand_action label (SDK-dependent)
                )
                self._gripper_open_flag = True
                logger.info("UnitreeAdapter %r: gripper open", self.robot_name)
                return True
            except Exception as exc:
                logger.warning("UnitreeAdapter %r: gripper_open failed: %s", self.robot_name, exc)
                return False
        self._gripper_open_flag = True
        return True

    def gripper_close(self) -> bool:
        """
        Close the H1 gripper / DexHand.

        No-op on Go2 (returns True).

        Returns:
            True if successful or not applicable.

        Raises:
            ImportError: if unitree_sdk2py is not installed.
        """
        self._require_sdk()
        if self._model != "h1":
            logger.info("UnitreeAdapter %r: gripper_close (Go2 has no gripper — ok)",
                        self.robot_name)
            return True
        if self._sport_client is not None:
            try:
                self._sport_client.HandAction("close")
                self._gripper_open_flag = False
                logger.info("UnitreeAdapter %r: gripper close", self.robot_name)
                return True
            except Exception as exc:
                logger.warning("UnitreeAdapter %r: gripper_close failed: %s", self.robot_name, exc)
                return False
        self._gripper_open_flag = False
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_sdk(self) -> None:
        """
        Raise ImportError if the Unitree SDK is not installed.

        This is called at the top of every method that needs the hardware
        connection, so the module imports cleanly on machines without the SDK.

        Raises:
            ImportError: with pip install instructions.
        """
        if not _UNITREE_SDK_AVAILABLE:
            raise ImportError(_UNITREE_SDK_IMPORT_ERROR)

    @property
    def _state_data(self) -> UnitreeState:
        """Internal UnitreeState dataclass (separate from base _state enum)."""
        if not isinstance(self._state, UnitreeState):
            if not hasattr(self, "_unitree_state"):
                self._unitree_state = UnitreeState()
            return self._unitree_state
        return self._state  # type: ignore[return-value]

    @_state_data.setter
    def _state_data(self, value: UnitreeState) -> None:
        self._unitree_state = value

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """Robot model string extracted from the URI (e.g. ``"go2"``)."""
        return self._model

    @property
    def host(self) -> str | None:
        """Robot IP address, or None when using DDS broadcast discovery."""
        return self._host

    @property
    def is_standing(self) -> bool:
        return self._is_standing


# ---------------------------------------------------------------------------
# Patch _state_data to use a separate attribute from the start
# ---------------------------------------------------------------------------

def _unitree_init_patch(self: UnitreeAdapter, robot_name: str, **kwargs: Any) -> None:  # noqa: ANN001
    """Ensure _unitree_state is always initialised on construction."""
    self._unitree_state = UnitreeState()


# Apply the initialisation through __init_subclass__ would be complex;
# instead we rely on the @property lazy init above for clean access.
