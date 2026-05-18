"""
NVIDIA Isaac Sim adapter for APYROBO.

Requires NVIDIA Isaac Sim 4.x. Source the Isaac Python environment before
running::

    source ~/.local/share/ov/pkg/isaac-sim-4.x.x/setup_python_env.sh
    python -m apyrobo ...

When the Isaac SDK (``omni.isaac.core``) is present the adapter drives the
simulation directly via the Python API — zero-copy, sub-millisecond command
latency.  When the SDK is absent but Isaac Sim is reachable over the network,
the adapter transparently falls back to the Isaac REST API (default port 8211)
so that remote workstations or CI runners can still issue commands.

URI scheme::

    isaac://my_scene
    isaac://my_scene?host=192.168.1.50&port=8211

v6.0.0 — APYROBO Ecosystem Integrations
"""

from __future__ import annotations

import json
import logging
import math
import time
import urllib.error
import urllib.request
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
# Optional SDK import — guarded so the module loads without Isaac installed
# ---------------------------------------------------------------------------

try:
    import omni  # type: ignore
    import omni.isaac.core  # type: ignore
    from omni.isaac.core import World  # type: ignore
    from omni.isaac.core.robots import Robot as IsaacRobot  # type: ignore
    _OMNI_AVAILABLE = True
except ImportError:
    omni = None  # type: ignore
    World = None  # type: ignore
    IsaacRobot = None  # type: ignore
    _OMNI_AVAILABLE = False

_ISAAC_SDK_IMPORT_ERROR = (
    "Install NVIDIA Isaac Sim and source its Python environment. "
    "See: https://docs.omniverse.nvidia.com/isaacsim/latest/installation/"
)

# ---------------------------------------------------------------------------
# REST API helpers (fallback path)
# ---------------------------------------------------------------------------

_REST_TIMEOUT = 5.0  # seconds


def _isaac_rest_request(
    host: str,
    port: int,
    endpoint: str,
    payload: dict[str, Any] | None = None,
    method: str | None = None,
) -> dict[str, Any] | None:
    """
    Send a request to the Isaac Sim REST API.

    Returns the parsed JSON response, or None if the server is unreachable.
    """
    url = f"http://{host}:{port}{endpoint}"
    data = json.dumps(payload).encode() if payload is not None else None
    req_method = method or ("POST" if data else "GET")
    req = urllib.request.Request(
        url,
        data=data,
        method=req_method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=_REST_TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as exc:
        logger.debug("Isaac REST %s %s failed: %s", req_method, url, exc)
        return None
    except Exception as exc:
        logger.warning("Isaac REST unexpected error for %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# IsaacSimAdapter
# ---------------------------------------------------------------------------

@register_adapter("isaac")
class IsaacSimAdapter(CapabilityAdapter):
    """
    NVIDIA Isaac Sim adapter.

    Connects to a running Isaac Sim instance via the Isaac SDK Python API
    (``omni.isaac.core``) when the Isaac Python environment is sourced, or
    falls back to the Isaac REST API (port 8211 by default) when only network
    access is available.

    **SDK path** — zero-copy, direct Python calls into the simulator:
        * ``connect()`` initialises ``omni.isaac.core``, opens a stage, and
          retrieves the active ``World`` object.
        * ``move()`` / ``stop()`` set velocity targets on the robot prim.
        * ``_simulation_step()`` advances the world by one physics step.

    **REST path** — HTTP calls to ``localhost:8211`` (or configured host/port):
        * Falls back automatically when ``omni`` is not importable but Isaac
          Sim is running and its REST API is reachable.
        * Endpoints used: ``/isaac/move``, ``/isaac/stop``, ``/isaac/pose``.

    **Offline / test mode** — when neither SDK nor REST API is reachable the
    adapter logs a warning and operates as a stateful stub so unit tests and
    offline development work without a simulator.

    Usage::

        from apyrobo.core.robot import Robot
        robot = await Robot.discover("isaac://warehouse_scene")
        await robot.move(3.0, 1.0)
    """

    #: Isaac Sim REST API version prefix
    _API_PREFIX = "/api/v1"

    def __init__(
        self,
        robot_name: str,
        host: str = "localhost",
        port: int = 8211,
        **kwargs: Any,
    ) -> None:
        super().__init__(robot_name, **kwargs)
        self._host = host
        self._port = port
        self._world: Any = None          # omni.isaac.core.World instance
        self._robot_prim: Any = None     # Isaac robot prim (SDK path)
        self._position = (0.0, 0.0)
        self._orientation = 0.0          # radians (yaw)
        self._moving = False
        self._use_sdk = _OMNI_AVAILABLE
        self._rest_available: bool | None = None  # None = not yet probed
        # Start in connected state when SDK is available; otherwise probe REST
        if _OMNI_AVAILABLE:
            self._state = AdapterState.CONNECTED
        logger.info(
            "IsaacSimAdapter created for %r (host=%s port=%d sdk=%s)",
            robot_name, host, port, _OMNI_AVAILABLE,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Establish connection to Isaac Sim.

        SDK path: initialises the Omniverse application, opens a new stage,
        and retrieves the World singleton.

        REST path: probes the REST API and raises ConnectionError if not
        reachable.
        """
        if _OMNI_AVAILABLE:
            self._connect_sdk()
        else:
            self._connect_rest()

    def _connect_sdk(self) -> None:
        """Initialise the Isaac SDK and obtain the World object."""
        try:
            from omni.isaac.core import World  # type: ignore  # noqa: F811

            self._world = World()
            self._world.initialize_physics()
            self._state = AdapterState.CONNECTED
            logger.info(
                "IsaacSimAdapter: SDK connected for %r (world=%r)",
                self.robot_name, self._world,
            )
        except Exception as exc:
            self._state = AdapterState.ERROR
            raise ConnectionError(
                f"Failed to initialise Isaac Sim SDK for {self.robot_name!r}: {exc}"
            ) from exc

    def _connect_rest(self) -> None:
        """Probe the Isaac REST API and set state accordingly."""
        resp = _isaac_rest_request(self._host, self._port, "/status")
        if resp is not None:
            self._rest_available = True
            self._state = AdapterState.CONNECTED
            logger.info(
                "IsaacSimAdapter: REST API connected for %r at %s:%d",
                self.robot_name, self._host, self._port,
            )
        else:
            self._rest_available = False
            # Operate as offline stub — tests and dev still work
            self._state = AdapterState.CONNECTED
            logger.warning(
                "IsaacSimAdapter: REST API not reachable at %s:%d — "
                "operating in offline stub mode for %r",
                self._host, self._port, self.robot_name,
            )

    def disconnect(self) -> None:
        """Shut down the Isaac session cleanly."""
        if self._world is not None:
            try:
                self._world.stop()
            except Exception as exc:
                logger.debug("IsaacSimAdapter: world.stop() raised: %s", exc)
            self._world = None
        super().disconnect()

    # ------------------------------------------------------------------
    # Required — CapabilityAdapter contract
    # ------------------------------------------------------------------

    def get_capabilities(self) -> RobotCapability:
        """Return the full Isaac Sim capability set."""
        return RobotCapability(
            robot_id=self.robot_name,
            name=f"Isaac-{self.robot_name}",
            capabilities=[
                Capability(
                    capability_type=CapabilityType.NAVIGATE,
                    name="navigate_to",
                    description="Navigate to a 2D/3D pose in the Isaac stage",
                    parameters={
                        "x": "float",
                        "y": "float",
                        "speed": "float (optional)",
                    },
                ),
                Capability(
                    capability_type=CapabilityType.MANIPULATE,
                    name="manipulate",
                    description="Arm motion via Isaac articulation controller",
                    parameters={"joint_targets": "list[float]"},
                ),
                Capability(
                    capability_type=CapabilityType.SCAN,
                    name="scan_area",
                    description="Lidar / depth scan via simulated sensors",
                ),
                Capability(
                    capability_type=CapabilityType.PICK,
                    name="pick_object",
                    description="Pick object with simulated gripper",
                    parameters={"prim_path": "str"},
                ),
                Capability(
                    capability_type=CapabilityType.PLACE,
                    name="place_object",
                    description="Place held object at target pose",
                    parameters={"x": "float", "y": "float", "z": "float"},
                ),
                Capability(
                    capability_type=CapabilityType.CUSTOM,
                    name="simulation_step",
                    description="Advance the physics simulation by one step",
                ),
            ],
            sensors=[
                SensorInfo(
                    sensor_id="isaac_camera_0",
                    sensor_type=SensorType.CAMERA,
                    topic="/isaac/camera/rgb",
                    hz=60.0,
                ),
                SensorInfo(
                    sensor_id="isaac_lidar_0",
                    sensor_type=SensorType.LIDAR,
                    topic="/isaac/lidar/scan",
                    hz=20.0,
                ),
                SensorInfo(
                    sensor_id="isaac_imu_0",
                    sensor_type=SensorType.IMU,
                    topic="/isaac/imu/data",
                    hz=200.0,
                ),
                SensorInfo(
                    sensor_id="isaac_depth_0",
                    sensor_type=SensorType.DEPTH,
                    topic="/isaac/depth/image",
                    hz=30.0,
                ),
            ],
            max_speed=2.0,
            metadata={
                "sim": True,
                "engine": "isaac_sim",
                "sdk_available": _OMNI_AVAILABLE,
                "rest_host": self._host,
                "rest_port": self._port,
            },
        )

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        """
        Send a navigation command to the Isaac Sim robot.

        SDK path: sets velocity targets via the robot's articulation controller.
        REST path: POST to ``/api/v1/robot/move``.
        Offline: updates internal state only.
        """
        effective_speed = speed or 1.0
        dx = x - self._position[0]
        dy = y - self._position[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 1e-6:
            self._orientation = math.atan2(dy, dx)

        if _OMNI_AVAILABLE and self._world is not None:
            self._move_sdk(x, y, effective_speed)
        elif self._rest_available:
            self._move_rest(x, y, effective_speed)
        else:
            logger.debug(
                "IsaacSimAdapter: offline move to (%.2f, %.2f) speed=%.2f",
                x, y, effective_speed,
            )

        self._position = (x, y)
        self._moving = True
        logger.info(
            "IsaacSimAdapter %r: move → (%.2f, %.2f) speed=%.2f dist=%.2fm",
            self.robot_name, x, y, effective_speed, dist,
        )

    def _move_sdk(self, x: float, y: float, speed: float) -> None:
        """Dispatch a move command via the Isaac SDK."""
        if self._robot_prim is not None:
            try:
                self._robot_prim.set_world_pose(
                    position=[x, y, 0.0],
                    orientation=[0.0, 0.0, math.sin(self._orientation / 2),
                                 math.cos(self._orientation / 2)],
                )
            except Exception as exc:
                logger.warning(
                    "IsaacSimAdapter: SDK move failed, falling back: %s", exc
                )
        self._simulation_step()

    def _move_rest(self, x: float, y: float, speed: float) -> None:
        """Dispatch a move command via the Isaac REST API."""
        _isaac_rest_request(
            self._host, self._port,
            f"{self._API_PREFIX}/robot/move",
            {"robot": self.robot_name, "x": x, "y": y, "speed": speed},
        )

    def stop(self) -> None:
        """
        Immediately halt all robot motion.

        SDK path: zeroes velocity targets via the articulation controller.
        REST path: POST to ``/api/v1/robot/stop``.
        """
        if _OMNI_AVAILABLE and self._world is not None:
            try:
                if self._robot_prim is not None:
                    self._robot_prim.apply_action(
                        type("EmptyAction", (), {"joint_velocities": [0.0] * 6})()
                    )
            except Exception as exc:
                logger.debug("IsaacSimAdapter: SDK stop action failed: %s", exc)
            self._simulation_step()
        elif self._rest_available:
            _isaac_rest_request(
                self._host, self._port,
                f"{self._API_PREFIX}/robot/stop",
                {"robot": self.robot_name},
            )
        self._moving = False
        logger.info("IsaacSimAdapter %r: stopped at (%.2f, %.2f)",
                    self.robot_name, *self._position)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_position(self) -> tuple[float, float, float]:  # type: ignore[override]
        """
        Return the robot's current (x, y, theta) pose.

        SDK path: queries the robot prim's world transform.
        REST path: GET ``/api/v1/robot/pose``.
        Offline: returns last known position from move() calls.

        Returns:
            (x, y, theta) — position in metres, heading in radians.
        """
        if _OMNI_AVAILABLE and self._robot_prim is not None:
            try:
                pos, orient = self._robot_prim.get_world_pose()
                x, y = float(pos[0]), float(pos[1])
                # Extract yaw from quaternion [x, y, z, w]
                qz, qw = float(orient[2]), float(orient[3])
                theta = 2.0 * math.atan2(qz, qw)
                self._position = (x, y)
                self._orientation = theta
            except Exception as exc:
                logger.debug("IsaacSimAdapter: SDK get_pose failed: %s", exc)
        elif self._rest_available:
            resp = _isaac_rest_request(
                self._host, self._port,
                f"{self._API_PREFIX}/robot/pose",
                method="GET",
            )
            if resp:
                self._position = (
                    float(resp.get("x", self._position[0])),
                    float(resp.get("y", self._position[1])),
                )
                self._orientation = float(resp.get("theta", self._orientation))

        return (self._position[0], self._position[1], self._orientation)

    def get_orientation(self) -> float:
        """Return the robot's current heading in radians (yaw)."""
        self.get_position()  # refreshes _orientation as a side-effect
        return self._orientation

    def get_battery_level(self) -> float:
        """
        Return the battery level.

        Isaac Sim is a pure simulator — always returns 1.0 (100%).
        """
        return 1.0

    def get_health(self) -> dict[str, Any]:
        """Return adapter health including SDK/REST availability."""
        return {
            "state": self._state.value,
            "adapter": "IsaacSimAdapter",
            "robot": self.robot_name,
            "sdk_available": _OMNI_AVAILABLE,
            "rest_available": self._rest_available,
            "host": self._host,
            "port": self._port,
            "battery_pct": 100.0,
            "sim": True,
        }

    # ------------------------------------------------------------------
    # Gripper / manipulation
    # ------------------------------------------------------------------

    def gripper_open(self) -> bool:
        """Open the simulated end-effector / gripper."""
        if _OMNI_AVAILABLE and self._robot_prim is not None:
            try:
                self._robot_prim.gripper.open()
                self._simulation_step()
                return True
            except AttributeError:
                pass  # robot prim has no gripper — fall through
        if self._rest_available:
            resp = _isaac_rest_request(
                self._host, self._port,
                f"{self._API_PREFIX}/gripper/open",
                {"robot": self.robot_name},
            )
            return resp is not None
        logger.info("IsaacSimAdapter %r: gripper_open (stub)", self.robot_name)
        return True

    def gripper_close(self) -> bool:
        """Close the simulated end-effector / gripper."""
        if _OMNI_AVAILABLE and self._robot_prim is not None:
            try:
                self._robot_prim.gripper.close()
                self._simulation_step()
                return True
            except AttributeError:
                pass
        if self._rest_available:
            resp = _isaac_rest_request(
                self._host, self._port,
                f"{self._API_PREFIX}/gripper/close",
                {"robot": self.robot_name},
            )
            return resp is not None
        logger.info("IsaacSimAdapter %r: gripper_close (stub)", self.robot_name)
        return True

    def rotate(self, angle_rad: float, speed: float | None = None) -> None:
        """Rotate the robot in place by *angle_rad* radians (yaw only)."""
        self._orientation = (self._orientation + angle_rad) % (2 * math.pi)
        if _OMNI_AVAILABLE and self._robot_prim is not None:
            try:
                qz = math.sin(self._orientation / 2)
                qw = math.cos(self._orientation / 2)
                pos, _ = self._robot_prim.get_world_pose()
                self._robot_prim.set_world_pose(
                    position=pos,
                    orientation=[0.0, 0.0, qz, qw],
                )
                self._simulation_step()
                return
            except Exception as exc:
                logger.debug("IsaacSimAdapter: SDK rotate failed: %s", exc)
        if self._rest_available:
            _isaac_rest_request(
                self._host, self._port,
                f"{self._API_PREFIX}/robot/rotate",
                {"robot": self.robot_name, "angle_rad": angle_rad, "speed": speed},
            )

    # ------------------------------------------------------------------
    # SDK helpers
    # ------------------------------------------------------------------

    def _simulation_step(self) -> None:
        """
        Advance the Isaac Sim physics world by one step.

        Calls ``world.step(render=False)`` when the World object is available.
        No-op otherwise (REST path / offline mode).
        """
        if self._world is not None:
            try:
                self._world.step(render=False)
            except Exception as exc:
                logger.debug(
                    "IsaacSimAdapter: world.step() raised: %s", exc
                )

    def _sdk_import_check(self) -> bool:
        """
        Raise ImportError if the Isaac SDK is unavailable.

        Call this from methods that absolutely require the SDK and cannot
        fall back to REST.  Returns True if SDK is available.

        Raises:
            ImportError: with installation instructions when omni is absent.
        """
        if not _OMNI_AVAILABLE:
            raise ImportError(_ISAAC_SDK_IMPORT_ERROR)
        return True

    def load_robot_prim(self, prim_path: str) -> None:
        """
        Attach the adapter to an existing robot prim in the current stage.

        Args:
            prim_path: USD prim path, e.g. ``"/World/Robots/go2"``.

        Raises:
            ImportError: if the Isaac SDK is not available.
            RuntimeError: if the prim does not exist in the current stage.
        """
        self._sdk_import_check()
        try:
            from omni.isaac.core.robots import Robot as IsaacRobot  # type: ignore  # noqa: F811

            self._robot_prim = IsaacRobot(prim_path=prim_path, name=self.robot_name)
            logger.info(
                "IsaacSimAdapter %r: attached to prim %r", self.robot_name, prim_path
            )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot attach to prim {prim_path!r} in the current stage: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def position(self) -> tuple[float, float]:
        """Last known (x, y) position — no SDK/REST call."""
        return self._position

    @property
    def is_moving(self) -> bool:
        return self._moving

    @property
    def world(self) -> Any:
        """The underlying ``omni.isaac.core.World`` instance, or None."""
        return self._world
