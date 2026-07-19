"""Real MuJoCo bridge for the ``mujoco://`` scheme.

Unlike the retired in-memory stand-in, this adapter loads an actual MuJoCo
model, steps real physics in a background thread, and drives the robot
through position-servo actuators. ``move()`` blocks until the base reaches
the target (like Nav2's action interface), so a skill graph's pick step
runs only after the robot has physically arrived. ``stop()`` from another
thread interrupts an in-flight move.

Requires the ``mujoco`` package (``pip install 'apyrobo[mujoco]'``). The
default scene is a planar mobile base with a suction-style gripper and one
graspable "package" box (``assets/pickplace_arena.xml``); pass
``model_path=`` for your own scene using the same element names:

- joints ``base_x``, ``base_y``, ``base_yaw`` with position actuators
  ``drive_x``, ``drive_y``, ``drive_yaw``
- site ``grip_site`` (grasp reach is measured from here)
- an inactive weld ``grasp_weld`` from the base to the graspable body
"""

from __future__ import annotations

import logging
import threading
import time
from importlib import resources
from typing import Any

from apyrobo.core.adapters import CapabilityAdapter, register_adapter
from apyrobo.core.schemas import (
    AdapterState,
    Capability,
    CapabilityType,
    RobotCapability,
)

logger = logging.getLogger(__name__)

_ARRIVAL_TOL = 0.05  # metres
_YAW_TOL = 0.05  # radians


class MuJoCoNotInstalledError(ImportError):
    """Raised when the mujoco:// scheme is used without the mujoco package."""


@register_adapter("mujoco")
class MuJoCoBridgeAdapter(CapabilityAdapter):
    """Live MuJoCo physics adapter for ``mujoco://<name>``.

    Keyword args (via ``Robot.discover(..., **kwargs)``):
        model_path: path to a MuJoCo XML scene (default: bundled arena)
        max_speed: setpoint speed cap in m/s (default 2.0)
        realtime: pace the physics thread to wall-clock time (default
            False — run as fast as possible; set True for viewers/demos)
        grasp_radius: max distance from grip_site for a grasp (default 0.4)
        move_timeout_extra: seconds of sim-time grace on blocking moves
            beyond the ideal travel time (default 10.0)
    """

    def __init__(self, robot_name: str, **kwargs: Any) -> None:
        super().__init__(robot_name, **kwargs)
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - exercised on bare installs
            raise MuJoCoNotInstalledError(
                "mujoco:// needs the MuJoCo bindings — install them with: "
                "pip install 'apyrobo[mujoco]'"
            ) from exc
        self._mj = mujoco

        model_path = kwargs.get("model_path")
        if model_path:
            self._model_path = str(model_path)
        else:
            self._model_path = str(
                resources.files("apyrobo.sim").joinpath("assets/pickplace_arena.xml")
            )
        self._model = mujoco.MjModel.from_xml_path(self._model_path)
        self._data = mujoco.MjData(self._model)

        self._jid = {
            name: self._name2id(mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ("base_x", "base_y", "base_yaw")
        }
        self._aid = {
            name: self._name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ("drive_x", "drive_y", "drive_yaw")
        }
        self._site_id = self._name2id(mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        self._weld_id = self._name2id(
            mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld", required=False
        )
        self._grasp_body_id = (
            int(self._model.eq_obj2id[self._weld_id])
            if self._weld_id is not None
            else None
        )

        self.max_speed = float(kwargs.get("max_speed", 2.0))
        self._realtime = bool(kwargs.get("realtime", False))
        self._grasp_radius = float(kwargs.get("grasp_radius", 0.4))
        self._lift_height = float(kwargs.get("lift_height", 0.08))
        self._move_timeout_extra = float(kwargs.get("move_timeout_extra", 10.0))

        self._lock = threading.Lock()
        self._arrived = threading.Condition(self._lock)
        # Setpoint the servo targets track; the goal it slews toward.
        self._setpoint = [0.0, 0.0]
        self._goal: tuple[float, float] | None = None
        self._yaw_goal: float | None = None
        self._speed = self.max_speed
        self._moving = False
        self._holding = False

        mujoco.mj_forward(self._model, self._data)
        self._setpoint = [self._qpos("base_x"), self._qpos("base_y")]

        self._state = AdapterState.CONNECTED
        self._shutdown = threading.Event()
        self._thread = threading.Thread(
            target=self._physics_loop, name=f"mujoco-{robot_name}", daemon=True
        )
        self._thread.start()
        logger.info(
            "MuJoCoBridgeAdapter %r: live physics from %s (nq=%d)",
            robot_name, self._model_path, self._model.nq,
        )

    # ------------------------------------------------------------------
    # Model lookups
    # ------------------------------------------------------------------

    def _name2id(self, objtype: Any, name: str, required: bool = True) -> int | None:
        oid = self._mj.mj_name2id(self._model, objtype, name)
        if oid < 0:
            if required:
                raise ValueError(
                    f"MuJoCo scene {self._model_path!r} has no element "
                    f"{name!r} — see mujoco_bridge.py for the naming contract"
                )
            return None
        return int(oid)

    def _qpos(self, joint: str) -> float:
        return float(self._data.qpos[self._model.jnt_qposadr[self._jid[joint]]])

    # ------------------------------------------------------------------
    # Physics thread
    # ------------------------------------------------------------------

    def _physics_loop(self) -> None:
        dt = float(self._model.opt.timestep)
        while not self._shutdown.is_set():
            with self._lock:
                if self._state == AdapterState.CONNECTED:
                    self._advance_setpoint(dt)
                    self._data.ctrl[self._aid["drive_x"]] = self._setpoint[0]
                    self._data.ctrl[self._aid["drive_y"]] = self._setpoint[1]
                    if self._yaw_goal is not None:
                        self._data.ctrl[self._aid["drive_yaw"]] = self._yaw_goal
                    self._mj.mj_step(self._model, self._data)
                    self._update_arrival()
            if self._realtime:
                time.sleep(dt)

    def _advance_setpoint(self, dt: float) -> None:
        if self._goal is None:
            return
        gx, gy = self._goal
        dx, dy = gx - self._setpoint[0], gy - self._setpoint[1]
        dist = (dx * dx + dy * dy) ** 0.5
        step = self._speed * dt
        if dist <= step:
            self._setpoint = [gx, gy]
        else:
            self._setpoint[0] += dx / dist * step
            self._setpoint[1] += dy / dist * step

    def _update_arrival(self) -> None:
        if not self._moving:
            return
        pos_ok = True
        if self._goal is not None:
            gx, gy = self._goal
            pos_ok = (
                self._setpoint == [gx, gy]
                and abs(self._qpos("base_x") - gx) < _ARRIVAL_TOL
                and abs(self._qpos("base_y") - gy) < _ARRIVAL_TOL
            )
        yaw_ok = (
            self._yaw_goal is None
            or abs(self._qpos("base_yaw") - self._yaw_goal) < _YAW_TOL
        )
        if pos_ok and yaw_ok:
            self._moving = False
            self._arrived.notify_all()

    def _require_connected(self) -> None:
        if self._state != AdapterState.CONNECTED:
            raise ConnectionError(
                f"MuJoCoBridgeAdapter {self.robot_name!r} is disconnected — "
                "commands fail fast rather than queue"
            )

    # ------------------------------------------------------------------
    # Required contract
    # ------------------------------------------------------------------

    def get_capabilities(self) -> RobotCapability:
        caps = [
            Capability(capability_type=CapabilityType.NAVIGATE, name="navigate_to"),
            Capability(capability_type=CapabilityType.ROTATE, name="rotate"),
        ]
        if self._weld_id is not None:
            caps.append(Capability(capability_type=CapabilityType.PICK, name="pick_object"))
            caps.append(Capability(capability_type=CapabilityType.PLACE, name="place_object"))
        return RobotCapability(
            robot_id=self.robot_name,
            name=f"MuJoCo-{self.robot_name}",
            capabilities=caps,
            metadata={
                "backend": "mujoco",
                "model_path": self._model_path,
                "mujoco_version": getattr(self._mj, "__version__", "unknown"),
                "real_physics": True,
            },
            max_speed=self.max_speed,
        )

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        """Drive the base to (x, y); blocks until arrival, stop(), or timeout."""
        self._require_connected()
        with self._lock:
            requested = self.max_speed if speed is None else float(speed)
            self._speed = max(0.01, min(requested, self.max_speed))
            dist = (
                (x - self._qpos("base_x")) ** 2 + (y - self._qpos("base_y")) ** 2
            ) ** 0.5
            self._goal = (float(x), float(y))
            self._moving = True
            deadline_sim = (
                float(self._data.time) + dist / self._speed + self._move_timeout_extra
            )
            while self._moving and self._state == AdapterState.CONNECTED:
                if float(self._data.time) >= deadline_sim:
                    logger.warning(
                        "MuJoCoBridgeAdapter %r: move(%.2f, %.2f) timed out at "
                        "(%.2f, %.2f)", self.robot_name, x, y,
                        self._qpos("base_x"), self._qpos("base_y"),
                    )
                    self._moving = False
                    break
                self._arrived.wait(timeout=0.05)

    def stop(self) -> None:
        """Halt: freeze the setpoint at the current pose. Safe when disconnected."""
        with self._lock:
            here = [self._qpos("base_x"), self._qpos("base_y")]
            self._setpoint = here
            self._goal = None
            self._yaw_goal = None
            self._data.ctrl[self._aid["drive_x"]] = here[0]
            self._data.ctrl[self._aid["drive_y"]] = here[1]
            self._data.ctrl[self._aid["drive_yaw"]] = self._qpos("base_yaw")
            self._data.qvel[:] = 0.0
            self._moving = False
            self._arrived.notify_all()

    # ------------------------------------------------------------------
    # Optional contract
    # ------------------------------------------------------------------

    def rotate(self, angle_rad: float, speed: float | None = None) -> None:
        """Rotate in place by *angle_rad*; blocks until settled or timeout."""
        self._require_connected()
        with self._lock:
            self._yaw_goal = self._qpos("base_yaw") + float(angle_rad)
            self._moving = True
            deadline_sim = float(self._data.time) + abs(angle_rad) / 0.5 + 5.0
            while self._moving and self._state == AdapterState.CONNECTED:
                if float(self._data.time) >= deadline_sim:
                    self._moving = False
                    break
                self._arrived.wait(timeout=0.05)
            self._yaw_goal = None

    def gripper_close(self) -> bool:
        """Grasp the scene's graspable body if it is within reach.

        Suction-style: activates the ``grasp_weld`` equality at the current
        relative pose, so the object is carried by the constraint solver —
        no finger contact tuning, but real physics from there on.
        """
        self._require_connected()
        if self._weld_id is None:
            logger.warning(
                "MuJoCoBridgeAdapter %r: scene has no grasp_weld — cannot grasp",
                self.robot_name,
            )
            return False
        with self._lock:
            if self._holding:
                return True
            base_id = int(self._model.eq_obj1id[self._weld_id])
            obj_id = int(self._model.eq_obj2id[self._weld_id])
            site = self._data.site_xpos[self._site_id]
            obj = self._data.xpos[obj_id]
            dist = float(sum((a - b) ** 2 for a, b in zip(site, obj)) ** 0.5)
            if dist > self._grasp_radius:
                logger.info(
                    "MuJoCoBridgeAdapter %r: object %.2fm from grip_site "
                    "(> %.2fm) — grasp refused", self.robot_name, dist,
                    self._grasp_radius,
                )
                return False
            self._weld_at_current_pose(base_id, obj_id)
            self._data.eq_active[self._weld_id] = 1
            self._holding = True
            return True

    def gripper_open(self) -> bool:
        """Release the grasp; the object is handed back to gravity."""
        self._require_connected()
        if self._weld_id is None:
            return False
        with self._lock:
            self._data.eq_active[self._weld_id] = 0
            self._holding = False
            return True

    def _weld_at_current_pose(self, body1: int, body2: int) -> None:
        """Point the weld's eq_data at the bodies' current relative pose."""
        import numpy as np

        p1, q1 = self._data.xpos[body1], self._data.xquat[body1]
        p2, q2 = self._data.xpos[body2], self._data.xquat[body2]
        q1_inv = np.zeros(4)
        self._mj.mju_negQuat(q1_inv, q1)
        rel_q = np.zeros(4)
        self._mj.mju_mulQuat(rel_q, q1_inv, q2)
        rel_p = np.zeros(3)
        self._mj.mju_rotVecQuat(rel_p, p2 - p1, q1_inv)
        # Suction lift: hold the object above its grasped height so it
        # clears the floor — the constraint solver carries it against
        # gravity, like a magnet crane.
        rel_p[2] += self._lift_height
        # weld eq_data: anchor(3) in body2 frame, relpose(3 pos + 4 quat),
        # torquescale(1)
        data = self._model.eq_data[self._weld_id]
        data[0:3] = 0.0
        data[3:6] = rel_p
        data[6:10] = rel_q

    def get_position(self) -> tuple[float, float]:
        with self._lock:
            return (self._qpos("base_x"), self._qpos("base_y"))

    def get_orientation(self) -> float:
        with self._lock:
            return self._qpos("base_yaw")

    def get_health(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state.value,
                "adapter": type(self).__name__,
                "robot": self.robot_name,
                "sim_time_s": float(self._data.time),
                "holding": self._holding,
                "moving": self._moving,
            }

    @property
    def is_moving(self) -> bool:
        with self._lock:
            return self._moving

    def object_position(self) -> tuple[float, float, float] | None:
        """World position of the scene's graspable body (None without one)."""
        if self._grasp_body_id is None:
            return None
        with self._lock:
            x, y, z = self._data.xpos[self._grasp_body_id]
            return (float(x), float(y), float(z))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        with self._lock:
            self._state = AdapterState.CONNECTED
        logger.info("MuJoCoBridgeAdapter %r: connected", self.robot_name)

    def disconnect(self) -> None:
        with self._lock:
            was_connected = self._state == AdapterState.CONNECTED
            self._state = AdapterState.DISCONNECTED
            self._moving = False
            self._arrived.notify_all()
        if was_connected:
            self._last_disconnect_time = time.time()
            self._notify_disconnect()
        logger.info("MuJoCoBridgeAdapter %r: disconnected", self.robot_name)

    def shutdown(self) -> None:
        """Stop the physics thread entirely (disconnect keeps it idling)."""
        self._shutdown.set()
        self._thread.join(timeout=2.0)
