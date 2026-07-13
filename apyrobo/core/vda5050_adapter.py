"""
VDA5050 master-control adapter — drive any VDA5050-compliant AMR fleet.

`vda5050://` implements the **master-controller side** of VDA 5050 (the
MQTT interface standard the industrial-AMR world is converging on: MiR,
OTTO/Rockwell, Seegrid, BlueBotics). No ROS is required on the robot —
this is APYROBO's first non-ROS base.

    robot = Robot.discover(
        "vda5050://AGV-0042",
        broker="10.0.0.5:1883",
        manufacturer="mir",
    )
    robot.move(x=12.0, y=3.5)     # → publishes a VDA5050 Order
    robot.stop()                  # → instantAction cancelOrder

Topic layout (VDA5050 §6.2), master side:

    {interface}/v{major}/{manufacturer}/{serial}/order           ← we publish
    {interface}/v{major}/{manufacturer}/{serial}/instantActions  ← we publish
    {interface}/v{major}/{manufacturer}/{serial}/state           → we consume
    {interface}/v{major}/{manufacturer}/{serial}/connection      → we consume
    {interface}/v{major}/{manufacturer}/{serial}/factsheet       → we consume

`move()` publishes a two-node order (current position → goal, one edge,
both released — VDA5050 §6.6's "first node must be trivially reachable")
and blocks until the state stream reports the goal node reached, an order
error, or `order_timeout_sec`. `stop()` publishes a HARD-blocking
`cancelOrder` instant action and never raises (safety contract §2).

Transport: paho-mqtt (`pip install 'apyrobo[vda5050]'`) by default; any
object satisfying the small `Vda5050Transport` duck type can be injected
via the ``transport=`` kwarg — the test suite drives the full protocol
against an in-process simulated AGV this way, brokerless.

Verified today against that simulated AGV and the conformance suite —
**not yet against physical hardware** (that is the Arc-1 gate; see
ROADMAP.md).
"""
from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from apyrobo.core.adapters import CapabilityAdapter, register_adapter
from apyrobo.core.schemas import (
    Capability,
    CapabilityType,
    RobotCapability,
)

logger = logging.getLogger(__name__)

try:  # optional dependency — tests inject a transport instead
    import paho.mqtt.client as _paho  # type: ignore
    _HAS_PAHO = True
except ImportError:  # pragma: no cover
    _paho = None
    _HAS_PAHO = False


class Vda5050Error(RuntimeError):
    """Raised on order rejection, transport failure, or timeout."""


# ---------------------------------------------------------------------------
# Transport — paho by default, injectable for tests
# ---------------------------------------------------------------------------

class _PahoTransport:
    """Thin wrapper mapping the adapter's needs onto paho-mqtt 2.x."""

    def __init__(self) -> None:
        if not _HAS_PAHO:
            raise ImportError(
                "paho-mqtt is required for vda5050:// — install with: "
                "pip install 'apyrobo[vda5050]'"
            )
        self.on_message: Callable[[str, str], None] | None = None
        self._client = _paho.Client(
            callback_api_version=_paho.CallbackAPIVersion.VERSION2,
            client_id=f"apyrobo-master-{uuid.uuid4().hex[:8]}",
        )
        self._client.on_message = self._dispatch

    def _dispatch(self, _client: Any, _userdata: Any, msg: Any) -> None:
        if self.on_message is not None:
            self.on_message(msg.topic, msg.payload.decode("utf-8", "replace"))

    def connect(self, host: str, port: int) -> None:
        self._client.connect(host, port)
        self._client.loop_start()

    def disconnect(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def subscribe(self, topic: str) -> None:
        self._client.subscribe(topic)

    def publish(self, topic: str, payload: str) -> None:
        self._client.publish(topic, payload, qos=0)


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

@register_adapter("vda5050")
class Vda5050Adapter(CapabilityAdapter):
    """Master-control adapter for a single VDA5050 AGV (see module docs)."""

    ORDER_TIMEOUT_SEC = 120.0

    def __init__(self, robot_name: str, **kwargs: Any) -> None:
        super().__init__(robot_name, **kwargs)
        self._broker: str = kwargs.get("broker", "localhost:1883")
        self._manufacturer: str = kwargs.get("manufacturer", "apyrobo")
        self._interface: str = kwargs.get("interface_name", "uagv")
        self._version: str = kwargs.get("protocol_version", "2.0.0")
        self._map_id: str = kwargs.get("map_id", "map")
        self._order_timeout_sec: float = float(
            kwargs.get("order_timeout_sec", self.ORDER_TIMEOUT_SEC)
        )
        self._transport = kwargs.get("transport") or _PahoTransport()

        major = self._version.split(".", 1)[0]
        self._base_topic = (
            f"{self._interface}/v{major}/{self._manufacturer}/{robot_name}"
        )

        self._lock = threading.Lock()
        self._header_id = 0
        self._position: tuple[float, float] = (0.0, 0.0)
        self._orientation = 0.0
        self._battery_pct: float | None = None
        self._agv_errors: list[dict[str, Any]] = []
        self._agv_online: bool | None = None
        self._factsheet: dict[str, Any] = {}
        self._driving = False

        # Active-order tracking for blocking move()
        self._active_order_id: str | None = None
        self._goal_node_id: str | None = None
        self._order_done = threading.Event()
        self._order_error: str | None = None

        logger.info(
            "Vda5050Adapter created for %s (broker=%s, topic=%s)",
            robot_name, self._broker, self._base_topic,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        host, _, port = self._broker.partition(":")
        self._transport.on_message = self._on_message
        self._transport.connect(host or "localhost", int(port or 1883))
        for sub in ("state", "connection", "factsheet"):
            self._transport.subscribe(f"{self._base_topic}/{sub}")
        super().connect()
        logger.info("Vda5050Adapter: connected to %s", self._broker)

    def disconnect(self) -> None:
        with contextlib.suppress(Exception):
            self._transport.disconnect()
        super().disconnect()

    # ------------------------------------------------------------------
    # Inbound: state / connection / factsheet
    # ------------------------------------------------------------------

    def _on_message(self, topic: str, payload: str) -> None:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Vda5050Adapter: non-JSON on %s", topic)
            return
        if not isinstance(data, dict):
            return
        suffix = topic.rsplit("/", 1)[-1]
        if suffix == "state":
            self._handle_state(data)
        elif suffix == "connection":
            self._agv_online = data.get("connectionState") == "ONLINE"
        elif suffix == "factsheet":
            self._factsheet = data

    def _handle_state(self, s: dict[str, Any]) -> None:
        with self._lock:
            pos = s.get("agvPosition") or {}
            if isinstance(pos, dict) and "x" in pos and "y" in pos:
                self._position = (float(pos["x"]), float(pos["y"]))
                self._orientation = float(pos.get("theta", self._orientation))
            batt = s.get("batteryState") or {}
            if isinstance(batt, dict) and "batteryCharge" in batt:
                self._battery_pct = float(batt["batteryCharge"])
            errors = s.get("errors") or []
            self._agv_errors = errors if isinstance(errors, list) else []
            self._driving = bool(s.get("driving", False))

            # Order progress (VDA5050 §6.10): done when the state references
            # our order, the node/edge horizons are empty, and lastNodeId is
            # the goal.
            if self._active_order_id and s.get("orderId") == self._active_order_id:
                fatal = [
                    e for e in self._agv_errors
                    if isinstance(e, dict) and e.get("errorLevel") == "FATAL"
                ]
                order_refs = [
                    e for e in self._agv_errors
                    if isinstance(e, dict) and any(
                        r.get("referenceKey") == "orderId"
                        and r.get("referenceValue") == self._active_order_id
                        for r in (e.get("errorReferences") or [])
                        if isinstance(r, dict)
                    )
                ]
                if fatal or order_refs:
                    err = (order_refs or fatal)[0]
                    self._order_error = str(
                        err.get("errorType") or err.get("errorDescription") or "AGV error"
                    )
                    self._order_done.set()
                elif (
                    not s.get("nodeStates")
                    and not s.get("edgeStates")
                    and s.get("lastNodeId") == self._goal_node_id
                ):
                    self._order_error = None
                    self._order_done.set()

    # ------------------------------------------------------------------
    # Outbound: order / instantActions
    # ------------------------------------------------------------------

    def _header(self) -> dict[str, Any]:
        with self._lock:
            self._header_id += 1
            hid = self._header_id
        return {
            "headerId": hid,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "version": self._version,
            "manufacturer": self._manufacturer,
            "serialNumber": self.robot_name,
        }

    def _publish(self, subtopic: str, message: dict[str, Any]) -> None:
        self._transport.publish(
            f"{self._base_topic}/{subtopic}", json.dumps(message)
        )

    def _build_order(
        self, x: float, y: float, speed: float | None
    ) -> dict[str, Any]:
        order_id = f"apyrobo-{uuid.uuid4().hex[:12]}"
        start_id, goal_id = f"{order_id}-n0", f"{order_id}-n1"
        sx, sy = self._position
        edge: dict[str, Any] = {
            "edgeId": f"{order_id}-e0",
            "sequenceId": 1,
            "released": True,
            "startNodeId": start_id,
            "endNodeId": goal_id,
            "actions": [],
        }
        if speed is not None:
            edge["maxSpeed"] = float(speed)
        return {
            **self._header(),
            "orderId": order_id,
            "orderUpdateId": 0,
            "nodes": [
                {
                    "nodeId": start_id,
                    "sequenceId": 0,
                    "released": True,
                    "nodePosition": {"x": sx, "y": sy, "mapId": self._map_id},
                    "actions": [],
                },
                {
                    "nodeId": goal_id,
                    "sequenceId": 2,
                    "released": True,
                    "nodePosition": {"x": float(x), "y": float(y), "mapId": self._map_id},
                    "actions": [],
                },
            ],
            "edges": [edge],
        }

    # ------------------------------------------------------------------
    # CapabilityAdapter interface
    # ------------------------------------------------------------------

    def move(self, x: float, y: float, speed: float | None = None) -> None:
        if not self.is_connected:
            raise Vda5050Error(
                "vda5050:// is not connected — call connect() first "
                "(commands are never queued for replay; wire-safety §4)"
            )
        order = self._build_order(x, y, speed)
        with self._lock:
            self._active_order_id = order["orderId"]
            self._goal_node_id = order["nodes"][1]["nodeId"]
            self._order_error = None
        self._order_done.clear()
        self._publish("order", order)
        logger.info(
            "Vda5050Adapter: order %s → (%.2f, %.2f)", order["orderId"], x, y
        )

        if not self._order_done.wait(timeout=self._order_timeout_sec):
            self.stop()
            raise Vda5050Error(
                f"order {order['orderId']} timed out after "
                f"{self._order_timeout_sec:.0f}s (goal ({x}, {y}))"
            )
        if self._order_error is not None:
            raise Vda5050Error(
                f"order {order['orderId']} failed on the AGV: {self._order_error}"
            )

    def stop(self) -> None:
        """Publish cancelOrder. Never raises — safety contract (§2)."""
        action = {
            **self._header(),
            "actions": [
                {
                    "actionType": "cancelOrder",
                    "actionId": f"apyrobo-cancel-{uuid.uuid4().hex[:8]}",
                    "blockingType": "HARD",
                }
            ],
        }
        with contextlib.suppress(Exception):
            self._publish("instantActions", action)
        with self._lock:
            self._active_order_id = None
            self._goal_node_id = None
        self._order_done.set()

    def cancel(self) -> None:
        self.stop()

    def get_capabilities(self) -> RobotCapability:
        agv_class = str(self._factsheet.get("typeSpecification", {}).get(
            "agvClass", "CARRIER"
        )) if isinstance(self._factsheet.get("typeSpecification"), dict) else "CARRIER"
        max_speed = 1.0
        phys = self._factsheet.get("physicalParameters")
        if isinstance(phys, dict) and "speedMax" in phys:
            with contextlib.suppress(TypeError, ValueError):
                max_speed = float(phys["speedMax"])
        return RobotCapability(
            robot_id=self.robot_name,
            name=f"VDA5050-{self.robot_name}",
            capabilities=[
                Capability(
                    capability_type=CapabilityType.NAVIGATE,
                    name="navigate_to",
                    description="VDA5050 order to a target node",
                ),
            ],
            sensors=[],
            max_speed=max_speed,
            metadata={
                "transport": "vda5050",
                "broker": self._broker,
                "manufacturer": self._manufacturer,
                "agv_class": agv_class,
            },
        )

    def get_position(self) -> tuple[float, float]:
        return self._position

    def get_orientation(self) -> float:
        return self._orientation

    def get_health(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "adapter": "Vda5050Adapter",
            "robot": self.robot_name,
            "broker": self._broker,
            "agv_online": self._agv_online,
            "battery_pct": self._battery_pct,
            "driving": self._driving,
            "errors": len(self._agv_errors),
        }

    @property
    def position(self) -> tuple[float, float]:
        return self._position
