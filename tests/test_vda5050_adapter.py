"""
vda5050:// master-control adapter, driven end-to-end against an
in-process simulated VDA5050 AGV (no broker, no paho — the transport is
injected). Covers: order shape, blocking move, rejection, timeout,
cancelOrder semantics, factsheet-derived capabilities, and full
conformance (`run_adapter_checks`).
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from apyrobo.conformance.adapter_checks import run_adapter_checks
from apyrobo.conformance.report import ConformanceReport
from apyrobo.core.robot import Robot
from apyrobo.core.vda5050_adapter import Vda5050Adapter, Vda5050Error

# ---------------------------------------------------------------------------
# In-process transport + simulated AGV
# ---------------------------------------------------------------------------

class FakeTransport:
    """Duck-typed Vda5050 transport routing messages to a SimulatedAgv."""

    def __init__(self, agv: SimulatedAgv) -> None:
        self.on_message: Callable[[str, str], None] | None = None
        self.agv = agv
        self.subscribed: list[str] = []
        self.published: list[tuple[str, dict[str, Any]]] = []
        self.connected = False
        agv.transport = self

    def connect(self, host: str, port: int) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def subscribe(self, topic: str) -> None:
        self.subscribed.append(topic)

    def publish(self, topic: str, payload: str) -> None:
        msg = json.loads(payload)
        self.published.append((topic, msg))
        # Deliver master → AGV synchronously, like a lossless local broker.
        self.agv.receive(topic, msg)

    def emit(self, topic: str, message: dict[str, Any]) -> None:
        """AGV → master."""
        if self.on_message is not None:
            self.on_message(topic, json.dumps(message))


class SimulatedAgv:
    """Minimal VDA5050-compliant AGV: accepts orders, drives, reports state."""

    def __init__(self, serial: str = "AGV-1", base: str = "uagv/v2/test/AGV-1") -> None:
        self.serial = serial
        self.base = base
        self.transport: FakeTransport | None = None
        self.position = (0.0, 0.0)
        self.battery = 87.0
        self.reject_orders = False
        self.silent = False           # accept orders but never progress
        self.cancelled: list[str] = []

    # -- master → AGV ---------------------------------------------------
    def receive(self, topic: str, msg: dict[str, Any]) -> None:
        if topic.endswith("/order"):
            self._handle_order(msg)
        elif topic.endswith("/instantActions"):
            for a in msg.get("actions", []):
                if a.get("actionType") == "cancelOrder":
                    self.cancelled.append(a["actionId"])

    def _state(self, **over: Any) -> dict[str, Any]:
        base = {
            "orderId": "",
            "lastNodeId": "",
            "nodeStates": [],
            "edgeStates": [],
            "agvPosition": {"x": self.position[0], "y": self.position[1],
                            "theta": 0.0, "mapId": "map"},
            "batteryState": {"batteryCharge": self.battery},
            "errors": [],
            "driving": False,
        }
        base.update(over)
        return base

    def _handle_order(self, order: dict[str, Any]) -> None:
        assert self.transport is not None
        oid = order["orderId"]
        goal = order["nodes"][-1]
        if self.reject_orders:
            self.transport.emit(f"{self.base}/state", self._state(
                orderId=oid,
                errors=[{
                    "errorType": "orderValidationError",
                    "errorLevel": "WARNING",
                    "errorReferences": [
                        {"referenceKey": "orderId", "referenceValue": oid}
                    ],
                }],
            ))
            return
        if self.silent:
            return
        gx = goal["nodePosition"]["x"]
        gy = goal["nodePosition"]["y"]
        # accepted + driving
        self.transport.emit(f"{self.base}/state", self._state(
            orderId=oid, driving=True,
            nodeStates=[{"nodeId": goal["nodeId"], "sequenceId": 2, "released": True}],
            edgeStates=[{"edgeId": order["edges"][0]["edgeId"], "sequenceId": 1,
                         "released": True}],
        ))
        # arrived
        self.position = (gx, gy)
        self.transport.emit(f"{self.base}/state", self._state(
            orderId=oid, lastNodeId=goal["nodeId"],
        ))

    # -- convenience -----------------------------------------------------
    def announce(self) -> None:
        assert self.transport is not None
        self.transport.emit(f"{self.base}/connection",
                            {"connectionState": "ONLINE"})
        self.transport.emit(f"{self.base}/factsheet", {
            "typeSpecification": {"agvClass": "CARRIER"},
            "physicalParameters": {"speedMax": 1.8},
        })
        self.transport.emit(f"{self.base}/state", self._state())


@pytest.fixture()
def rig():
    agv = SimulatedAgv()
    transport = FakeTransport(agv)
    adapter = Vda5050Adapter(
        "AGV-1", transport=transport, manufacturer="test",
        order_timeout_sec=2.0,
    )
    adapter.connect()
    agv.announce()
    return adapter, transport, agv


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

class TestWiring:
    def test_scheme_registered_via_robot_discover(self):
        agv = SimulatedAgv()
        robot = Robot.discover(
            "vda5050://AGV-1", transport=FakeTransport(agv), manufacturer="test"
        )
        assert isinstance(robot._adapter, Vda5050Adapter)

    def test_connect_subscribes_state_connection_factsheet(self, rig):
        _, transport, _ = rig
        assert sorted(t.rsplit("/", 1)[-1] for t in transport.subscribed) == [
            "connection", "factsheet", "state",
        ]

    def test_topic_base_follows_vda5050_layout(self, rig):
        adapter, _, _ = rig
        assert adapter._base_topic == "uagv/v2/test/AGV-1"

    def test_announce_populates_position_battery_online(self, rig):
        adapter, _, _ = rig
        assert adapter.get_position() == (0.0, 0.0)
        health = adapter.get_health()
        assert health["battery_pct"] == 87.0
        assert health["agv_online"] is True


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

class TestOrders:
    def test_move_publishes_wellformed_order(self, rig):
        adapter, transport, _ = rig
        adapter.move(3.0, 4.0, speed=0.7)
        topic, order = next(
            (t, m) for t, m in transport.published if t.endswith("/order")
        )
        assert topic == "uagv/v2/test/AGV-1/order"
        nodes, edges = order["nodes"], order["edges"]
        assert [n["sequenceId"] for n in nodes] == [0, 2]
        assert edges[0]["sequenceId"] == 1
        assert all(n["released"] for n in nodes) and edges[0]["released"]
        assert nodes[1]["nodePosition"]["x"] == 3.0
        assert edges[0]["maxSpeed"] == 0.7
        assert order["orderUpdateId"] == 0
        assert order["serialNumber"] == "AGV-1"

    def test_move_blocks_until_goal_and_updates_position(self, rig):
        adapter, _, _ = rig
        adapter.move(3.0, 4.0)
        assert adapter.get_position() == (3.0, 4.0)

    def test_rejected_order_raises(self, rig):
        adapter, _, agv = rig
        agv.reject_orders = True
        with pytest.raises(Vda5050Error, match="orderValidationError"):
            adapter.move(1.0, 1.0)

    def test_unanswered_order_times_out_and_cancels(self, rig):
        adapter, _, agv = rig
        agv.silent = True
        adapter._order_timeout_sec = 0.2
        with pytest.raises(Vda5050Error, match="timed out"):
            adapter.move(1.0, 1.0)
        assert agv.cancelled, "timeout must publish cancelOrder"

    def test_consecutive_orders_have_distinct_ids_and_increasing_headers(self, rig):
        adapter, transport, _ = rig
        adapter.move(1.0, 0.0)
        adapter.move(2.0, 0.0)
        orders = [m for t, m in transport.published if t.endswith("/order")]
        assert orders[0]["orderId"] != orders[1]["orderId"]
        assert orders[1]["headerId"] > orders[0]["headerId"]
        # Second order starts from where the first ended.
        assert orders[1]["nodes"][0]["nodePosition"]["x"] == 1.0


# ---------------------------------------------------------------------------
# Safety semantics
# ---------------------------------------------------------------------------

class TestSafety:
    def test_stop_publishes_hard_cancel(self, rig):
        adapter, transport, agv = rig
        adapter.stop()
        _, msg = next(
            (t, m) for t, m in transport.published
            if t.endswith("/instantActions")
        )
        action = msg["actions"][0]
        assert action["actionType"] == "cancelOrder"
        assert action["blockingType"] == "HARD"
        assert agv.cancelled

    def test_stop_never_raises_when_disconnected(self, rig):
        adapter, _, _ = rig
        adapter.disconnect()
        adapter.stop()  # must not raise
        adapter.stop()  # idempotent

    def test_move_fails_fast_when_disconnected(self, rig):
        adapter, _, _ = rig
        adapter.disconnect()
        with pytest.raises(Vda5050Error, match="not connected"):
            adapter.move(1.0, 1.0)

    def test_fatal_agv_error_fails_the_order(self, rig):
        adapter, transport, agv = rig

        def fatal_order(order: dict[str, Any]) -> None:
            transport.emit(f"{agv.base}/state", agv._state(
                orderId=order["orderId"],
                errors=[{"errorType": "motorFault", "errorLevel": "FATAL"}],
            ))
        agv._handle_order = fatal_order  # type: ignore[method-assign]
        with pytest.raises(Vda5050Error, match="motorFault"):
            adapter.move(1.0, 1.0)


# ---------------------------------------------------------------------------
# Capabilities + conformance
# ---------------------------------------------------------------------------

class TestCapabilitiesAndConformance:
    def test_capabilities_use_factsheet(self, rig):
        adapter, _, _ = rig
        caps = adapter.get_capabilities()
        assert caps.max_speed == 1.8
        assert caps.metadata["agv_class"] == "CARRIER"
        assert caps.metadata["transport"] == "vda5050"

    def test_full_conformance_suite_passes(self, rig):
        adapter, _, _ = rig
        report = ConformanceReport(target="vda5050://AGV-1", kind="adapter")
        run_adapter_checks(adapter, report)
        must_fails = [
            c for c in report.checks
            if c.level == "MUST" and c.status == "fail"
        ]
        assert report.conformant, f"MUST failures: {must_fails}"
