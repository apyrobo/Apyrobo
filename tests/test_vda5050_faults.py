"""
vda5050:// fault-state and reconnect behaviour, driven against a small
in-process simulated AGV (no broker, no paho — the transport is injected).

Covers what the happy-path suite in test_vda5050_adapter.py does not:
AGV errors that should and should not fail an order, recovery once errors
clear, connection-state changes, stale state from an abandoned order,
broker failures on connect and publish, and disconnect → connect cycles.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from apyrobo.core.vda5050_adapter import Vda5050Adapter, Vda5050Error

BASE = "uagv/v2/test/AGV-1"


# ---------------------------------------------------------------------------
# In-process transport + simulated AGV
# ---------------------------------------------------------------------------

class FaultyTransport:
    """Duck-typed transport that can be told to fail connect or publish."""

    def __init__(self, agv: FaultAgv) -> None:
        self.on_message: Callable[[str, str], None] | None = None
        self.agv = agv
        self.subscribed: list[str] = []
        self.published: list[tuple[str, dict[str, Any]]] = []
        self.connects = 0
        self.fail_connect = False
        self.fail_publish = False
        agv.transport = self

    def connect(self, host: str, port: int) -> None:
        if self.fail_connect:
            raise ConnectionRefusedError(f"{host}:{port} refused")
        self.connects += 1

    def disconnect(self) -> None:
        pass

    def subscribe(self, topic: str) -> None:
        self.subscribed.append(topic)

    def publish(self, topic: str, payload: str) -> None:
        if self.fail_publish:
            raise OSError("broker connection lost")
        msg = json.loads(payload)
        self.published.append((topic, msg))
        self.agv.receive(topic, msg)

    def emit(self, subtopic: str, message: Any) -> None:
        """AGV → master. ``message`` may be a raw string to send as-is."""
        assert self.on_message is not None
        payload = message if isinstance(message, str) else json.dumps(message)
        self.on_message(f"{BASE}/{subtopic}", payload)


class FaultAgv:
    """AGV whose order handling can be swapped per test via ``on_order``."""

    def __init__(self) -> None:
        self.transport: FaultyTransport | None = None
        self.position = (0.0, 0.0)
        self.on_order: Callable[[dict[str, Any]], None] = self.drive
        self.cancelled = 0

    def receive(self, topic: str, msg: dict[str, Any]) -> None:
        if topic.endswith("/order"):
            self.on_order(msg)
        elif topic.endswith("/instantActions"):
            self.cancelled += sum(
                1 for a in msg.get("actions", [])
                if a.get("actionType") == "cancelOrder"
            )

    def state(self, **over: Any) -> dict[str, Any]:
        s: dict[str, Any] = {
            "orderId": "",
            "lastNodeId": "",
            "nodeStates": [],
            "edgeStates": [],
            "agvPosition": {"x": self.position[0], "y": self.position[1],
                            "theta": 0.0, "mapId": "map"},
            "batteryState": {"batteryCharge": 80.0},
            "errors": [],
            "driving": False,
        }
        s.update(over)
        return s

    def drive(self, order: dict[str, Any], errors: list[Any] | None = None) -> None:
        """Accept the order and arrive at its goal node."""
        assert self.transport is not None
        goal = order["nodes"][-1]
        self.position = (goal["nodePosition"]["x"], goal["nodePosition"]["y"])
        self.transport.emit("state", self.state(
            orderId=order["orderId"], lastNodeId=goal["nodeId"],
            errors=errors or [],
        ))

    def ignore(self, order: dict[str, Any]) -> None:
        """Accept the order but never report progress."""


@pytest.fixture()
def rig():
    agv = FaultAgv()
    transport = FaultyTransport(agv)
    adapter = Vda5050Adapter(
        "AGV-1", transport=transport, manufacturer="test",
        order_timeout_sec=2.0,
    )
    adapter.connect()
    transport.emit("connection", {"connectionState": "ONLINE"})
    transport.emit("state", agv.state())
    return adapter, transport, agv


# ---------------------------------------------------------------------------
# AGV fault states
# ---------------------------------------------------------------------------

class TestFaultStates:
    def test_warning_unrelated_to_the_order_does_not_fail_it(self, rig):
        adapter, _, agv = rig
        warning = {"errorType": "batteryLow", "errorLevel": "WARNING"}
        agv.on_order = lambda o: agv.drive(o, errors=[warning])
        adapter.move(2.0, 1.0)
        assert adapter.get_position() == (2.0, 1.0)
        assert adapter.get_health()["errors"] == 1

    def test_fatal_error_fails_the_order_and_shows_in_health(self, rig):
        adapter, transport, agv = rig

        def fatal(order: dict[str, Any]) -> None:
            transport.emit("state", agv.state(
                orderId=order["orderId"],
                errors=[{"errorDescription": "laser scanner fault",
                         "errorLevel": "FATAL"}],
            ))
        agv.on_order = fatal
        with pytest.raises(Vda5050Error, match="laser scanner fault"):
            adapter.move(1.0, 0.0)
        assert adapter.get_health()["errors"] == 1

    def test_error_referencing_another_order_is_not_ours(self, rig):
        adapter, _, agv = rig
        other = {
            "errorType": "orderUpdateError",
            "errorLevel": "WARNING",
            "errorReferences": [
                {"referenceKey": "orderId", "referenceValue": "someone-else"}
            ],
        }
        agv.on_order = lambda o: agv.drive(o, errors=[other])
        adapter.move(1.0, 1.0)  # must complete, not raise
        assert adapter.get_position() == (1.0, 1.0)

    def test_order_succeeds_again_once_the_fault_clears(self, rig):
        adapter, transport, agv = rig

        def fatal(order: dict[str, Any]) -> None:
            transport.emit("state", agv.state(
                orderId=order["orderId"],
                errors=[{"errorType": "motorFault", "errorLevel": "FATAL"}],
            ))
        agv.on_order = fatal
        with pytest.raises(Vda5050Error, match="motorFault"):
            adapter.move(1.0, 0.0)

        # Operator clears the fault; the AGV reports a clean state.
        transport.emit("state", agv.state())
        assert adapter.get_health()["errors"] == 0
        agv.on_order = agv.drive
        adapter.move(3.0, 0.0)
        assert adapter.get_position() == (3.0, 0.0)

    def test_malformed_errors_field_is_treated_as_no_errors(self, rig):
        adapter, transport, agv = rig
        transport.emit("state", agv.state(errors="not-a-list"))
        assert adapter.get_health()["errors"] == 0

    def test_garbage_payloads_are_ignored(self, rig):
        adapter, transport, agv = rig
        transport.emit("state", "{not json")
        transport.emit("state", "[1, 2, 3]")
        transport.emit("connection", "null")
        health = adapter.get_health()
        assert health["agv_online"] is True
        assert health["battery_pct"] == 80.0


# ---------------------------------------------------------------------------
# Connection state and stale messages
# ---------------------------------------------------------------------------

class TestConnectionState:
    @pytest.mark.parametrize("state", ["OFFLINE", "CONNECTIONBROKEN"])
    def test_agv_going_offline_is_reported(self, rig, state):
        adapter, transport, _ = rig
        transport.emit("connection", {"connectionState": state})
        assert adapter.get_health()["agv_online"] is False

    def test_agv_coming_back_online_is_reported(self, rig):
        adapter, transport, _ = rig
        transport.emit("connection", {"connectionState": "CONNECTIONBROKEN"})
        transport.emit("connection", {"connectionState": "ONLINE"})
        assert adapter.get_health()["agv_online"] is True

    def test_late_completion_of_an_abandoned_order_does_not_finish_a_new_one(
        self, rig
    ):
        adapter, transport, agv = rig
        agv.on_order = agv.ignore
        adapter._order_timeout_sec = 0.2
        with pytest.raises(Vda5050Error, match="timed out"):
            adapter.move(1.0, 0.0)
        first = [m for t, m in transport.published if t.endswith("/order")][0]

        def replay_stale(order: dict[str, Any]) -> None:
            # The AGV finally reports the first order as done, then goes quiet.
            transport.emit("state", agv.state(
                orderId=first["orderId"],
                lastNodeId=first["nodes"][-1]["nodeId"],
            ))
        agv.on_order = replay_stale
        with pytest.raises(Vda5050Error, match="timed out"):
            adapter.move(5.0, 0.0)
        assert agv.cancelled == 2  # each timeout published cancelOrder


# ---------------------------------------------------------------------------
# Broker failures and reconnects
# ---------------------------------------------------------------------------

class TestBrokerFailures:
    def test_refused_connect_leaves_adapter_disconnected(self):
        agv = FaultAgv()
        transport = FaultyTransport(agv)
        transport.fail_connect = True
        adapter = Vda5050Adapter("AGV-1", transport=transport, manufacturer="test")
        with pytest.raises(ConnectionRefusedError):
            adapter.connect()
        assert not adapter.is_connected
        with pytest.raises(Vda5050Error, match="not connected"):
            adapter.move(1.0, 1.0)
        adapter.stop()  # must not raise

    def test_stop_never_raises_when_publish_fails(self, rig):
        adapter, transport, _ = rig
        transport.fail_publish = True
        adapter.stop()
        adapter.stop()

    def test_reconnect_resubscribes_and_orders_work_again(self, rig):
        adapter, transport, agv = rig
        adapter.move(1.0, 0.0)
        adapter.disconnect()
        assert not adapter.is_connected

        adapter.connect()
        assert adapter.is_connected
        assert transport.connects == 2
        assert sorted(t.rsplit("/", 1)[-1] for t in transport.subscribed[-3:]) == [
            "connection", "factsheet", "state",
        ]
        adapter.move(2.0, 0.0)
        assert adapter.get_position() == (2.0, 0.0)

        orders = [m for t, m in transport.published if t.endswith("/order")]
        assert orders[0]["orderId"] != orders[1]["orderId"]
        assert orders[1]["headerId"] > orders[0]["headerId"]
        # The order after reconnect starts from the last reported position.
        assert orders[1]["nodes"][0]["nodePosition"]["x"] == 1.0

    def test_commands_are_not_replayed_after_reconnect(self, rig):
        adapter, transport, _ = rig
        adapter.disconnect()
        with pytest.raises(Vda5050Error, match="not connected"):
            adapter.move(4.0, 4.0)
        adapter.connect()
        assert not [t for t, _ in transport.published if t.endswith("/order")]
