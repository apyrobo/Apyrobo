"""
Mocked tests for apyrobo/swarm/ros2_bus.py.

Patches all ROS2 modules at sys.modules level so the swarm bus can be
imported and exercised without a real ROS2 installation.
"""

from __future__ import annotations

import json
import sys
import time
from unittest.mock import MagicMock, call, patch
import pytest

# ---------------------------------------------------------------------------
# ROS2 sys.modules mocks — must happen before any ros2_bus import
# ---------------------------------------------------------------------------

_rclpy_mock = MagicMock()
_rclpy_node_mock = MagicMock()
_rclpy_qos_mock = MagicMock()
_std_msgs_mock = MagicMock()
_std_msgs_msg_mock = MagicMock()

_rclpy_mock.node = _rclpy_node_mock
_rclpy_mock.qos = _rclpy_qos_mock

# String msg: make it a real class so ros_msg.data = ... actually works
class _FakeString:
    def __init__(self):
        self.data = ""

_std_msgs_msg_mock.String = _FakeString

ros2_mocks = {
    "rclpy": _rclpy_mock,
    "rclpy.node": _rclpy_node_mock,
    "rclpy.qos": _rclpy_qos_mock,
    "std_msgs": _std_msgs_mock,
    "std_msgs.msg": _std_msgs_msg_mock,
}
for mod, mock in ros2_mocks.items():
    sys.modules.setdefault(mod, mock)

# Also need geometry_msgs etc. for robot dependency chain
_geometry_msgs_mock = MagicMock()
_geometry_msgs_msg_mock = MagicMock()
_nav_msgs_mock = MagicMock()
_nav_msgs_msg_mock = MagicMock()
_sensor_msgs_mock = MagicMock()
_sensor_msgs_msg_mock = MagicMock()
_nav2_msgs_mock = MagicMock()
_nav2_msgs_action_mock = MagicMock()
_action_msgs_mock = MagicMock()
_action_msgs_msg_mock = MagicMock()
_builtin_interfaces_mock = MagicMock()
_builtin_interfaces_msg_mock = MagicMock()
_tf_transformations_mock = MagicMock()

extra_mocks = {
    "geometry_msgs": _geometry_msgs_mock,
    "geometry_msgs.msg": _geometry_msgs_msg_mock,
    "nav_msgs": _nav_msgs_mock,
    "nav_msgs.msg": _nav_msgs_msg_mock,
    "sensor_msgs": _sensor_msgs_mock,
    "sensor_msgs.msg": _sensor_msgs_msg_mock,
    "nav2_msgs": _nav2_msgs_mock,
    "nav2_msgs.action": _nav2_msgs_action_mock,
    "action_msgs": _action_msgs_mock,
    "action_msgs.msg": _action_msgs_msg_mock,
    "builtin_interfaces": _builtin_interfaces_mock,
    "builtin_interfaces.msg": _builtin_interfaces_msg_mock,
    "tf_transformations": _tf_transformations_mock,
}
for mod, mock in extra_mocks.items():
    sys.modules.setdefault(mod, mock)

# Force re-import
if "apyrobo.swarm.ros2_bus" in sys.modules:
    del sys.modules["apyrobo.swarm.ros2_bus"]

import apyrobo.swarm.ros2_bus as bus_module

assert bus_module._HAS_ROS2, (
    "ros2_bus._HAS_ROS2 should be True with mocked rclpy"
)

ROS2SwarmBus = bus_module.ROS2SwarmBus
SWARM_TOPIC = bus_module.SWARM_TOPIC

from apyrobo.swarm.bus import SwarmBus, SwarmMessage
from apyrobo.core.robot import Robot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node():
    node = MagicMock()
    node.create_publisher.return_value = MagicMock()
    node.create_subscription.return_value = MagicMock()
    return node


def _make_robot(robot_id: str) -> Robot:
    """Create a minimal mock Robot with just enough attributes."""
    robot = MagicMock(spec=Robot)
    robot.robot_id = robot_id
    cap = MagicMock()
    cap.robot_id = robot_id
    robot.capabilities.return_value = cap
    return robot


def _make_bus(topic=SWARM_TOPIC):
    node = _make_node()
    bus = ROS2SwarmBus(node, topic=topic)
    return bus, node


def _make_wire_msg(sender, target, payload, msg_type="generic", timestamp=None):
    """Build a JSON _FakeString as would arrive from the ROS topic."""
    wire = {
        "sender": sender,
        "target": target,
        "msg_type": msg_type,
        "payload": payload,
        "timestamp": timestamp or time.time(),
    }
    msg = _FakeString()
    msg.data = json.dumps(wire)
    return msg


# ===========================================================================
# _swarm_qos
# ===========================================================================

class TestSwarmQos:
    def test_returns_something(self):
        result = bus_module._swarm_qos()
        assert result is not None


# ===========================================================================
# Construction
# ===========================================================================

class TestROS2SwarmBusInit:
    def test_basic_construction(self):
        bus, node = _make_bus()
        assert isinstance(bus, ROS2SwarmBus)
        assert isinstance(bus, SwarmBus)

    def test_creates_publisher(self):
        bus, node = _make_bus()
        node.create_publisher.assert_called_once()

    def test_creates_subscriber(self):
        bus, node = _make_bus()
        node.create_subscription.assert_called_once()

    def test_custom_topic(self):
        bus, node = _make_bus(topic="/my/swarm")
        assert bus._topic == "/my/swarm"

    def test_default_topic(self):
        bus, _ = _make_bus()
        assert bus._topic == SWARM_TOPIC

    def test_own_robot_ids_empty_at_start(self):
        bus, _ = _make_bus()
        assert bus._own_robot_ids == set()

    def test_message_log_empty_at_start(self):
        bus, _ = _make_bus()
        # The base class SwarmBus may log a join message on init; we just
        # ensure the log attribute exists and is a list.
        assert isinstance(bus._message_log, list)


# ===========================================================================
# register / unregister
# ===========================================================================

class TestRegisterUnregister:
    def test_register_adds_to_own_ids(self):
        bus, _ = _make_bus()
        robot = _make_robot("r1")
        bus.register(robot)
        assert "r1" in bus._own_robot_ids

    def test_register_adds_to_base_robots(self):
        bus, _ = _make_bus()
        robot = _make_robot("r2")
        bus.register(robot)
        assert "r2" in bus._robots

    def test_unregister_removes_from_own_ids(self):
        bus, _ = _make_bus()
        robot = _make_robot("r3")
        bus.register(robot)
        bus.unregister("r3")
        assert "r3" not in bus._own_robot_ids

    def test_unregister_removes_from_base_robots(self):
        bus, _ = _make_bus()
        robot = _make_robot("r4")
        bus.register(robot)
        bus.unregister("r4")
        assert "r4" not in bus._robots

    def test_unregister_nonexistent_is_safe(self):
        bus, _ = _make_bus()
        bus.unregister("ghost")  # Should not raise


# ===========================================================================
# send
# ===========================================================================

class TestSend:
    def test_send_publishes_to_dds(self):
        bus, _ = _make_bus()
        robot_a = _make_robot("a")
        robot_b = _make_robot("b")
        bus.register(robot_a)
        bus.register(robot_b)

        bus._pub.publish.reset_mock()
        bus.send("a", "b", {"action": "move"})

        bus._pub.publish.assert_called_once()
        published = bus._pub.publish.call_args[0][0]
        wire = json.loads(published.data)
        assert wire["sender"] == "a"
        assert wire["target"] == "b"
        assert wire["payload"] == {"action": "move"}
        assert wire["msg_type"] == "generic"

    def test_send_with_custom_msg_type(self):
        bus, _ = _make_bus()
        robot_a = _make_robot("x")
        robot_b = _make_robot("y")
        bus.register(robot_a)
        bus.register(robot_b)

        bus._pub.publish.reset_mock()
        bus.send("x", "y", {"task": "pick"}, msg_type="command")

        wire = json.loads(bus._pub.publish.call_args[0][0].data)
        assert wire["msg_type"] == "command"

    def test_send_to_unregistered_target_raises(self):
        bus, _ = _make_bus()
        robot_a = _make_robot("a")
        bus.register(robot_a)

        with pytest.raises(ValueError, match="not registered"):
            bus.send("a", "nonexistent", {})

    def test_send_delivers_locally(self):
        bus, _ = _make_bus()
        robot_a = _make_robot("a")
        robot_b = _make_robot("b")
        bus.register(robot_a)
        bus.register(robot_b)

        received = []
        bus.on_message("b", lambda msg: received.append(msg))
        bus.send("a", "b", {"hello": "world"})
        assert len(received) == 1
        assert received[0].payload == {"hello": "world"}


# ===========================================================================
# broadcast
# ===========================================================================

class TestBroadcast:
    def test_broadcast_publishes_to_dds(self):
        bus, _ = _make_bus()
        robot = _make_robot("broadcaster")
        bus.register(robot)

        bus._pub.publish.reset_mock()
        bus.broadcast("broadcaster", {"status": "ready"})

        bus._pub.publish.assert_called_once()
        wire = json.loads(bus._pub.publish.call_args[0][0].data)
        assert wire["sender"] == "broadcaster"
        assert wire["target"] is None
        assert wire["payload"] == {"status": "ready"}

    def test_broadcast_with_msg_type(self):
        bus, _ = _make_bus()
        bus.broadcast("anon", {"event": "ping"}, msg_type="status")
        wire = json.loads(bus._pub.publish.call_args[0][0].data)
        assert wire["msg_type"] == "status"

    def test_broadcast_delivers_locally_to_other_robots(self):
        bus, _ = _make_bus()
        robot_a = _make_robot("a")
        robot_b = _make_robot("b")
        bus.register(robot_a)
        bus.register(robot_b)

        received = []
        bus.on_message("b", lambda msg: received.append(msg))
        bus.broadcast("a", {"ping": True})

        assert any(m.payload.get("ping") is True for m in received)


# ===========================================================================
# _publish_to_dds
# ===========================================================================

class TestPublishToDds:
    def test_wire_format_has_timestamp(self):
        bus, _ = _make_bus()
        before = time.time()
        bus._publish_to_dds("sender1", "target1", {"k": "v"}, "generic")
        after = time.time()

        wire = json.loads(bus._pub.publish.call_args[0][0].data)
        assert before <= wire["timestamp"] <= after

    def test_wire_format_broadcast_has_null_target(self):
        bus, _ = _make_bus()
        bus._publish_to_dds("sender1", None, {}, "generic")
        wire = json.loads(bus._pub.publish.call_args[0][0].data)
        assert wire["target"] is None

    def test_wire_format_valid_json(self):
        bus, _ = _make_bus()
        bus._publish_to_dds("s", "t", {"nested": {"a": 1}}, "custom")
        raw = bus._pub.publish.call_args[0][0].data
        parsed = json.loads(raw)
        assert parsed["payload"]["nested"]["a"] == 1


# ===========================================================================
# _on_ros2_message
# ===========================================================================

class TestOnRos2Message:
    def test_message_from_own_robot_is_ignored(self):
        bus, _ = _make_bus()
        robot = _make_robot("local_bot")
        bus.register(robot)

        received = []
        bus.on_any(lambda msg: received.append(msg))

        # Reset the listener count (register broadcast already added 1)
        received.clear()

        ros_msg = _make_wire_msg("local_bot", None, {"data": 1})
        bus._on_ros2_message(ros_msg)

        # Should be filtered out
        assert len(received) == 0

    def test_remote_broadcast_delivered_to_local_robots(self):
        bus, _ = _make_bus()
        robot_a = _make_robot("local_a")
        bus.register(robot_a)

        received_a = []
        bus.on_message("local_a", lambda msg: received_a.append(msg))

        ros_msg = _make_wire_msg("remote_robot", None, {"event": "hello"})
        bus._on_ros2_message(ros_msg)

        assert len(received_a) == 1
        assert received_a[0].sender == "remote_robot"
        assert received_a[0].payload["event"] == "hello"

    def test_remote_targeted_message_delivered_to_correct_robot(self):
        bus, _ = _make_bus()
        robot_a = _make_robot("local_a")
        robot_b = _make_robot("local_b")
        bus.register(robot_a)
        bus.register(robot_b)

        received_a = []
        received_b = []
        bus.on_message("local_a", lambda msg: received_a.append(msg))
        bus.on_message("local_b", lambda msg: received_b.append(msg))

        ros_msg = _make_wire_msg("remote", "local_a", {"task": "dock"})
        bus._on_ros2_message(ros_msg)

        assert len(received_a) == 1
        assert len(received_b) == 0
        assert received_a[0].target == "local_a"

    def test_remote_message_not_for_our_robots_not_delivered_locally(self):
        bus, _ = _make_bus()
        robot_a = _make_robot("local_a")
        bus.register(robot_a)

        received = []
        bus.on_message("local_a", lambda msg: received.append(msg))

        ros_msg = _make_wire_msg("remote", "other_robot", {"x": 1})
        bus._on_ros2_message(ros_msg)

        assert len(received) == 0

    def test_global_handlers_always_called_for_remote_messages(self):
        bus, _ = _make_bus()

        global_received = []
        bus.on_any(lambda msg: global_received.append(msg))

        # Remove any messages from register() calls above
        global_received.clear()

        ros_msg = _make_wire_msg("remote_x", None, {"val": 42})
        bus._on_ros2_message(ros_msg)

        assert any(m.payload.get("val") == 42 for m in global_received)

    def test_invalid_json_is_silently_ignored(self):
        bus, _ = _make_bus()
        bad_msg = _FakeString()
        bad_msg.data = "{ not valid json }"

        # Should not raise
        bus._on_ros2_message(bad_msg)

    def test_message_appended_to_log(self):
        bus, _ = _make_bus()
        initial_len = len(bus._message_log)

        ros_msg = _make_wire_msg("remote_z", None, {})
        bus._on_ros2_message(ros_msg)

        assert len(bus._message_log) == initial_len + 1
        assert bus._message_log[-1].sender == "remote_z"

    def test_missing_sender_defaults_to_empty_string(self):
        bus, _ = _make_bus()
        wire = {"target": None, "msg_type": "generic", "payload": {}, "timestamp": time.time()}
        msg = _FakeString()
        msg.data = json.dumps(wire)
        # Should not raise
        bus._on_ros2_message(msg)

    def test_missing_payload_defaults_to_empty_dict(self):
        bus, _ = _make_bus()
        wire = {"sender": "r", "target": None, "msg_type": "generic", "timestamp": time.time()}
        msg = _FakeString()
        msg.data = json.dumps(wire)
        bus._on_ros2_message(msg)
        assert bus._message_log[-1].payload == {}

    def test_handler_exception_is_swallowed(self):
        bus, _ = _make_bus()
        robot = _make_robot("local_c")
        bus.register(robot)

        def bad_handler(msg):
            raise RuntimeError("handler blew up")

        bus.on_message("local_c", bad_handler)

        # Remote broadcast → should trigger handler but swallow exception
        ros_msg = _make_wire_msg("remote", None, {})
        bus._on_ros2_message(ros_msg)  # Should not raise

    def test_global_handler_exception_is_swallowed(self):
        bus, _ = _make_bus()

        def bad_global(msg):
            raise ValueError("boom")

        bus.on_any(bad_global)

        ros_msg = _make_wire_msg("remote", None, {})
        bus._on_ros2_message(ros_msg)  # Should not raise

    def test_timestamp_from_wire_is_preserved(self):
        bus, _ = _make_bus()
        ts = 1_700_000_000.0
        ros_msg = _make_wire_msg("remote", None, {}, timestamp=ts)
        bus._on_ros2_message(ros_msg)
        assert bus._message_log[-1].timestamp == ts

    def test_multiple_local_robots_all_receive_broadcast(self):
        bus, _ = _make_bus()
        r1 = _make_robot("local_1")
        r2 = _make_robot("local_2")
        bus.register(r1)
        bus.register(r2)

        recv1, recv2 = [], []
        bus.on_message("local_1", lambda m: recv1.append(m))
        bus.on_message("local_2", lambda m: recv2.append(m))

        ros_msg = _make_wire_msg("far_robot", None, {"cmd": "wave"})
        bus._on_ros2_message(ros_msg)

        assert len(recv1) == 1
        assert len(recv2) == 1


# ===========================================================================
# SwarmMessage properties (shared with base class)
# ===========================================================================

class TestSwarmMessage:
    def test_is_broadcast_true_when_no_target(self):
        msg = SwarmMessage(sender="a", target=None, payload={})
        assert msg.is_broadcast is True

    def test_is_broadcast_false_when_target_set(self):
        msg = SwarmMessage(sender="a", target="b", payload={})
        assert msg.is_broadcast is False

    def test_repr(self):
        msg = SwarmMessage(sender="a", target=None, payload={})
        assert "ALL" in repr(msg)
        msg2 = SwarmMessage(sender="a", target="b", payload={})
        assert "b" in repr(msg2)


# ===========================================================================
# Inherited SwarmBus behaviours still work via ROS2SwarmBus
# ===========================================================================

class TestInheritedSwarmBusBehaviours:
    def test_on_any_handler(self):
        bus, _ = _make_bus()
        robot = _make_robot("rx")
        bus.register(robot)

        received = []
        bus.on_any(lambda msg: received.append(msg))
        received.clear()  # clear register join message

        bus.broadcast("rx", {"ping": 1})
        # Should appear via DDS publish AND via local deliver
        bus._pub.publish.assert_called()

    def test_robot_count_property(self):
        bus, _ = _make_bus()
        assert bus.robot_count == 0
        bus.register(_make_robot("q1"))
        assert bus.robot_count == 1

    def test_robot_ids_property(self):
        bus, _ = _make_bus()
        bus.register(_make_robot("q1"))
        bus.register(_make_robot("q2"))
        ids = bus.robot_ids
        assert "q1" in ids
        assert "q2" in ids

    def test_get_robot(self):
        bus, _ = _make_bus()
        robot = _make_robot("q3")
        bus.register(robot)
        assert bus.get_robot("q3") is robot

    def test_get_capabilities(self):
        bus, _ = _make_bus()
        robot = _make_robot("q4")
        bus.register(robot)
        caps = bus.get_capabilities("q4")
        assert caps.robot_id == "q4"

    def test_message_log_property(self):
        bus, _ = _make_bus()
        log = bus.message_log
        assert isinstance(log, list)

    def test_repr(self):
        bus, _ = _make_bus()
        r = repr(bus)
        assert "SwarmBus" in r

    def test_heartbeat_broadcasts(self):
        bus, _ = _make_bus()
        robot = _make_robot("hb1")
        bus.register(robot)

        bus._pub.publish.reset_mock()
        bus.heartbeat("hb1", health={"battery": 80})

        # heartbeat triggers a broadcast which triggers a DDS publish
        bus._pub.publish.assert_called()

    def test_detect_dropouts(self):
        bus, _ = _make_bus()
        robot = _make_robot("old_bot")
        bus.register(robot)

        # Manually set a stale heartbeat time
        bus._last_heartbeat["old_bot"] = time.time() - 100.0
        dropouts = bus.detect_dropouts(timeout_s=5.0)
        assert "old_bot" in dropouts

    def test_publish_world_state(self):
        bus, _ = _make_bus()
        robot = _make_robot("ws1")
        bus.register(robot)

        bus.publish_world_state("ws1", {"x": 1.0, "y": 2.0})
        ws = bus.get_world_state()
        assert "ws1" in ws
        assert ws["ws1"]["x"] == 1.0

    def test_get_all_capabilities(self):
        bus, _ = _make_bus()
        r1 = _make_robot("c1")
        r2 = _make_robot("c2")
        bus.register(r1)
        bus.register(r2)
        all_caps = bus.get_all_capabilities()
        assert "c1" in all_caps
        assert "c2" in all_caps
