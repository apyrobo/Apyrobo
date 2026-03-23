"""
Comprehensive tests for apyrobo/swarm/bus.py and apyrobo/swarm/safety.py.

Covers: SwarmBus (register/unregister, send, broadcast, handlers, heartbeat,
detect_dropouts, world state, queries, repr) and SwarmSafety (positions,
proximity, deadlock, check_all, properties, repr).
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from apyrobo.core.robot import Robot
from apyrobo.swarm.bus import SwarmBus, SwarmMessage
from apyrobo.swarm.safety import (
    SwarmSafety,
    ProximityViolation,
    DeadlockDetected,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_robot(uri: str) -> Robot:
    return Robot.discover(uri)


def make_bus_with_robots(*uris: str) -> tuple[SwarmBus, list[Robot]]:
    bus = SwarmBus()
    robots = []
    for uri in uris:
        r = make_robot(uri)
        bus.register(r)
        robots.append(r)
    return bus, robots


# ---------------------------------------------------------------------------
# SwarmMessage
# ---------------------------------------------------------------------------

class TestSwarmMessage:
    def test_is_broadcast_true(self):
        msg = SwarmMessage(sender="r1", target=None, payload={})
        assert msg.is_broadcast is True

    def test_is_broadcast_false(self):
        msg = SwarmMessage(sender="r1", target="r2", payload={})
        assert msg.is_broadcast is False

    def test_repr(self):
        msg = SwarmMessage(sender="r1", target=None, payload={}, msg_type="status")
        r = repr(msg)
        assert "r1" in r
        assert "ALL" in r
        assert "status" in r

    def test_repr_targeted(self):
        msg = SwarmMessage(sender="r1", target="r2", payload={})
        r = repr(msg)
        assert "r2" in r


# ---------------------------------------------------------------------------
# SwarmBus
# ---------------------------------------------------------------------------

class TestSwarmBus:
    def setup_method(self):
        self.r1 = make_robot("mock://tb4")
        self.r2 = make_robot("mock://tb5")
        self.bus = SwarmBus()

    def test_register(self):
        self.bus.register(self.r1)
        assert self.r1.robot_id in self.bus.robot_ids
        assert self.bus.robot_count == 1

    def test_register_broadcasts_join_event(self):
        received = []
        self.bus.on_any(received.append)
        self.bus.register(self.r1)
        # Should receive the robot_joined system message
        join_msgs = [m for m in received if m.payload.get("event") == "robot_joined"]
        assert len(join_msgs) == 1

    def test_unregister(self):
        self.bus.register(self.r1)
        self.bus.unregister(self.r1.robot_id)
        assert self.r1.robot_id not in self.bus.robot_ids
        assert self.bus.robot_count == 0

    def test_unregister_broadcasts_left_event(self):
        self.bus.register(self.r1)
        received = []
        self.bus.on_any(received.append)
        self.bus.unregister(self.r1.robot_id)
        left_msgs = [m for m in received if m.payload.get("event") == "robot_left"]
        assert len(left_msgs) == 1

    def test_send_to_registered_target(self):
        self.bus.register(self.r1)
        self.bus.register(self.r2)
        received = []
        self.bus.on_message(self.r2.robot_id, received.append)
        self.bus.send(self.r1.robot_id, self.r2.robot_id, {"data": "hello"}, "test")
        assert len(received) == 1
        assert received[0].payload["data"] == "hello"

    def test_send_to_unregistered_raises_value_error(self):
        self.bus.register(self.r1)
        with pytest.raises(ValueError, match="not registered"):
            self.bus.send(self.r1.robot_id, "nonexistent_robot", {})

    def test_broadcast(self):
        self.bus.register(self.r1)
        self.bus.register(self.r2)
        r1_received = []
        r2_received = []
        self.bus.on_message(self.r1.robot_id, r1_received.append)
        self.bus.on_message(self.r2.robot_id, r2_received.append)

        self.bus.broadcast(self.r1.robot_id, {"data": "broadcast"})
        # r1 is sender — should not receive own broadcast
        # r2 should receive it
        assert len(r2_received) >= 1
        broadcast_msgs = [m for m in r2_received if m.payload.get("data") == "broadcast"]
        assert len(broadcast_msgs) == 1

    def test_on_message_handler_delivery(self):
        self.bus.register(self.r1)
        self.bus.register(self.r2)
        msgs_for_r2 = []
        self.bus.on_message(self.r2.robot_id, msgs_for_r2.append)
        self.bus.send(self.r1.robot_id, self.r2.robot_id, {"x": 42})
        assert len(msgs_for_r2) == 1
        assert msgs_for_r2[0].payload["x"] == 42

    def test_on_any_global_handler(self):
        self.bus.register(self.r1)
        self.bus.register(self.r2)
        all_msgs = []
        self.bus.on_any(all_msgs.append)
        self.bus.send(self.r1.robot_id, self.r2.robot_id, {"y": 99})
        send_msgs = [m for m in all_msgs if m.payload.get("y") == 99]
        assert len(send_msgs) == 1

    def test_heartbeat(self):
        self.bus.register(self.r1)
        self.bus.register(self.r2)
        r2_received = []
        self.bus.on_message(self.r2.robot_id, r2_received.append)

        self.bus.heartbeat(self.r1.robot_id, health={"battery": 90})
        heartbeat_msgs = [m for m in r2_received if m.payload.get("event") == "heartbeat"]
        assert len(heartbeat_msgs) == 1

    def test_heartbeat_unknown_robot_ignored(self):
        # Should not raise
        self.bus.heartbeat("nonexistent_robot", {})

    def test_detect_dropouts_remove_false(self):
        self.bus.register(self.r1)
        # Force stale heartbeat
        self.bus._last_heartbeat[self.r1.robot_id] = time.time() - 100
        stale = self.bus.detect_dropouts(timeout_s=5.0, remove=False)
        assert self.r1.robot_id in stale
        # Robot should still be in the bus
        assert self.r1.robot_id in self.bus.robot_ids

    def test_detect_dropouts_remove_true(self):
        self.bus.register(self.r1)
        self.bus._last_heartbeat[self.r1.robot_id] = time.time() - 100
        stale = self.bus.detect_dropouts(timeout_s=5.0, remove=True)
        assert self.r1.robot_id in stale
        # Robot should be removed
        assert self.r1.robot_id not in self.bus.robot_ids

    def test_detect_no_dropouts(self):
        self.bus.register(self.r1)
        # Fresh heartbeat
        self.bus._last_heartbeat[self.r1.robot_id] = time.time()
        stale = self.bus.detect_dropouts(timeout_s=5.0)
        assert len(stale) == 0

    def test_publish_world_state(self):
        self.bus.register(self.r1)
        self.bus.register(self.r2)
        state = {"position": [1.0, 2.0], "battery": 80}
        self.bus.publish_world_state(self.r1.robot_id, state)
        world = self.bus.get_world_state()
        assert self.r1.robot_id in world
        assert world[self.r1.robot_id]["battery"] == 80

    def test_get_world_state_empty(self):
        assert self.bus.get_world_state() == {}

    def test_robot_ids(self):
        self.bus.register(self.r1)
        self.bus.register(self.r2)
        ids = self.bus.robot_ids
        assert self.r1.robot_id in ids
        assert self.r2.robot_id in ids

    def test_robot_count(self):
        assert self.bus.robot_count == 0
        self.bus.register(self.r1)
        assert self.bus.robot_count == 1

    def test_get_robot_found(self):
        self.bus.register(self.r1)
        robot = self.bus.get_robot(self.r1.robot_id)
        assert robot is self.r1

    def test_get_robot_not_found(self):
        with pytest.raises(KeyError, match="not in swarm"):
            self.bus.get_robot("nonexistent")

    def test_get_capabilities_found(self):
        self.bus.register(self.r1)
        caps = self.bus.get_capabilities(self.r1.robot_id)
        assert caps is not None

    def test_get_capabilities_not_found(self):
        with pytest.raises(KeyError, match="not in swarm"):
            self.bus.get_capabilities("nonexistent")

    def test_get_all_capabilities(self):
        self.bus.register(self.r1)
        self.bus.register(self.r2)
        all_caps = self.bus.get_all_capabilities()
        assert self.r1.robot_id in all_caps
        assert self.r2.robot_id in all_caps

    def test_message_log(self):
        self.bus.register(self.r1)
        initial_count = len(self.bus.message_log)
        self.bus.broadcast(self.r1.robot_id, {"msg": "test"})
        assert len(self.bus.message_log) == initial_count + 1

    def test_repr(self):
        r = repr(self.bus)
        assert "SwarmBus" in r
        assert "robots=" in r
        assert "messages=" in r

    def test_handler_exception_is_caught(self):
        self.bus.register(self.r1)
        self.bus.register(self.r2)

        def bad_handler(msg):
            raise RuntimeError("handler error")

        self.bus.on_message(self.r2.robot_id, bad_handler)
        # Should not raise
        self.bus.send(self.r1.robot_id, self.r2.robot_id, {"safe": True})

    def test_global_handler_exception_is_caught(self):
        self.bus.register(self.r1)

        def bad_global(msg):
            raise RuntimeError("global handler error")

        self.bus.on_any(bad_global)
        # Should not raise
        self.bus.broadcast(self.r1.robot_id, {"x": 1})


# ---------------------------------------------------------------------------
# SwarmSafety
# ---------------------------------------------------------------------------

class TestSwarmSafety:
    def setup_method(self):
        self.r1 = make_robot("mock://tb4")
        self.r2 = make_robot("mock://tb5")
        self.bus = SwarmBus()
        self.bus.register(self.r1)
        self.bus.register(self.r2)
        self.safety = SwarmSafety(self.bus, min_distance=1.0)

    def test_update_position(self):
        self.safety.update_position(self.r1.robot_id, 0.0, 0.0)
        assert self.safety.get_position(self.r1.robot_id) == (0.0, 0.0)

    def test_get_position_not_found(self):
        assert self.safety.get_position("nonexistent") is None

    def test_check_proximity_no_violations(self):
        self.safety.update_position(self.r1.robot_id, 0.0, 0.0)
        self.safety.update_position(self.r2.robot_id, 5.0, 0.0)
        violations = self.safety.check_proximity()
        assert len(violations) == 0

    def test_check_proximity_with_violation(self):
        self.safety.update_position(self.r1.robot_id, 0.0, 0.0)
        self.safety.update_position(self.r2.robot_id, 0.3, 0.0)  # 0.3m < 1.0m
        violations = self.safety.check_proximity()
        assert len(violations) == 1
        a, b, dist = violations[0]
        assert dist < 1.0

    def test_enforce_proximity_no_violation(self):
        self.safety.update_position(self.r1.robot_id, 0.0, 0.0)
        self.safety.update_position(self.r2.robot_id, 5.0, 0.0)
        # Should not raise
        self.safety.enforce_proximity()

    def test_enforce_proximity_raises(self):
        self.safety.update_position(self.r1.robot_id, 0.0, 0.0)
        self.safety.update_position(self.r2.robot_id, 0.2, 0.0)
        with pytest.raises(ProximityViolation, match="apart"):
            self.safety.enforce_proximity()

    def test_would_violate_proximity_true(self):
        self.safety.update_position(self.r2.robot_id, 2.0, 0.0)
        would_viol, other_id, dist = self.safety.would_violate_proximity(
            self.r1.robot_id, 1.5, 0.0  # 0.5m from r2
        )
        assert would_viol is True
        assert other_id == self.r2.robot_id
        assert dist < 1.0

    def test_would_violate_proximity_false(self):
        self.safety.update_position(self.r2.robot_id, 10.0, 0.0)
        would_viol, other_id, dist = self.safety.would_violate_proximity(
            self.r1.robot_id, 0.0, 0.0
        )
        assert would_viol is False
        assert other_id is None

    def test_set_waiting_set(self):
        self.safety.set_waiting(self.r1.robot_id, self.r2.robot_id)
        assert self.safety._waiting_on[self.r1.robot_id] == self.r2.robot_id
        assert self.r1.robot_id in self.safety._wait_start

    def test_set_waiting_clear(self):
        self.safety.set_waiting(self.r1.robot_id, self.r2.robot_id)
        self.safety.set_waiting(self.r1.robot_id, None)
        assert self.r1.robot_id not in self.safety._waiting_on
        assert self.r1.robot_id not in self.safety._wait_start

    def test_check_deadlock_no_cycle(self):
        r3 = make_robot("mock://tb4")  # just need another ID
        # Linear wait: r1 -> r2 -> r3 (no cycle)
        self.safety.set_waiting(self.r1.robot_id, self.r2.robot_id)
        cycles = self.safety.check_deadlock()
        assert len(cycles) == 0

    def test_check_deadlock_simple_cycle(self):
        self.safety.set_waiting(self.r1.robot_id, self.r2.robot_id)
        self.safety.set_waiting(self.r2.robot_id, self.r1.robot_id)
        cycles = self.safety.check_deadlock()
        assert len(cycles) > 0

    def test_check_deadlock_longer_cycle(self):
        # r1 -> r2 -> r3 -> r1
        r3 = make_robot("mock://tb4")
        self.bus.register(r3)
        self.safety.set_waiting(self.r1.robot_id, self.r2.robot_id)
        self.safety.set_waiting(self.r2.robot_id, r3.robot_id)
        self.safety.set_waiting(r3.robot_id, self.r1.robot_id)
        cycles = self.safety.check_deadlock()
        assert len(cycles) > 0

    def test_enforce_deadlock_no_deadlock(self):
        # Should not raise
        self.safety.enforce_deadlock()

    def test_enforce_deadlock_raises(self):
        self.safety.set_waiting(self.r1.robot_id, self.r2.robot_id)
        self.safety.set_waiting(self.r2.robot_id, self.r1.robot_id)
        with pytest.raises(DeadlockDetected):
            self.safety.enforce_deadlock()

    def test_resolve_deadlock_no_deadlock_returns_none(self):
        result = self.safety.resolve_deadlock()
        assert result is None

    def test_resolve_deadlock_resolves_cycle(self):
        self.safety.set_waiting(self.r1.robot_id, self.r2.robot_id)
        self.safety.set_waiting(self.r2.robot_id, self.r1.robot_id)
        action = self.safety.resolve_deadlock()
        assert action is not None
        assert action["event"] == "deadlock_resolved"
        assert "released_robot" in action
        # After resolution, deadlock should be gone
        cycles = self.safety.check_deadlock()
        assert len(cycles) == 0

    def test_check_all_safe(self):
        self.safety.update_position(self.r1.robot_id, 0.0, 0.0)
        self.safety.update_position(self.r2.robot_id, 5.0, 0.0)
        result = self.safety.check_all()
        assert result["safe"] is True
        assert len(result["proximity_violations"]) == 0
        assert len(result["deadlocks"]) == 0
        assert result["robot_count"] == 2

    def test_check_all_not_safe(self):
        self.safety.update_position(self.r1.robot_id, 0.0, 0.0)
        self.safety.update_position(self.r2.robot_id, 0.1, 0.0)
        result = self.safety.check_all()
        assert result["safe"] is False
        assert len(result["proximity_violations"]) > 0

    def test_violations_property(self):
        self.safety.update_position(self.r1.robot_id, 0.0, 0.0)
        self.safety.update_position(self.r2.robot_id, 0.1, 0.0)
        self.safety.check_proximity()
        violations = self.safety.violations
        assert len(violations) > 0
        assert violations[0]["type"] == "proximity"

    def test_positions_property(self):
        self.safety.update_position(self.r1.robot_id, 1.0, 2.0)
        positions = self.safety.positions
        assert self.r1.robot_id in positions
        assert positions[self.r1.robot_id] == (1.0, 2.0)

    def test_repr(self):
        r = repr(self.safety)
        assert "SwarmSafety" in r
        assert "min_dist=" in r
        assert "violations=" in r
