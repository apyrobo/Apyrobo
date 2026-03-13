"""
ROS 2 DDS Swarm Bus — real inter-robot communication via ROS 2 topics.

Replaces the in-memory SwarmBus with actual DDS pub/sub so that robots
running on different machines (or different Docker containers) can
communicate through the ROS 2 network.

Uses a shared topic (/apyrobo/swarm) with JSON-serialised messages.
Each robot subscribes to the topic and filters messages by target.

Usage (inside Docker):
    from apyrobo.swarm.ros2_bus import ROS2SwarmBus
    bus = ROS2SwarmBus(node)
    bus.register(robot)
    bus.broadcast("robot_a", {"status": "ready"})

Requires rclpy + std_msgs.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import RobotCapability
from apyrobo.swarm.bus import SwarmBus, SwarmMessage, MessageHandler

logger = logging.getLogger(__name__)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import String

    _HAS_ROS2 = True
except ImportError:
    _HAS_ROS2 = False


# Default topic for swarm communication
SWARM_TOPIC = "/apyrobo/swarm"

# QoS: reliable + transient-local so late joiners get recent messages
def _swarm_qos() -> Any:
    if not _HAS_ROS2:
        return None
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        depth=50,
    )


if _HAS_ROS2:

    class ROS2SwarmBus(SwarmBus):
        """
        DDS-backed swarm bus using ROS 2 topics.

        Extends the in-memory SwarmBus — local handlers still work,
        but messages are also published to a shared ROS 2 topic so
        robots on other machines receive them.

        Message format on the wire (JSON):
        {
            "sender": "robot_a",
            "target": null | "robot_b",
            "msg_type": "generic",
            "payload": { ... },
            "timestamp": 1234567890.123
        }
        """

        def __init__(self, node: Node, topic: str = SWARM_TOPIC) -> None:
            super().__init__()
            self._node = node
            self._topic = topic
            self._own_robot_ids: set[str] = set()  # robots registered on THIS node

            qos = _swarm_qos()

            # Publisher
            self._pub = self._node.create_publisher(String, self._topic, qos)

            # Subscriber
            self._sub = self._node.create_subscription(
                String, self._topic, self._on_ros2_message, qos,
            )

            logger.info("ROS2SwarmBus ready on topic %s", self._topic)

        def register(self, robot: Robot) -> None:
            """Register a robot — also announces via DDS."""
            super().register(robot)
            self._own_robot_ids.add(robot.robot_id)

        def unregister(self, robot_id: str) -> None:
            """Unregister — also announces via DDS."""
            super().unregister(robot_id)
            self._own_robot_ids.discard(robot_id)

        def send(self, sender: str, target: str, message: dict[str, Any],
                 msg_type: str = "generic") -> None:
            """Send targeted message — published to DDS + local delivery."""
            # Local delivery (in-memory handlers)
            super().send(sender, target, message, msg_type)
            # DDS publish (for remote robots)
            self._publish_to_dds(sender, target, message, msg_type)

        def broadcast(self, sender: str, message: dict[str, Any],
                      msg_type: str = "generic") -> None:
            """Broadcast — published to DDS + local delivery."""
            super().broadcast(sender, message, msg_type)
            self._publish_to_dds(sender, None, message, msg_type)

        def _publish_to_dds(self, sender: str, target: str | None,
                            payload: dict[str, Any], msg_type: str) -> None:
            """Serialise and publish a message to the DDS topic."""
            wire_msg = {
                "sender": sender,
                "target": target,
                "msg_type": msg_type,
                "payload": payload,
                "timestamp": time.time(),
            }
            ros_msg = String()
            ros_msg.data = json.dumps(wire_msg)
            self._pub.publish(ros_msg)

        def _on_ros2_message(self, ros_msg: String) -> None:
            """
            Callback for incoming DDS messages.

            Filters out messages from our own robots (already handled locally)
            and delivers messages from remote robots to local handlers.
            """
            try:
                wire = json.loads(ros_msg.data)
            except json.JSONDecodeError:
                logger.warning("ROS2SwarmBus: invalid JSON on topic")
                return

            sender = wire.get("sender", "")
            target = wire.get("target")
            msg_type = wire.get("msg_type", "generic")
            payload = wire.get("payload", {})
            timestamp = wire.get("timestamp", time.time())

            # Skip messages from our own robots (already delivered locally)
            if sender in self._own_robot_ids:
                return

            # Build a SwarmMessage and deliver to local handlers
            msg = SwarmMessage(
                sender=sender,
                target=target,
                payload=payload,
                msg_type=msg_type,
                timestamp=timestamp,
            )

            # Only deliver if the target is one of our robots (or broadcast)
            if msg.is_broadcast:
                for rid in self._own_robot_ids:
                    for handler in self._handlers.get(rid, []):
                        try:
                            handler(msg)
                        except Exception as e:
                            logger.warning("Handler error for %s: %s", rid, e)
            elif target in self._own_robot_ids:
                for handler in self._handlers.get(target, []):
                    try:
                        handler(msg)
                    except Exception as e:
                        logger.warning("Handler error for %s: %s", target, e)

            # Global handlers always see remote messages
            for handler in self._global_handlers:
                try:
                    handler(msg)
                except Exception as e:
                    logger.warning("Global handler error: %s", e)

            self._message_log.append(msg)
            logger.debug("ROS2SwarmBus: received from remote %s → %s",
                         sender, target or "ALL")

else:

    class ROS2SwarmBus(SwarmBus):  # type: ignore[no-redef]
        """Stub — falls back to in-memory SwarmBus."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            logger.warning("rclpy not available — using in-memory SwarmBus")
            super().__init__()
