"""
SwarmBus — message passing layer between multiple robots.

In production this uses ROS 2 DDS topics for inter-robot communication.
The in-memory implementation works for testing and single-machine sims.

Usage:
    bus = SwarmBus()
    bus.register(robot_a)
    bus.register(robot_b)
    bus.broadcast(sender="robot_a", message={"type": "status", "battery": 80})
    bus.send(sender="robot_a", target="robot_b", message={"type": "task_offer"})
"""

from __future__ import annotations

import logging
import time
import threading
from collections import defaultdict
from typing import Any, Callable

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import RobotCapability

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

class SwarmMessage:
    """A message passed between robots in the swarm."""

    def __init__(
        self,
        sender: str,
        target: str | None,
        payload: dict[str, Any],
        msg_type: str = "generic",
        timestamp: float | None = None,
    ) -> None:
        self.sender = sender
        self.target = target  # None = broadcast
        self.payload = payload
        self.msg_type = msg_type
        self.timestamp = timestamp or time.time()

    @property
    def is_broadcast(self) -> bool:
        return self.target is None

    def __repr__(self) -> str:
        dest = "ALL" if self.is_broadcast else self.target
        return f"<SwarmMessage {self.sender}→{dest} type={self.msg_type}>"


# Type for message handlers
MessageHandler = Callable[[SwarmMessage], None]


# ---------------------------------------------------------------------------
# SwarmBus
# ---------------------------------------------------------------------------

class SwarmBus:
    """
    In-memory message bus for swarm communication.

    Robots register with the bus and can send targeted or broadcast
    messages.  Handlers are called synchronously for simplicity.

    In production, this would be backed by ROS 2 DDS topics for
    true distributed communication across machines.
    """

    def __init__(self) -> None:
        self._robots: dict[str, Robot] = {}
        self._capabilities: dict[str, RobotCapability] = {}
        self._handlers: dict[str, list[MessageHandler]] = defaultdict(list)
        self._global_handlers: list[MessageHandler] = []
        self._message_log: list[SwarmMessage] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, robot: Robot) -> None:
        """
        Register a robot with the swarm.

        Queries its capabilities and makes it available for task assignment.
        """
        with self._lock:
            rid = robot.robot_id
            self._robots[rid] = robot
            self._capabilities[rid] = robot.capabilities()
            logger.info("SwarmBus: registered robot %s (%s)",
                        rid, self._capabilities[rid].name)

            # Announce to existing members
            self._deliver(SwarmMessage(
                sender=rid, target=None,
                payload={"event": "robot_joined", "robot_id": rid},
                msg_type="system",
            ))

    def unregister(self, robot_id: str) -> None:
        """Remove a robot from the swarm."""
        with self._lock:
            self._robots.pop(robot_id, None)
            self._capabilities.pop(robot_id, None)
            self._handlers.pop(robot_id, None)
            logger.info("SwarmBus: unregistered robot %s", robot_id)

            self._deliver(SwarmMessage(
                sender=robot_id, target=None,
                payload={"event": "robot_left", "robot_id": robot_id},
                msg_type="system",
            ))

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def send(self, sender: str, target: str, message: dict[str, Any],
             msg_type: str = "generic") -> None:
        """Send a targeted message to a specific robot."""
        if target not in self._robots:
            raise ValueError(f"Target robot {target!r} is not registered in the swarm")
        msg = SwarmMessage(sender=sender, target=target, payload=message, msg_type=msg_type)
        self._deliver(msg)

    def broadcast(self, sender: str, message: dict[str, Any],
                  msg_type: str = "generic") -> None:
        """Broadcast a message to all robots in the swarm."""
        msg = SwarmMessage(sender=sender, target=None, payload=message, msg_type=msg_type)
        self._deliver(msg)

    def on_message(self, robot_id: str, handler: MessageHandler) -> None:
        """Register a handler for messages sent to a specific robot."""
        self._handlers[robot_id].append(handler)

    def on_any(self, handler: MessageHandler) -> None:
        """Register a handler that sees all messages (for logging/monitoring)."""
        self._global_handlers.append(handler)

    def _deliver(self, msg: SwarmMessage) -> None:
        """Route a message to the correct handlers."""
        self._message_log.append(msg)

        # Global handlers see everything
        for handler in self._global_handlers:
            try:
                handler(msg)
            except Exception as e:
                logger.warning("Global handler error: %s", e)

        if msg.is_broadcast:
            # Deliver to all registered robots except sender
            for rid in self._robots:
                if rid != msg.sender:
                    for handler in self._handlers.get(rid, []):
                        try:
                            handler(msg)
                        except Exception as e:
                            logger.warning("Handler error for %s: %s", rid, e)
        else:
            # Targeted delivery
            for handler in self._handlers.get(msg.target, []):
                try:
                    handler(msg)
                except Exception as e:
                    logger.warning("Handler error for %s: %s", msg.target, e)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def robot_ids(self) -> list[str]:
        """IDs of all registered robots."""
        return list(self._robots.keys())

    @property
    def robot_count(self) -> int:
        return len(self._robots)

    def get_robot(self, robot_id: str) -> Robot:
        """Get a robot by ID."""
        if robot_id not in self._robots:
            raise KeyError(f"Robot {robot_id!r} not in swarm")
        return self._robots[robot_id]

    def get_capabilities(self, robot_id: str) -> RobotCapability:
        """Get cached capabilities for a robot."""
        if robot_id not in self._capabilities:
            raise KeyError(f"Robot {robot_id!r} not in swarm")
        return self._capabilities[robot_id]

    def get_all_capabilities(self) -> dict[str, RobotCapability]:
        """Get capabilities for all robots."""
        return dict(self._capabilities)

    @property
    def message_log(self) -> list[SwarmMessage]:
        """All messages that have passed through the bus."""
        return list(self._message_log)

    def __repr__(self) -> str:
        return f"<SwarmBus robots={self.robot_count} messages={len(self._message_log)}>"
