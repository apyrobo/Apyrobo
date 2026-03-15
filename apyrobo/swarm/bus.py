from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Any, Callable

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import RobotCapability

logger = logging.getLogger(__name__)


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
        self.target = target
        self.payload = payload
        self.msg_type = msg_type
        self.timestamp = timestamp or time.time()

    @property
    def is_broadcast(self) -> bool:
        return self.target is None

    def __repr__(self) -> str:
        dest = "ALL" if self.is_broadcast else self.target
        return f"<SwarmMessage {self.sender}→{dest} type={self.msg_type}>"


MessageHandler = Callable[[SwarmMessage], None]


class SwarmBus:
    """In-memory bus with heartbeat, reassignment hooks, and gossip support."""

    def __init__(self) -> None:
        self._robots: dict[str, Robot] = {}
        self._capabilities: dict[str, RobotCapability] = {}
        self._handlers: dict[str, list[MessageHandler]] = defaultdict(list)
        self._global_handlers: list[MessageHandler] = []
        self._message_log: list[SwarmMessage] = []
        self._last_heartbeat: dict[str, float] = {}
        self._world_state: dict[str, dict[str, Any]] = defaultdict(dict)
        self._lock = threading.Lock()

    def register(self, robot: Robot) -> None:
        with self._lock:
            rid = robot.robot_id
            self._robots[rid] = robot
            self._capabilities[rid] = robot.capabilities()
            self._last_heartbeat[rid] = time.time()
            self._deliver(SwarmMessage(
                sender=rid,
                target=None,
                payload={"event": "robot_joined", "robot_id": rid},
                msg_type="system",
            ))

    def unregister(self, robot_id: str) -> None:
        with self._lock:
            self._robots.pop(robot_id, None)
            self._capabilities.pop(robot_id, None)
            self._handlers.pop(robot_id, None)
            self._last_heartbeat.pop(robot_id, None)
            self._deliver(SwarmMessage(
                sender=robot_id,
                target=None,
                payload={"event": "robot_left", "robot_id": robot_id},
                msg_type="system",
            ))

    def send(self, sender: str, target: str, message: dict[str, Any], msg_type: str = "generic") -> None:
        if target not in self._robots:
            raise ValueError(f"Target robot {target!r} is not registered in the swarm")
        self._deliver(SwarmMessage(sender=sender, target=target, payload=message, msg_type=msg_type))

    def broadcast(self, sender: str, message: dict[str, Any], msg_type: str = "generic") -> None:
        self._deliver(SwarmMessage(sender=sender, target=None, payload=message, msg_type=msg_type))

    def on_message(self, robot_id: str, handler: MessageHandler) -> None:
        self._handlers[robot_id].append(handler)

    def on_any(self, handler: MessageHandler) -> None:
        self._global_handlers.append(handler)

    def _deliver(self, msg: SwarmMessage) -> None:
        self._message_log.append(msg)
        for handler in self._global_handlers:
            try:
                handler(msg)
            except Exception as e:
                logger.warning("Global handler error: %s", e)

        if msg.is_broadcast:
            for rid in self._robots:
                if rid != msg.sender:
                    for handler in self._handlers.get(rid, []):
                        try:
                            handler(msg)
                        except Exception as e:
                            logger.warning("Handler error for %s: %s", rid, e)
        else:
            for handler in self._handlers.get(msg.target, []):
                try:
                    handler(msg)
                except Exception as e:
                    logger.warning("Handler error for %s: %s", msg.target, e)

    # SW-03: heartbeat & liveness detection
    def heartbeat(self, robot_id: str, health: dict[str, Any] | None = None) -> None:
        if robot_id not in self._robots:
            return
        self._last_heartbeat[robot_id] = time.time()
        self.broadcast(
            sender=robot_id,
            message={"event": "heartbeat", "robot_id": robot_id, "health": health or {}},
            msg_type="status",
        )

    def detect_dropouts(self, timeout_s: float = 5.0, remove: bool = False) -> list[str]:
        now = time.time()
        stale = [rid for rid, ts in self._last_heartbeat.items() if (now - ts) > timeout_s]
        for rid in stale:
            self.broadcast(
                sender="bus",
                message={"event": "robot_dropout", "robot_id": rid, "age_s": now - self._last_heartbeat[rid]},
                msg_type="system",
            )
            if remove:
                self.unregister(rid)
        return stale

    # SW-10: gossip world-state
    def publish_world_state(self, robot_id: str, state: dict[str, Any]) -> None:
        self._world_state[robot_id] = dict(state)
        self.broadcast(
            sender=robot_id,
            message={"event": "world_state_gossip", "robot_id": robot_id, "state": state},
            msg_type="gossip",
        )

    def get_world_state(self) -> dict[str, dict[str, Any]]:
        return {rid: dict(state) for rid, state in self._world_state.items()}

    @property
    def robot_ids(self) -> list[str]:
        return list(self._robots.keys())

    @property
    def robot_count(self) -> int:
        return len(self._robots)

    def get_robot(self, robot_id: str) -> Robot:
        if robot_id not in self._robots:
            raise KeyError(f"Robot {robot_id!r} not in swarm")
        return self._robots[robot_id]

    def get_capabilities(self, robot_id: str) -> RobotCapability:
        if robot_id not in self._capabilities:
            raise KeyError(f"Robot {robot_id!r} not in swarm")
        return self._capabilities[robot_id]

    def get_all_capabilities(self) -> dict[str, RobotCapability]:
        return dict(self._capabilities)

    @property
    def message_log(self) -> list[SwarmMessage]:
        return list(self._message_log)

    def __repr__(self) -> str:
        return f"<SwarmBus robots={self.robot_count} messages={len(self._message_log)}>"
