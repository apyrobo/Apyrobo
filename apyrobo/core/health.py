"""
Connection health monitor for the ros2:// adapter.

Watches the /odom timestamp and fires disconnect/reconnect callbacks
when the robot goes silent or comes back online. Reconnects automatically
with exponential backoff.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ConnectionHealth:
    """
    Monitors a ROS2Adapter's /odom subscription and fires events on timeout.

    The adapter's odom callback must call ``record_odom()`` on every message.
    A background thread compares the elapsed time against ``timeout_seconds``
    and enters a reconnect loop when the gap is exceeded.

    Usage::

        health = ConnectionHealth(adapter, timeout_seconds=5.0)
        health.on_disconnect(lambda: print("Robot lost! Pausing tasks..."))
        health.on_reconnect(lambda: print("Robot back online."))
        health.start()
    """

    def __init__(
        self,
        adapter: Any,
        timeout_seconds: float = 5.0,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
        max_retries: int | None = None,
        *,
        _check_interval: float = 1.0,
        _reconnect_verify_timeout: float = 2.0,
    ) -> None:
        self._adapter = adapter
        self._timeout_seconds = timeout_seconds
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._max_retries = max_retries
        self._check_interval = _check_interval
        self._reconnect_verify_timeout = _reconnect_verify_timeout

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Initialise to now so the monitor doesn't fire immediately on start.
        self._last_odom_time: float = time.monotonic()
        self._healthy: bool = True

        self._disconnect_callbacks: list[Callable[[], None]] = []
        self._reconnect_callbacks: list[Callable[[], None]] = []
        self._give_up_callbacks: list[Callable[[], None]] = []

        self._monitor_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_odom(self) -> None:
        """Update the last-seen odom timestamp. Call from the adapter's odom callback."""
        with self._lock:
            self._last_odom_time = time.monotonic()

    def start(self) -> None:
        """Begin monitoring in a background thread. Safe to call multiple times."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="apyrobo-health-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def stop(self) -> None:
        """Stop monitoring cleanly and join the background thread."""
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

    @property
    def is_healthy(self) -> bool:
        """True if odom was received within timeout_seconds."""
        with self._lock:
            return self._healthy

    def on_disconnect(self, callback: Callable[[], None]) -> None:
        """Register a callback invoked when odom timeout is exceeded."""
        self._disconnect_callbacks.append(callback)

    def on_reconnect(self, callback: Callable[[], None]) -> None:
        """Register a callback invoked when odom resumes after a disconnect."""
        self._reconnect_callbacks.append(callback)

    def on_give_up(self, callback: Callable[[], None]) -> None:
        """Register a callback invoked when max_retries is exhausted."""
        self._give_up_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Monitor thread
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self._check_interval)
            if self._stop_event.is_set():
                break

            with self._lock:
                age = time.monotonic() - self._last_odom_time
                currently_healthy = self._healthy

            if age > self._timeout_seconds and currently_healthy:
                with self._lock:
                    self._healthy = False
                self._fire_disconnect(age)
                self._reconnect_loop()

    def _reconnect_loop(self) -> None:
        attempt = 0
        while not self._stop_event.is_set():
            if self._max_retries is not None and attempt >= self._max_retries:
                logger.warning(
                    "ConnectionHealth: max retries (%d) exhausted for %s",
                    self._max_retries,
                    getattr(self._adapter, "robot_name", "unknown"),
                )
                self._fire_give_up()
                return

            delay = self._backoff_delay(attempt)
            logger.debug(
                "ConnectionHealth: attempt %d — waiting %.2fs before connect",
                attempt, delay,
            )
            self._stop_event.wait(timeout=delay)
            if self._stop_event.is_set():
                return

            t_connect = time.monotonic()
            try:
                self._adapter.connect()
            except Exception as exc:
                logger.warning("ConnectionHealth: connect() raised: %s", exc)

            # Wait up to _reconnect_verify_timeout for a fresh odom message.
            deadline = time.monotonic() + self._reconnect_verify_timeout
            reconnected = False
            while time.monotonic() < deadline and not self._stop_event.is_set():
                with self._lock:
                    if self._last_odom_time > t_connect:
                        reconnected = True
                        break
                self._stop_event.wait(timeout=0.05)

            if reconnected:
                with self._lock:
                    self._healthy = True
                self._fire_reconnect(attempt)
                return

            attempt += 1

    # ------------------------------------------------------------------
    # Backoff
    # ------------------------------------------------------------------

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with ±20% jitter, capped at backoff_max."""
        delay = self._backoff_base * (2 ** attempt)
        delay = min(delay, self._backoff_max)
        jitter = delay * 0.2 * (2.0 * random.random() - 1.0)
        return max(0.0, delay + jitter)

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def _fire_disconnect(self, odom_age: float) -> None:
        robot_id = getattr(self._adapter, "robot_name", "unknown")
        logger.warning(
            "ConnectionHealth: robot %s lost (odom age %.1fs)", robot_id, odom_age,
        )
        try:
            from apyrobo.observability import emit_event
            emit_event(
                "robot.disconnected",
                robot_id=robot_id,
                timestamp=time.time(),
                last_odom_age_s=round(odom_age, 3),
            )
        except Exception:
            pass
        for cb in self._disconnect_callbacks:
            try:
                cb()
            except Exception:
                logger.exception("ConnectionHealth: disconnect callback raised")

    def _fire_reconnect(self, attempt: int) -> None:
        robot_id = getattr(self._adapter, "robot_name", "unknown")
        logger.info(
            "ConnectionHealth: robot %s reconnected (attempt %d)", robot_id, attempt,
        )
        try:
            from apyrobo.observability import emit_event
            emit_event(
                "robot.reconnected",
                robot_id=robot_id,
                timestamp=time.time(),
                reconnect_attempt=attempt,
            )
        except Exception:
            pass
        for cb in self._reconnect_callbacks:
            try:
                cb()
            except Exception:
                logger.exception("ConnectionHealth: reconnect callback raised")

    def _fire_give_up(self) -> None:
        for cb in self._give_up_callbacks:
            try:
                cb()
            except Exception:
                logger.exception("ConnectionHealth: give_up callback raised")
