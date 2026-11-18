"""apyrobo mcp shuts the shared ROS 2 node down before the process exits.

On ros2:// the executor spins in a daemon thread. Left running into
interpreter exit, the process aborts with exit -6 ("terminate called without
an active exception") even though the robot was already stopped. The CLI now
stops the executor, joins the spin thread and shuts rclpy down first.
"""

from __future__ import annotations

import argparse
import sys
import threading
import types
from unittest.mock import MagicMock

import pytest

from apyrobo import cli
from apyrobo.core import ros2_bridge


def _args() -> argparse.Namespace:
    return argparse.Namespace(robot="mock://tb", policy=None, max_speed=None,
                              no_go=None, provider="rule", policy_file=None)


class _Server:
    """Stands in for the MCP server: serve() returns as if stdin closed."""

    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.toolbox = MagicMock()
        self.toolbox.enforcer.policy.name = "default"
        self.toolbox.enforcer.policy.max_speed = 0.5
        self.toolbox.enforcer.policy.collision_zones = []
        self.toolbox.enforcer.policy.allowed_areas = []
        self.toolbox.enforcer.stop.side_effect = lambda: order.append("stop")

    def serve(self, watch_client: bool = True) -> None:
        self.order.append("serve")

    def join_halt(self) -> None:
        self.order.append("join_halt")

    def reader_running(self) -> bool:
        return False


def _fake_bridge(monkeypatch, started: bool, order: list[str], fail: bool = False):
    manager = MagicMock()
    manager.started.return_value = started

    def shutdown() -> None:
        order.append("ros2_shutdown")
        if fail:
            raise RuntimeError("context already shut down")

    manager.shutdown.side_effect = shutdown
    module = types.SimpleNamespace(_ROS2NodeManager=manager)
    monkeypatch.setitem(sys.modules, "apyrobo.core.ros2_bridge", module)
    return manager


def _run(monkeypatch, order: list[str]) -> None:
    monkeypatch.setattr("apyrobo.mcp_server.build_server",
                        lambda **_kw: _Server(order))
    monkeypatch.setattr(cli, "_utf8_stdio", lambda: None)
    cli.cmd_mcp(_args())


def test_ros2_node_is_shut_down_after_the_robot_is_stopped(monkeypatch):
    order: list[str] = []
    manager = _fake_bridge(monkeypatch, started=True, order=order)
    _run(monkeypatch, order)
    manager.shutdown.assert_called_once()
    assert order == ["serve", "stop", "join_halt", "ros2_shutdown"]


def test_no_ros2_shutdown_when_no_ros2_robot_started(monkeypatch):
    order: list[str] = []
    manager = _fake_bridge(monkeypatch, started=False, order=order)
    _run(monkeypatch, order)
    manager.shutdown.assert_not_called()


def test_no_ros2_shutdown_when_the_bridge_was_never_imported(monkeypatch):
    order: list[str] = []
    monkeypatch.delitem(sys.modules, "apyrobo.core.ros2_bridge", raising=False)
    _run(monkeypatch, order)
    assert "ros2_shutdown" not in order
    assert "apyrobo.core.ros2_bridge" not in sys.modules


def test_a_failing_ros2_shutdown_does_not_fail_the_exit(monkeypatch, caplog):
    order: list[str] = []
    _fake_bridge(monkeypatch, started=True, order=order, fail=True)
    _run(monkeypatch, order)  # no exception, no SystemExit
    assert "ROS 2 shutdown failed" in caplog.text


class TestNodeManagerShutdown:
    """The manager stops the executor and joins the spin thread first."""

    @pytest.fixture
    def manager(self, monkeypatch):
        m = ros2_bridge._ROS2NodeManager
        rclpy = MagicMock()
        rclpy.ok.return_value = True
        monkeypatch.setattr(ros2_bridge, "rclpy", rclpy, raising=False)
        release = threading.Event()
        spin = threading.Thread(target=release.wait, daemon=True)
        spin.start()
        executor = MagicMock()
        executor.shutdown.side_effect = lambda timeout_sec=None: release.set()
        node = MagicMock()
        monkeypatch.setattr(m, "_instance", object())
        monkeypatch.setattr(m, "_executor", executor)
        monkeypatch.setattr(m, "_node", node)
        monkeypatch.setattr(m, "_spin_thread", spin)
        yield m, rclpy, executor, node, spin

    def test_shutdown_joins_the_spin_thread_and_clears_state(self, manager):
        m, rclpy, executor, node, spin = manager
        assert m.started()
        m.shutdown(timeout_s=2.0)
        executor.shutdown.assert_called_once_with(timeout_sec=2.0)
        assert not spin.is_alive()
        node.destroy_node.assert_called_once()
        rclpy.shutdown.assert_called_once()
        assert not m.started()
        assert m._executor is None and m._spin_thread is None and m._node is None

    def test_shutdown_twice_is_harmless(self, manager):
        m, rclpy, *_ = manager
        m.shutdown(timeout_s=2.0)
        rclpy.ok.return_value = False
        m.shutdown(timeout_s=2.0)
        rclpy.shutdown.assert_called_once()
