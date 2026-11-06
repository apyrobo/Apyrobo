"""rotate on an adapter that cannot turn in place is an error, not 'rotated'."""
from __future__ import annotations

import json

from apyrobo.core.adapters import CapabilityAdapter, MockAdapter
from apyrobo.mcp_server import build_server


class NoRotateAdapter(MockAdapter):
    """A mock:// adapter that keeps the base no-op rotate, as ROS2Adapter does."""

    rotate = CapabilityAdapter.rotate


def call(name, msg_id=2, **arguments):
    return {
        "jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def initialize():
    return {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "t", "version": "0"}},
    }


def make(tmp_path, *, can_rotate):
    log = tmp_path / "audit.jsonl"
    server = build_server(audit_log=log)
    server.handle(initialize())
    robot = server.toolbox.robot
    if not can_rotate:
        robot._adapter = NoRotateAdapter(robot_name=robot._adapter.robot_name)
    return server, log


def text(reply):
    return reply["result"]["content"][0]["text"]


def test_rotate_without_rotation_is_an_error_and_sends_nothing(tmp_path):
    server, log = make(tmp_path, can_rotate=False)
    adapter = server.toolbox.robot._adapter
    heading = adapter.get_orientation()
    reply = server.handle(call("rotate", angle_rad=1.0))
    assert reply["result"]["isError"] is True
    assert "rotate is not available" in text(reply)
    assert "NoRotateAdapter" in text(reply)
    assert "did not move" in text(reply)
    assert "rotated" not in text(reply)
    assert adapter.get_orientation() == heading
    line = json.loads(log.read_text().splitlines()[-1])
    assert line["tool"] == "rotate"
    assert line["decision"] == "failed"
    assert "not available" in line["detail"]


def test_rotate_still_works_on_mock(tmp_path):
    server, log = make(tmp_path, can_rotate=True)
    assert server.toolbox.rotate_available() is True
    reply = server.handle(call("rotate", angle_rad=0.5))
    assert reply["result"]["isError"] is False
    assert json.loads(text(reply))["status"] == "rotated"


def test_ros2_adapter_has_no_rotate_of_its_own():
    """docs/mcp.md says rotate is an error on ros2://; that holds while ROS2Adapter
    keeps the base rotate. Read from source, so no rclpy is needed."""
    import ast
    import pathlib

    import apyrobo.core.ros2_bridge as bridge

    tree = ast.parse(pathlib.Path(bridge.__file__).read_text())
    cls = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "ROS2Adapter"
    )
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert "move" in methods
    assert "rotate" not in methods
