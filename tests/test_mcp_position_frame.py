"""robot_status says which frame the position is in, when the adapter knows."""

from __future__ import annotations

from apyrobo.mcp_server import build_server


def _status(server):
    return server.toolbox.call("robot_status", {})


def test_robot_status_reports_the_adapters_frame():
    server = build_server()
    server.toolbox.robot._adapter.position_frame = "odom"
    assert _status(server)["position_frame"] == "odom"


def test_robot_status_leaves_the_field_out_when_the_adapter_has_no_frame():
    assert "position_frame" not in _status(build_server())
