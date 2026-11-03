"""The error for ros2:// without ROS 2 tells a ROS user what to do."""

from __future__ import annotations

import pytest

from apyrobo.core.adapters import _ADAPTER_REGISTRY


def _message() -> str:
    adapter = _ADAPTER_REGISTRY["ros2"]
    if adapter.__name__ != "_ROS2Unavailable":
        pytest.skip("ROS 2 is importable here, so the stub is not installed")
    with pytest.raises(RuntimeError) as err:
        adapter("turtlebot4")
    return str(err.value)


def test_says_to_source_ros_first():
    assert "source /opt/ros/<distro>/setup.bash" in _message()


def test_keeps_docker_as_another_way_not_the_only_way():
    message = _message()
    assert "only available" not in message
    assert "docker compose" in message


def test_points_at_the_docs_and_keeps_the_import_cause():
    message = _message()
    assert "docs/mcp.md" in message
    assert "import failed:" in message
