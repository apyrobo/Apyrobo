"""docs/concepts/adapter_pattern.md: its code runs and every scheme it names exists."""

from __future__ import annotations

import re
from pathlib import Path
from unittest import mock

import apyrobo.core.isaac_adapter  # noqa: F401  (registers isaac_sim://)
import apyrobo.core.unitree_adapter  # noqa: F401  (registers unitree://)
from apyrobo import Robot
from apyrobo.core.adapters import _ADAPTER_REGISTRY

DOC = Path(__file__).resolve().parent.parent / "docs" / "concepts" / "adapter_pattern.md"
PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def _blocks() -> list[str]:
    return PYTHON_BLOCK.findall(DOC.read_text(encoding="utf-8"))


def test_the_runnable_example_drives_the_mock_robot(capsys):
    block = next(b for b in _blocks() if "MockAdapter(" in b)
    namespace: dict = {}
    exec(block, namespace)
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "completed"
    assert namespace["robot"].get_position() == (2.0, 3.0)


def test_the_hardware_swap_block_runs():
    block = next(b for b in _blocks() if "unitree://" in b)
    real_discover = Robot.discover
    discovered: list[str] = []

    def discover(uri: str, *args, **kwargs):
        discovered.append(uri)
        if uri.startswith("ros2://"):  # needs rclpy, not in the unit-test env
            assert "ros2" in _ADAPTER_REGISTRY
            return real_discover("mock://turtlebot4")
        return real_discover(uri, *args, **kwargs)

    namespace: dict = {"Robot": Robot}
    with mock.patch.object(Robot, "discover", side_effect=discover):
        exec(block, namespace)
    assert discovered == ["unitree://go2@192.168.1.10", "ros2://turtlebot4"]
    assert type(namespace["robot"]._adapter).__name__ == "IsaacSimAdapter"


def test_every_scheme_the_doc_names_is_registered():
    text = DOC.read_text(encoding="utf-8")
    schemes = set(re.findall(r"`([a-z_0-9]+)://", text))
    schemes |= set(re.findall(r"\"([a-z_0-9]+)://", text))
    assert {"mock", "ros2", "unitree", "isaac_sim", "gazebo"} <= schemes
    missing = sorted(s for s in schemes if s not in _ADAPTER_REGISTRY)
    assert missing == []
