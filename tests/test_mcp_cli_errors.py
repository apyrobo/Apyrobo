"""apyrobo mcp turns adapter start-up failures into a readable error."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys

import pytest

from apyrobo import cli


def _args(**overrides) -> argparse.Namespace:
    base = dict(robot="mock://tb", policy=None, max_speed=None, no_go=None,
                provider="rule", policy_file=None)
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("exc", [
    RuntimeError("The ros2:// adapter is unavailable"),
    ConnectionError("broker unreachable"),
    TimeoutError("robot did not answer"),
])
def test_adapter_start_failure_is_a_readable_error(monkeypatch, capsys, exc):
    def boom(**_kwargs):
        raise exc

    monkeypatch.setattr("apyrobo.mcp_server.build_server", boom)
    with pytest.raises(SystemExit) as info:
        cli.cmd_mcp(_args())
    assert info.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith(f"Error: {exc}")
    assert "Traceback" not in err


@pytest.mark.skipif(importlib.util.find_spec("rclpy") is not None,
                    reason="rclpy is installed, ros2:// would start")
def test_ros2_without_rclpy_exits_cleanly():
    proc = subprocess.run(
        [sys.executable, "-m", "apyrobo", "mcp", "--robot", "ros2://tb3"],
        input="", capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 1
    assert proc.stderr.startswith("Error: The ros2:// adapter is unavailable")
    assert "Traceback" not in proc.stderr
