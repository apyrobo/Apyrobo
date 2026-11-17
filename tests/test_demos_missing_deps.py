"""The MuJoCo and Nav2 demos stop with a short hint, not a traceback, when
their dependency is missing (a plain ``pip install apyrobo`` has neither
MuJoCo nor ROS 2)."""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Make the named modules unimportable in the child, then run the demo as
# __main__. A None entry in sys.modules makes ``import name`` raise ImportError.
_RUNNER = (
    "import runpy, sys\n"
    "for name in sys.argv[2].split(','):\n"
    "    if name:\n"
    "        sys.modules[name] = None\n"
    "path = sys.argv[1]\n"
    "sys.argv = [path]\n"
    "runpy.run_path(path, run_name='__main__')\n"
)


def _run(demo: str, hide: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _RUNNER, str(ROOT / demo), hide],
        capture_output=True, text=True, timeout=120, cwd=ROOT,
    )


def test_mujoco_demo_without_mujoco_prints_install_hint():
    proc = _run("demos/mujoco_pickplace/demo.py", "mujoco")
    assert proc.returncode == 1
    out = proc.stdout + proc.stderr
    assert "Traceback" not in out
    assert "pip install 'apyrobo[mujoco]'" in out


def test_mujoco_demo_without_ffmpeg_says_so(tmp_path):
    # MuJoCo itself may or may not be installed here; fake an importable one
    # so the ffmpeg check is what stops the demo.
    fake = tmp_path / "mujoco.py"
    fake.write_text("")
    # PATH holds only tmp_path, so there is no ffmpeg on it.
    env = dict(os.environ, PATH=str(tmp_path), PYTHONPATH=str(tmp_path))
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(ROOT / "demos/mujoco_pickplace/demo.py"), ""],
        capture_output=True, text=True, timeout=120, cwd=ROOT, env=env,
    )
    assert proc.returncode == 1
    out = proc.stdout + proc.stderr
    assert "Traceback" not in out
    assert "ffmpeg" in out


def test_nav2_demo_without_ros2_points_at_the_sim():
    proc = _run("demos/nav2_gazebo/demo.py", "rclpy")
    assert proc.returncode == 1
    out = proc.stdout + proc.stderr
    assert "Traceback" not in out
    assert "ros2:// adapter is unavailable" in out
    assert "--profile gazebo-nav run --rm gazebo-nav-demo" in out
    assert "mock://" in out
