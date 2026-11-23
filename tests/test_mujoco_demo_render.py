"""demos/mujoco_pickplace/demo.py: drawing a frame must not stall the physics.

The demo records a video while the task runs. Rendering is slow without a
GPU (software OpenGL takes ~100 ms a frame), and the demo used to hold the
simulation lock for all of it, so the physics thread starved, the first move
timed out after 60 s and the demo ended "NOT delivered". These tests stand in
a slow fake renderer and ffmpeg, so they need MuJoCo but no OpenGL.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import threading
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos" / "mujoco_pickplace" / "demo.py"


def _load_demo():
    spec = importlib.util.spec_from_file_location("mujoco_pickplace_demo", DEMO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _SlowRenderer:
    """Stands in for mujoco.Renderer; render() is as slow as software GL."""

    def __init__(self, adapter=None, render_s: float = 0.15):
        self.adapter = adapter
        self.render_s = render_s
        self.renders = 0
        self.locked_during_render = []

    def update_scene(self, data, camera=None):
        if self.adapter is not None:
            assert self.adapter._lock.locked(), "the scene must be read under the lock"

    def render(self):
        import numpy as np

        if self.adapter is not None:
            self.locked_during_render.append(self.adapter._lock.locked())
        time.sleep(self.render_s)
        self.renders += 1
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def close(self):
        pass


class _FakeAdapter:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = object()


def test_grab_frame_reads_the_scene_under_the_lock_and_renders_outside_it():
    demo = _load_demo()
    adapter = _FakeAdapter()
    renderer = _SlowRenderer(adapter, render_s=0.0)
    demo.grab_frame(adapter, renderer)
    assert renderer.locked_during_render == [False]


class _FakeStdin:
    def write(self, _data):
        pass

    def close(self):
        pass


class _FakeFfmpeg:
    def __init__(self, *_args, **_kwargs):
        self.stdin = _FakeStdin()

    def wait(self):
        return 0


def test_demo_delivers_with_slow_rendering(monkeypatch, tmp_path, capsys):
    mujoco = pytest.importorskip("mujoco")
    demo = _load_demo()

    made = []

    def slow_renderer(model, height, width):
        r = _SlowRenderer()
        made.append(r)
        return r

    monkeypatch.setattr(mujoco, "Renderer", slow_renderer)
    monkeypatch.setattr(demo.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(demo.subprocess, "Popen", _FakeFfmpeg)
    monkeypatch.setattr(sys, "argv", ["demo.py", "--out", str(tmp_path / "x.mp4")])

    code = demo.main()
    out = capsys.readouterr().out
    assert code == 0, out
    assert "Task: completed" in out
    assert "Delivered" in out and "NOT delivered" not in out
    assert made and made[0].renders > 0


def test_demo_without_opengl_prints_the_headless_setting(monkeypatch, tmp_path, capsys):
    mujoco = pytest.importorskip("mujoco")
    demo = _load_demo()

    def no_gl(model, height, width):
        raise mujoco.FatalError(
            "an OpenGL platform library has not been loaded into this process"
        )

    monkeypatch.setattr(mujoco, "Renderer", no_gl)
    monkeypatch.setattr(demo.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(sys, "argv", ["demo.py", "--out", str(tmp_path / "x.mp4")])

    assert demo.main() == 1
    err = capsys.readouterr().err
    assert "MUJOCO_GL=egl" in err
    assert "MUJOCO_GL=osmesa" in err
    assert not (tmp_path / "x.mp4").exists()


def test_readme_names_the_headless_setting():
    text = (ROOT / "demos" / "mujoco_pickplace" / "README.md").read_text()
    assert "No display needed" not in text
    assert "MUJOCO_GL=egl" in text
