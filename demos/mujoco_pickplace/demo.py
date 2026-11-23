#!/usr/bin/env python3
"""
Demo: Natural Language → real MuJoCo physics → pick-and-place, rendered
=======================================================================
The rule agent plans "deliver package from A to B", the skill graph
executes on the live ``mujoco://`` bridge, and real physics carries the
package: the base drives over (blocking, speed-limited moves), the
suction gripper welds the box at its current pose and lifts it, and on
release gravity puts it back down. Every frame in the video is rendered
from the same simulation state the skills acted on — nothing is staged.

    pip install 'apyrobo[mujoco]'
    python demos/mujoco_pickplace/demo.py            # writes demo.mp4
    python demos/mujoco_pickplace/demo.py --out x.mp4 --fps 30

Rendering is offscreen (mujoco.Renderer); ffmpeg encodes the stream.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import threading
import time

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import TaskStatus
from apyrobo.skills.agent import Agent

TASK = "deliver package from (0.85, 0.42) to (-1.0, -0.8)"
WIDTH, HEIGHT = 960, 540


HEADLESS_HINT = (
    "MuJoCo could not create an OpenGL context to render the video. On a machine "
    "without a display, pick a headless backend before running the demo:\n"
    "  MUJOCO_GL=egl python demos/mujoco_pickplace/demo.py     (GPU or Mesa EGL)\n"
    "  MUJOCO_GL=osmesa python demos/mujoco_pickplace/demo.py  (software, needs libosmesa)"
)


def grab_frame(adapter, renderer):
    """Render one frame of the live sim.

    Only the scene update reads the simulation, so only it holds the
    adapter's lock. ``render()`` draws from the renderer's own copy of the
    scene; holding the lock through it (slow with software rendering) would
    stall the physics thread and the robot's moves would time out.
    """
    with adapter._lock:
        renderer.update_scene(adapter._data, camera="overview")
    return renderer.render()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(pathlib.Path(__file__).parent / "demo.mp4"))
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    # After argparse so --help works without MuJoCo or ffmpeg.
    try:
        import mujoco
    except ImportError:
        print("This demo needs MuJoCo, which a plain 'pip install apyrobo' does not include.\n"
              "Install it with: pip install 'apyrobo[mujoco]'", file=sys.stderr)
        return 1
    if shutil.which("ffmpeg") is None:
        print("This demo needs ffmpeg on your PATH to write the video "
              "(e.g. apt install ffmpeg, or brew install ffmpeg).", file=sys.stderr)
        return 1

    print("① Discovering the robot: mujoco:// (live physics, realtime pacing)")
    robot = Robot.discover("mujoco://pickplace", realtime=True)
    adapter = robot._adapter
    print(f"   model: {adapter._model_path}")
    print(f"   package at {tuple(round(v, 2) for v in adapter.object_position())}")

    agent = Agent(provider="rule")
    print(f"② Planning: {TASK!r} (rule agent, no LLM)")
    graph = agent.plan(TASK, robot)
    for skill in graph.get_execution_order():
        print(f"   → {skill.skill_id}")

    result: dict = {}

    def run_task() -> None:
        result["task"] = agent.execute(task=TASK, robot=robot)

    worker = threading.Thread(target=run_task)

    print(f"③ Executing while recording {args.out} …")
    try:
        renderer = mujoco.Renderer(adapter._model, height=HEIGHT, width=WIDTH)
    except mujoco.FatalError:
        adapter.shutdown()
        print(HEADLESS_HINT, file=sys.stderr)
        return 1
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(args.fps),
            "-i", "-",
            "-vf", "format=yuv420p", args.out,
        ],
        stdin=subprocess.PIPE,
    )
    assert ffmpeg.stdin is not None

    worker.start()
    frame_dt = 1.0 / args.fps
    settle_until: float | None = None
    frames = 0
    try:
        while True:
            t0 = time.monotonic()
            frame = grab_frame(adapter, renderer)
            ffmpeg.stdin.write(frame.tobytes())
            frames += 1
            if not worker.is_alive():
                if settle_until is None:
                    settle_until = time.monotonic() + 1.5  # let the box land on camera
                elif time.monotonic() >= settle_until:
                    break
            time.sleep(max(0.0, frame_dt - (time.monotonic() - t0)))
    finally:
        ffmpeg.stdin.close()
        ffmpeg.wait()
        renderer.close()
        worker.join(timeout=5.0)
        adapter.shutdown()

    task_result = result.get("task")
    px, py, pz = adapter.object_position()
    print(f"④ Task: {task_result.status.value if task_result else 'n/a'} — "
          f"package now at ({px:.2f}, {py:.2f}, {pz:.2f}), {frames} frames")
    ok = (
        task_result is not None
        and task_result.status == TaskStatus.COMPLETED
        and abs(px - (-1.0)) < 0.5
        and abs(py - (-0.8)) < 0.5
    )
    print("Delivered ✓" if ok else "NOT delivered ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
