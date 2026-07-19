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
import subprocess
import sys
import threading
import time

from apyrobo.core.robot import Robot
from apyrobo.core.schemas import TaskStatus
from apyrobo.skills.agent import Agent

TASK = "deliver package from (0.85, 0.42) to (-1.0, -0.8)"
WIDTH, HEIGHT = 960, 540


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(pathlib.Path(__file__).parent / "demo.mp4"))
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    import mujoco  # after argparse so --help works without it

    print(f"① Discovering the robot: mujoco:// (live physics, realtime pacing)")
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
    renderer = mujoco.Renderer(adapter._model, height=HEIGHT, width=WIDTH)
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
            with adapter._lock:
                renderer.update_scene(adapter._data, camera="overview")
                frame = renderer.render()
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
