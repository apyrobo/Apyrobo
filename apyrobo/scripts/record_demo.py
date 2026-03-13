#!/usr/bin/env python3
"""
APYROBO Demo Recorder — captures the MVP demo for proof of concept.

Runs the full demo while recording:
    1. Screen capture (Gazebo window) via ffmpeg
    2. Terminal output to a log file
    3. Execution metrics to a JSON report

Output goes to /workspace/recordings/

Usage:
    python3 scripts/record_demo.py
    python3 scripts/record_demo.py --task "deliver package to (5, 3)"
    python3 scripts/record_demo.py --duration 120 --no-screen

Prerequisites:
    - Gazebo must be running (via scripts/launch.sh)
    - ffmpeg must be installed (included in Dockerfile)
    - DISPLAY must be set (for screen capture)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/workspace")

RECORDINGS_DIR = Path("/workspace/recordings")


def setup_recording_dir() -> Path:
    """Create a timestamped recording directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rec_dir = RECORDINGS_DIR / f"demo_{timestamp}"
    rec_dir.mkdir(parents=True, exist_ok=True)
    return rec_dir


def start_screen_recording(rec_dir: Path, duration: int) -> subprocess.Popen | None:
    """Start ffmpeg screen capture in the background."""
    display = os.environ.get("DISPLAY")
    if not display:
        print("  ⚠ DISPLAY not set — skipping screen capture")
        return None

    output_file = rec_dir / "screen_capture.mp4"
    cmd = [
        "ffmpeg",
        "-y",                          # overwrite
        "-f", "x11grab",              # X11 screen capture
        "-framerate", "15",           # 15 fps (lighter)
        "-video_size", "1920x1080",   # adjust if needed
        "-i", display,                # capture this display
        "-t", str(duration),          # max duration
        "-c:v", "libx264",           # H.264 codec
        "-preset", "ultrafast",       # fast encoding
        "-crf", "28",                 # quality (lower = better, 28 = ok for demo)
        "-pix_fmt", "yuv420p",       # compatible pixel format
        str(output_file),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        print(f"  ✓ Screen recording started → {output_file.name}")
        return proc
    except FileNotFoundError:
        print("  ⚠ ffmpeg not found — skipping screen capture")
        return None
    except Exception as e:
        print(f"  ⚠ Screen capture failed: {e}")
        return None


def stop_screen_recording(proc: subprocess.Popen | None) -> None:
    """Stop ffmpeg gracefully."""
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
        print("  ✓ Screen recording stopped")
    except Exception as e:
        print(f"  ⚠ Error stopping recording: {e}")
        proc.kill()


def run_recorded_demo(task: str, rec_dir: Path) -> dict:
    """Run the demo and capture all output + metrics."""

    from apyrobo import Robot, Agent, SafetyEnforcer, SafetyPolicy
    from apyrobo.core.schemas import TaskStatus
    from apyrobo.skills.executor import SkillStatus

    # ── Metrics collection ──
    metrics = {
        "task": task,
        "timestamp": datetime.now().isoformat(),
        "events": [],
        "result": None,
        "duration_seconds": 0,
        "robot_start_position": None,
        "robot_end_position": None,
    }

    log_file = rec_dir / "terminal_output.log"
    log_lines = []

    def log(msg):
        line = f"[{time.time():.1f}] {msg}"
        print(line)
        log_lines.append(line)

    # ── Execute demo ──
    log("═" * 50)
    log("  APYROBO MVP Demo — Recording")
    log("═" * 50)
    log("")

    log("Discovering robot...")
    robot = Robot.discover("gazebo://turtlebot4")
    caps = robot.capabilities()

    log(f"Robot: {caps.name}")
    log(f"  Capabilities: {[c.name for c in caps.capabilities]}")
    log(f"  Sensors: {[s.sensor_id for s in caps.sensors]}")

    start_pos = robot._adapter.position
    metrics["robot_start_position"] = list(start_pos)
    log(f"  Position: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
    log("")

    # Safety
    enforcer = SafetyEnforcer(robot, policy=SafetyPolicy(
        name="demo", max_speed=0.8,
    ))
    log(f"Safety: {enforcer.policy}")
    log("")

    # Agent
    agent = Agent(provider="auto")
    log(f"Task: {task!r}")
    log("-" * 50)

    t0 = time.time()

    def on_event(event):
        elapsed = time.time() - t0
        icon = {
            SkillStatus.PENDING: "⏳",
            SkillStatus.RUNNING: "🔄",
            SkillStatus.COMPLETED: "✅",
            SkillStatus.FAILED: "❌",
        }.get(event.status, "  ")
        line = f"  {icon} [{elapsed:6.1f}s] {event.skill_id}: {event.message}"
        log(line)
        metrics["events"].append({
            "timestamp": elapsed,
            "skill_id": event.skill_id,
            "status": event.status.value,
            "message": event.message,
        })

    result = agent.execute(task=task, robot=robot, on_event=on_event)
    duration = time.time() - t0

    log("-" * 50)
    log(f"Result: {result.status.value}")
    log(f"  Steps: {result.steps_completed}/{result.steps_total}")
    log(f"  Duration: {duration:.1f}s")

    end_pos = robot._adapter.position
    metrics["robot_end_position"] = list(end_pos)
    log(f"  Start: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
    log(f"  End:   ({end_pos[0]:.2f}, {end_pos[1]:.2f})")
    log("")

    # Safety report
    if enforcer.violations:
        log(f"Safety violations: {len(enforcer.violations)}")
    else:
        log("Safety: No violations ✓")
    if enforcer.interventions:
        log(f"Safety interventions: {len(enforcer.interventions)}")
    else:
        log("Safety: No interventions ✓")

    robot.stop()

    # ── Save metrics ──
    metrics["result"] = {
        "status": result.status.value,
        "steps_completed": result.steps_completed,
        "steps_total": result.steps_total,
        "confidence": result.confidence,
        "error": result.error,
        "recovery_actions": [a.value for a in result.recovery_actions_taken],
    }
    metrics["duration_seconds"] = duration
    metrics["safety_violations"] = len(enforcer.violations)
    metrics["safety_interventions"] = len(enforcer.interventions)

    return metrics, log_lines


def run_benchmark(task: str, trials: int, rec_dir: Path) -> None:
    """Run the demo N times and collect aggregate metrics."""

    from apyrobo import Robot, Agent
    from apyrobo.core.schemas import TaskStatus

    print(f"\n  Running {trials} benchmark trials...")
    print(f"  Task: {task!r}")
    print()

    all_results = []

    for i in range(trials):
        print(f"  Trial {i+1}/{trials}...", end=" ", flush=True)
        try:
            robot = Robot.discover("gazebo://turtlebot4")
            agent = Agent(provider="auto")
            t0 = time.time()
            result = agent.execute(task=task, robot=robot)
            duration = time.time() - t0
            robot.stop()

            trial_data = {
                "trial": i + 1,
                "status": result.status.value,
                "duration": duration,
                "steps": result.steps_completed,
                "success": result.status == TaskStatus.COMPLETED,
            }
            all_results.append(trial_data)
            status_icon = "✓" if trial_data["success"] else "✗"
            print(f"{status_icon} {result.status.value} ({duration:.1f}s)")

            # Brief pause between trials
            time.sleep(2.0)

        except Exception as e:
            print(f"✗ Error: {e}")
            all_results.append({
                "trial": i + 1, "status": "error",
                "duration": 0, "steps": 0, "success": False,
            })

    # Aggregate
    successes = sum(1 for r in all_results if r["success"])
    durations = [r["duration"] for r in all_results if r["success"]]
    avg_duration = sum(durations) / len(durations) if durations else 0

    benchmark = {
        "task": task,
        "trials": trials,
        "success_rate": successes / trials if trials > 0 else 0,
        "successes": successes,
        "failures": trials - successes,
        "avg_duration_seconds": avg_duration,
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
    }

    benchmark_file = rec_dir / "benchmark.json"
    with open(benchmark_file, "w") as f:
        json.dump(benchmark, f, indent=2)

    print()
    print(f"  ═══ Benchmark Results ═══")
    print(f"  Success rate: {successes}/{trials} ({benchmark['success_rate']*100:.0f}%)")
    print(f"  Avg duration: {avg_duration:.1f}s")
    print(f"  Report saved: {benchmark_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APYROBO Demo Recorder")
    parser.add_argument("--task", default="deliver package from (1, 0) to (3, 0)")
    parser.add_argument("--duration", type=int, default=120,
                        help="Max screen recording duration (seconds)")
    parser.add_argument("--no-screen", action="store_true",
                        help="Skip screen capture")
    parser.add_argument("--benchmark", type=int, default=0,
                        help="Run N benchmark trials after demo")
    args = parser.parse_args()

    rec_dir = setup_recording_dir()
    print(f"Recording to: {rec_dir}")
    print()

    # Start screen capture
    screen_proc = None
    if not args.no_screen:
        screen_proc = start_screen_recording(rec_dir, args.duration)
    print()

    try:
        # Run the demo
        metrics, log_lines = run_recorded_demo(args.task, rec_dir)

        # Save files
        metrics_file = rec_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  ✓ Metrics saved: {metrics_file.name}")

        log_file = rec_dir / "terminal_output.log"
        with open(log_file, "w") as f:
            f.write("\n".join(log_lines))
        print(f"  ✓ Log saved: {log_file.name}")

        # Run benchmark if requested
        if args.benchmark > 0:
            run_benchmark(args.task, args.benchmark, rec_dir)

    except KeyboardInterrupt:
        print("\n  Interrupted!")
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stop_screen_recording(screen_proc)

    print(f"\n  All recordings in: {rec_dir}")
    print(f"  Files: {[f.name for f in rec_dir.iterdir()]}")
