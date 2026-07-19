#!/usr/bin/env bash
# Re-render demo.mp4 / demo.gif for the MuJoCo pick-and-place demo.
# Runs anywhere MuJoCo does (macOS/Linux, no display needed — offscreen).
#
# Requirements: pip install 'apyrobo[mujoco]'; ffmpeg on PATH.
set -euo pipefail
cd "$(dirname "$0")/../.."

command -v ffmpeg >/dev/null || { echo "error: ffmpeg not found" >&2; exit 1; }
python3 -c "import mujoco" 2>/dev/null || {
    echo "error: mujoco not installed — pip install 'apyrobo[mujoco]'" >&2
    exit 1
}

python3 demos/mujoco_pickplace/demo.py --out demos/mujoco_pickplace/demo.mp4

echo "[record] rendering demo.gif …"
ffmpeg -y -loglevel error -i demos/mujoco_pickplace/demo.mp4 \
    -vf "fps=15,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
    demos/mujoco_pickplace/demo.gif
echo "[record] done: demos/mujoco_pickplace/demo.mp4 + demo.gif"
