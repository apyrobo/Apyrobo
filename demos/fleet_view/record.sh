#!/usr/bin/env bash
# Re-render demo.gif / demo.mp4 for the fleet view.
#
# Uses the headless renderer (render.py), which drives the demo's real sim
# and planner and draws frames with Pillow — no browser or screen recorder
# needed, so the video regenerates deterministically after any change.
#
# Requirements:
#   - apyrobo importable by `python`  (from repo root: pip install -e .)
#   - Pillow                          (pip install Pillow)
#   - ffmpeg on PATH
#
# For the *interactive* browser version instead, run:
#   pip install -e '.[websocket]'
#   npm --prefix packages/apyrobo-client-ts install
#   npm --prefix packages/apyrobo-client-ts run build
#   python demos/fleet_view/server.py
#   → open http://localhost:8420/demos/fleet_view/index.html
set -euo pipefail
cd "$(dirname "$0")/../.."

python -c "import apyrobo, PIL" 2>/dev/null || {
    echo "error: need apyrobo + Pillow — run: pip install -e . Pillow" >&2
    exit 1
}
command -v ffmpeg >/dev/null || {
    echo "error: ffmpeg not found — install from https://ffmpeg.org" >&2
    exit 1
}

exec python demos/fleet_view/render.py
