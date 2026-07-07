#!/usr/bin/env bash
# Re-render demo.gif / demo.mp4 for the orchestration-flow demo.
#
# flow.py runs the REAL pipeline (Agent → SkillGraph → SafetyEnforcer →
# SkillExecutor) and draws each stage with Pillow, so the video regenerates
# deterministically after any change.
#
# Requirements:
#   - apyrobo importable by `python`  (from repo root: pip install -e .)
#   - Pillow                          (pip install Pillow)
#   - ffmpeg on PATH
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

exec python demos/orchestration_flow/flow.py
