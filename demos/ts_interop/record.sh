#!/usr/bin/env bash
# Re-render demo.gif / demo.mp4 for the TS interop demo (mock robot — runs
# anywhere, no ROS or Docker needed).
#
# Requirements:
#   - apyrobo installed (`pip install -e .`)
#   - the TS client built (packages/apyrobo-client-ts: npm install && npm run build)
#   - vhs (https://github.com/charmbracelet/vhs) with ttyd + ffmpeg
set -euo pipefail
cd "$(dirname "$0")/../.."

command -v vhs >/dev/null || {
    echo "error: vhs not found — install from https://github.com/charmbracelet/vhs" >&2
    exit 1
}
command -v apyrobo >/dev/null || python3 -c "import apyrobo" 2>/dev/null || {
    echo "error: apyrobo not installed — pip install -e ." >&2
    exit 1
}
[ -f packages/apyrobo-client-ts/dist/index.js ] || {
    echo "error: TS client not built — cd packages/apyrobo-client-ts && npm install && npm run build" >&2
    exit 1
}

echo "[record] recording the demo run with vhs…"
vhs demos/ts_interop/demo.tape
echo "[record] done: demos/ts_interop/demo.gif + demo.mp4"
