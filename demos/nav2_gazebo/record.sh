#!/usr/bin/env bash
# Re-render demo.gif / demo.mp4 for the Nav2-in-Gazebo demo.
#
# LINUX ONLY — Gazebo Classic does not run under Docker Desktop on macOS.
# CI renders this via .github/workflows/demo-recording.yml (workflow_dispatch)
# and uploads the results as artifacts.
#
# Requirements:
#   - docker + compose v2
#   - vhs (https://github.com/charmbracelet/vhs) with ttyd + ffmpeg
set -euo pipefail
cd "$(dirname "$0")/../.."

COMPOSE=(docker compose -f docker/docker-compose.yml --profile gazebo-nav)

command -v vhs >/dev/null || {
    echo "error: vhs not found — install from https://github.com/charmbracelet/vhs" >&2
    exit 1
}

cleanup() { "${COMPOSE[@]}" down --volumes >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "[record] building the gazebo image…"
"${COMPOSE[@]}" build

echo "[record] starting the sim (gzserver + RSP + Nav2/SLAM)…"
"${COMPOSE[@]}" up -d gazebo-nav-sim

echo "[record] waiting for the sim to be healthy (navigate_to_pose live)…"
for _ in $(seq 1 60); do
    status=$(docker inspect -f '{{.State.Health.Status}}' apyrobo-gazebo-nav-sim 2>/dev/null || echo starting)
    [ "$status" = healthy ] && break
    [ "$status" = unhealthy ] && { echo "[record] FATAL: sim unhealthy" >&2; exit 1; }
    sleep 5
done
[ "$status" = healthy ] || { echo "[record] FATAL: sim never became healthy" >&2; exit 1; }

echo "[record] recording the demo run with vhs…"
vhs demos/nav2_gazebo/demo.tape

echo "[record] done — demos/nav2_gazebo/demo.gif + demo.mp4"
