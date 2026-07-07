#!/usr/bin/env bash
# Build the reference TypeScript client and launch the live fleet view.
#
# Requirements:
#   - apyrobo with the websocket extra:  pip install -e '.[websocket]'
#   - node + npm (to build packages/apyrobo-client-ts)
#
# This is an interactive browser demo, so there's nothing to pre-render —
# the script just builds the client and starts the server, then prints the
# URL to open.
set -euo pipefail
cd "$(dirname "$0")/../.."

python -c "import apyrobo, websockets" 2>/dev/null || {
    echo "error: need apyrobo + websockets — run: pip install -e '.[websocket]'" >&2
    exit 1
}

if [ ! -f packages/apyrobo-client-ts/dist/index.js ]; then
    echo "Building the reference TypeScript client…"
    npm --prefix packages/apyrobo-client-ts install
    npm --prefix packages/apyrobo-client-ts run build
fi

echo "Open http://localhost:8420/demos/fleet_view/index.html once the server is up."
exec python demos/fleet_view/server.py
