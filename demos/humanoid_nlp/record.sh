#!/usr/bin/env bash
# Re-render demo.gif / demo.mp4 for this demo.
#
# Requirements:
#   - vhs        (https://github.com/charmbracelet/vhs — brew install vhs)
#   - apyrobo importable by `python` (from the repo root: pip install -e .)
set -euo pipefail
cd "$(dirname "$0")/../.."

python -c "import apyrobo" 2>/dev/null || {
    echo "error: apyrobo is not importable — run 'pip install -e .' first" >&2
    exit 1
}
command -v vhs >/dev/null || {
    echo "error: vhs not found — install from https://github.com/charmbracelet/vhs" >&2
    exit 1
}

exec vhs demos/humanoid_nlp/demo.tape
