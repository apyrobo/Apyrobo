#!/usr/bin/env bash
# Usage: scripts/bump_version.sh <new-version>
# Updates the version string in pyproject.toml, apyrobo/__version__.py and both
# versions in server.json, and turns the CHANGELOG.md [Unreleased] section into
# the section for the new version, leaving a fresh empty [Unreleased] above it.
# Refuses, changing nothing, if [Unreleased] is empty or the version already has
# a section. It never commits, tags or uploads.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <new-version>  (e.g. $0 1.1.0)" >&2
    exit 1
fi

NEW="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TODAY="$(date +%Y-%m-%d)"

# Validate semver-ish format
if ! [[ "$NEW" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: version must be in X.Y.Z format, got '$NEW'" >&2
    exit 1
fi

# 1. CHANGELOG.md and server.json first: they can refuse, and then nothing changed
python3 - "$ROOT" "$NEW" "$TODAY" <<'PY'
import json
import re
import sys
from pathlib import Path

root, new, today = Path(sys.argv[1]), sys.argv[2], sys.argv[3]


def fail(message):
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


changelog = root / "CHANGELOG.md"
text = changelog.read_text(encoding="utf-8")
if re.search(rf"^## \[{re.escape(new)}\]", text, re.MULTILINE):
    fail(f"CHANGELOG.md already has a section for {new}")
heading = re.search(r"^## \[Unreleased\][^\n]*\n", text, re.MULTILINE)
if heading is None:
    fail("CHANGELOG.md has no '## [Unreleased]' section")
following = re.search(r"^## \[", text[heading.end():], re.MULTILINE)
end = heading.end() + following.start() if following else len(text)
body = text[heading.end():end]
if not any(
    line.strip() and line.strip() != "---" and not line.startswith("#")
    and "<!--" not in line
    for line in body.splitlines()
):
    fail("the [Unreleased] section of CHANGELOG.md is empty; nothing to release")

server_path = root / "server.json"
server = json.loads(server_path.read_text(encoding="utf-8"))
server["version"] = new
for package in server.get("packages", []):
    if package.get("registryType") == "pypi":
        package["version"] = new

changelog.write_text(
    text[:heading.start()]
    + f"## [Unreleased]\n\n---\n\n## [{new}] - {today}\n"
    + text[heading.end():],
    encoding="utf-8",
)
server_path.write_text(
    json.dumps(server, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
PY
echo "  CHANGELOG.md    → [Unreleased] is now [$NEW] - $TODAY"
echo "  server.json     → $NEW"

# 2. pyproject.toml — first occurrence of version = "..."
sed -i.bak "s/^version = \".*\"/version = \"$NEW\"/" "$ROOT/pyproject.toml"
rm -f "$ROOT/pyproject.toml.bak"
echo "  pyproject.toml  → $NEW"

# 3. apyrobo/__version__.py — single-source-of-truth version file
sed -i.bak "s/__version__ = \".*\"/__version__ = \"$NEW\"/" "$ROOT/apyrobo/__version__.py"
rm -f "$ROOT/apyrobo/__version__.py.bak"
echo "  __version__.py  → $NEW"

# 4. apyrobo/__init__.py — backward-compat inline string (if still present)
if grep -q '__version__ = "' "$ROOT/apyrobo/__init__.py" 2>/dev/null; then
    sed -i.bak "s/__version__ = \".*\"/__version__ = \"$NEW\"/" "$ROOT/apyrobo/__init__.py"
    rm -f "$ROOT/apyrobo/__init__.py.bak"
    echo "  __init__.py     → $NEW"
fi

echo ""
echo "Bumped version to $NEW. Next:"
echo "  python scripts/release_check.py"
echo "  git add pyproject.toml apyrobo/__version__.py apyrobo/__init__.py server.json CHANGELOG.md"
echo "  git commit -m \"release: bump version to $NEW\""
echo "  git tag v$NEW && git push origin v$NEW"
