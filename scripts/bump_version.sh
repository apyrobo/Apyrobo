#!/usr/bin/env bash
# Usage: scripts/bump_version.sh <new-version>
# Updates the version string in pyproject.toml and apyrobo/__init__.py.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <new-version>  (e.g. $0 1.1.0)" >&2
    exit 1
fi

NEW="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Validate semver-ish format
if ! [[ "$NEW" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: version must be in X.Y.Z format, got '$NEW'" >&2
    exit 1
fi

# pyproject.toml — first occurrence of version = "..."
sed -i.bak "s/^version = \".*\"/version = \"$NEW\"/" "$ROOT/pyproject.toml"
rm -f "$ROOT/pyproject.toml.bak"

# apyrobo/__init__.py
sed -i.bak "s/__version__ = \".*\"/__version__ = \"$NEW\"/" "$ROOT/apyrobo/__init__.py"
rm -f "$ROOT/apyrobo/__init__.py.bak"

echo "Bumped version to $NEW in pyproject.toml and apyrobo/__init__.py"
echo ""
echo "Don't forget to update CHANGELOG.md with a [${NEW}] section!"
echo ""
echo "Next steps:"
echo "  git add pyproject.toml apyrobo/__init__.py CHANGELOG.md"
echo "  git commit -m \"release: bump version to $NEW\""
echo "  git tag v$NEW && git push origin v$NEW"
