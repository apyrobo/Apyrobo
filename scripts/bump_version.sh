#!/usr/bin/env bash
# Usage: scripts/bump_version.sh <new-version>
# Updates the version string in pyproject.toml, apyrobo/__version__.py,
# and prepends a stub section to CHANGELOG.md.
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

# 1. pyproject.toml — first occurrence of version = "..."
sed -i.bak "s/^version = \".*\"/version = \"$NEW\"/" "$ROOT/pyproject.toml"
rm -f "$ROOT/pyproject.toml.bak"
echo "  pyproject.toml  → $NEW"

# 2. apyrobo/__version__.py — single-source-of-truth version file
sed -i.bak "s/__version__ = \".*\"/__version__ = \"$NEW\"/" "$ROOT/apyrobo/__version__.py"
rm -f "$ROOT/apyrobo/__version__.py.bak"
echo "  __version__.py  → $NEW"

# 3. apyrobo/__init__.py — backward-compat inline string (if still present)
if grep -q '__version__ = "' "$ROOT/apyrobo/__init__.py" 2>/dev/null; then
    sed -i.bak "s/__version__ = \".*\"/__version__ = \"$NEW\"/" "$ROOT/apyrobo/__init__.py"
    rm -f "$ROOT/apyrobo/__init__.py.bak"
    echo "  __init__.py     → $NEW"
fi

# 4. CHANGELOG.md — prepend a stub section after the header block
STUB="## [$NEW] - $TODAY

### Added
- <!-- describe new features -->

### Changed
- <!-- describe changes -->

### Fixed
- <!-- describe bug fixes -->

---

"

# Insert stub after the first "---" separator in CHANGELOG.md
CHANGELOG="$ROOT/CHANGELOG.md"
if [[ -f "$CHANGELOG" ]]; then
    # Find line number of first "---" separator and insert after it
    FIRST_HR=$(grep -n "^---$" "$CHANGELOG" | head -1 | cut -d: -f1)
    if [[ -n "$FIRST_HR" ]]; then
        # Write lines up to and including the separator, then the stub, then the rest
        {
            head -n "$FIRST_HR" "$CHANGELOG"
            echo ""
            printf '%s' "$STUB"
            tail -n +"$((FIRST_HR + 1))" "$CHANGELOG"
        } > "$CHANGELOG.tmp"
        mv "$CHANGELOG.tmp" "$CHANGELOG"
        echo "  CHANGELOG.md    → stub added for $NEW"
    else
        echo "  CHANGELOG.md    → no '---' found; prepend stub manually"
    fi
fi

echo ""
echo "Bumped version to $NEW in all three files."
echo ""
echo "Fill in the CHANGELOG.md stub for [$NEW], then:"
echo "  git add pyproject.toml apyrobo/__version__.py CHANGELOG.md"
echo "  git commit -m \"release: bump version to $NEW\""
echo "  git tag v$NEW && git push origin v$NEW"
