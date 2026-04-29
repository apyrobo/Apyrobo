#!/usr/bin/env bash
# clean_worktrees.sh — prune merged .claude/worktrees/ branches
#
# Usage:
#   ./scripts/clean_worktrees.sh            # dry run (shows what would be removed)
#   ./scripts/clean_worktrees.sh --force    # actually remove merged worktrees
#   ./scripts/clean_worktrees.sh --help

set -euo pipefail

FORCE=0
HELP=0

for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --help|-h) HELP=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

if [[ $HELP -eq 1 ]]; then
  echo "Usage: $0 [--force]"
  echo ""
  echo "Prune .claude/worktrees/ entries whose branch is already merged into main."
  echo "Runs as a dry-run by default. Pass --force to actually remove them."
  exit 0
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Ensure main is up to date before checking merge status
echo "Fetching origin/main..."
git fetch origin main --quiet

removed=0
kept=0
skipped=0

# Parse worktree list (porcelain format)
# Each worktree block: worktree <path>\nHEAD <sha>\nbranch <refname>\n\n
while IFS= read -r line; do
  if [[ "$line" == worktree\ * ]]; then
    wt_path="${line#worktree }"
    wt_branch=""
    wt_head=""
  elif [[ "$line" == branch\ * ]]; then
    wt_branch="${line#branch refs/heads/}"
  elif [[ "$line" == HEAD\ * ]]; then
    wt_head="${line#HEAD }"
  elif [[ -z "$line" && -n "$wt_path" ]]; then
    # End of a worktree block — evaluate it
    # Only process .claude/worktrees/ paths
    if [[ "$wt_path" == *"/.claude/worktrees/"* ]]; then
      if [[ -z "$wt_branch" ]]; then
        # Detached HEAD or bare — skip
        echo "  SKIP  (detached) $wt_path"
        skipped=$((skipped + 1))
      elif git merge-base --is-ancestor "$wt_head" origin/main 2>/dev/null; then
        # Branch is merged into main
        if [[ $FORCE -eq 1 ]]; then
          echo "  REMOVE  $wt_path  [$wt_branch]"
          git worktree remove --force "$wt_path" 2>/dev/null || echo "    (already gone)"
          git branch -D "$wt_branch" 2>/dev/null || true
          removed=$((removed + 1))
        else
          echo "  DRY-RUN would remove: $wt_path  [$wt_branch]"
          removed=$((removed + 1))
        fi
      else
        echo "  KEEP  $wt_path  [$wt_branch]"
        kept=$((kept + 1))
      fi
    fi
    wt_path=""
    wt_branch=""
    wt_head=""
  fi
done < <(git worktree list --porcelain)

# Prune stale remote-tracking refs
echo ""
echo "Pruning stale remote-tracking refs..."
git remote prune origin

echo ""
if [[ $FORCE -eq 1 ]]; then
  echo "Done. Removed: $removed  Kept: $kept  Skipped: $skipped"
else
  echo "Dry run. Would remove: $removed  Would keep: $kept  Skipped: $skipped"
  echo "Run with --force to actually remove."
fi
