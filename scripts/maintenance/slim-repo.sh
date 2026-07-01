#!/usr/bin/env bash
#
# slim-repo.sh — one-time git history purge to shrink clone size.
#
# WHY: the clone was >450MB because (a) the gh-pages branch committed the whole
# docs site (incl. a 42MB llms-full.txt) on every deploy, and (b) develop's
# history carried heavy scratch notebooks + generated data dumps that were later
# deleted from HEAD but still live in history. See docs/maintenance/repo-slimming.md.
#
# WHAT IT DOES (on a fresh --mirror clone, never your working checkout):
#   1. drops the gh-pages branch (docs now deploy via the Pages *artifact* flow)
#   2. strips the proven dead-weight paths from ALL branch history
#   3. repacks and reports before/after size
#   4. force-pushes the rewritten history back  (only with --push)
#
# DESTRUCTIVE: this rewrites every commit SHA. After pushing, everyone must
# re-clone and every open PR must be re-based. Run only with maintainer sign-off,
# and temporarily lift branch protection on the default branch first.
#
# Usage:
#   scripts/maintenance/slim-repo.sh <remote-url>            # dry run (inspect only)
#   scripts/maintenance/slim-repo.sh <remote-url> --push     # rewrite + force-push
#
set -euo pipefail

REMOTE="${1:?usage: slim-repo.sh <remote-url> [--push]}"
PUSH="${2:-}"
WORK="eegdash-slim.git"

# Paths purged from history. All of these are already absent from HEAD; this
# only reclaims their dead history. Add more here if new dead weight is found.
PURGE_ARGS=(
  --invert-paths
  --path notebooks/          # scratch + tutorial notebooks w/ saved outputs (none in HEAD)
  --path consolidated/       # generated dataset JSON dumps
  --path .verify_cache/
  --path .verify_cache_python/
  --path .verify_cache_test/
)

if ! git filter-repo --version >/dev/null 2>&1; then
  echo "ERROR: git-filter-repo not found. Install with: pipx install git-filter-repo" >&2
  exit 1
fi

rm -rf "$WORK"
echo ">> Mirroring $REMOTE into $WORK ..."
git clone --mirror "$REMOTE" "$WORK"
cd "$WORK"

echo ">> Size BEFORE:"; du -sh .

# Docs no longer live in a branch — drop gh-pages entirely.
git update-ref -d refs/heads/gh-pages 2>/dev/null || true

echo ">> Rewriting history (stripping dead-weight paths) ..."
git filter-repo --force "${PURGE_ARGS[@]}"

git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ">> Size AFTER:"; du -sh .

if [ "$PUSH" = "--push" ]; then
  echo ">> Force-pushing rewritten history (mirror) to $REMOTE ..."
  git push --force --mirror "$REMOTE"
  echo ">> Done. Tell everyone to re-clone; delete stale gh-pages if it lingers."
else
  echo ">> Dry run complete. Inspect $(pwd), then push with:"
  echo "     git -C $(pwd) push --force --mirror $REMOTE"
fi
