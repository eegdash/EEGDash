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
  --path notebooks/               # scratch + tutorial notebooks w/ saved outputs (none in HEAD)
  --path consolidated/            # generated dataset JSON dumps
  --path .verify_cache/
  --path .verify_cache_python/
  --path .verify_cache_test/
  --path docs/architecture2.pptx  # 2MB deck, deleted from HEAD
)

# The default branch. Its file tree MUST be identical before and after the
# rewrite — the safety gate below aborts if a still-live file would be removed.
DEFAULT_BRANCH="develop"

if ! git filter-repo --version >/dev/null 2>&1; then
  echo "ERROR: git-filter-repo not found. Install with: pipx install git-filter-repo" >&2
  exit 1
fi

rm -rf "$WORK"
echo ">> Mirroring $REMOTE into $WORK ..."
git clone --mirror "$REMOTE" "$WORK"
cd "$WORK"

echo ">> Size BEFORE:"; du -sh .

# Record the default branch's file tree BEFORE rewriting. Because every purged
# path is already absent from HEAD, this tree must be byte-identical afterward.
TREE_BEFORE="$(git rev-parse "refs/heads/${DEFAULT_BRANCH}^{tree}")"

# gh-pages is already deleted on the remote; drop it locally too if it lingers.
git update-ref -d refs/heads/gh-pages 2>/dev/null || true

echo ">> Rewriting history (stripping dead-weight paths) ..."
git filter-repo --force "${PURGE_ARGS[@]}"

git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ">> Size AFTER:"; du -sh .

# SAFETY GATE: the working files on the default branch must be unchanged.
TREE_AFTER="$(git rev-parse "refs/heads/${DEFAULT_BRANCH}^{tree}")"
if [ "$TREE_BEFORE" != "$TREE_AFTER" ]; then
  echo "ABORT: ${DEFAULT_BRANCH} file tree changed by the rewrite ($TREE_BEFORE -> $TREE_AFTER)." >&2
  echo "       A still-live file would be removed. NOT pushing. Inspect $(pwd)." >&2
  exit 1
fi
echo ">> Safety gate OK: ${DEFAULT_BRANCH} file tree unchanged ($TREE_AFTER)."

if [ "$PUSH" = "--push" ]; then
  echo ">> Force-pushing rewritten history (mirror) to $REMOTE ..."
  echo ">> (Lift branch protection on ${DEFAULT_BRANCH} first, or this rejects.)"
  git push --force --mirror "$REMOTE"
  echo ">> Done. KEEP this mirror ($(pwd)) as a backup until you've verified the"
  echo "   remote. Tell collaborators to re-clone; fork owners to re-sync."
else
  echo ">> Dry run complete (safety gate passed). Inspect $(pwd), then push with:"
  echo "     git -C $(pwd) push --force --mirror $REMOTE"
  echo ">> KEEP $(pwd) as a backup of the pre-rewrite remote until you're done."
fi
