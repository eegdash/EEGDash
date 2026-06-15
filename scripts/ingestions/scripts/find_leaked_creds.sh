#!/usr/bin/env bash
# Find leaked credentials in commit messages, file contents, and the
# index. Exit 1 if any are found; 0 otherwise.
#
# Patterns target what's been observed leaking historically:
#   - EEGDASH_ADMIN_TOKEN=<20+ alphanumerics>
#   - MONGO_INITDB_ROOT_PASSWORD=<8+ chars>
#   - 40+ char hex strings (CI_TOKEN shape)
#
# Add new patterns at the top; the script reports per-pattern matches.

set -u  # not -e: we want to count matches, not bail on first miss

if ! git_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  printf '[find_leaked_creds] Not in a git repo — refusing to declare clean.\n' >&2
  exit 2
fi
cd "$git_root" >/dev/null

PATTERNS=(
  'EEGDASH_ADMIN_TOKEN[[:space:]]*[=:][[:space:]]*"?[A-Za-z0-9]{20,}'
  'MONGO_INITDB_ROOT_PASSWORD[[:space:]]*[=:][[:space:]]*"?[^[:space:]"]{8,}'
  'ADMIN_TOKEN[[:space:]]*[=:][[:space:]]*"?[A-Za-z0-9]{20,}'
  'CI_TOKEN[[:space:]]*[=:][[:space:]]*"?[a-f0-9]{40,}'
  # Generic AWS-style key shape (no separator)
  'AKIA[0-9A-Z]{16}'
)

total=0

scan() {
  local label="$1" cmd="$2"
  for pat in "${PATTERNS[@]}"; do
    matches=$(eval "$cmd" 2>/dev/null | grep -E "$pat" || true)
    if [[ -n "$matches" ]]; then
      printf '\n=== %s — pattern: %s ===\n' "$label" "$pat"
      printf '%s\n' "$matches"
      lines=$(printf '%s\n' "$matches" | wc -l)
      total=$((total + lines))
    fi
  done
}

scan "Commit messages" "git log --all --format=%B"
scan "Tracked file contents" "git grep -I -n '' -- . ':(exclude)*.snirf' ':(exclude)*.tmet' ':(exclude)*.edf' ':(exclude)*.bdf' ':(exclude)*.set' ':(exclude)*.fif' ':(exclude)*.vhdr' ':(exclude)*.cnt' ':(exclude)*.nwb' 2>/dev/null"
scan "Staged changes"  "git diff --cached"

if [[ $total -gt 0 ]]; then
  printf '\n[find_leaked_creds] Found %d suspect match(es). Investigate above.\n' "$total" >&2
  exit 1
fi

printf '[find_leaked_creds] Clean.\n'
exit 0
