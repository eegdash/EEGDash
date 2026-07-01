# Repo slimming — keeping `git clone` small

The repository grew to **>450 MB to clone**. This documents why, what changed to
prevent it, and the one-time purge that shrinks it to a few MB.

## What bloated it

| Source | Where | History weight |
|---|---|---|
| `llms-full.txt` (42 MB) committed on every docs deploy (67×) | `gh-pages` branch | ~1.9 GB uncompressed |
| whole docs site snapshot (incl. that file) | `gh-pages` (single commit) | 258 MB |
| `notebooks/scratch_features*.ipynb` with saved outputs, committed 10+× | `develop` history | ~183 MB |
| `consolidated/scidb_datasets.json`, `.verify_cache/*.snirf` | `develop` history | ~20 MB |

`develop`'s **working tree is only ~8 MB** — everything above is dead weight in
*history* (files already deleted from HEAD) plus the auto-generated `gh-pages`
branch, which a default `git clone` fetches along with every other branch.

## Prevention (already in this PR — no action needed)

1. **Docs deploy no longer uses a branch.** `.github/workflows/doc.yaml` now
   publishes via the GitHub Pages *artifact* flow (`actions/upload-pages-artifact`
   + `actions/deploy-pages`) instead of `peaceiris/actions-gh-pages`. The site is
   an ephemeral artifact, so it never lands in a clone again.
2. **`.gitignore`** now blocks `notebooks/scratch_*.ipynb` and `consolidated/`.

### One-time repo setting

Set **Settings → Pages → Source = "GitHub Actions"** (was "Deploy from a branch").
The custom domain (`eegdash.org`) is written into the artifact as a `CNAME` file
by the workflow; you can also keep it set under Settings → Pages.

## The purge (maintainer action — DESTRUCTIVE, requires admin)

This rewrites every commit SHA. **Everyone must re-clone afterward and every open
PR must be re-based.** Do it during a quiet window with team sign-off.

```bash
pipx install git-filter-repo         # once

# 1. Dry run — mirrors the remote, purges, reports before/after size. No push.
scripts/maintenance/slim-repo.sh https://github.com/eegdash/eegdash

# 2. Lift branch protection on `develop` in repo settings (temporarily).

# 3. Rewrite + force-push all branches, and drop gh-pages:
scripts/maintenance/slim-repo.sh https://github.com/eegdash/eegdash --push

# 4. Re-enable branch protection. Delete the gh-pages branch if it lingers:
git push https://github.com/eegdash/eegdash --delete gh-pages
```

After the push, GitHub garbage-collects the unreachable objects; new clones are
small immediately (they only receive objects reachable from the rewritten refs).

### Expected result

Measured on a full mirror of all 68 branches: **149 MB → ~19 MB** (~87% smaller;
the local unpacked `.git` was ~450 MB). The gh-pages branch is gone and the dead
notebook/cache/JSON history is stripped. The ~19 MB floor is legitimate
source-file revision history — going lower would mean squashing away all git
history/blame, which is not recommended.

To also prune your own already-cloned copy without re-cloning:

```bash
git remote prune origin
git reflog expire --expire=now --all && git gc --prune=now --aggressive
```
