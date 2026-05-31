# Testing-corpus update required: digest snapshots

The cheap-metadata-resolver work (`docs/superpowers/specs/2026-05-31-cheap-metadata-resolver-design.md`)
changes digest output **additively**: every Record now carries `duration_seconds`
and `_metadata_provenance` gains a `duration_seconds` entry. Later phases also flip
some `_metadata_provenance.ntimes` values from `mne_fallback` → `binary_parser`/
`sidecar_arithmetic` as cheap header paths replace the MNE reads.

The digest **golden-master snapshots** live in the external, SHA-pinned
`eegdash-testing-data` corpus (fetched via `eegdash.testing.data_file`, pin in
`eegdash/testing.py`), not in this repo. The regenerated snapshot JSONs are vendored
here under `digest_snapshots/outputs/<id>/` as the artifact to apply.

## To land this in CI

1. Copy `docs/superpowers/corpus-updates/digest_snapshots/outputs/*` over the matching
   files in the `eegdash-testing-data` repo.
2. Publish a new corpus tarball (bump its version, e.g. `0.2.0` → `0.3.0`).
3. Bump `VERSION` + the `registry` SHA256 in `eegdash/testing.py` to the new tarball.

Until then, `tests/digest/test_snapshot.py` passes **locally** (the local pooch cache
was regenerated) but will fail in CI against the old pinned corpus — this is the
expected, documented golden-master/corpus-bump coupling, not a regression.

## What changed in the snapshots

- `*_records.json`: each record gains `"duration_seconds"` and
  `_metadata_provenance` gains `"duration_seconds"`.
- Verified the delta is additive-only (no existing field value changed except the
  redacted `digested_at` timestamp).
