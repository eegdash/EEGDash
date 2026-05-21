# RecordEnumerator merge — Stage 3 plan (orchestrator collapse)

Stages 1 + 2 landed on `record-enumerator-merge`. The Module, the
shared `write_dataset_outputs` helper, and the Adapter wiring (via a
stage-2D tempfile bridge) all work. 223 tests pass.

Stage 3 — the actual orchestrator collapse — requires moving ~900 LOC
across files. This doc says why I stopped before attempting it and
what the next session needs to do it safely.

## Why I stopped here

The four candidates I considered for a quick Stage 3:

### Option A — Route fallback paths through the Adapter

`digest_dataset`'s 3 fallback sites currently say
`return digest_from_manifest(...)`. Routing them via
`ManifestEnumerator.enumerate()` would exercise the Adapter in
production… but the Adapter delegates to `digest_from_manifest` via
a **tempfile bridge** (Stage 2D), so the net effect is:

```
digest_dataset  →  ManifestEnumerator.enumerate()
                       → bridge writes JSON to tempfile
                       → bridge reads JSON back
                       → returns EnumerationResult
                  →  write_dataset_outputs writes JSON AGAIN
```

That's a **double write**. ~50 MB of unnecessary disk I/O per
NEMAR-scale dataset. Acceptable for tests; bad in production. Net
LOC delta: ~+10 (more verbose). No structural benefit.

**Reject.**

### Option B — Extract `_enumerate_via_bids` + `_enumerate_via_manifest`

Cleanest. The legacy function bodies (262 + 628 = **890 LOC**) move
into private helpers that return `EnumerationResult`. The Adapters
call them directly (no round-trip). `digest_dataset` shrinks to ~50
LOC. `digest_from_manifest` deletes entirely.

**Risk**: 890 LOC of mechanical extraction. Each variable rename, each
preserved local-scope binding, each error-return-shape preservation
is a chance for behavioural drift. The characterisation tests +
LOC canary catch some classes of drift (the function exists, returns
something, LOC budget) but **don't catch byte-level JSON drift**.

For the production paths (OpenNeuro / NEMAR), byte-level drift would
mean ingesting subtly-wrong Records for hundreds of datasets. The
existing JSON files in MongoDB are the contract; the refactor must
produce byte-identical output.

**Doable, but only with a fixture pre-snapshot. See "What stage 3
needs" below.**

### Option C — Add `return_only=True` kwarg

Add a kwarg to both legacy functions: when True, skip the JSON write,
return `EnumerationResult` instead of summary dict. The Adapters call
with `return_only=True`. Net LOC delta: ~+15.

**Reject** — confusing dual-mode functions, signature drift, two
return shapes from one function. Style guide forbids this kind of
mode-flag.

### Option D — Move bodies + ADD characterisation tests in one commit

A safer version of Option B: extract the bodies AND add a fixture-
based byte-comparison test that asserts the new code path produces
identical JSON to a saved snapshot.

**Doable, the right design, but it's a 2-3 hour focused refactor**
that needs:
- 1 hour to extract + sanity-test
- 30 min to write the snapshot-comparison test
- 30 min to fix any drift the test surfaces

Not a quick win.

## What Stage 3 needs

To execute Stage 3 safely, the next session should:

### Pre-flight (DONE — landed in this branch)

1. ~~Pick a real CC0 fixture dataset~~ ✓ `ds_snapshot_vhdr` constructed
   from the existing CC0 VHDR triple, wrapped in a minimal BIDS root
   (`dataset_description.json` + `participants.tsv` + `sub-xp101/eeg/*`).
2. ~~Run the current `digest_dataset` against it~~ ✓ produced 4 JSON
   files in `tests/fixtures/digest_snapshots/outputs/ds_snapshot_vhdr/`.
   Snapshot committed (force-added against `*.json` gitignore).
3. ~~Snapshot comparison test~~ ✓ `tests/test_digest_snapshot.py` runs
   `digest_dataset` against the fixture, sanitizes non-deterministic
   fields (timestamps, absolute paths), and compares the 4 JSON files
   field-by-field to the committed snapshot. 6 assertions across the
   4 files + record count + fingerprint stability.

**Manifest-only snapshot is NOT done yet** — the manifest path is
exercised only via the Stage 2D bridge in tests. A future Stage 3
should add a parallel `ds_snapshot_manifest` fixture (a synthetic
`manifest.json` describing a few files) to cover the second algorithm.
The BIDS path is the production hot path; the snapshot we have
guards the right thing first.

### Extract bodies (60-90 min)

4. Extract `_enumerate_via_bids(dataset_dir, dataset_id, source,
   source_adapter, digested_at, manifest_data) -> EnumerationResult`
   from `digest_dataset`'s body (lines 2594-2769).
5. Extract `_enumerate_via_manifest(...)` from `digest_from_manifest`
   similarly.
6. Wire the Adapter classes to call these directly (delete the
   tempfile bridge `_delegate_via_legacy`).

### Slim the orchestrator (30 min)

7. Rewrite `digest_dataset` as ~50 LOC:
   ```python
   def digest_dataset(dataset_id, input_dir, output_dir):
       # skip-check, path setup
       source = detect_source(dataset_dir)
       digested_at = datetime.now(timezone.utc).isoformat()
       _repair_participants_tsv_ids(dataset_dir)
       source_adapter = get_source_adapter(source, dataset_id, dataset_dir)
       enumerator = get_record_enumerator(
           dataset_id, dataset_dir, source, source_adapter, digested_at,
       )
       result = enumerator.enumerate()
       if not result.records:
           # surface a useful status — error or empty depending on
           # whether result.errors is populated
           ...
       return write_dataset_outputs(output_dir / dataset_id, result, ...)
   ```
8. Delete `digest_from_manifest` entirely.

### Verify (30 min)

9. Run the snapshot comparison test against the fixture from step 2.
   Both JSON outputs must be byte-identical to the snapshot.
10. Run the full 223-test suite. Adjust the LOC canary baselines in
    `tests/test_digest_helpers.py` — `digest_dataset` drops to ~50,
    `digest_from_manifest` to 0 (deleted).
11. Run `mutmut` on the new `_enumerate_via_bids` to confirm the
    extraction didn't introduce test gaps.

### Total time

~3 hours of focused work; can't be done in 200-token bursts because
mistakes compound and the LOC canary doesn't catch byte-level drift.

## What Stages 1+2 already delivered

Even without Stage 3, the value on `record-enumerator-merge`:

- **Named Seam.** `RecordEnumerator` is a tested, documented Module.
  6 of 7 Sources (per ADR 0001) have a path through it.
- **Shared write helper.** `write_dataset_outputs` owns the 4-file
  output contract. Future schema changes land in one place.
- **Drift between paths closed.** Both legacy functions emit the
  same summary fields now. The audit-1 F1 fix-and-miss pattern
  can't recur.
- **Adapter pattern proven in production-adjacent code.** Stage 2D
  bridge runs the legacy via tempfile + reads back — slower but
  end-to-end correct.

Stages 1+2 are coherent and ship-ready as a PR.

## Decision matrix for the next session

| If you have... | Then do... |
|---|---|
| 3-4 hours of focused time + a fixture dataset | Full Stage 3 (Option D) |
| 30 min, want incremental progress | Add the snapshot tests (pre-flight only) |
| no time but want to land Stages 1+2 | merge `record-enumerator-merge` as-is — Stage 3 follow-up |

The branch is a coherent stopping point. The merge is half-done in a
*structurally correct* way: the Seam exists, both bodies go through
the shared write helper, the Adapter classes are wired and tested via
the bridge. Stage 3 finishes the merge but isn't required for value.
