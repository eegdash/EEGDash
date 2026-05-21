# RecordEnumerator merge — Stage 2 + 3 plan

Stage 1 landed (commit on `record-enumerator-merge`): the
`record_enumerator.py` Module exists, the factory dispatches correctly,
14 unit tests pin every fallback rule. The Adapter `enumerate()`
bodies are stubs.

Stages 2 + 3 wire the Adapters to the existing algorithm bodies and
then collapse the orchestrator. Doing them in **two carefully-bounded
commits** keeps the diff reviewable.

## Stage 2 — wire the Adapters via private helpers

### Goal

Both Adapter `enumerate()` methods produce a real `EnumerationResult`
by delegating to the existing algorithm body in `3_digest.py`, but
**without** doing the JSON writes (the orchestrator owns those in
Stage 3). The old top-level functions still exist and still write JSON
the same way they do today.

### Mechanical refactor

In `3_digest.py`, extract from `digest_dataset` (lines ~2542-2868) and
`digest_from_manifest` (lines ~1893-2540):

1. **Body of `digest_dataset` → `_enumerate_from_bids(dataset_dir, dataset_id, source, source_adapter, digested_at) -> EnumerationResult`**
   - Take the existing body, skip the JSON-write section (lines ~2787-2867)
   - Return `EnumerationResult(dataset_meta=dataset_doc, records=records, errors=errors, montages=montages, digest_method="bids_filesystem")` instead of the summary dict
   - Skip-check + path setup stays in the public function

2. **Body of `digest_from_manifest` → `_enumerate_from_manifest(dataset_dir, dataset_id, source, source_adapter, digested_at) -> EnumerationResult`**
   - Same shape: skip JSON-write (lines ~2502-2538), return result
   - `digest_method="manifest_only"`

3. **New shared helper `_write_dataset_outputs(dataset_output_dir, result: EnumerationResult, dataset_id, source, digested_at) -> dict`**
   - Takes an `EnumerationResult`
   - Writes `_dataset.json`, `_records.json`, `_montages.json`,
     `_summary.json`
   - Returns the summary dict (preserves the current return shape of
     `digest_dataset` and `digest_from_manifest`)
   - The montages write happens unconditionally — `digest_from_manifest`
     used to NOT write it; this is a small behaviour change (an empty
     `_montages.json` now exists for manifest-only datasets too). If
     downstream tooling can't handle that, revisit.

4. **Old public functions become thin wrappers**:

   ```python
   def digest_dataset(dataset_id, input_dir, output_dir):
       # skip-check + path setup + detect_source + source_adapter
       result = _enumerate_from_bids(dataset_dir, ..., source_adapter, ...)
       return _write_dataset_outputs(dataset_output_dir, result, ...)

   def digest_from_manifest(dataset_id, input_dir, output_dir):
       # path setup
       result = _enumerate_from_manifest(dataset_dir, ...)
       return _write_dataset_outputs(dataset_output_dir, result, ...)
   ```

5. **Wire the Enumerator Adapters**:

   ```python
   class BIDSFilesystemEnumerator(RecordEnumerator):
       def enumerate(self) -> EnumerationResult:
           # Import inside method to avoid cycle
           from importlib.util import spec_from_file_location, module_from_spec
           # ... or just `from ._digest_helpers import _enumerate_from_bids`
           # once both have been moved into a clean module
           return _enumerate_from_bids(
               self.dataset_dir, self.dataset_id, self.source,
               self.source_adapter, self.digested_at,
           )
   ```

   Same shape for `ManifestEnumerator.enumerate()`.

### Verification

- All 215 existing tests still pass.
- `_write_dataset_outputs` writes the same JSON shapes as before
  (verified by characterisation test against the existing snapshots).
- The 4 mega-function LOC canary baselines in
  `tests/test_digest_helpers.py` shift downward as the bodies move
  into the new helpers — update the baselines.

### Risks

1. **Behaviour drift** — the JSON-write section reads from variables
   that are still in scope (`records`, `errors`, `montages`,
   `dataset_doc`). Extracting them via a shared helper means passing
   those explicitly. If any field gets renamed or dropped, the JSON
   output drifts. Mitigation: byte-compare the JSON output of a
   fixture dataset before/after.

2. **Montages.json now written for manifest-only** — `digest_from_manifest`
   never wrote `_montages.json` (manifest path produces no montages).
   `_write_dataset_outputs` writes it unconditionally. The file has
   `montage_count: 0, montages: []` for manifest-only datasets. If
   `4_validate_output.py` or `5_inject.py` chokes on that, either
   add a "skip empty montages" branch to the helper or accept the
   change and update those consumers.

3. **Circular import** — `record_enumerator` would need to import the
   private helpers from `3_digest.py`. Either:
   - Move the helpers into a `_digest_helpers.py` module that
     `3_digest.py` and `record_enumerator.py` both import, or
   - Use late imports inside the Adapter methods (uglier but works).

   Prefer option A.

## Stage 3 — collapse the orchestrator

### Goal

`digest_dataset` and `digest_from_manifest` disappear as top-level
functions. The orchestrator becomes a single function (~50 LOC) that
uses the Enumerator factory.

### Mechanical refactor

```python
def digest_dataset(dataset_id, input_dir, output_dir):
    # Skip check
    output_dir_path = output_dir / dataset_id
    if output_dir_path.exists():
        return {"status": "skipped", "dataset_id": dataset_id,
                "reason": "already digested"}
    dataset_dir = input_dir / dataset_id
    if not dataset_dir.exists():
        return {"status": "skipped", "dataset_id": dataset_id,
                "reason": "directory not found"}

    # Detect + setup
    source = detect_source(dataset_dir)
    digested_at = datetime.now(timezone.utc).isoformat()
    _repair_participants_tsv_ids(dataset_dir)
    source_adapter = get_source_adapter(source, dataset_id, dataset_dir)

    # Enumerate via the Adapter factory (handles BIDS vs manifest dispatch)
    try:
        enumerator = get_record_enumerator(
            dataset_id, dataset_dir, source, source_adapter, digested_at,
        )
        result = enumerator.enumerate()
    except (OSError, ValueError, KeyError) as exc:
        return {"status": "error", "dataset_id": dataset_id,
                "error": f"Enumeration failed: {exc}"}

    if not result.records:
        return {"status": "empty", "dataset_id": dataset_id,
                "reason": "no records produced"}

    # Write outputs
    dataset_output_dir = output_dir / dataset_id
    return _write_dataset_outputs(
        dataset_output_dir, result, dataset_id, source, digested_at,
    )
```

`digest_from_manifest` is deleted entirely; its body lives in
`_enumerate_from_manifest` (Stage 2) and is reached via
`ManifestEnumerator.enumerate`.

### Verification

- All 215 tests pass.
- Run the orchestrator against a real CC0 fixture dataset (e.g.
  `tests/fixtures/eeg/ds002893`) and byte-compare the produced JSON
  with the pre-merge output.
- LOC canary: `digest_dataset` drops from 327 → ~50.
  `digest_from_manifest` from 647 → 0 (deleted).
  Update baselines.

### Outcome

- One orchestrator function. One dispatch point. Cross-algorithm bug
  fixes happen in `_write_dataset_outputs` or the per-algorithm
  private helper — no risk of fixing one path and missing the other.
- `3_digest.py` net LOC drops by ~900 (most of the body moves to
  the shared `_write_dataset_outputs` + the private helpers).
- The two algorithms remain separately implemented; the SEAM is
  named and observable.

## What this delivers vs the stage 1 stop point

**Stage 1 (done)**: the Module + factory + tests exist. The orchestrator
in `3_digest.py` doesn't use them yet.

**Stage 2 (planned)**: Enumerator methods work; the old public functions
still exist as thin wrappers; both paths produce identical JSON output
as before.

**Stage 3 (planned)**: the orchestrator is unified; the legacy
top-level functions disappear; `3_digest.py` shrinks by ~900 LOC.

## Total commits expected for the merge

```
6ef02a7e8  feat: RecordEnumerator scaffolding (stage 1 — done)
<next>     refactor: extract _enumerate_from_bids + _enumerate_from_manifest
<next+1>   refactor: shared _write_dataset_outputs; wire Enumerator Adapters
<next+2>   refactor: collapse orchestrator; delete digest_from_manifest
```

Three more commits, each ~150-300 lines of mechanical changes, each
independently verifiable by characterisation tests + LOC canary.

The 5-stage shape was a clean way to describe it but in practice it's
3 medium commits, not 5 small ones. Each is reviewable in isolation.

## Why stop here?

Stage 1 delivered value:
- The Seam is named and documented
- The factory + fallback rules are tested in isolation
- A future session has a clear plan to execute

Stage 2-3 are mechanical and would benefit from:
- A dedicated review pass (loud diff)
- A characterisation-test pre-run on a real fixture dataset to lock
  the byte output before the refactor

Worth doing both, but not in a 200-token-rush. The 215-test suite +
LOC canary + characterisation tests are the safety net; using them
deliberately is the work.
