# Rollout & performance — cheap-exact metadata resolver

## Performance: no slowness (in fact, faster)

The cheap paths replace full readers, so they are strictly cheaper:

| Path | Before | After | Note |
|------|--------|-------|------|
| VHDR `n_times` | `read_raw_brainvision` (~3.7 ms) | regex + one `stat` (~0.26 ms) | **~14× faster** (`tests/perf/test_cheap_resolver_perf.py`) |
| SNIRF `n_times` | full `time[:]` read | `time.shape[0]` | O(1) metadata, no array load |
| SET `n_times` | `loadmat` whole file | `variable_names=['EEG']` + 50 MB embedded ceiling | no unrelated vars; oversized embedded skipped |
| Sidecar `ntimes` (all formats) | n/a (None) | `round(sfreq × RecordingDuration)` | zero binary access |

Existing absolute ceilings still pass unchanged: `parse_vhdr_metadata` median
~0.25 ms (ceiling 5 ms); snapshot digest e2e ~6.8 ms (ceiling 10 s). The cheap
paths never construct an MNE `Raw` for `n_times` — proven structurally by
`tests/parsers/test_cheap_paths.py::test_vhdr_cascade_skips_mne_fallback` and
`...::test_snirf_h5py_does_not_read_full_time_vector`, not just by timing.

## Correctness gate (the "infer OR check" contract)

Cheap `n_times` is certified **equal to MNE** within tolerance by
`tests/validation/test_mne_equivalence.py` (`@network @slow`). The VHDR file-size
arithmetic declines to emit a value when the `.eeg` size is not an exact multiple
of `nchans × dtype_bytes` (e.g. truncated stubs) rather than guessing — a wrong
value never ships.

## Rollout (full re-digest)

1. Land Phases 1–2 (done) under golden-master protection. Digest output changes
   are additive: every record gains `duration_seconds`, and VHDR records' `ntimes`
   provenance flips `mne_fallback → binary_parser` (value identical).
2. **Bump the testing corpus** — the committed digest snapshots live in the
   SHA-pinned `eegdash-testing-data` repo, not in-tree. Apply the regenerated
   `*_records.json` under `docs/superpowers/corpus-updates/digest_snapshots/outputs/`,
   publish a new corpus tarball, and bump `VERSION`/SHA in `eegdash/testing.py`.
   Until then `tests/digest/test_snapshot.py` is green locally but red in CI
   against the old pin (the expected golden-master/corpus coupling).
3. **Measure → re-digest → measure.** Run `coverage_report.py` on the current
   output (baseline in `coverage/coverage_before.json`), re-run stages 1–3 over all
   NEMAR + OpenNeuro datasets (shallow clone, no signal), then re-run
   `coverage_report.py --json after.json` and diff. Expected lift: `ntimes`
   83.8 % → high-90s (VHDR 76.8 %→~100 % from file-size; long tail from sidecar
   arithmetic), `duration_seconds` 0 % → ≈ `ntimes` coverage.
4. Re-inject via the existing stage-5 create/update/skip logic — newly-filled
   fields change the record fingerprint, so updates flow naturally.

## Remaining gaps (future phases)

- **`.mefd` (MEF3) `ntimes` 3.9 %** — sidecar arithmetic (Phase 1) lifts records
  whose `ieeg.json` has `RecordingDuration`; a header `number_of_samples` reader is
  a further enhancement.
- **New-format header parsers** (CTF `.ds`, KIT `.con/.sqd`, NWB, BTi `.pdf`) and
  enumeration of `.cnt/.cdt/.mff/.bin/.lay` — Phase 3. All already receive
  sidecar-arithmetic `ntimes`/`duration` today; the parsers add header-exactness.
