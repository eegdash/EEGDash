# Robustness programme — session 5 continuation (2026-05-22)

PROGRESS-7 closed three of four open points from PROGRESS-6 and
introduced ROADMAP.md with explicit P0/P1/P2/P3 priorities.
This document catalogues the first P0 item closed.

## ROADMAP.md introduced

Anchor document at `ROBUSTNESS/ROADMAP.md`. Four priority tiers:

- **P0** — Operational unblockers (real-world drivers)
- **P1** — Operational visibility
- **P2** — Architectural moves (no immediate driver)
- **P3** — Hygiene / speculative

Plus working agreements (snapshot gate, no `--no-verify`, commit
cadence, ADRs for deferrals). Future sessions consult ROADMAP.md
to pick the next-leverage work.

## P0.1 — Cascade-with-provenance — DONE

The 4-step technical-metadata cascade in `_extract_technical_metadata`
(introduced in PROGRESS-7) now emits a provenance dict alongside the
4 metadata values. Each Record carries `_metadata_provenance` —
a mapping of `{field_name -> source_name | None}`.

### Five provenance sources

| Constant | Source | Used for |
|---|---|---|
| `_PROV_MNE_BIDS` | `EEGBIDSDataset` attribute getters + `channel_labels` | step 1 |
| `_PROV_MODALITY_SIDECAR` | raw modality JSON sidecar walk | step 2 |
| `_PROV_CHANNELS_TSV` | raw `channels.tsv` walk | step 3 |
| `_PROV_BINARY_PARSER` | direct format parsers (.vhdr / .snirf / .mefd / .edf / .bdf / .set) | step 4 |
| `_PROV_MNE_FALLBACK` | MNE for VHDR n_times + FIF metadata | step 4 |

### Implementation choices

- **First-writer-wins** semantics mirror the cascade's existing
  `X = X or new_X` pattern. The second step that *could* fill a field
  finds it already set and never overwrites.
- **`_clamp_metadata_extremes` clears provenance on reject.** When
  sfreq <= 0 or nchans <= 0 the value is replaced with None; the
  provenance entry is also cleared to maintain the invariant
  *"provenance is None iff value is None"* for the final Record.
- **Warned-not-rejected values keep provenance.** sfreq > 1MHz logs
  a warning but the value stays; so does the provenance.
- **Manifest path Records bypass the cascade.** They have no
  `_metadata_provenance` field. Pinned by test.

### Snapshot update

`tests/fixtures/digest_snapshots/outputs/ds_snapshot_vhdr/ds_snapshot_vhdr_records.json`
regenerated. The one Record now carries:

```json
"_metadata_provenance": {
    "sampling_frequency": "binary_parser",
    "nchans": "binary_parser",
    "ntimes": "mne_fallback",
    "ch_names": "binary_parser"
}
```

The VHDR fixture has no sidecar JSONs, so the cascade falls through
to step 4 (binary parser) for sfreq / nchans / ch_names. The .vhdr
header doesn't include n_times, so MNE reads it from the binary
companion. This is **exactly** the expected behaviour for that
fixture; the snapshot test pins it.

### 12 new unit tests in `tests/test_metadata_provenance.py`

- `_empty_provenance` has all 4 fields, all None
- `_stamp_provenance` records source on fill
- `_stamp_provenance` honours first-writer-wins
- `_stamp_provenance` skips when value unchanged
- `_stamp_provenance` skips when value stays None
- `_clamp_metadata_extremes` clears sfreq provenance when sfreq rejected
- `_clamp_metadata_extremes` clears nchans provenance when nchans rejected
- `_clamp_metadata_extremes` keeps provenance for suspicious-but-kept values
- `_clamp_metadata_extremes` provenance kwarg is optional (back-compat)
- BIDS snapshot Record has `_metadata_provenance` with valid sources
- BIDS snapshot provenance matches expected cascade-fallthrough
- Manifest snapshot Records have NO `_metadata_provenance`

### Driver met

The original support-diagnosis problem: when a Record has wrong
`sampling_frequency`, a runtime / support engineer can now read the
record's `_metadata_provenance.sampling_frequency` and immediately
know which extractor produced the bad value. Workflow:

- `"mne_bids"` → file a bug against the BIDS sidecar JSON
- `"modality_sidecar"` → check the dataset's `*_eeg.json` content
- `"channels_tsv"` → check the dataset's `*_channels.tsv` content
- `"binary_parser"` → file a bug against our `_set/_vhdr/_snirf/_mef3`
  parsers
- `"mne_fallback"` → file a bug against MNE-Python

### What enables next

P1.1 (DigestTelemetry) now has its payload: each per-Record event
carries the provenance dict. The telemetry Module emits structured
events; the provenance answers the per-event "where did this come
from?" question.

## Numbers

| Metric | Start of P0.1 | End |
|---|---:|---:|
| Tests passing | 235 | **247** (+12) |
| Snapshot assertions | 12 | 12 (BIDS regenerated, manifest unchanged) |
| Provenance helpers | 0 | 2 (`_empty_provenance`, `_stamp_provenance`) |
| Provenance sources | 0 | 5 |
| `_extract_technical_metadata` LOC | 140 | **265** (grew; provenance tracking adds bookkeeping) |
| `extract_record` LOC | 189 | 189 (unchanged — one new line + minor edits) |

The cascade helper grew to 265 LOC because tracking provenance at
each step requires capturing before/after values. This is acceptable
— the function reads as **N steps, each annotated with its source
name** rather than the previous "4 steps with no audit trail." A
follow-up could split the cascade into 4 per-step sub-helpers (each
returning value + provenance) but the current shape reads top-down
clearly enough.

## What's still open after this session

P0.2 (mutmut nightly CI promotion), P1.1 (DigestTelemetry — now
unblocked), P1.2 (per-helper unit tests for the 17 PROGRESS-7
helpers), P2 architectural moves, P3 hygiene. See ROADMAP.md for
the prioritized list.
