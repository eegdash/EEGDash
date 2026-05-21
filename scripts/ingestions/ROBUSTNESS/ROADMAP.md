# Roadmap — what comes next, prioritized

Anchor document. Updated when work lands; consult before starting a
new session. Every item has: leverage rationale, expected effort,
prerequisites, and a clear definition-of-done.

The ordering is **priority by leverage / urgency**, not arbitrary.

---

## P0 — Operational unblockers

These have real-world drivers (support tickets, security gates,
known bugs) and the next session should start here.

### P0.1 — Cascade-with-provenance ⭐ NEXT

**Driver**: Today when a Record ends up with the wrong
`sampling_frequency`, support cannot tell whether the BIDS sidecar
was wrong, the binary header reader misread the file, or
`mne_bids` returned something unexpected. The 4-step recovery
cascade in `_extract_technical_metadata` is correct but invisible.

**Outcome**: each Record gains a `_metadata_provenance` field that
maps each technical-metadata key (`sampling_frequency`, `nchans`,
`ntimes`, `ch_names`) to the cascade step that produced it. Five
possible values per field: `"mne_bids"`, `"modality_sidecar"`,
`"channels_tsv"`, `"binary_parser"`, `"mne_fallback"`, or `null`.

**Why this first**:
- Real operational driver (support diagnosis is hard today)
- Plugs into `_extract_technical_metadata` (140 LOC) which is the
  helper extracted in PROGRESS-7
- Enables P1 (DigestTelemetry — downstream consumer of provenance)
- Bounded scope; no new modules

**Estimated effort**: 4-6 hours.

**Definition of done**:
- `_extract_technical_metadata` returns provenance dict alongside
  values
- `extract_record` stamps provenance onto the Record as
  `_metadata_provenance`
- Snapshot tests updated (intentional change)
- Unit tests pin provenance for each of the 5 sources
- PROGRESS-8.md catalogues the work

### P0.2 — Mutmut nightly CI promotion

**Driver**: 137 mutants in `_vhdr_parser.py:96-232` are still
untimed-out from the session-4 baseline. The cycle time is also
unmeasured (cached re-runs should be < 2 min).

**Outcome**: `mutmut run --simultaneous-mutants 4` lands as a nightly
CI job. Full kill-ratio established for `_vhdr_parser.py`. Same
treatment for `_set_parser.py`, `_snirf_parser.py`, `_mef3_parser.py`
in sequence.

**Estimated effort**: 2 hours (script + CI config + first nightly).

**Definition of done**: nightly run posts a report; the 4 real-gap
mutants documented in `findings-phase-4.md` either killed or
explicitly accepted as won't-fix in an ADR.

---

## P1 — Operational visibility

Builds on P0.1. Enables debugging-at-scale.

### P1.1 — DigestTelemetry

**Driver**: For a 1000-dataset CI run, the only forensic tool is
grep through stdout. Questions like "Which datasets had Records
with `sampling_frequency = None`?" require re-running.

**Outcome**: New Module that emits structured per-Record events.
Shape: `(event_kind, dataset_id, record_id, payload)`. Backend:
NDJSON file or per-run SQLite, plug-replaceable.

**Prerequisite**: P0.1 (provenance is the payload).

**Estimated effort**: 6-8 hours.

**Definition of done**: a 5-dataset ingest run produces a queryable
artifact; sample queries answer the operational questions above.

### P1.2 — Per-helper unit tests for the 17 new helpers

**Driver**: The decomposition helpers are covered transitively by
the snapshot tests. When a snapshot fails, the failure points at
`extract_record` but the bug is in `_clamp_metadata_extremes`.

**Outcome**: direct unit tests for each helper, with edge-case
fixtures (malformed participants.tsv, BIDS inheritance with no
matching session_base, FIF without continuations, etc).

**Estimated effort**: 4-6 hours.

**Definition of done**: each helper has at least 2 direct tests
(happy path + at least one edge case); refactor failures point at
the offending helper, not the orchestrator.

---

## P2 — Architectural moves (no immediate driver)

These have leverage but need a real driver to commit to.

### P2.1 — Modality layout Seam

**Driver**: 4 modality extractors in `_montage.py` (EEG/iEEG/MEG/fNIRS)
+ a stubbed EMG. Each is bespoke; adding a new modality is a fork
inside a 1077-LOC file. The Seam is implicit.

**Real driver**: EMG stub completion + future modalities.

**Estimated effort**: 6-8 hours.

### P2.2 — Format metadata parser Seam

**Driver**: 6 parsers (`_set`, `_vhdr`, `_snirf`, `_mef3` + 2
inlined). Implicit shared contract; smaller drift than P2.1.

**Real driver**: lower urgency. Pick up after P2.1.

**Estimated effort**: 4 hours.

### P2.3 — Pipeline orchestration

**Driver**: 5 numbered stages share an implicit JSON contract;
9 CI workflows duplicate the stage order.

**Real driver**: planned new stage (e.g., `4.5_anonymise.py`).
Until then: speculative.

**Estimated effort**: 8-12 hours.

### P2.4 — Source-listing Seam — DEFERRED per ADR 0001

Revisit only when ≥ 2 Sources are exercised in production CI.
Currently 1 (OpenNeuro + NEMAR both via `list_git_files`).

---

## P3 — Hygiene / speculative

Low-leverage cleanups; tackle opportunistically.

### P3.1 — `_read_participants_demographics` split (102 → 4 × ~25 LOC)

102 LOC, just above the 100-ceiling. Could split into
`_count_subjects`, `_extract_ages`, `_extract_sex_distribution`,
`_extract_handedness_distribution`. Cohesion is high; only worth
splitting if column-extraction logic grows.

**Estimated effort**: 1 hour.

### P3.2 — Content-addressed Records / idempotency

Separate `digest_hash` (content) from `digested_at` (metadata).
Enables byte-comparing two digest runs of the same dataset.

**Driver**: today there's no specific complaint; would benefit
characterisation tests. Hygiene.

### P3.3 — Schema migration framework

A Migrations Module with one rule per past schema change.
Only precedent so far: `correct_storage_inplace` in
`_source_inference.py`. Speculative until the next field change.

---

## Mega-function LOC ceilings (current state)

The Phase 8 style-guide ceiling is 80 LOC per function. After
PROGRESS-7:

| Function | Current | Status |
|---|---:|---|
| `digest_dataset` | 110 | ⚠️ above ceiling |
| `digest_from_manifest` | 69 | ✅ |
| `extract_record` | 189 | ⚠️ above ceiling |
| `extract_dataset_metadata` | 205 | ⚠️ above ceiling |
| `_enumerate_via_bids` | 109 | ⚠️ marginally above |
| `_enumerate_via_manifest` | 222 | ⚠️ above ceiling |
| `_extract_technical_metadata` | 140 | ⚠️ above ceiling (cohesive cascade) |
| `_read_participants_demographics` | 102 | ⚠️ marginally above (P3.1) |
| `_build_dep_keys` | 98 | ✅ |
| **All other helpers** | < 100 | ✅ |

The remaining over-ceiling functions are mostly *orchestrators*
that read top-down as narratives. The next round of leverage is
in **observability** (provenance + telemetry) rather than further
LOC reduction.

---

## Working agreements

- **Snapshot tests are the gate**. Both `ds_snapshot_vhdr` (BIDS-fs)
  and `ds_snapshot_manifest` (manifest) must stay byte-identical
  unless the change is *intentional*. When intentional, update the
  snapshot in the same commit + cite the reason in the commit message.
- **No `--no-verify`** unless explicitly requested by the user.
- **Pre-commit hooks** stay enabled; codespell + ruff + nested-function
  checks are part of the contract.
- **Commit cadence**: every helper extraction + every behaviour change
  is its own commit, runnable in isolation against the test suite.
- **ADRs for deferrals**: any time we decide *not* to do something
  for a load-bearing reason, write an ADR (cf. ADR 0001).
- **Progress logs**: `PROGRESS-N.md` at the end of each session.
  PROGRESS-1 through PROGRESS-7 exist; PROGRESS-8 lands with P0.1.
