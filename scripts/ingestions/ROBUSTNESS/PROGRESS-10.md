# Robustness programme — session 5 continuation (2026-05-22)

PROGRESS-9 closed all P0 + P1 items. This round consumes the full P2
tier: P2.1 + P2.2 done with code refactors; P2.3 documented + deferred
via ADR 0002; P2.4 stays deferred via ADR 0001.

## ROADMAP status after this round

| Tier | Item | Status |
|---|---|---|
| **P0.1** | cascade-with-provenance | ✅ (PROGRESS-8) |
| **P0.2** | mutmut nightly CI | ✅ (PROGRESS-9) |
| **P1.1** | DigestTelemetry | ✅ (PROGRESS-9) |
| **P1.2** | per-helper unit tests | ✅ (PROGRESS-9) |
| **P2.1** | modality layout Seam | ✅ (this round) |
| **P2.2** | format-parser Seam | ✅ (this round) |
| **P2.3** | pipeline orchestration | ⏸️ deferred per **ADR 0002** (contract documented) |
| **P2.4** | source-listing Seam | ⏸️ deferred per ADR 0001 |
| **P3.1** | `_read_participants_demographics` split | ⏳ open (1h, low value) |
| **P3.2** | content-addressed Records | ⏳ open (speculative) |
| **P3.3** | schema migration framework | ⏳ open (speculative) |

**P0 + P1 + P2 fully consumed** — every item either closed with code
or documented as deferred with an ADR + revisit triggers. P3 is
deliberately speculative.

## P2.1 — Modality layout Seam (closed)

`_montage.py`'s 4 TSV-based extractors (EEG, iEEG, EMG, fNIRS) were
thin wrappers over `_extract_tsv_layout` with different kwargs.
Replaced by:

- `_ModalityConfig` frozen dataclass (modality, tsv_pattern, extras,
  min_sensors, coord_suffix, template_fallback)
- `_TSV_MODALITY_CONFIGS` dict — 4 named configs
- `_extract_layout_for_config(data_file, bids_root, config)` — one
  helper that runs the TSV pipeline + the EEG-only template fallback
- `extract_layout(datatype)` dispatches via the configs

MEG stays special-cased (~190 LOC of FIF header streaming doesn't
fit the TSV pattern). Adding a new TSV-based modality is now one
config entry, not a function + dispatch entry.

The 4 redundant wrapper functions (`extract_eeg_layout`,
`extract_ieeg_layout`, `extract_emg_layout`, `extract_fnirs_layout`)
deleted — no external callers per `grep -r`.

## P2.2 — Format-parser Seam (closed)

The 6 binary-header parsers shared an implicit contract before this
round (same `path → dict | None` signature, same field semantics,
documented only in individual docstrings). Now the Seam is named:

- **`FormatParserResult`** TypedDict (`total=False`) — documents the
  5 possible keys: `sampling_frequency`, `nchans`, `ch_names`,
  `n_times`, `n_samples` (alias).
- **`FormatParser`** Protocol — types the callable signature.
- **`_format_parser_registry.py`** Module — owns the extension →
  parser registry. `get_parser_for_extension(ext)` is the factory.
- `_parse_edf_with_mne` **moved** from `3_digest.py` into the
  registry (was a 60-LOC inline function; now lives with the other
  parsers it conceptually belongs to).

FIF stays special-cased in `3_digest.py` because its parser returns
`(dict, is_split_bool)` — the bool feeds the split-FIF integrity
check downstream. Documented in the registry's header.

`3_digest.py:_extract_technical_metadata` now calls
`get_parser_for_extension(ext)` instead of having an inline 8-line
parser-dispatch dict.

7 new unit tests in `tests/test_format_parser_registry.py`: registry
coverage (6 extensions), unknown-extension returns None, EDF/BDF
share parser, missing-file + broken-symlink return None,
`FormatParserResult` accepts all 5 keys.

## P2.3 — Pipeline orchestration (deferred)

The full orchestrator (`PipelineStage` Protocol + registry) is
deferred per **ADR 0002** — no driver. The 5 numbered stages share
an implicit JSON contract; the 9 CI workflows duplicate the stage
order but are stable.

Partial close instead:

1. **`ROBUSTNESS/PIPELINE-CONTRACT.md`** — names the Seam without
   inventing Adapter classes. Documents each stage's input + output
   JSON shape (Stage 1: `manifest.json`; Stage 2: cloned BIDS tree;
   Stage 3: 4-file output via `write_dataset_outputs`; Stage 4:
   validation report; Stage 5: MongoDB upserts).
2. **ADR 0002** — records the deferral with anti-recommendations:
   no `PipelineStage` Protocol, no orchestrator class, no unified
   CLI, no DAG library. Revisit when ≥ 1 of: new stage planned;
   pipeline grows past 5 stages; partial-reprocess capability
   needed; CVE requires updating all 9 workflows at once.

Same pragmatic pattern as ADR 0001: name the Seam in docs, prevent
re-litigation with anti-recommendations, tie revisit triggers to
real operational drivers.

## Numbers

| Metric | Pre-P2 | After P2 |
|---|---:|---:|
| Tests passing | 304 | **311** |
| ROADMAP P0+P1+P2 items closed | 4/7 | **7/7** (P2.4 deferred via ADR 0001 too) |
| Big-picture audit items closed | 3/5 | **3/5** (P2.3 / P2.4 deferred match audit deferrals) |
| ADRs recorded | 1 | **2** |
| New Modules | — | `_format_parser_registry.py` |
| New docs | — | `PIPELINE-CONTRACT.md`, `ADRs/0002` |

## Commits added this round

```
9a5a0b243  P2.3 — pipeline contract documented + ADR 0002 deferral
6886f47d9  P2.2 — explicit format-parser Seam (registry + Protocol)
248a7c7d5  mark P2.1 done in ROADMAP
d5db5b233  P2.1 — config-driven modality layout dispatch
```

Three code commits (P2.1, P2.2) + one docs commit (P2.3 + ADR 0002).
33 total commits on `record-enumerator-merge` branch.

## What's left

Only P3 items, all deliberately speculative:

1. **P3.1** — `_read_participants_demographics` split (1h, low value).
2. **P3.2** — content-addressed Records (no driver).
3. **P3.3** — schema migration framework (speculative until next
   field change).

Plus the operational follow-ups from earlier rounds:

- First nightly mutmut run produces baselines for `_set_parser.py`,
  `_snirf_parser.py`, `_mef3_parser.py` — should land in nightly
  CI within 24h of merge.
- `findings-phase-4.md` updated with the new kill ratios.
- Optional: real-gap mutants from findings-phase-4 §"Real test gaps
  with no production driver" — case-insensitive regex tests, escaped-
  comma channel name tests.

## Branch position

```
maturate-code
  └── ingestion-phase4-and-8-deeper
        └── record-enumerator-merge   ← HEAD (33 commits)
              All P0 + P1 + P2 items closed or deferred via ADR.
              311 tests pass. 12 snapshot assertions byte-stable.
              2 ADRs (0001, 0002). 4 progress docs (5, 6, 7, 8, 9, 10).
```

The arc reads as: maturity ladder from "duplicated mega-functions" →
"named Seams + explicit contracts + operational telemetry +
deferral-by-ADR for everything without a driver". Ship-ready as a
single PR or split by tier (Stages 1-2 / Stage 3 / open-points / P0
/ P1 / P2).
