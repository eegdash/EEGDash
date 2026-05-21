# Robustness programme — session 5 continuation (2026-05-22)

PROGRESS-8 closed ROADMAP P0.1 (cascade-with-provenance). This
document closes the remaining P0/P1 items.

## ROADMAP status — end of this round

| Tier | Item | Status |
|---|---|---|
| **P0** | P0.1 cascade-with-provenance | ✅ (PROGRESS-8) |
| **P0** | P0.2 mutmut nightly CI | ✅ (this round) |
| **P1** | P1.1 DigestTelemetry | ✅ (this round) |
| **P1** | P1.2 per-helper unit tests | ✅ (this round) |
| **P2** | P2.1 Modality layout Seam | ⏳ open (no driver yet) |
| **P2** | P2.2 Format metadata parser Seam | ⏳ open (no driver yet) |
| **P2** | P2.3 Pipeline orchestration | ⏳ open (no driver yet) |
| **P2** | P2.4 Source-listing Seam | ⏸️ deferred per ADR 0001 |
| **P3** | P3.1 `_read_participants_demographics` split | ⏳ open (1h, low value) |
| **P3** | P3.2 content-addressed Records | ⏳ open (speculative) |
| **P3** | P3.3 schema migration framework | ⏳ open (speculative) |

**Every P0 + P1 item closed.** P2 items have no immediate driver
("revisit when a new modality lands" / "revisit when a new stage is
planned"). P3 items are speculative.

## P0.2 — mutmut nightly CI

New workflow `.github/workflows/ingestions-mutmut-nightly.yml`:

- **Schedule**: 02:17 UTC daily (off the :00/:30 marks per CronCreate
  guidance). `workflow_dispatch` lets the user override the target
  file for ad-hoc runs.
- **Matrix**: 4 parsers (`_vhdr / _set / _snirf / _mef3`), each gets
  its own job + cache + ratio gate.
- **Pinned mutmut 2.x** (3.x has the config-parser bug from
  `findings-phase-4.md`).
- **Runner swap**: `sed` updates `paths_to_mutate` + `runner` in
  `pyproject.toml` per parser. Convention `_vhdr_parser.py →
  tests/test_vhdr_parser.py` keeps the swap mechanical.
- **Parallelism**: `--simultaneous-mutants 4` (5-8 min per parser
  vs 25 min single-threaded).
- **Ratio gate**: parses `mutmut results` output, fails the job if
  kill ratio < 60% (the floor from `findings-phase-4.md`).
- **Artefact upload**: `.mutmut-cache` retained for 30 days; lets
  follow-up `mutmut show <id>` runs locally.

Mechanism shipped; first nightly run produces the baselines for
`_set_parser.py / _snirf_parser.py / _mef3_parser.py` (session 4
only had the partial `_vhdr_parser.py` baseline at 69.5%).

## P1.1 — DigestTelemetry

New `digest_telemetry.py` Module. Operational visibility that's been
open since the session-4 big-picture audit. Now any 1000-dataset
ingest run produces a queryable event stream.

### Module shape

```
TelemetryEvent(event_kind, dataset_id, payload, record_id?, timestamp)
TelemetryEmitter (ABC)
  NullEmitter  — default, no-op
  NDJSONEmitter — append one JSON line per event
get_emitter() / configure_telemetry() / reset_telemetry()
auto_configure_from_env()  # via $EEGDASH_TELEMETRY_PATH
```

### Event kinds

- ``dataset_started`` — once per ``digest_dataset`` call
  (payload: source, dataset_dir)
- ``dataset_finished`` — once per ``digest_dataset`` call
  (payload: status, record_count, error_count, digest_method,
  integrity_issues_count, montage_count)
- ``record_built`` — one per successful ``extract_record`` (payload:
  bids_relpath, datatype, **metadata_provenance from P0.1**, sfreq,
  nchans, ntimes)
- ``record_failed`` — one per ``extract_record`` exception or split-
  FIF skip (payload: bids_file, error)

### Default behaviour: OFF

`NullEmitter` is the process-global default. No file is written, no
events emitted, no memory overhead. The pre-P1.1 behaviour is exactly
preserved when telemetry is disabled.

### Opt-in

Set `$EEGDASH_TELEMETRY_PATH=/path/to/events.ndjson` before the ingest
run. `auto_configure_from_env()` (called at module import) installs an
`NDJSONEmitter`. The events file accumulates across runs (append
mode), so re-running picks up where the prior run left off.

### Queries enabled

The provenance from P0.1 is the payload of `record_built`, so:

- *"Which datasets had Records with `sampling_frequency = None`?"* →
  filter on `payload.sampling_frequency is null`
- *"What fraction of Records get nchans from binary_parser vs
  channels.tsv?"* → group by
  `payload.metadata_provenance.nchans`
- *"Which datasets had a high record_failed rate?"* → count by
  `event_kind == "record_failed"`, group by `dataset_id`

All from a single ingest run — no re-execution required.

## P1.2 — Per-helper unit tests

`tests/test_digest_extraction_helpers.py` — **43 new tests** for the
17 helpers landed in PROGRESS-6 + PROGRESS-7. Coverage:

- **§1 BIDS-fs metadata readers** (16 tests):
  - ``_read_bids_readme`` (5)
  - ``_read_participants_demographics`` (7)
  - ``_build_global_storage_info`` (4)
- **§2 BIDS-fs dep_keys** (5 tests):
  - ``_build_dep_keys`` covers alongside-recording, BIDS-inheritance
    session-base, `.fdt`-companion, split-FIF detection, broken-
    symlink integrity flag.
- **§3 Manifest orchestration** (14 tests):
  - ``_determine_manifest_storage_base`` (7)
  - ``_collect_bids_entities_from_paths`` (4)
  - ``_is_bids_data_zip`` (3)
- **§4 Manifest builders** (8 tests):
  - ``_build_regular_manifest_record`` (3)
  - ``_build_ctf_ds_records`` (2)
  - ``_build_zip_extracted_records`` (1)
  - ``_build_subject_zip_record`` (2)

Previously the helpers were covered transitively by snapshot tests —
failures pointed at orchestrators. Now refactor failures point at the
helper.

**Gotcha caught**: case-insensitive filesystems (macOS APFS) collapse
``README`` and ``readme`` into one file, so the priority test uses
different *extensions* (``README`` vs ``README.md``) instead of
different *cases*. Documented inline.

## Numbers

| Metric | Pre-P0.1 | End of this round |
|---|---:|---:|
| Tests passing | 235 | **304** (+69) |
| Snapshot assertions | 12 | 12 (unchanged) |
| ROADMAP P0 items | 0/2 | **2/2** |
| ROADMAP P1 items | 0/2 | **2/2** |
| Big-picture audit items closed | 1/5 | **3/5** (RecordEnumerator + cascade-with-provenance + telemetry) |
| CI workflows | 12 | 13 (+ mutmut nightly) |

## Commits added this round

```
95c3bfcae  P1.1 DigestTelemetry — structured per-Record event stream
5052400bf  P0.2 mutmut nightly workflow for all 4 parsers
b8521687d  P1.2 — 43 per-helper unit tests for PROGRESS-7 decomposition
4ddbf7bcb  PROGRESS-8 (P0.1 close)
7e469469b  P0.1 cascade-with-provenance
f2a53ae15  ROADMAP — explicit priorities
```

Six commits on top of where P0.1 started. 28 total on
`record-enumerator-merge` branch.

## What's still open (in priority order)

The roadmap has been consumed down to P2 and P3:

1. **P2.1 Modality layout Seam** — 4-modality dispatch in `_montage.py`
   (1077 LOC). Worth doing when EMG stub completion is on the roadmap
   or a new modality is added.
2. **P2.2 Format metadata parser Seam** — 6 parsers (`_set/_vhdr/
   _snirf/_mef3` + 2 inlined) share an implicit contract. Smaller
   drift than P2.1; pick up after P2.1.
3. **P2.3 Pipeline orchestration** — explicit `PipelineStage` Protocol
   across the 5 numbered stages. Only worth doing if a new stage
   (e.g., `4.5_anonymise.py`) is planned.
4. **P3.1 `_read_participants_demographics` split** — 102 LOC, just
   above the 100 ceiling. Cohesion is high; only worth splitting if
   column-extraction logic grows.
5. **P3.2 content-addressed Records** — separate `digest_hash`
   (content) from `digested_at` (metadata). Hygiene; no specific
   driver.
6. **P3.3 schema migration framework** — speculative until the next
   Record-field change.

## Branch position

```
maturate-code
  └── ingestion-phase4-and-8-deeper
        └── record-enumerator-merge   ← HEAD (28 commits)
              Stages 1+2 (Seam + write helper)
              Stage 3 (orchestrator collapse)
              Open-points round (manifest snapshot + 17 helpers)
              Cascade-with-provenance (P0.1)
              Mutmut nightly CI (P0.2)
              Per-helper unit tests (P1.2)
              DigestTelemetry (P1.1)
```

Ready as a PR. The P2/P3 work is properly deferred — each has no
immediate operational driver and ADR-style documentation in
`ROADMAP.md` to prevent re-litigation.
