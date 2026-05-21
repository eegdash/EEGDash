# Pipeline contract — the 5-stage ingest

ROADMAP P2.3 partial close. The 5 numbered stages share an
implicit JSON contract between them; this document makes the contract
explicit. The full Pipeline-Orchestrator refactor (a `PipelineStage`
Protocol + a generic orchestrator) is deferred per
[ADR 0002](ADRs/0002-pipeline-orchestration-deferred.md) — there's no
driver yet.

## Stage diagram

```
                                  ┌─────────────────────────┐
                                  │ <source>-datasets.json  │  (catalogue from 1_fetch)
                                  │ <dataset_id>/           │   each dataset's working dir
                                  │   manifest.json         │   listing of files + sizes
                                  └─────────────────────────┘
                                            │
1_fetch_sources/<source>.py  ──────────────►│
   (e.g., openneuro.py, nemar.py)           │  WRITES manifest.json into
                                            │  each dataset's directory.
                                            │
                                            ▼
                                  ┌─────────────────────────┐
                                  │ <dataset_id>/           │  (manifest + cloned tree)
                                  │   manifest.json         │   normalised + ZIP peek
                                  │   <BIDS tree>           │   git-annex / git mirror
                                  └─────────────────────────┘
                                            │
2_clone.py  ────────────────────────────────►│
   per-source fetch + manifest normalisation │  Adds storage_base, _zip_contents,
                                              │  download_url where missing.
                                              ▼
                                  ┌─────────────────────────┐
                                  │ <dataset_id>/           │
                                  │   <dataset_id>_dataset.json
                                  │   <dataset_id>_records.json
                                  │   <dataset_id>_montages.json
                                  │   <dataset_id>_summary.json
                                  └─────────────────────────┘
                                            │
3_digest.py  ───────────────────────────────►│
   BIDS walk OR manifest walk → 4 JSON files │  Per ROADMAP P0.1, records now carry
                                              │  _metadata_provenance.
                                              ▼
                                  ┌─────────────────────────┐
                                  │ Validation report       │
                                  │   per-record schema errors
                                  │   per-dataset stats
                                  └─────────────────────────┘
                                            │
4_validate_output.py  ──────────────────────►│
   Schema validation against eegdash.schemas │
                                              ▼
                                  ┌─────────────────────────┐
                                  │ MongoDB                 │
                                  │   datasets, records,    │
                                  │   montages collections  │
                                  └─────────────────────────┘
                                            │
5_inject.py  ───────────────────────────────►│
   Upsert into MongoDB (idempotent)         │
```

## Per-stage JSON contracts

### Stage 1 — `1_fetch_sources/<source>.py`

**Input**: source-specific (API credentials, dataset list URL, etc.)

**Output (per dataset)**:
```
<input_dir>/<dataset_id>/manifest.json
```

`manifest.json` schema (informal, accreted over time — fields are
optional unless flagged required):

```jsonc
{
    "dataset_id": "ds002893",           // required
    "source": "openneuro",              // required
    "name": "...",
    "license": "CC0",
    "authors": ["..."],
    "demographics": {
        "subjects_count": 50,
        "ages": [25, 30, ...]
    },
    "bids_subject_count": 50,
    "modalities": ["eeg"],
    "recording_modality": "eeg",
    "tasks": ["rest"],
    "sessions": ["01"],
    "external_links": {
        "source_url": "https://...",
        "osf_url": "https://..."
    },
    "identifiers": {"doi": "..."},
    "dataset_doi": "...",
    "storage_base": "s3://.../dataset",   // ROADMAP P2.4 candidate
    "files": [                            // required (may be empty)
        {
            "path": "sub-01/eeg/sub-01_eeg.edf",
            "size": 1048576,
            "download_url": "https://...",
            "_zip_contents": [...]        // only when peeked
        }
    ],
    "zip_contents": [...]                 // separate from per-file _zip_contents
}
```

### Stage 2 — `2_clone.py`

**Input**: `manifest.json` (from Stage 1) + per-source fetch state

**Output (per dataset)**:
- Updates `manifest.json` in place: adds `storage_base`,
  normalises `_zip_contents`, fills `download_url` where missing.
- For git-clonable sources (OpenNeuro, NEMAR, GIN): clones the BIDS
  tree under `<input_dir>/<dataset_id>/`.

### Stage 3 — `3_digest.py`

**Input**: `<input_dir>/<dataset_id>/` containing either
- a BIDS tree (preferred), OR
- a `manifest.json` describing the files

**Output (4 JSON files per dataset)**:
```
<output_dir>/<dataset_id>/<dataset_id>_dataset.json
<output_dir>/<dataset_id>/<dataset_id>_records.json
<output_dir>/<dataset_id>/<dataset_id>_montages.json
<output_dir>/<dataset_id>/<dataset_id>_summary.json
```

Shape: see `eegdash.schemas.create_dataset` / `create_record` for the
authoritative TypedDicts. Stamped fields:

- **Dataset**: `ingestion_fingerprint`, `digested_at`, `source`,
  `recording_modality`, demographics aggregates, `storage` info.
- **Records**: `dataset`, `bids_relpath`, `storage`,
  `recording_modality`, technical metadata (`sampling_frequency`,
  `nchans`, `ntimes`, `ch_names`), `_metadata_provenance` (P0.1),
  `_data_integrity_issues` (per-file).
- **Montages**: deduplicated by hash; `representative_dataset` +
  `representative_subject` stamped once per hash.
- **Summary**: `status`, counts, file paths.

Behaviour split: the BIDS path goes through
`BIDSFilesystemEnumerator → _enumerate_via_bids`; the manifest-only
path goes through `ManifestEnumerator → _enumerate_via_manifest`.
Both write via `write_dataset_outputs`.

### Stage 4 — `4_validate_output.py`

**Input**: the 4 JSON files from Stage 3.

**Output**: validation report (passed/failed datasets, per-record
schema errors). Doesn't modify the JSON files.

### Stage 5 — `5_inject.py`

**Input**: the 4 JSON files from Stage 3.

**Output**: MongoDB writes into `datasets`, `records`, `montages`
collections. Idempotent — re-running upserts. The runtime
`correct_storage_inplace` (in `eegdash.dataset._source_inference`)
self-heals mis-routed records on read.

## Telemetry stream (P1.1)

Independent of the per-stage outputs: when
`$EEGDASH_TELEMETRY_PATH` is set, Stage 3 writes one NDJSON event
per dataset / record to that file. Consumers (Stage 4 / 5 / nightly
dashboards) can join on `dataset_id` + `record_id`.

## Working with the contract

When adding a new stage between two existing stages (e.g., an
`anonymise` step between 3 and 4):

1. **Read** this document to find the upstream output shape.
2. **Read** the downstream consumer's expected input shape.
3. **Write** a new numbered script (e.g., `3.5_anonymise.py`) that
   reads from the upstream stage's output directory and writes to a
   parallel sibling directory with the same JSON shapes.
4. **Wire** it into the 9 CI workflows (each currently hard-codes
   the stage order).
5. **Update** this document to reference the new stage.

Step 4 is the brittle one — it's the driver for [ADR 0002](ADRs/0002-pipeline-orchestration-deferred.md)
to be revisited. If you find yourself touching > 3 CI workflows for
a new stage, consider revisiting the orchestrator design.
