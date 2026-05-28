# Ingestion pipeline — domain map

`scripts/ingestions/` turns datasets discovered on remote repositories into
validated documents written to the EEGDash API/database. It is a **data-processing
pipeline**, not a rich behavioural domain — so we use *DDD-lite* (shared vocabulary
+ explicit stage boundaries) rather than full Domain-Driven Design. See
[`docs/adr/0001-evolutionary-typed-pipeline.md`](docs/adr/0001-evolutionary-typed-pipeline.md).

The canonical vocabulary lives in [`models.py`](models.py) (a glossary + re-export
surface; the entity contracts themselves are owned by `eegdash.schemas`).

## Stages

Each stage has one input artifact, one output artifact, and owns exactly one I/O
boundary. Stages are wired through `record_enumerator` (digest) and the numbered
entry scripts; they never reach across each other's boundaries.

| # | Stage | Entry script | Input → Output artifact | I/O boundary it owns |
|---|-------|--------------|-------------------------|----------------------|
| 1 | Fetch sources | `1_fetch_sources/*.py` | (query) → **SourceListing** / `manifest.json` | remote repo HTTP (Zenodo/OSF/OpenNeuro/Figshare/SciDB/DataRN/NEMAR) |
| 2 | Clone | `2_clone.py` | **CloneManifest** → **LocalDataset** (on disk) | filesystem / git-annex / S3 |
| 3 | Digest | `3_digest.py` | **LocalDataset** → **DigestBundle** (`*_dataset.json` + `*_records.json` + `*_montages.json`) | filesystem reads + BIDS/format parsing |
| 4 | Validate | `4_validate_output.py` | **DigestBundle** → **ValidationReport** | filesystem reads only (pure) |
| 5 | Inject | `5_inject.py` | **DigestBundle** + remote state → **InjectionPlan** → API writes | EEGDash API / MongoDB |

## Where each artifact lives today

- **DigestBundle** → `record_enumerator.EnumerationResult` (canonical name: `DigestBundle`)
- **ValidationReport** → `_validate.ValidationResult` (canonical name: `ValidationReport`)
- **InjectionPlan** → `_inject_plan.InjectionPlan`
- **entities** (`Dataset`, `Record`, `Montage`) + `create_dataset`/`create_record` → `eegdash.schemas`

## Module layout (intentionally flat — see ADR 0002)

```
3_digest.py          thin stage-3 orchestrator (was 2804 lines, now ~520)
  _digest_runner     multiprocessing watchdog (subprocess isolation + timeouts)
  record_enumerator  dispatches BIDS-filesystem vs manifest digest (the Seam)
  _bids_digest       BIDS-filesystem record synthesis
  _manifest_digest   manifest (API-only) record synthesis
  _record_extractor  per-record BIDS extraction
  _dataset_metadata  dataset-level metadata extraction
  _bids_path         pure BIDS path-parsing / file classification
  _source_id         dataset-id → source inference
parsers              _vhdr_parser, _set_parser, _snirf_parser, _mef3_parser, _format_parser_registry, _montage
foundations          _http, _parser_utils, _file_utils, _fingerprint, _constants, digest_telemetry
configs              _digest_config, _clone_config, _inject_config, _validate_config (pydantic-settings)
```

## Testing safety net

- **Golden masters** freeze current behaviour before any refactor:
  `tests/digest/test_snapshot.py` (byte-level digest output, incl. the montage path),
  `tests/inject/test_inject_plan_golden.py` (the create/update/skip decision),
  `tests/unit/test_validate.py` (full validation report).
- **Idempotency / fingerprint** stability: `tests/acceptance/`.
- **Megafunction canary**: `tests/digest/test_helpers.py` (LOC ceilings/floors).
