# EEGDash Data Management Redesign Plan (Draft)

This document is a maintainer-facing plan to simplify EEGDash’s data management across:

- **Ingestion/listings (CI)**: fetching dataset listings from external sources.
- **Digestion (metadata extraction)**: turning local BIDS datasets into DB records.
- **Runtime (users + CI tests)**: querying records, downloading files, and loading with MNE.

It is optimized for:
- **Less bespoke code** (prefer stable library APIs).
- **Clear contracts** (record schema + path rules).
- **Robustness** (handle OpenNeuro non-BIDS quirks like `run="5F"`).

---

## 0) Locked Decisions (for this redesign)

- **Non-BIDS runs are supported end-to-end**: preserve `entities.run` (e.g., `"5F"`) and keep a sanitized `entities_mne.run` for MNE-BIDS.
- **Introduce `record_id`** as the canonical stable identifier (keep `data_name` for backward compatibility during migration).
- **Keep an explicit `variant` field** (do not infer behavior from URLs).
- **Fallback reads must preserve BIDS behavior**: when we cannot use `mne_bids.read_raw_bids`, still apply `events.tsv` annotations and `channels.tsv` (bads + channel metadata) when present.

### 0.1) Implementation status (current PR branch)

**BREAKING CHANGE**: v1 compatibility has been removed. All records must now be v2 format.

Implemented (library-only, no DB changes yet):
- `eegdash/records.py`: Pure v2 record schema with `RecordBuilder` and `create_record()` for creating records. No default buckets - `storage.base` must be explicitly provided.
- `eegdash/downloader.py`: simplified downloader (no path rewriting); added basic integrity checks (re-download if local size mismatches remote).
- `eegdash/dataset/base.py`: requires v2 records with explicit storage.base; path resolution inlined. Prefers `mne_bids.read_raw_bids()` when representable, otherwise direct reads + best-effort sidecar application.

Test/runtime reliability improvements:
- `eegdash/__init__.py` + `eegdash/api.py`: lazy exports to avoid importing heavy optional deps at import time.
- `eegdash/paths.py`: defaults cache to `./.eegdash_cache` for reproducible local runs/CI.
- `tests/test_dataset.py` + `tests/test_minirelease.py`: make the live API + S3 test matrix opt-in (env vars) to keep the default suite fast/stable.

## 1) Current State (What’s driving complexity)

### Runtime path rewriting is scattered
Today, the “physical storage layout” differs across sources (OpenNeuro vs challenge buckets), so runtime code rewrites paths:
- Removes `dsXXXXXX/` prefixes for some buckets.
- Converts `.set → .bdf` for some buckets.
- Rewrites dependency paths separately from raw file paths.

This increases surface area and makes caching brittle.

### MNE-BIDS is strict about `run`
`mne_bids.BIDSPath` and `find_matching_paths()` reject non-numeric runs (e.g., `run="5F"`), which exist in OpenNeuro.
We want to support these at **runtime**, not only in digestion.

### CI caching is overly complex
Workflows create and cache multiple directories and hash a manifest file that doesn’t exist in this repo, making cache
invalidation and portability harder than necessary.

---

## 2) Target End State (Simplified architecture)

### 2.1 Record Contract: separate logical BIDS from physical storage
**Core idea:** records should not require runtime “guessing” how to map to storage.

#### Invariants
- `dataset` is the stable dataset identifier (e.g., `ds002718`).
- `bids_relpath` is **dataset-root relative** and **never** includes the dataset id prefix.
  - Example: `sub-01/eeg/sub-01_task-rest_run-5F_eeg.vhdr`
- `entities.run` preserves the original value (may be non-numeric).
- `entities_mne.run` is either a numeric string or `None` (safe for `mne_bids.BIDSPath`).

#### Minimal schema v2 (proposal)
Store the minimum fields needed to eliminate runtime rewriting:

- `schema_version`: `int` (always `2`)
- `record_id`: `str` (canonical stable id; UUID or deterministic hash)
- `variant`: `str` (e.g., `openneuro_raw`, `challenge_l100_bdf_mini`)
- `dataset`: `str`
- `data_name`: `str` (stable unique key; keep if already used in DB/API)
- `bids_relpath`: `str` (dataset-root relative)
- `datatype`: `str` (e.g., `eeg`)
- `suffix`: `str` (e.g., `eeg`)
- `extension`: `str` (e.g., `.vhdr`, `.set`, `.bdf`)
- `entities`: `dict` (subject/session/task/run… with **original** values)
- `entities_mne`: `dict` (same, but `run` sanitized for MNE-BIDS)
- `storage`: `dict`
  - `backend`: `"s3"` (future: `"https"`, `"datalad"`, etc.)
  - `base`: `str` (dataset root on remote; **already includes** the dataset/prefix)
    - OpenNeuro example: `s3://openneuro.org/ds002718`
    - Challenge example: `s3://nmdatasets/NeurIPS25/R5_mini_L100_bdf`
  - `raw_key`: `str` (relative to `storage.base`; usually equals `bids_relpath`)
  - `dep_keys`: `list[str]` (relative to `storage.base`)
- `cache`: `dict`
  - `dataset_subdir`: `str` (where this dataset variant lives locally; e.g., `ds002718` or `ds005509-bdf-mini`)
  - `raw_relpath`: `str` (relative to `cache.dataset_subdir`; usually equals `bids_relpath`)
  - `dep_relpaths`: `list[str]` (relative to `cache.dataset_subdir`)

This is intentionally explicit and removes runtime path hacks.

### 2.2 One place for file resolution (library-level)
Introduce a single “resolver” layer:
- Input: DB record (+ optional dataset-level overrides)
- Output:
  - Local path(s) to ensure exist (raw + deps)
  - Remote URI(s) to download from

All path rules live here; `downloader` becomes “download URI → local path”.

### 2.3 Runtime loading strategy (support non-BIDS `run`)
Goal: keep using `mne_bids.read_raw_bids()` when possible, but still support OpenNeuro quirks.

- If `entities_mne.run` is valid (numeric or `None` and path is representable):
  - Use `mne_bids.read_raw_bids(BIDSPath(...))` for best behavior.
- If `entities.run` is non-numeric and the filename cannot be represented by `BIDSPath`:
  - Read directly from the local raw file path using `mne.io.read_raw_*`.
  - Then *optionally* apply sidecars:
    - `events.tsv`: use `mne_bids.events_file_to_annotation_kwargs` and set `raw.set_annotations(...)`.
    - `channels.tsv`: mark bads by reading TSV (minimal behavior), or keep this step optional.
  - Continue to attach participants.tsv extras (this doesn’t require a run index).

### 2.4 Offline discovery should not depend on `find_matching_paths()`
Because `find_matching_paths()` rejects `run="5F"`, offline mode must use tolerant discovery:
- Use `EEGBIDSDataset`’s glob-based fallback (or a shared tolerant scanner) to enumerate files.
- Build records using the same “record contract” fields as DB records (at least `bids_relpath`, `entities`, `entities_mne`).

---

## 3) CI Data Management (simplify and make deterministic)

### 3.1 One cache root
Standardize CI + user cache to one variable:
- `EEGDASH_CACHE_DIR` (primary)

Optionally set:
- `MNE_DATA=$EEGDASH_CACHE_DIR` for MNE’s own caches.

Avoid caching multiple directories; cache only the single resolved cache root.

### 3.2 Fix cache invalidation input
Current workflows hash `consolidated/datasets_consolidated.json`, which isn’t present.
Pick one of:
- Add a real consolidated manifest file to this repo and hash it.
- Or hash existing listings files that actually exist (e.g., `consolidated/openneuro_datasets.json`).

### 3.3 Make data/network failures actionable
Add a lightweight “data reachability” smoke step (API + S3 HEAD) so failures are clearly attributed:
- API unreachable / rate-limited
- S3 object missing
- Auth misconfiguration

---

## 4) Execution Plan (phased, minimizes breakage)

### Phase A — Library refactor (backward compatible)
- Add a record resolver layer (new module) that supports:
  - Schema v1 records (current DB)
  - Schema v2 records (future)
- Update runtime download code to call the resolver, removing scattered path hacks.
- Update runtime loading to support non-numeric `run`:
  - Avoid constructing `BIDSPath(run="5F")`.
  - Implement the “read direct file” fallback.
- Update offline mode to avoid `mne_bids.find_matching_paths()` and use tolerant discovery.
- Add tests:
  - Offline discovery with `run="5F"` should not crash.
  - Runtime loader should not crash when `run="5F"` exists in record.

### Phase B — Digestion/ingestion emits schema v2 records
- Update digestion scripts to write:
  - `bids_relpath`, `entities`, `entities_mne`
  - `storage.base`, `storage.raw_key`, `storage.dep_keys`
  - `cache.dataset_subdir`, `cache.raw_relpath`, `cache.dep_relpaths`
- Keep exporting legacy fields temporarily (or provide a compatibility shim) while API/DB transitions.

### Phase C — DB/API migration
- Introduce schema v2 in DB (new fields added; old fields kept).
- Backfill existing records where possible:
  - OpenNeuro: `storage.base = s3://openneuro.org/<dataset>`
  - Challenge: `storage.base = <challenge-prefix>`, `raw_key` stripped correctly, extension corrected.
- Update API to serve v2 fields (and optionally both v1/v2 during transition).

### Phase D — Delete legacy hacks
- Once v2 fields are fully populated and stable:
  - Remove regex-based stripping and extension conversion.
  - Remove dependency path rewriting.
  - Treat storage layout as fully data-driven from records.

---

## 5) Open Questions (remaining)

- None currently. Update this section as new constraints appear.

---

## 6) Three-step Deliverables (split from all markdown action items)

Sources reviewed for these deliverables:
- `DATA_MANAGEMENT_PLAN.md`
- `REVISION_IMPROVING.MD`
- `scripts/ingestions/README.md`
- `mongodb-eegdash-server/README.md`

### Phase A — Library refactor (backward compatible)

**Deliverable A1 — Define schema v2 + compatibility adapter**
1. Update schema docs (fields + invariants) and add `record_id`/`variant` to the target contract.
2. Add a v2 record representation + `v1 → v2` in-memory adapter (no DB changes yet).
3. Add unit tests that adapt representative v1 records (OpenNeuro + challenge) and assert resolved URIs + local paths are stable.

**Deliverable A2 — Centralize path resolution (resolver layer)**
1. Implement a single resolver API that returns `{raw_uri, dep_uris, raw_path, dep_paths}` from `(record, cache_dir)`.
2. Refactor runtime download code to use the resolver (stop doing path rewriting in downloader/base dataset).
3. Add regression tests ensuring the same record yields the same cache layout across online/offline modes.

**Deliverable A3 — Runtime loading fallback with sidecars (supports `run="5F"`)**
1. Implement a loader that chooses `mne_bids.read_raw_bids` when representable, else reads directly from the raw file path.
2. In the fallback path, apply `events.tsv` → annotations and `channels.tsv` → bads/channel metadata when present.
3. Add tests using a tiny local BIDS fixture with `run="5F"` that asserts annotations + bad channels are applied.

**Deliverable A4 — Offline discovery without `find_matching_paths()`**
1. Replace offline discovery to use tolerant filesystem scanning + entity parsing (no `mne_bids.find_matching_paths`).
2. Emit offline records through the same adapter/resolver contract (so online/offline behave identically).
3. Add tests that build an offline dataset containing `run="5F"` and confirm it does not crash and resolves paths correctly.

**Deliverable A5 — CI cache simplification + reachability smoke checks**
1. Standardize CI to one cache root (`EEGDASH_CACHE_DIR`) and cache only that directory.
2. Fix cache invalidation inputs to hash files that exist (choose a single manifest/listings file).
3. Add a lightweight smoke step that reports API/S3 reachability failures distinctly (actionable errors).

### Phase B — Digestion/ingestion emits schema v2 records

**Deliverable B1 — Digestion scripts emit v2 fields (plus legacy during transition)**
1. Implement a shared “record builder” that produces v2 fields (`record_id`, `variant`, `bids_relpath`, `storage.*`, `cache.*`).
2. Update `scripts/ingestions/3_digest.py` to output v2 + keep v1 fields for compatibility.
3. Add a schema validation step (or unit tests) that asserts v2 outputs are internally consistent (URIs/paths/entities).

**Deliverable B2 — Align ingestion docs with reality**
1. Audit `scripts/ingestions/README.md` for script names/paths and update commands to match current repo structure.
2. Document the new v2 record fields and how they flow through “digest → upload → runtime”.
3. Add a short “migration notes” section covering coexistence of v1/v2 fields during rollout.

### Phase C — DB/API migration

**Deliverable C1 — API accepts/serves v2 fields**
1. Extend the API server to store and return v2 fields (keep old fields for backward compatibility).
2. Update server docs (`mongodb-eegdash-server/README.md`) to document new fields and recommended filters (e.g., `variant`).
3. Add an integration test or smoke script that inserts a v2 record in staging and reads it back successfully.

**Deliverable C2 — Backfill/migration tooling**
1. Write a migration script that backfills v2 fields for existing records (OpenNeuro + challenge).
2. Run migration on staging and validate: no runtime path rewrites required for migrated records.
3. Define a cutover criterion (e.g., ≥99% records have v2 fields) and then schedule prod migration.

### Phase D — Remove legacy hacks

**Deliverable D1 — Delete legacy path hacks**
1. Remove regex-based path stripping, `.set→.bdf` hacks, and dependency path rewriting from runtime code.
2. Remove v1-only compatibility branches once v2 coverage is sufficient.
3. Update docs + tests to rely on v2 behavior only (keep a pinned “legacy” branch/tag if needed).
