# 08 — NEMAR / neuroschema Adoption

Status: Draft, 2026-05-24
Companion: `04-ROADMAP.md`, `ADRs/0003-separate-ingestions-repository.md`

## Summary

`nemarOrg/neuroschema` v0.3.0 defines the canonical metadata contract
EEGDash should consume from. Adopting it deletes about 1,150 lines
(~88%) of NEMAR-specific code, drops `mne` + `numba` from the NEMAR
path, and retires three helper scripts. Most of the schema fields
already exist server-side; what's left is two surfacing tasks on
NEMAR's side and a `schemas.py` migration on ours.

## Where things stand

- `data.nemar.org` publishes a catalogue and per-dataset
  `metadata.json` (123 datasets, schema_version 0.3.0).
- `nemarOrg/nemar-cli` is the Cloudflare Workers backend behind it
  (D1 + S3 + git-annex via DataLad).
- `nemarOrg/neuroschema` is the JSON Schema repo, BSD-3,
  DataCite-4.6 aligned. Each platform writes only its own extension
  collection.
- `sccn/NEMAR-pipeline` runs the QA pass at SDSC and produces the
  per-recording numbers EEGDash currently re-derives with MNE.

## What neuroschema already specifies (we can stop deriving)

Core dataset (`schema/core/dataset.schema.json`):
`dataset_id`, `name`, `source`, `recording_modality`, `bids_version`,
`license`, structured `authors[]` and `funding[]`, `tasks`, `datatypes`,
`sessions`, `has_hed_annotations`, `hed_version`, `has_event_annotations`,
`demographics` (subjects_count, age_min/max/mean, sex_distribution,
handedness_distribution), `data_summary`, `signal_defaults`
(BIDS-inheritable), `keywords`, `related_identifiers`, `contributors`,
`dates`, `rights`, `language`, `external_links`.

Core record (`schema/core/record.schema.json`): `dataset`, `bids_relpath`,
`modality`, `datatype`, `suffix`, `file_extension`, `entities`,
`signal_properties` (BIDS-override), `signal_summary` (nchans, ntimes,
recording_duration, channel_type_counts), `provenance`.

Extensions: `eegdash` (storage, tags, ages), `nemar` (citation_count,
hed_annotation, on_brainlife, processed), `dataQuality` (quality_score,
quality_flags), `rawDetail` (per-channel arrays), `dataCite`,
`dataCategories`, `signalJourney`.

## Two gaps to file against neuroschema before we adopt

These are bugs in v0.3.0 that block EEGDash adoption:

1. `record.schema.json` line 20: `"pattern": "^ds[0-9]+$"` on the
   `dataset` foreign key. Rejects every NEMAR `nm*` and `on*` record.
   Should be `^[a-z]{2}[0-9]+$` to match `dataset_id` in core.
2. `eegdash.schema.json` line 14: `storage.backend` enum is
   `["s3", "https", "local"]`. EEGDash uses `nemar` (git-annex
   pointer resolution) as a fourth backend. Add `"nemar"`.

Both are one-line PRs.

## Two endpoints NEMAR still needs to expose

Everything else in §"What neuroschema already specifies" is in the
shape we need. These two are computed internally but not yet on the
public JSON:

1. `GET /{id}/{version}/records.json` — array of neuroschema record
   documents. Source is `sccn/NEMAR-pipeline/eeg_nemar_dataqual.m`
   which already reads every data file. Surfacing it removes the
   entire MNE branch from `3_digest.py`.
2. `bytes_url` field on each `manifest.json` entry that resolves to
   the data bytes (not the git-annex pointer at
   raw.githubusercontent.com). The CLI's upload flow already produces
   the S3 key; this surfaces it on read.

## EEGDash work

### Phase A — no NEMAR change required

Land now.

- Delete `patch_nemar_source.py` and `patch_nemar_records_storage.py`.
  Both are one-time fixups; the records they target are already
  patched in production. Verify with `db.datasets.distinct("source",
  {"dataset_id": /^nm/})` first. Saves 390 lines.
- Rewrite `1_fetch_sources/nemar.py` to call `data.nemar.org` and
  parse the neuroschema document. Replaces the GitHub-org scan and
  the sidecar fetch. From 360 lines to ~70.
- Switch `2_clone.py` NEMAR branch to use `manifest.json` directly.
  No shallow clone, no git-annex pointer size extraction. Use the
  git checksum as the dataset fingerprint. Saves ~120 lines.

Total: ~800 lines removed, no upstream dependency.

### Phase B — after NEMAR adds records.json + bytes_url

- Replace MNE header inspection in `3_digest.py` (NEMAR branch) with
  one HTTP fetch of `records.json`. Saves ~150 lines.
- Drop `_file_utils.get_annex_file_key` from the NEMAR path; trust
  `bytes_url`. Saves ~70 lines.
- Mark `mne` and `numba` as optional extras in `pyproject.toml` so
  NEMAR-only runs no longer need them.

### Phase C — fold in neuroschema's nemar + dataQuality extensions

- Read `nemar.citation_count` from `metadata.json`. Delete
  `inject_nemar_citations.py` (207 lines) and the daily cron.
- Read `nemar.hed_annotation` and `dataset.has_hed_annotations`
  instead of the local `events.tsv` grep. Saves ~20 lines.
- Read the `dataQuality` extension (managed by `nemar-pipeline`)
  instead of scraping the NEMAR dashboard for retention/ICA numbers.
  Saves ~60 lines in `docs/source/conf.py`.

### Phase D — schemas.py migration

`neuroschema/.context/plan.md` Phase 4 is "Update EEGDash `schemas.py`
to align with neuroschema." The adapter table in
`neuroschema/docs/adapter-eegdash.md` already maps every current field.

Move per-channel arrays (`channel_names`, `channel_types`,
`channel_tsv`, `eeg_json`, `participant_tsv`, `bidsdependencies`) into
the `ext_rawDetail` collection. Surface them via a lookup join when
the application needs them. Keeps the `records` collection lean for
discovery queries.

Map flat fields to nested neuroschema structure during the next
ingestion. Existing records get migrated by a one-shot adapter
script.

## Why this is mostly surfacing, not new work

| Field EEGDash derives | Already computed by |
|---|---|
| subject_count, modalities, age_min/max, file_size, total_files, tasks | `nemar-cli/backend/src/services/dataset-metadata-columns.ts` |
| nchans, sampling_frequency, n_times, channel_type_counts | `sccn/NEMAR-pipeline/eeg_nemar_dataqual.m` |
| citation_count | `nemarOrg/nemar-citations` |
| Funding, EthicsApprovals, Acknowledgements, BIDSVersion, HEDVersion | legacy `dataexplorer_dataset` MySQL rows |
| description, keywords, related_identifiers, funding | `nemar-cli/backend/src/services/llm-enrich.ts` (OpenRouter) |
| frame retention, ICA components, channel rejection | `nemar-pipeline` MAT outputs |
| bytes URL (S3 key) | `nemar-cli/backend/src/services/manifest.ts` |

`nemar.org`'s own frontend bridges these same layers. Exposing them
once in neuroschema-shaped JSON lets both consumers stop reimplementing
the bridge.

## Code-removal totals

| Phase | Lines | Notes |
|---|---|---|
| A | -800 | Independent of NEMAR. |
| B | -220 | After records.json + bytes_url. |
| C | -287 | After nemar + dataQuality extensions land in metadata.json. |
| D | varies | schemas.py rewrite; net positive long-term. |
| Total before D | -1,307 | |

Plus: drops `GIT_LFS_SKIP_SMUDGE` from the NEMAR path, drops
`mne`/`numba` as required deps, removes the legacy patch script
maintenance burden.

## Single-message ask to NEMAR

> Two endpoints would let EEGDash retire most of its NEMAR ingestion
> code: `GET /{id}/{version}/records.json` (per-file neuroschema
> records — same shape as `core/record.schema.json`), and a
> `bytes_url` field on each `manifest.json` entry so we don't have
> to resolve git-annex pointers ourselves.
>
> Two one-line PRs against neuroschema would also help: fix the
> `record.dataset` pattern to accept `nm*` IDs, and add `"nemar"` to
> the `eegdash.storage.backend` enum.
>
> Once those are in, three NEMAR-side helper scripts on our side
> (citations, source patch, storage patch) and the entire MNE branch
> of our digestion stage can come out. Happy to send PRs.

## Open questions

1. Where does the canonical schema live long-term — in
   `nemarOrg/neuroschema` as a Git submodule or pinned package?
2. How does the QA pipeline at SDSC publish into `metadata.json`?
   Does the latter block on QA, or ship eagerly with
   `dataQuality.quality_flags: ["pending"]`?
3. Do `on*` (OpenNeuro mirror) datasets get the same QA pipeline
   coverage as `nm*`, or do we keep the OpenNeuro-direct path warm
   for those IDs?
4. Is there a planned `nemar-py` SDK surface for the same JSON, and
   should EEGDash pull it as a dependency instead of writing its own
   client?
5. Cache-control plan: `If-None-Match` on `metadata.json` would let
   EEGDash drop its `ingestion_fingerprint` layer for NEMAR entirely.
6. Webhook contract: `backend/src/routes/webhooks.ts` exists. Is
   EEGDash allowed to subscribe to dataset add/update events?

## Verification

Pre-change, snapshot `digestion_output/{id}/*.json` for 10
representative NEMAR datasets (small/large, EEG/MEG, single/multi-session,
with/without HED). After each phase, regenerate and `git diff --stat`
the snapshot. Only intended field changes should show.

Then dry-run `5_inject.py --database eegdash_dev --dry-run` and
confirm zero schema errors against the existing MongoDB shape.

After Phase D, validate every dataset against
`schema/neuroschema.schema.json` using the validator in
`neuroschema/src/neuroschema/validate.py`.
