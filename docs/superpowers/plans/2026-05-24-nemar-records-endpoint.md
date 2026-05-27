# Issue draft — `GET /<id>/<version>/records.json` (+ `manifest.bytes_url`)

> Intended target: `nemarOrg/nemar-cli` as a single GitHub issue.
> Style aim: short enough to discuss in one read, structured enough
> to act on, with a stability section the implementer doesn't have
> to invent. Companion file:
> `eegdash/scripts/ingestions/ROBUSTNESS/08-NEMAR-API-SIMPLIFICATION.md`.

---

## Motivation

Every public consumer of `data.nemar.org` currently re-derives two pieces
of metadata that NEMAR's own systems already compute:

1. **Per-recording header summary** (`nchans`, `sampling_frequency`,
   `n_times`, `channel_type_counts`). The QA pipeline at SDSC
   (`sccn/NEMAR-pipeline/nemar_dataqual.m`) reads every data file to
   render the dashboard widgets. EEGDash re-runs `mne.io.read_raw_*`
   on the same files in its ingestion pass. The `nemar.org` frontend
   asks the legacy PHP `dataexplorer` API. Same numbers, three
   readers.
2. **Stable bytes URL** for each file. `manifest.json` carries the
   git-annex key + size + checksum, but the `url` field still points
   at `raw.githubusercontent.com`, which serves a 130-byte pointer
   for large files instead of the data. Consumers resolve the key to
   `s3://nemar/<key>` by convention.

The central manifest epic (#559) shipped the pattern we'd reuse here:
worker dispatches a `repository_dispatch` event to a central workflow
on `nemarDatasets/.github`, the workflow checks out the dataset at the
tag, writes an artifact to S3, posts back via webhook, worker records
the result. The shape works; we'd add one new artifact and one new
field to an existing artifact.

`nemarOrg/neuroschema` v0.3.0 already specifies the record document:
`schema/core/record.schema.json` + `schema/definitions/signalSummary.schema.json`.
This issue is mostly surfacing, not new design.

## Proposal

Add one route, one artifact, one field. No new infra; reuse the
manifest workflow's plumbing.

```
[Tag pushed on nemarDatasets/<id>]
      ↓ (existing GH App webhook, #559)
[POST /webhooks/dataset-release]
      ↓
[triggerManifestGeneration()]    ← adds bytes_url (extant artifact)
[triggerRecordsGeneration()]     ← NEW dispatch, parallel
      ↓
[generate-manifest.yml + generate-records.yml on nemarDatasets/.github]
      ↓
[s3://nemar/<id>/version/v<X.Y.Z>.json           ← manifest, now with bytes_url]
[s3://nemar/<id>/version/v<X.Y.Z>-records.json   ← NEW]
      ↓
[POST /webhooks/manifest-ready]
[POST /webhooks/records-ready]   ← NEW
      ↓
[GET /<id>/<version>/records.json]   ← NEW route on data.ts
```

### `records.json` shape (per `neuroschema/schema/core/record.schema.json` v0.3.0)

```json
[
  {
    "schema_version": "0.3.0",
    "doc_type": "record",
    "dataset": "nm000110",
    "bids_relpath": "sub-chb01/eeg/sub-chb01_task-rest_run-01_eeg.edf",
    "modality": "EEG",
    "datatype": "eeg",
    "suffix": "eeg",
    "file_extension": ".edf",
    "entities": { "subject": "chb01", "task": "rest", "run": "01" },
    "signal_summary": {
      "nchans": 23,
      "ntimes": 921600,
      "recording_duration": 3600.0,
      "channel_type_counts": { "EEG": 21, "EOG": 2 }
    },
    "provenance": {
      "extracted_at": "2026-05-24T06:14:00Z",
      "extractor_version": "0.1.0"
    }
  }
]
```

`signal_properties` is intentionally absent at the record level when
nothing diverges from `dataset.signal_defaults` — BIDS inheritance
applies (see `neuroschema/docs/inheritance.md`).

Per-channel arrays (full `ch_names`, sidecar JSON, TSV content) belong
in the `rawDetail` extension, served from a separate endpoint. Not in
scope here.

### `manifest.json` addition

One backward-compatible field per file entry:

```json
{
  "path": "sub-chb01/eeg/sub-chb01_task-rest_run-01_eeg.edf",
  "size": 124573612,
  "checksum_algorithm": "git",
  "checksum": "abc123...",
  "url": "https://raw.githubusercontent.com/nemarDatasets/nm000110/v1.0.1/...",
  "bytes_url": "https://nemar.s3.us-east-2.amazonaws.com/SHA256E-s124573612--abc123.edf"
}
```

For git-tracked files (`checksum_algorithm: "git"`), `bytes_url`
equals the existing `url` field. For annex-keyed files, it resolves
the key to the S3 public URL.

Old consumers ignore unknown fields; new consumers stop guessing the
S3 URL.

## Stability

This is the section that decides whether the rollout goes well. None
of it is novel — most of it mirrors how #557 handled the manifest
cutover.

### Feature flag

`RECORDS_VIA_CENTRAL_WORKFLOW` (Workers env var, default `false`).
Same shape as `MANIFEST_VIA_CENTRAL_WORKFLOW`. Flip per env:

- dev — flip first, leave for 7 days
- prod — flip after dev has zero `records_jobs.status='failed'` rows
  for a week and one EEGDash ingestion run has consumed the new
  endpoint end-to-end

With the flag off, `triggerRecordsGeneration()` is a no-op and the
`/records.json` route returns 404 with `reason: "endpoint_disabled"`.

### Idempotency

`records_jobs (dataset_id, version)` is unique. Re-dispatching the
same `(id, version)` either:

- Reuses the existing row if the artifact already exists on S3 and
  `status='uploaded'` (HEAD check on every redispatch — see "S3
  liveness probe" below).
- Replaces the row if status is `failed` or older than a configurable
  TTL (default 7 days for `dispatched`).

The S3 key is content-addressed by `(dataset_id, version)`; a re-run
overwrites in place. Consumers see at most one version of the truth
per `(id, version)`.

### Callback HMAC

Same one-shot HMAC pattern as `/webhooks/manifest-ready`:
`callback_token = HMAC(MANIFEST_CALLBACK_SECRET, "{dataset_id}:{version}:{nonce}")`.
Store `sha256(callback_token)` in `records_jobs.callback_token_hash`,
never the secret itself. Compare on the receiving side; reject 403 on
mismatch. One-shot via a 5-minute Workers cache key on the
`(dataset_id, version, nonce)` triple.

### S3 liveness probe

Before the worker returns 200 from `/records.json`, it `HEAD`s the S3
URL. If the HEAD fails or the size is implausibly small (<128 bytes
suggests a truncated upload), serve 502 with `reason: "artifact_unreachable"`
and log the run id so an operator can replay.

If a published-but-records-pending version is hit before the workflow
finishes, serve 404 with `reason: "records_pending"` and a `Retry-After:
60` header. Don't fake a partial response.

### Schema validation in CI

Add one workflow step to `nemar-cli/.github/workflows/test.yml`:

```yaml
- name: Validate emitter output against neuroschema
  run: |
    cd nemarDatasets/.github
    python -m scripts.emit_records --bids tests/fixtures/bids_root --dataset nm099999 --out /tmp/r.json
    python -m neuroschema.validate --doc-type record --array /tmp/r.json
```

Bumps to neuroschema that break the emitter's output get caught
before merge instead of after deploy. Pin the neuroschema version
explicitly; treat un-pinned upgrades as a separate PR.

### Graceful degradation

If the GitHub Actions workflow is down (GH incident, expired App
token, runner outage), the dispatch path fails fast:

1. `triggerRecordsGeneration()` catches the 4xx/5xx from GitHub and
   inserts `records_jobs.status='failed'` with the error body.
2. `/records.json` returns the last known good artifact (S3 HEAD
   on the previous version's key, if any).
3. A `nemar admin records replay --since 24h` command re-dispatches
   every `failed` row from the last day.

The publish flow does NOT block on records generation. A dataset
publishes when manifest + DOI succeed; records emission is fire-and-forget
with a separate completion signal. This decouples user-facing
publication latency from the (longer) records emission time.

### Rate limits

- GitHub `repository_dispatch` is 5,000 req/hour per token. One
  dispatch per published version means we'd hit the limit at ~83
  publish events per minute sustained, which is well above expected
  load.
- S3 PUT is unconstrained for our usage profile.
- Workers free tier on `data.nemar.org` was the bottleneck during the
  manifest epic; same fix applies — serve `Cache-Control: public,
  max-age=300` and let CF cache hit ratio climb above 90%.

### Observability

Three log lines per dispatch, structured (one JSON line each):

```
{"event":"records.dispatch","dataset_id":"nm099999","version":"1.0.0","ts":"...","run_id":null}
{"event":"records.ready","dataset_id":"nm099999","version":"1.0.0","run_id":"...","totals":{...}}
{"event":"records.served","dataset_id":"nm099999","version":"1.0.0","cache":"miss"}
```

One dashboard panel: `count(records.ready) / count(records.dispatch)`
over the last 24h. If it drops below 0.95, page someone.

### Compatibility window

Both `records.json` and the existing `manifest.json` (without
`bytes_url`) keep working through the cutover. The plan is:

| Week | Records endpoint | Manifest `bytes_url` |
|---|---|---|
| 1 | flag on in dev | added in dev manifests |
| 2 | dev only, monitoring | dev only |
| 3 | flag on in prod for new versions | added in prod for new versions |
| 4–6 | backfill old prod versions via re-dispatch | backfill via same script |
| 7+ | EEGDash + frontend cut over | EEGDash + frontend cut over |

Old `manifest.json` consumers see `bytes_url` show up as an
additional field. The git-annex key resolution they already do still
works. No coordinated cutover needed.

### Rollback

If `/records.json` causes any prod issue (slow S3, runaway worker
CPU, unbounded D1 growth), flip `RECORDS_VIA_CENTRAL_WORKFLOW=false`.
Route returns 404, dispatch is a no-op, `records_jobs` rows stop
arriving. No data is lost; old `manifest.json` is untouched.

If `bytes_url` causes any issue, the field is purely additive — strip
it from the serializer with a one-line revert. No consumer is
required to read it.

## Out of scope (explicitly)

- LLM enrichment of records (covered by existing
  `services/llm-enrich.ts`).
- `dataQuality` extension (frame retention, ICA stats). Separate
  issue. Owner is `nemar-pipeline`.
- Per-subject demographics. Separate issue.
- `rawDetail` extension (per-channel arrays, full sidecar JSON).
  Separate consumer; would need its own endpoint and access policy.
- MEF3 and CTF (.ds) header parsers. First version of the emitter
  covers EDF / BDF / BrainVision / EEGLAB / FIF / SNIRF. MEF3 / CTF
  follow as v0.2 of the emitter.

## Companion fixes (separate, small PRs)

Two one-line PRs against `nemarOrg/neuroschema` unblock EEGDash
adoption. Could land before this issue ships, after, or independently:

1. `schema/core/record.schema.json` line 20:
   `"pattern": "^ds[0-9]+$"` rejects every `nm*` / `on*` record.
   Change to `"^[a-z]{2}[0-9]+$"` (matches `dataset_id` in
   `dataset.schema.json`).
2. `schema/extensions/eegdash.schema.json` line 14:
   `storage.backend` enum missing `"nemar"`. EEGDash uses `nemar` as
   a fourth backend (git-annex key resolution).

## Acceptance

- [ ] `GET https://dev-data.nemar.org/nm099999/1.0.0/records.json`
      returns a JSON array; every element validates against
      `neuroschema/schema/core/record.schema.json` v0.3.x.
- [ ] `GET https://dev-data.nemar.org/nm099999/1.0.0/manifest.json`
      has `bytes_url` on every file entry; 5 sampled entries per
      dataset across 3 datasets `HEAD` 200 with the expected size.
- [ ] `RECORDS_VIA_CENTRAL_WORKFLOW=false` keeps the route off and
      the dispatch a no-op; flip-on covered by integration test.
- [ ] `records_jobs.status='uploaded'` for the test dataset after one
      `nemar admin e2e-test --verbose` run.
- [ ] CI step validates emitter output against neuroschema; PR is
      blocked if the schema check fails.
- [ ] Both neuroschema PRs reviewed and either merged or scheduled
      with an owner.
- [ ] `/review-pr` clean.

## Open questions

1. **Header readers in the central workflow vs. SDSC.** The QA
   pipeline already reads each file with EEGLAB and dumps numbers to
   MAT files. Re-reading with MNE in the central workflow duplicates
   that work. Option: mount or sync the SDSC outputs and have the
   emitter parse the MAT files instead. Faster, no MNE dep in the
   workflow, but introduces a cross-system coupling. This proposal
   assumes the simpler "re-read with MNE" path; flag if the SDSC
   path is preferred.
2. **ETag / If-None-Match on the JSON endpoints.** Cheap; lets
   downstream consumers (EEGDash daily ingestion, frontend cache)
   skip unchanged datasets without fetching the body. Drop in this
   epic or a follow-up?
3. **`on*` (OpenNeuro mirror) coverage.** Does the records emission
   fire for `on*` datasets too, or only `nm*`? EEGDash would prefer
   parity, but if `on*` BIDS validation differs upstream, scoping
   `on*` out for v1 is fine.
4. **Webhook subscription for third parties.** EEGDash would gladly
   subscribe to a `records-ready` event instead of polling. The
   webhook infrastructure exists (`backend/src/routes/webhooks.ts`).
   In scope for this epic or follow-up?

## Downstream impact

EEGDash drops ~1,300 lines of NEMAR-specific code once this lands
(detailed in
`eegdash/scripts/ingestions/ROBUSTNESS/08-NEMAR-API-SIMPLIFICATION.md`).
The `nemar.org` frontend can read both artifacts directly and stop
asking the legacy PHP `dataexplorer` API.
