# Live integration testing against the EEGdash cluster

C6.3. Tests at `tests/test_inject_integration_live.py` exercise the
full Stage 5 path against the production cluster's API Gateway:
``digest output → POST /admin/eegdash_dev/... → MongoDB → GET back``.

## When to run

These tests are **off by default** — both PR-fast CI and the local
`pytest` run skip them. Opt in by setting env vars:

```bash
export EEGDASH_INTEGRATION_API_URL="https://data.eegdash.org"
export EEGDASH_INTEGRATION_ADMIN_TOKEN="<admin Bearer token>"
pytest tests/test_inject_integration_live.py
```

Or run the whole suite in integration mode:

```bash
pytest -m integration
```

## What they cover

5 tests against the live cluster:

1. **API health** — GET `/health` returns ``{"status": "healthy", "mongodb": "connected"}``
2. **Dataset list** — GET `/api/eegdash_dev/datasets` returns real data (sanity check the database has content)
3. **Dataset round-trip** — POST a synthetic test dataset, GET it back, verify fields
4. **Record round-trip** — POST a synthetic record to `/admin/eegdash_dev/records/upsert`, GET it back filtered by dataset_id
5. **Idempotent upsert** — POST the same Record twice, verify no duplicate (composite key works)

## Cluster topology (`sccn` host)

```
   ┌─────────────────────────────────────────────────┐
   │   data.eegdash.org (TLS, public)                │
   │   ──> Caddy reverse proxy (ports 80/443)        │
   │                                                  │
   │       ──> eegdash-api (FastAPI, internal :3000) │
   │              │                                   │
   │              ├──> mongodb-production (:27017)   │
   │              │      databases:                   │
   │              │        eegdash (prod, 1.3 GB)    │
   │              │        eegdash_dev (262 MB) ←target│
   │              │        eegdash_archive (empty)    │
   │              │                                   │
   │              └──> eegdash-redis (rate limiting) │
   └─────────────────────────────────────────────────┘
```

The integration tests target **`eegdash_dev`** — explicitly NOT
production. All test documents use a unique ``c6_smoke_<unix_ts>_<uuid8>``
prefix so a concurrent run never collides + orphans are visible.

## Cleanup limitation

The Caddy reverse proxy blocks DELETE methods at the edge:

```
# caddy/Caddyfile ~line 180
@dangerous_methods {
  method TRACE TRACK DELETE PUT CONNECT PROPFIND SEARCH
}
respond @dangerous_methods 405
```

This means the test's automatic cleanup **cannot use the admin DELETE
endpoint** through the public URL. Two workarounds:

### Option A — Cleanup command env var

Set `EEGDASH_INTEGRATION_CLEANUP_CMD` to a shell template with
`{dataset_id}` substitution. The test invokes it after each test:

```bash
export EEGDASH_INTEGRATION_CLEANUP_CMD='ssh sccn "docker exec mongodb-production mongosh '\''mongodb://admin:PASS@localhost:27017/eegdash_dev?authSource=admin'\'' --quiet --eval \"db.datasets.deleteOne({dataset_id: \\\"{dataset_id}\\\"}); db.records.deleteMany({dataset: \\\"{dataset_id}\\\"})\""'
```

Suits a dev who has SSH access to the cluster.

### Option B — Periodic orphan sweep

If the cleanup command isn't set, the tests still pass but leave
test documents in `eegdash_dev`. They're identifiable by prefix.
Clean up periodically:

```bash
ssh sccn 'docker exec mongodb-production mongosh \
  "mongodb://admin:PASS@localhost:27017/eegdash_dev?authSource=admin" \
  --quiet --eval "
    const ds = db.datasets.deleteMany({dataset_id: /^c6_smoke_/});
    const rec = db.records.deleteMany({dataset: /^c6_smoke_/});
    print(\"Deleted \" + ds.deletedCount + \" datasets, \" + rec.deletedCount + \" records\");
  "'
```

### Option C — Long-term fix

Either:
- **Update the Caddy config** to allow DELETE on `/admin/*` paths
  with the Bearer token (small Caddy change; needs ops review)
- **Add a POST-based delete endpoint** to the API
  (e.g., `POST /admin/{db}/datasets/{id}/delete`) that bypasses the
  Caddy method filter

Tracked as a separate concern; the integration tests work today with
either Option A or Option B cleanup.

## Production credential warning

The `docker-compose.yml` in `~/eegdash-competition/` on the cluster
has hardcoded credentials (MongoDB root password, admin token, CI
token). Anyone with SSH access to the `sccn` host can read them. If
this conversation log or the test file is shared, the credentials are
exposed. Rotate them and move to `.env` after this work lands.

## How this relates to other test layers

| Layer | What it tests | When it runs |
|---|---|---|
| Unit (mocked) — `test_inject_api_calls.py` | inject_datasets/_records/_montages with respx-mocked Gateway | Every PR (CI gate) |
| E2E (subprocess, --dry-run) — `test_pipeline_e2e.py` | digest → validate → inject --dry-run against snapshot | Every PR (CI gate) |
| **Integration (live)** — `test_inject_integration_live.py` | real network → real Gateway → real MongoDB | Opt-in via env var |

The mocked layer catches our own assumptions about the API. The
integration layer catches drift between those assumptions and the
real Gateway's behaviour. Found-bug-per-fixture pattern from C5.1
applies here too — if the Gateway updates its response shape, this
layer fires before production.

## When to extend

Add a new integration test when:
- A new Stage 5 endpoint is added (e.g., bulk-update, schema-migration)
- The API Gateway changes its request/response shape
- A bug in production data is suspected to come from the
  inject path (the integration tests run against `eegdash_dev`
  with prod-shape data — closest you can get without touching prod)

Don't add an integration test for:
- New parser features (use unit tests + snapshot)
- New BIDS sidecar fields (covered by `test_bids_sidecar_enrichment.py`)
- Cleanup of orphan data (script, not a test)
