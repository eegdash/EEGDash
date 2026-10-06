# Experimental NEMAR metadata backend

EEGDash's existing service remains the default. Explicit opt-in selects a public,
read-only **dataset discovery** backend; it is not automatic outage failover:

```python
from eegdash import EEGDash

catalog = EEGDash(backend="nemar")
first_page = catalog.find_datasets(limit=20)
metadata = catalog.get_dataset("nm000132")
summary = catalog.search_datasets(limit=20)  # no filters in this initial adapter
# Rollback: create EEGDash() instead. No persistent configuration is changed.
```

## Supported contract

- Unfiltered `find_datasets(limit=1..1000)` over NEMAR REST `/datasets`, with
  offset pagination (at most five requests of 200 rows). `limit` is an explicit
  result cap, **not** a claim of full catalog coverage. `search_datasets` without
  filters formats these results as the existing summary DataFrame.
- `get_dataset` and exact `find_datasets({"dataset_id": "nm000132"})` use
  `data.nemar.org/<canonical-id>/metadata.json`. Canonical `nmNNNNNN` and
  `onNNNNNN` IDs only; no prefix substitution or inferred alias equivalence.
  Only a rich-metadata HTTP 404 returns `None` / an empty exact-match list.
- Rich neuroschema 0.4 documents retain their original fields, structured authors,
  related identifiers, license, provenance, extensions and version DOI inventory.
  The reported `latest_snapshot` is exposed as `version`; its matching DOI is
  `version_doi`, separately from the concept `dataset_doi`. This endpoint is
  mutable: `metadata_scope="current"` does **not** mean historical version-pinned
  metadata. Catalog rows have `metadata_scope="catalog"` and preserve the raw
  upstream row under `nemar_catalog`. Catalog and rich metadata have different
  detail levels; unknown fields are `None`, not inferred facts. Both include
  `metadata_url` and a UTC `metadata_retrieved_at` acquisition timestamp (not an
  upstream modification time or an atomic snapshot identifier).
- `source` remains the upstream-reported source; `provider="nemar"` identifies
  the service. A missing citation stays unknown rather than being fabricated;
  supplied citations, authors and DOI identifiers are preserved. For example,
  `on000117` reports NEMAR `version="v1.0.0"` while its `IsDerivedFrom` DOI
  identifies OpenNeuro `ds000117.v1.1.0`; these are not interchangeable versions.
  Age bounds are not manufactured participant observations. Missing modality is not default EEG.

## Explicitly unsupported

All Mongo/friendly filters (including license, modality, task and source), `$in`,
source-ID aliases (`ds*`), counts, recordings, participants, signal loading,
aggregations and writes. Unsupported client operations raise
`NemarUnsupportedOperation`, not empty results or dataset totals masquerading as
record totals. Participant/aggregation-specific methods are not part of this
client API; trying such an absent method raises `AttributeError`, while record
queries for participant fields raise `NemarUnsupportedOperation`. Dataset-list
`skip`, text-search `q`, and historical `version` keyword arguments are not
supported and raise `TypeError`; internal offset paging is not a public skip API.
`EEGDashDataset` does not accept this metadata backend, and its existing recording
loader behavior is unchanged. Do not pass these metadata documents as loadable
records. The summary DataFrame omits provenance columns; use `find_datasets` or
`get_dataset` when provenance matters. `api_url`, non-default `database` and
explicit `auth_token` cannot be combined with this backend. No EEGDash environment credentials, proxies or
netrc authentication are forwarded to NEMAR.

The existing ingestion mapper was inspected and its source-field mappings are
used here without the ingestion-specific inferred defaults. The existing
`records.json` fast path supplies signal summaries only and tolerates missing
artifacts; that is not faithful enough for runtime record queries/counts. Neither
it nor `nemar-py`'s downloader establishes EEGDash record/storage parity. They are
left unchanged; no duplicate download protocol or artificial storage paths are
introduced. Record support requires separate verified identity/completeness and
version-policy work.

## Bounded reads and failure behavior

Each request: 5s connect/15s socket-read timeout, checked 30s streaming deadline,
2 MiB decoded body maximum, no redirects and no automatic retries. The deadline
is checked between chunks, not a hard wall-clock interruption of a blocked socket;
a socket read may add up to its timeout. HTTP errors (including 429/Retry-After)
propagate without retrying, so callers must honor Retry-After before trying again.
Malformed/degraded/explicitly partial metadata, pagination inconsistencies,
changing totals and duplicate IDs raise `NemarContractError`; no partial list escapes on error.
Live offset pagination is not an atomic catalog snapshot and cannot detect every
concurrent replacement. Use neither this list nor its page counts as a migration
inventory. There is no implicit caching, fallback to EEGDash, or catalog parity
claim.

## Tests

Offline contract tests live in `tests/unit_tests/test_nemar_backend.py`. Two tiny
public metadata probes are disabled unless explicitly enabled:

```sh
EEGDASH_TEST_NEMAR_METADATA=1 pytest tests/integration/test_nemar_metadata_backend.py
```

These request one catalog row and one rich metadata document only. No recordings,
models, manifests, events, Zarr arrays or participant files are downloaded.
