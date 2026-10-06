Experimental NEMAR metadata backend
===================================

EEGDash's existing service remains the default. Explicit opt-in selects public,
read-only **dataset discovery**; it is not automatic outage failover and does not
switch ``EEGDashDataset`` recording loading.

.. code-block:: python

   from eegdash import EEGDash

   catalog = EEGDash(backend="nemar")
   first_page = catalog.find_datasets(limit=20)
   metadata = catalog.get_dataset("nm000132")
   summary = catalog.search_datasets(limit=20)  # no filters
   # Rollback: construct EEGDash() instead; no persistent setting changes.

Supported discovery
-------------------

Unfiltered ``find_datasets(limit=1..1000)`` reads at most five catalog pages of
200 rows. The limit is a result cap, not a complete catalog inventory or a record
count. ``search_datasets`` provides the existing summary DataFrame, which omits
provenance columns; use dictionary results when provenance matters.

``get_dataset`` and exact ``find_datasets({"dataset_id": "nm000132"})`` retrieve
current rich metadata for canonical ``nmNNNNNN`` / ``onNNNNNN`` IDs. Source-ID
aliases such as ``ds*`` are unsupported; no prefix substitution is attempted.
Only rich-metadata HTTP 404 denotes absence (``None`` / an empty exact-match list).

Rich neuroschema 0.4 documents preserve original scientific metadata, structured
authors, license, related identifiers and provenance. Unknown fields are not
invented. ``provider="nemar"`` identifies the service, while ``source`` remains
upstream-reported. The concept ``dataset_doi`` is separate from ``version_doi``
for the reported latest snapshot. For example, ``on000117`` reports NEMAR
``v1.0.0`` while its derived-from OpenNeuro DOI identifies ``ds000117.v1.1.0``.

``metadata_scope="current"`` does not mean immutable version-pinned metadata.
Catalog rows instead have ``metadata_scope="catalog"`` and retain their raw row
in ``nemar_catalog``. Both include ``metadata_url`` and UTC acquisition time
``metadata_retrieved_at``; this is not an upstream modification or atomic snapshot
time. Age bounds are not participant observations; missing modality is not EEG.

Unsupported operations
----------------------

Mongo/friendly filters (including license, modality, task and source), ``$in``,
counts, recordings, participants, signal loading, aggregations and writes are not
supported. Existing record/count/write methods raise ``NemarUnsupportedOperation``
rather than returning empty or misleading results. Absent participant/aggregation
methods raise ``AttributeError``. Unsupported ``skip``, ``q`` and historical
``version`` keywords raise ``TypeError``. Do not pass these metadata documents as
loadable records. Custom ``api_url``, non-default ``database`` or explicit
``auth_token`` cannot be combined with this backend. EEGDash environment
credentials, proxies and netrc authentication are not forwarded to NEMAR.

Failure behavior and limits
---------------------------

Each request has 5s connect/15s socket-read timeouts, a checked 30s streaming
deadline and a 2 MiB decoded response cap, with no redirects or automatic retries.
The deadline is checked between chunks; a blocked socket may extend wall time by
its read timeout. HTTP errors propagate, including 429; callers must honor
``Retry-After``. Malformed, degraded or explicitly partial metadata, duplicate IDs,
changing totals and inconsistent pagination raise ``NemarContractError``. No
partial list escapes after an error. Offset pagination is mutable and cannot
detect every concurrent replacement: it is not an atomic migration inventory.
There is no implicit cache, fallback, catalog completeness or recording-parity
claim. The existing ingestion mapper, records.json fast path and signal
downloader remain unchanged.

The detailed repository contract and tiny opt-in metadata test instructions are
available in `docs/nemar_backend.md
<https://github.com/EEGDash/EEGDash/blob/develop/docs/nemar_backend.md>`_.
