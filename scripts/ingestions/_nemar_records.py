"""NEMAR ``records.json`` fast-path — exact upstream ``signal_summary``.

NEMAR serves, per published dataset version,
``GET https://data.nemar.org/<id>/<version>/records.json`` — a JSON array of
neuroschema ``record`` docs, one per primary signal file, each carrying a
``signal_summary`` block (``nchans``/``ntimes``/``recording_duration``/
``sampling_frequency``/``channel_type_counts``) resolved server-side
(sidecar-first → biosigIO header read → ``null``).

This module exposes a single lookup, :func:`signal_summary`, that the digest
cascade consults *first* for NEMAR datasets so it can use those exact values
instead of re-reading headers locally. It is intentionally tiny and total:

* gated to NEMAR ids (``_source_from_dataset_id == "nemar"``) and to the
  ``EEGDASH_NEMAR_RECORDS`` kill-switch — both checked *outside* the cache so
  the network is never touched for OpenNeuro/Zenodo or when disabled;
* the per-dataset fetch is memoised for the process lifetime (digest runs one
  dataset per subprocess, so the cache is effectively per-dataset);
* it **never raises** — any 404 / malformed body / network failure resolves to
  an empty index, and every field simply falls through to the existing cascade.

Version: the ``latest`` alias, so there is no version plumbing. A ``bids_relpath``
that is absent from ``latest`` (e.g. a renamed file in an older clone) is not
matched and falls through to the cascade — safe by construction.
"""

from __future__ import annotations

import os
from functools import cache

from _http import request_json
from _source_id import _source_from_dataset_id

BASE_URL = "https://data.nemar.org"
_ENV_DISABLE = "EEGDASH_NEMAR_RECORDS"


@cache
def _records_index(dataset_id: str) -> dict[str, dict]:
    """Fetch ``records.json`` once per dataset → ``{bids_relpath: signal_summary}``.

    Total: returns ``{}`` on any non-200, non-list body, or transport failure.
    """
    url = f"{BASE_URL}/{dataset_id}/latest/records.json"
    payload, response = request_json("GET", url, timeout=30.0, retries=3)
    if response is None or response.status_code != 200 or not isinstance(payload, list):
        return {}
    index: dict[str, dict] = {}
    for record in payload:
        if not isinstance(record, dict):
            continue
        relpath = record.get("bids_relpath")
        summary = record.get("signal_summary")
        if isinstance(relpath, str) and isinstance(summary, dict):
            index[relpath] = summary
    return index


def signal_summary(dataset_id: str, bids_relpath: str) -> dict | None:
    """Return NEMAR's ``signal_summary`` for one file, or ``None``.

    ``None`` for non-NEMAR ids, when ``EEGDASH_NEMAR_RECORDS=0``, or when the
    file is absent from ``records.json`` / the fetch failed.
    """
    if os.environ.get(_ENV_DISABLE) == "0":
        return None
    if _source_from_dataset_id(dataset_id) != "nemar":
        return None
    return _records_index(dataset_id).get(bids_relpath)


__all__ = ["signal_summary"]
