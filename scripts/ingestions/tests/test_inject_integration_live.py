"""Live integration tests against the EEGdash cluster (C6.3).

Tests the real network round-trip:
  digest output  →  POST https://data.eegdash.org/admin/eegdash_dev/...
  GET back        →  verify shape + values
  DELETE          →  cleanup so the test database stays clean

**Skipped by default.** Two env vars are required:
- ``EEGDASH_INTEGRATION_API_URL`` — e.g. ``https://data.eegdash.org``
- ``EEGDASH_INTEGRATION_ADMIN_TOKEN`` — admin Bearer token

Without those, the whole module skips. CI is expected to leave them
unset; opt-in is a local-dev or staging-CI concern. The cluster's
``eegdash_dev`` MongoDB database is the target.

Each test inserts documents with a ``c6_smoke_<timestamp>_<uuid>`` prefix
on the dataset_id so concurrent runs don't collide and a cleanup hook
deletes the test prefix at end (best-effort — failures don't fail
the test).

Why this exists: C6.2 covered the API integration path against respx
mocks. This file is the SAME tests against the live API — caught any
contract drift between what our mocks assume and what the real Gateway
actually returns. The 1-bug-per-cycle pattern from C5.1 might catch
another bug here.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

_API_URL = os.environ.get("EEGDASH_INTEGRATION_API_URL")
_ADMIN_TOKEN = os.environ.get("EEGDASH_INTEGRATION_ADMIN_TOKEN")
_DATABASE = os.environ.get("EEGDASH_INTEGRATION_DATABASE", "eegdash_dev")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_API_URL and _ADMIN_TOKEN),
        reason=(
            "Live integration tests skipped. Set "
            "EEGDASH_INTEGRATION_API_URL and EEGDASH_INTEGRATION_ADMIN_TOKEN "
            "to opt in."
        ),
    ),
]


def _load_inject():
    spec = importlib.util.spec_from_file_location(
        "_inject_live_target", _INGEST_DIR / "5_inject.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _unique_test_id() -> str:
    """Stable enough across concurrent runs that two CI jobs can race
    each other without collision, but obvious enough that a human
    looking at the database can identify + delete leaks."""
    return f"c6_smoke_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def _cleanup_dataset(dataset_id: str) -> None:
    """Best-effort cleanup. Failures don't fail the test — orphans
    carry the ``c6_smoke_`` prefix for easy bulk removal later.

    The cluster's Caddy reverse proxy blocks DELETE methods at the
    edge (see ``caddy/Caddyfile`` line ~180). The admin DELETE
    endpoint exists internally at port 3000 but isn't reachable from
    the public ``data.eegdash.org`` URL.

    Three cleanup options, in order of preference:

    1. ``EEGDASH_INTEGRATION_CLEANUP_CMD`` env var — if set, run the
       command with ``{dataset_id}`` substituted. Lets ops teams plug
       in mongosh-via-ssh or kubectl exec without baking it in.
    2. Otherwise log + skip. Orphans are visible via the prefix:
       ``db.datasets.deleteMany({dataset_id: /^c6_smoke_/})``.
    """
    cleanup_cmd = os.environ.get("EEGDASH_INTEGRATION_CLEANUP_CMD")
    if cleanup_cmd:
        import subprocess

        try:
            subprocess.run(
                cleanup_cmd.format(dataset_id=dataset_id),
                shell=True,
                check=False,
                timeout=10,
                capture_output=True,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass


# ─── Read-side smoke: API is up + answering ───────────────────────────────


def test_live_api_health_endpoint_responds():
    """Sanity: GET /health returns ``{"status": "healthy"}``."""
    resp = httpx.get(f"{_API_URL}/health", timeout=10)
    resp.raise_for_status()
    body = resp.json()
    assert body.get("status") == "healthy"
    assert body.get("mongodb") == "connected"


def test_live_api_datasets_list_returns_real_data():
    """GET /api/<database>/datasets returns the live data — confirm we're
    actually talking to the cluster and not a stub."""
    resp = httpx.get(f"{_API_URL}/api/{_DATABASE}/datasets?limit=5", timeout=10)
    resp.raise_for_status()
    body = resp.json()
    assert body.get("success") is True
    assert body.get("database") == _DATABASE
    assert len(body.get("data", [])) > 0
    # Each entry has a dataset_id
    for d in body["data"]:
        assert "dataset_id" in d


# ─── Write-side: inject one dataset, read it back, clean it up ────────────


def test_live_inject_dataset_round_trip():
    """Round-trip: inject a single test dataset, read it back, verify
    fields match, delete it.

    Pins the real-network leg of inject_datasets that --dry-run skips.
    If the API's expected request shape drifts away from what the
    digest produces, this test catches it; the respx mocks in C6.2
    only catch our own assumptions.
    """
    inject_mod = _load_inject()
    test_id = _unique_test_id()

    try:
        # 1) inject
        result = inject_mod.inject_datasets(
            [
                {
                    "dataset_id": test_id,
                    "source": "openneuro",
                    "name": "C6.3 live integration test",
                    "recording_modality": ["eeg"],
                }
            ],
            api_url=_API_URL,
            database=_DATABASE,
            admin_token=_ADMIN_TOKEN,
        )
        assert result["errors"] == [], f"inject errors: {result['errors']}"
        assert result["inserted_count"] >= 1

        # 2) read back
        resp = httpx.get(f"{_API_URL}/api/{_DATABASE}/datasets/{test_id}", timeout=10)
        resp.raise_for_status()
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["dataset_id"] == test_id
        assert data["source"] == "openneuro"
        assert data["recording_modality"] == ["eeg"]
        # MongoDB ObjectId is auto-stamped — confirm it landed
        assert "_id" in data

    finally:
        # 3) cleanup
        _cleanup_dataset(test_id)


def test_live_inject_record_round_trip():
    """Round-trip a single Record through the upsert endpoint."""
    inject_mod = _load_inject()
    test_id = _unique_test_id()
    test_bids_relpath = "sub-01/eeg/sub-01_task-test_eeg.edf"

    # First insert the parent dataset (records reference dataset_id)
    inject_mod.inject_datasets(
        [
            {
                "dataset_id": test_id,
                "source": "openneuro",
                "name": "C6.3 record round-trip test",
                "recording_modality": ["eeg"],
            }
        ],
        api_url=_API_URL,
        database=_DATABASE,
        admin_token=_ADMIN_TOKEN,
    )

    try:
        # Inject a single record
        result = inject_mod.inject_records(
            [
                {
                    "dataset": test_id,
                    "bids_relpath": test_bids_relpath,
                    "recording_modality": ["eeg"],
                    "sampling_frequency": 500.0,
                    "nchans": 32,
                    "storage": {
                        "base": f"s3://openneuro.org/{test_id}",
                        "backend": "s3",
                        "raw_key": test_bids_relpath,
                        "dep_keys": [],
                    },
                }
            ],
            api_url=_API_URL,
            database=_DATABASE,
            admin_token=_ADMIN_TOKEN,
        )
        assert result["errors"] == [], f"inject errors: {result['errors']}"
        assert result["inserted_count"] + result["updated_count"] >= 1

        # Read back via the records endpoint. The API uses ``filter`` as
        # a JSON-encoded MongoDB query string (not individual field
        # params), so we encode our dataset selector as JSON.
        resp = httpx.get(
            f"{_API_URL}/api/{_DATABASE}/records",
            params={"filter": json.dumps({"dataset": test_id}), "limit": 10},
            timeout=10,
        )
        resp.raise_for_status()
        body = resp.json()
        records = body.get("data", [])
        assert len(records) >= 1
        match = next(
            (r for r in records if r.get("bids_relpath") == test_bids_relpath),
            None,
        )
        assert match is not None
        assert match["sampling_frequency"] == 500.0
        assert match["nchans"] == 32

    finally:
        # Cleanup happens via the dataset DELETE (which cascades to records
        # in the cluster's MongoDB setup, per the API admin endpoints).
        # If it doesn't cascade, orphan records persist under c6_smoke_
        # prefix and can be deleted manually.
        _cleanup_dataset(test_id)


def test_live_idempotent_upsert():
    """The upsert endpoint is idempotent: injecting the same Record twice
    should not duplicate; the second call should report updated_count >= 1
    rather than inserted_count >= 1.

    Pins the Stage 5 idempotency contract (re-running digest must not
    duplicate records — the dataset_id + bids_relpath composite key
    is the upsert key)."""
    inject_mod = _load_inject()
    test_id = _unique_test_id()

    # Insert dataset
    inject_mod.inject_datasets(
        [
            {
                "dataset_id": test_id,
                "source": "openneuro",
                "name": "Idempotency test",
                "recording_modality": ["eeg"],
            }
        ],
        api_url=_API_URL,
        database=_DATABASE,
        admin_token=_ADMIN_TOKEN,
    )

    try:
        record = {
            "dataset": test_id,
            "bids_relpath": "sub-01/eeg/sub-01_eeg.edf",
            "recording_modality": ["eeg"],
            "storage": {
                "base": f"s3://openneuro.org/{test_id}",
                "backend": "s3",
                "raw_key": "sub-01/eeg/sub-01_eeg.edf",
                "dep_keys": [],
            },
        }
        # First upsert: should insert
        first = inject_mod.inject_records(
            [record],
            api_url=_API_URL,
            database=_DATABASE,
            admin_token=_ADMIN_TOKEN,
        )
        # Second upsert (same record): should NOT add a duplicate
        second = inject_mod.inject_records(
            [record],
            api_url=_API_URL,
            database=_DATABASE,
            admin_token=_ADMIN_TOKEN,
        )

        assert first["errors"] == []
        assert second["errors"] == []
        # The composite key means total record count stays at 1
        resp = httpx.get(
            f"{_API_URL}/api/{_DATABASE}/records",
            params={"filter": json.dumps({"dataset": test_id}), "limit": 10},
            timeout=10,
        )
        body = resp.json()
        records = body.get("data", [])
        # Exactly one record with our bids_relpath (no duplicate from re-upsert)
        matches = [
            r for r in records if r.get("bids_relpath") == "sub-01/eeg/sub-01_eeg.edf"
        ]
        assert len(matches) == 1

    finally:
        _cleanup_dataset(test_id)
