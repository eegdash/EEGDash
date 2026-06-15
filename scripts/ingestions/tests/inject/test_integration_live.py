"""Live integration tests against the EEGdash cluster (C6.3).

Skipped unless EEGDASH_INTEGRATION_API_URL and EEGDASH_INTEGRATION_ADMIN_TOKEN are set.
Each test uses a ``c6_smoke_<timestamp>_<uuid>`` dataset_id prefix to avoid collisions.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid

import httpx
import pytest
from _helpers import load_inject

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


def _unique_test_id() -> str:
    """Collision-safe test dataset id with an obvious ``c6_smoke_`` prefix for leak cleanup."""
    return f"c6_smoke_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def _cleanup_dataset(dataset_id: str) -> None:
    """Best-effort cleanup via EEGDASH_INTEGRATION_CLEANUP_CMD env var.

    Failures don't fail the test. Orphans carry the ``c6_smoke_`` prefix
    for easy bulk removal: ``db.datasets.deleteMany({dataset_id: /^c6_smoke_/})``.
    """
    cleanup_cmd = os.environ.get("EEGDASH_INTEGRATION_CLEANUP_CMD")
    if cleanup_cmd:
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
    """GET /health returns ``{"status": "healthy"}``."""
    resp = httpx.get(f"{_API_URL}/health", timeout=10)
    resp.raise_for_status()
    body = resp.json()
    assert body.get("status") == "healthy"
    assert body.get("mongodb") == "connected"


def test_live_api_datasets_list_returns_real_data():
    """GET /api/<database>/datasets returns live data from the cluster."""
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
    """Inject a single dataset, read it back, verify fields, then delete it."""
    inject_mod = load_inject()
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

        resp = httpx.get(f"{_API_URL}/api/{_DATABASE}/datasets/{test_id}", timeout=10)
        resp.raise_for_status()
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["dataset_id"] == test_id
        assert data["source"] == "openneuro"
        assert data["recording_modality"] == ["eeg"]
        assert "_id" in data

    finally:
        _cleanup_dataset(test_id)


def test_live_inject_record_round_trip():
    """Round-trip a single Record through the upsert endpoint."""
    inject_mod = load_inject()
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
        _cleanup_dataset(test_id)


def test_live_idempotent_upsert():
    """Injecting the same Record twice must not duplicate it (composite-key upsert)."""
    inject_mod = load_inject()
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
        first = inject_mod.inject_records(
            [record],
            api_url=_API_URL,
            database=_DATABASE,
            admin_token=_ADMIN_TOKEN,
        )
        second = inject_mod.inject_records(
            [record],
            api_url=_API_URL,
            database=_DATABASE,
            admin_token=_ADMIN_TOKEN,
        )

        assert first["errors"] == []
        assert second["errors"] == []
        resp = httpx.get(
            f"{_API_URL}/api/{_DATABASE}/records",
            params={"filter": json.dumps({"dataset": test_id}), "limit": 10},
            timeout=10,
        )
        body = resp.json()
        records = body.get("data", [])
        matches = [
            r for r in records if r.get("bids_relpath") == "sub-01/eeg/sub-01_eeg.edf"
        ]
        assert len(matches) == 1

    finally:
        _cleanup_dataset(test_id)
