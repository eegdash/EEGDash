"""HTTP API integration tests for 5_inject.py (C6.2).

Stage 5's actual upsert path was never tested — only ``--dry-run``.
This file mocks the API Gateway endpoints with respx and verifies the
inject functions send the right requests with the right payloads.

This is the equivalent of "MongoDB integration tests" because the
ingest pipeline doesn't talk to MongoDB directly — it goes through
the EEGdash API Gateway which writes to MongoDB on its behalf.
"""

from __future__ import annotations

import importlib.util
import json as _json

import httpx
import pytest
import respx
from _helpers import INGEST_DIR as _INGEST_DIR

from eegdash.testing import data_file


def _load_inject():
    """Lazy-load 5_inject.py (digit-prefixed filename forces this)."""
    spec = importlib.util.spec_from_file_location(
        "_inject_target", _INGEST_DIR / "5_inject.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── inject_datasets ──────────────────────────────────────────────────────


@respx.mock
def test_inject_datasets_posts_to_bulk_endpoint():
    """A single-dataset call hits /admin/<database>/datasets/bulk with
    the dataset list as the JSON body."""
    inject_mod = _load_inject()
    route = respx.post("https://api.example.com/admin/eegdash_dev/datasets/bulk").mock(
        return_value=httpx.Response(200, json={"inserted_count": 1})
    )

    datasets = [
        {
            "dataset_id": "ds-001",
            "source": "openneuro",
            "name": "Test",
            "recording_modality": ["eeg"],
        }
    ]
    result = inject_mod.inject_datasets(
        datasets,
        api_url="https://api.example.com",
        database="eegdash_dev",
        admin_token="fake-token",
    )

    assert route.called
    assert result["inserted_count"] == 1
    assert result["errors"] == []


@respx.mock
def test_inject_datasets_batches_at_batch_size_boundary():
    """A list larger than batch_size produces multiple POSTs."""
    inject_mod = _load_inject()
    route = respx.post("https://api.example.com/admin/eegdash_dev/datasets/bulk").mock(
        return_value=httpx.Response(200, json={"inserted_count": 50})
    )

    datasets = [
        {
            "dataset_id": f"ds-{i:03d}",
            "source": "openneuro",
            "name": "T",
            "recording_modality": ["eeg"],
        }
        for i in range(75)  # > batch_size=50
    ]
    result = inject_mod.inject_datasets(
        datasets,
        api_url="https://api.example.com",
        database="eegdash_dev",
        admin_token="fake-token",
        batch_size=50,
    )
    # 75 items / batch 50 = 2 calls
    assert route.call_count == 2
    assert result["inserted_count"] == 100  # 2 x 50 (mock returns 50 each)
    assert result["errors"] == []


@respx.mock
def test_inject_datasets_sends_authorization_header():
    """The admin_token shows up as Bearer in the Authorization header."""
    inject_mod = _load_inject()
    route = respx.post("https://api.example.com/admin/eegdash_dev/datasets/bulk").mock(
        return_value=httpx.Response(200, json={"inserted_count": 1})
    )

    inject_mod.inject_datasets(
        [
            {
                "dataset_id": "ds-001",
                "source": "openneuro",
                "name": "T",
                "recording_modality": ["eeg"],
            }
        ],
        api_url="https://api.example.com",
        database="eegdash_dev",
        admin_token="my-secret-token",
    )

    assert route.called
    headers = route.calls.last.request.headers
    assert "Authorization" in headers
    assert "my-secret-token" in headers["Authorization"]


@respx.mock
def test_inject_datasets_request_body_is_the_list_payload():
    """The POST body is a JSON list with the dataset documents
    (after _sanitize_for_json)."""

    inject_mod = _load_inject()
    route = respx.post("https://api.example.com/admin/eegdash_dev/datasets/bulk").mock(
        return_value=httpx.Response(200, json={"inserted_count": 1})
    )

    datasets = [
        {
            "dataset_id": "ds-001",
            "source": "openneuro",
            "name": "Test name",
            "recording_modality": ["eeg"],
        }
    ]
    inject_mod.inject_datasets(
        datasets,
        api_url="https://api.example.com",
        database="eegdash_dev",
        admin_token="fake-token",
    )

    # Read the body that respx captured
    body = _json.loads(route.calls.last.request.content)
    assert isinstance(body, list)
    assert len(body) == 1
    assert body[0]["dataset_id"] == "ds-001"


# ─── inject_records ───────────────────────────────────────────────────────


@respx.mock
def test_inject_records_uses_upsert_endpoint():
    """Records use /admin/<database>/records/upsert (different endpoint
    from datasets/bulk, because Records are upserted by composite key)."""
    inject_mod = _load_inject()
    route = respx.post("https://api.example.com/admin/eegdash_dev/records/upsert").mock(
        return_value=httpx.Response(200, json={"inserted_count": 1, "updated_count": 0})
    )

    records = [
        {
            "dataset": "ds-001",
            "bids_relpath": "sub-01/eeg/sub-01_eeg.edf",
            "recording_modality": ["eeg"],
            "storage": {
                "base": "s3://openneuro.org/ds-001",
                "backend": "s3",
                "raw_key": "sub-01/eeg/sub-01_eeg.edf",
                "dep_keys": [],
            },
        }
    ]
    result = inject_mod.inject_records(
        records,
        api_url="https://api.example.com",
        database="eegdash_dev",
        admin_token="fake-token",
        batch_size=1000,
    )
    assert route.called
    assert result["inserted_count"] >= 1
    assert result["errors"] == []


@respx.mock
def test_inject_records_reports_both_inserted_and_updated():
    """The upsert response distinguishes new from existing — both
    counts propagate to the summary."""
    inject_mod = _load_inject()
    respx.post("https://api.example.com/admin/eegdash_dev/records/upsert").mock(
        return_value=httpx.Response(200, json={"inserted_count": 7, "updated_count": 3})
    )

    records = [
        {
            "dataset": "ds-001",
            "bids_relpath": f"sub-{i:02d}/eeg/sub-{i:02d}_eeg.edf",
            "recording_modality": ["eeg"],
            "storage": {
                "base": "s3://openneuro.org/ds-001",
                "backend": "s3",
                "raw_key": f"sub-{i:02d}/eeg/sub-{i:02d}_eeg.edf",
                "dep_keys": [],
            },
        }
        for i in range(10)
    ]
    result = inject_mod.inject_records(
        records,
        api_url="https://api.example.com",
        database="eegdash_dev",
        admin_token="fake-token",
    )
    # The mock returns 7+3; result should reflect at least these counts
    assert result["inserted_count"] >= 7
    assert result["updated_count"] >= 3


# ─── inject_montages ──────────────────────────────────────────────────────


@respx.mock
def test_inject_montages_posts_to_bulk_endpoint():
    """Montages use /admin/<database>/montages/bulk."""
    inject_mod = _load_inject()
    route = respx.post("https://api.example.com/admin/eegdash_dev/montages/bulk").mock(
        return_value=httpx.Response(200, json={"inserted_count": 1})
    )

    montages = [
        {
            "hash": "abc123def456",
            "modality": "eeg",
            "n_sensors": 32,
            "sensors": [{"name": "Cz", "x": 0.0, "y": 0.0, "z": 0.1}],
        }
    ]
    result = inject_mod.inject_montages(
        montages,
        api_url="https://api.example.com",
        database="eegdash_dev",
        admin_token="fake-token",
    )
    assert route.called
    assert result["inserted_count"] == 1


# ─── Full e2e: digest → inject (mocked) ───────────────────────────────────


@respx.mock
def test_inject_runs_against_snapshot_with_mocked_api():
    """End-to-end against the committed snapshot fixture, with all 3
    inject endpoints mocked. Verifies that running 5_inject.py without
    --dry-run actually exercises the full request pipeline."""
    inject_mod = _load_inject()

    # Mock all 3 endpoints
    respx.post("https://api.example.com/admin/eegdash_dev/datasets/bulk").mock(
        return_value=httpx.Response(200, json={"inserted_count": 2})
    )
    respx.post("https://api.example.com/admin/eegdash_dev/records/upsert").mock(
        return_value=httpx.Response(200, json={"inserted_count": 4, "updated_count": 0})
    )
    respx.post("https://api.example.com/admin/eegdash_dev/montages/bulk").mock(
        return_value=httpx.Response(200, json={"inserted_count": 0})
    )

    # Also mock the per-dataset fetch endpoint that inject uses to
    # check for existing fingerprints
    respx.get(url__regex=r"https://api\.example\.com/api/eegdash_dev/datasets/.*").mock(
        return_value=httpx.Response(404)
    )

    # Load the snapshot Records + Datasets + Montages
    snapshot_root = data_file("digest_snapshots/outputs")
    if not snapshot_root.exists():
        pytest.skip("snapshot fixture not available")

    datasets, records, montages = [], [], []
    for ds_dir in snapshot_root.iterdir():
        ds_path = ds_dir / f"{ds_dir.name}_dataset.json"
        rec_path = ds_dir / f"{ds_dir.name}_records.json"
        mon_path = ds_dir / f"{ds_dir.name}_montages.json"
        if ds_path.exists():
            datasets.append(_json.loads(ds_path.read_text()))
        if rec_path.exists():
            records.extend(_json.loads(rec_path.read_text()).get("records", []))
        if mon_path.exists():
            payload = _json.loads(mon_path.read_text())
            if isinstance(payload, dict):
                montages.extend(payload.get("montages", []))
            elif isinstance(payload, list):
                montages.extend(payload)

    # Inject them against the mocked API
    ds_result = inject_mod.inject_datasets(
        datasets,
        api_url="https://api.example.com",
        database="eegdash_dev",
        admin_token="fake-token",
    )
    rec_result = inject_mod.inject_records(
        records,
        api_url="https://api.example.com",
        database="eegdash_dev",
        admin_token="fake-token",
    )

    assert ds_result["errors"] == []
    assert rec_result["errors"] == []
    assert ds_result["inserted_count"] >= 2
    assert rec_result["inserted_count"] >= 4
