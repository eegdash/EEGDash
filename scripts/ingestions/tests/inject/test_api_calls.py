"""HTTP API integration tests for 5_inject.py — mocks API Gateway endpoints with respx."""

from __future__ import annotations

import json as _json

import httpx
import pytest
import respx
from _helpers import load_inject

from eegdash.testing import data_file

# ─── inject_datasets ──────────────────────────────────────────────────────


@respx.mock
def test_inject_datasets_posts_to_bulk_endpoint():
    """Single-dataset call hits /admin/<database>/datasets/bulk."""
    inject_mod = load_inject()
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
    """List larger than batch_size produces multiple POSTs."""
    inject_mod = load_inject()
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
    """admin_token appears as Bearer in the Authorization header."""
    inject_mod = load_inject()
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
    """POST body is a JSON list of dataset documents (after _sanitize_for_json)."""
    inject_mod = load_inject()
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

    body = _json.loads(route.calls.last.request.content)
    assert isinstance(body, list)
    assert len(body) == 1
    assert body[0]["dataset_id"] == "ds-001"


# ─── inject_records ───────────────────────────────────────────────────────


@respx.mock
def test_inject_records_uses_upsert_endpoint():
    """Records use /admin/<database>/records/upsert (upserted by composite key)."""
    inject_mod = load_inject()
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
    """Both inserted_count and updated_count propagate to the summary."""
    inject_mod = load_inject()
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
    assert result["inserted_count"] >= 7
    assert result["updated_count"] >= 3


# ─── inject_montages ──────────────────────────────────────────────────────


@respx.mock
def test_inject_montages_posts_to_bulk_endpoint():
    """Montages POST to /admin/<database>/montages/bulk."""
    inject_mod = load_inject()
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
    """End-to-end against the snapshot fixture with all 3 inject endpoints mocked."""
    inject_mod = load_inject()

    respx.post("https://api.example.com/admin/eegdash_dev/datasets/bulk").mock(
        return_value=httpx.Response(200, json={"inserted_count": 2})
    )
    respx.post("https://api.example.com/admin/eegdash_dev/records/upsert").mock(
        return_value=httpx.Response(200, json={"inserted_count": 4, "updated_count": 0})
    )
    respx.post("https://api.example.com/admin/eegdash_dev/montages/bulk").mock(
        return_value=httpx.Response(200, json={"inserted_count": 0})
    )

    respx.get(url__regex=r"https://api\.example\.com/api/eegdash_dev/datasets/.*").mock(
        return_value=httpx.Response(404)
    )

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
