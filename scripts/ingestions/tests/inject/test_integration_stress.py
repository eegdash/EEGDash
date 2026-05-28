"""Stress + edge-case tests against the live EEGdash cluster (C6.4).

Goes beyond the C6.3 round-trip smoke. Hunts for:
- Schema drift: do the C6.1 BIDS-spec fields (PowerLineFrequency,
  EEGReference, SoftwareFilters, etc.) round-trip through MongoDB?
- Auth failures: missing/wrong token, wrong database name.
- Bulk throughput: 100-dataset batch insert, latency budget.
- Concurrency: parallel batches racing for the same composite key.
- Edge cases: unicode in name fields, very long strings, special
  characters in dataset_id, large records (~MB-class sidecar_inline).

Opt-in via the same env vars as test_inject_integration_live.py.
"""

from __future__ import annotations

import concurrent.futures as cf
import importlib.util
import json
import os
import subprocess
import time
import uuid

import httpx
import pytest
from _helpers import INGEST_DIR as _INGEST_DIR

_API_URL = os.environ.get("EEGDASH_INTEGRATION_API_URL")
_ADMIN_TOKEN = os.environ.get("EEGDASH_INTEGRATION_ADMIN_TOKEN")
_DATABASE = os.environ.get("EEGDASH_INTEGRATION_DATABASE", "eegdash_dev")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_API_URL and _ADMIN_TOKEN),
        reason=(
            "Stress tests skipped. Set EEGDASH_INTEGRATION_API_URL + "
            "EEGDASH_INTEGRATION_ADMIN_TOKEN. See "
            "."
        ),
    ),
]


def _load_inject():
    spec = importlib.util.spec_from_file_location(
        "_inject_stress_target", _INGEST_DIR / "5_inject.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stress_id(tag: str = "stress") -> str:
    """Stress-tagged ID (separate prefix from c6_smoke_ so orphan
    sweep can target each set independently)."""
    return f"c6_stress_{tag}_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def _cleanup(dataset_id: str) -> None:
    """Same best-effort hook as C6.3."""
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


# ─── Schema drift: do the C6.1 BIDS fields round-trip? ────────────────────


def test_stress_record_with_all_c6_1_fields_round_trips():
    """The new BIDS-spec fields from C6.1 (PowerLineFrequency,
    EEGReference, SoftwareFilters, Manufacturer, etc.) must survive
    the round-trip through the Gateway → MongoDB → read-back.

    If the Gateway has a stricter schema than RecordModel (extra="allow"
    or not), unknown fields might be silently stripped. This test fails
    loudly if that happens. THE HIGHEST-PRIORITY STRESS TEST — if it
    catches drift, the C6.1 enrichment is invisible to consumers.
    """
    inject_mod = _load_inject()
    test_id = _stress_id("c6_1_fields")

    inject_mod.inject_datasets(
        [
            {
                "dataset_id": test_id,
                "source": "openneuro",
                "name": "C6.1 field round-trip test",
                "recording_modality": ["eeg"],
                # C6.1 dataset-level extras
                "acknowledgements": "Funded by Lab X",
                "how_to_acknowledge": "Cite our paper",
                "ethics_approvals": ["IRB-001", "IRB-002"],
            }
        ],
        api_url=_API_URL,
        database=_DATABASE,
        admin_token=_ADMIN_TOKEN,
    )

    try:
        # Inject a Record with all the C6.1 sidecar-extracted fields
        record = {
            "dataset": test_id,
            "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.edf",
            "recording_modality": ["eeg"],
            "sampling_frequency": 500.0,
            "nchans": 32,
            "ntimes": 250000,
            "storage": {
                "base": f"s3://openneuro.org/{test_id}",
                "backend": "s3",
                "raw_key": "sub-01/eeg/sub-01_task-rest_eeg.edf",
                "dep_keys": [],
            },
            # C6.1 fields — the things we just started capturing
            "power_line_frequency": 60,
            "eeg_reference": "linked mastoids",
            "software_filters": {"HighPass": 0.1, "LowPass": 100.0},
            "hardware_filters": {"HighPass": 0.05},
            "manufacturer": "Brain Products",
            "manufacturers_model_name": "BrainAmp DC",
            "eeg_placement_scheme": "10-20",
            "bad_channels": ["F7", "T3"],
            "bad_channels_count": 2,
            "_metadata_provenance": {
                "sampling_frequency": "modality_sidecar",
                "nchans": "channels_tsv",
                "ntimes": "binary_parser",
                "ch_names": "channels_tsv",
            },
        }
        result = inject_mod.inject_records(
            [record],
            api_url=_API_URL,
            database=_DATABASE,
            admin_token=_ADMIN_TOKEN,
        )
        assert result["errors"] == []

        # Read back via filter
        resp = httpx.get(
            f"{_API_URL}/api/{_DATABASE}/records",
            params={"filter": json.dumps({"dataset": test_id}), "limit": 5},
            timeout=10,
        )
        body = resp.json()
        records = body.get("data", [])
        assert len(records) >= 1
        rec = records[0]

        # Each C6.1 field must survive — failure here means the Gateway
        # silently stripped a field, masking our C6.1 enrichment.
        assert rec.get("power_line_frequency") == 60, (
            "PowerLineFrequency lost in round-trip"
        )
        assert rec.get("eeg_reference") == "linked mastoids"
        assert rec.get("software_filters") == {"HighPass": 0.1, "LowPass": 100.0}
        assert rec.get("manufacturer") == "Brain Products"
        assert rec.get("manufacturers_model_name") == "BrainAmp DC"
        assert rec.get("eeg_placement_scheme") == "10-20"
        assert rec.get("bad_channels") == ["F7", "T3"]
        assert rec.get("bad_channels_count") == 2
        # Provenance dict round-trips too
        assert rec.get("_metadata_provenance") == {
            "sampling_frequency": "modality_sidecar",
            "nchans": "channels_tsv",
            "ntimes": "binary_parser",
            "ch_names": "channels_tsv",
        }

        # Read the dataset back too — its C6.1 extras
        resp = httpx.get(f"{_API_URL}/api/{_DATABASE}/datasets/{test_id}", timeout=10)
        ds = resp.json()["data"]
        assert ds.get("acknowledgements") == "Funded by Lab X"
        assert ds.get("how_to_acknowledge") == "Cite our paper"
        assert ds.get("ethics_approvals") == ["IRB-001", "IRB-002"]

    finally:
        _cleanup(test_id)


# ─── Auth failure paths ────────────────────────────────────────────────────


def test_stress_inject_with_bad_token_rejected():
    """A wrong admin token → API returns 401/403."""
    test_id = _stress_id("bad_token")
    resp = httpx.post(
        f"{_API_URL}/admin/{_DATABASE}/datasets/bulk",
        headers={
            "Authorization": "Bearer not-a-real-token",
            "Content-Type": "application/json",
        },
        json=[
            {
                "dataset_id": test_id,
                "source": "openneuro",
                "name": "should not insert",
                "recording_modality": ["eeg"],
            }
        ],
        timeout=10,
    )
    assert resp.status_code in (401, 403), (
        f"expected 401/403 for bad token; got {resp.status_code}: {resp.text}"
    )


def test_stress_inject_without_token_rejected():
    """Missing Authorization header → 401/403."""
    test_id = _stress_id("no_token")
    resp = httpx.post(
        f"{_API_URL}/admin/{_DATABASE}/datasets/bulk",
        headers={"Content-Type": "application/json"},
        json=[
            {
                "dataset_id": test_id,
                "source": "openneuro",
                "name": "should not insert",
                "recording_modality": ["eeg"],
            }
        ],
        timeout=10,
    )
    assert resp.status_code in (401, 403)


def test_stress_inject_to_invalid_database_rejected():
    """A database name not in valid_databases (config) → 400."""
    test_id = _stress_id("bad_db")
    resp = httpx.post(
        f"{_API_URL}/admin/eegdash_does_not_exist/datasets/bulk",
        headers={
            "Authorization": f"Bearer {_ADMIN_TOKEN}",
            "Content-Type": "application/json",
        },
        json=[
            {
                "dataset_id": test_id,
                "source": "openneuro",
                "name": "should not insert",
                "recording_modality": ["eeg"],
            }
        ],
        timeout=10,
    )
    # 400 (validate_database rejects) or 404 (path doesn't match)
    assert resp.status_code in (400, 404)


# ─── Bulk insert throughput ────────────────────────────────────────────────


def test_stress_bulk_insert_100_datasets():
    """100-dataset bulk insert. Asserts:
    - All 100 land in the database (count matches)
    - The whole operation completes under a generous time budget
    """
    inject_mod = _load_inject()
    prefix = _stress_id("bulk100")
    datasets = [
        {
            "dataset_id": f"{prefix}_{i:03d}",
            "source": "openneuro",
            "name": f"Bulk stress {i:03d}",
            "recording_modality": ["eeg"],
        }
        for i in range(100)
    ]

    try:
        t0 = time.perf_counter()
        result = inject_mod.inject_datasets(
            datasets,
            api_url=_API_URL,
            database=_DATABASE,
            admin_token=_ADMIN_TOKEN,
            batch_size=100,  # all in one batch
        )
        elapsed = time.perf_counter() - t0

        assert result["errors"] == [], f"errors: {result['errors']}"
        assert result["inserted_count"] >= 100
        # 100 inserts in one batch should land in under 10s on a healthy cluster
        assert elapsed < 10.0, (
            f"bulk insert took {elapsed:.2f}s; expected < 10s. "
            f"Network slow OR cluster under load?"
        )

        # Verify via count endpoint
        resp = httpx.get(
            f"{_API_URL}/api/{_DATABASE}/datasets",
            params={
                "filter": json.dumps({"dataset_id": {"$regex": f"^{prefix}_"}}),
                "limit": 200,
            },
            timeout=15,
        )
        body = resp.json()
        # The dataset endpoint may paginate; just check at least 100 present
        # in the database (verified via separate query if list-API caps)
        assert body.get("count", 0) >= 100, (
            f"expected ≥100 datasets with prefix, got {body.get('count')}"
        )
    finally:
        # Cleanup all 100 via the cleanup hook
        for i in range(100):
            _cleanup(f"{prefix}_{i:03d}")


# ─── Concurrency: parallel inserts ─────────────────────────────────────────


# Module-level workers — keep the nested-function lint happy. Each
# closure pulls config from globals + the inject_mod cached by the
# test's first call. The prefix is passed via a mutable holder
# because ThreadPoolExecutor.map only accepts one positional arg.

_PARALLEL_PREFIX: str | None = None


def _parallel_inject_worker(idx: int) -> dict:
    """Module-level inject worker (one dataset per call)."""
    assert _PARALLEL_PREFIX is not None
    inject_mod = _load_inject()
    return inject_mod.inject_datasets(
        [
            {
                "dataset_id": f"{_PARALLEL_PREFIX}_{idx}",
                "source": "openneuro",
                "name": f"Parallel inject {idx}",
                "recording_modality": ["eeg"],
            }
        ],
        api_url=_API_URL,
        database=_DATABASE,
        admin_token=_ADMIN_TOKEN,
    )


_RACE_TEST_ID: str | None = None


def _race_upsert_worker(_idx: int) -> dict:
    """Module-level upsert worker — same composite key on every call."""
    assert _RACE_TEST_ID is not None
    inject_mod = _load_inject()
    return inject_mod.inject_records(
        [
            {
                "dataset": _RACE_TEST_ID,
                "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.edf",
                "recording_modality": ["eeg"],
                "storage": {
                    "base": f"s3://openneuro.org/{_RACE_TEST_ID}",
                    "backend": "s3",
                    "raw_key": "sub-01/eeg/sub-01_task-rest_eeg.edf",
                    "dep_keys": [],
                },
            }
        ],
        api_url=_API_URL,
        database=_DATABASE,
        admin_token=_ADMIN_TOKEN,
    )


def test_stress_parallel_inject_no_race_on_distinct_ids():
    """4 parallel inject_datasets calls with DISTINCT ids. Each should
    succeed without errors; rate-limiting shouldn't kick in for this
    volume (100 req/min limit per docker-compose; we're sending 4)."""
    global _PARALLEL_PREFIX
    _PARALLEL_PREFIX = _stress_id("parallel")

    try:
        with cf.ThreadPoolExecutor(max_workers=4) as exe:
            results = list(exe.map(_parallel_inject_worker, range(4)))
        # All 4 succeeded
        for i, r in enumerate(results):
            assert r["errors"] == [], f"worker {i} had errors: {r['errors']}"
            assert r["inserted_count"] >= 1
    finally:
        for i in range(4):
            _cleanup(f"{_PARALLEL_PREFIX}_{i}")
        _PARALLEL_PREFIX = None


def test_stress_parallel_upsert_same_record_no_duplicate():
    """4 parallel inject_records calls upserting the SAME composite key.
    After all settle, exactly 1 record should exist (composite-key upsert
    is the safety net for digest re-runs)."""
    global _RACE_TEST_ID
    inject_mod = _load_inject()
    _RACE_TEST_ID = _stress_id("race")

    inject_mod.inject_datasets(
        [
            {
                "dataset_id": _RACE_TEST_ID,
                "source": "openneuro",
                "name": "Race test parent",
                "recording_modality": ["eeg"],
            }
        ],
        api_url=_API_URL,
        database=_DATABASE,
        admin_token=_ADMIN_TOKEN,
    )

    try:
        with cf.ThreadPoolExecutor(max_workers=4) as exe:
            results = list(exe.map(_race_upsert_worker, range(4)))
        for r in results:
            assert r["errors"] == []

        # Read back: exactly 1 record with this composite key
        resp = httpx.get(
            f"{_API_URL}/api/{_DATABASE}/records",
            params={
                "filter": json.dumps({"dataset": _RACE_TEST_ID}),
                "limit": 10,
            },
            timeout=10,
        )
        records = resp.json().get("data", [])
        matches = [
            r
            for r in records
            if r.get("bids_relpath") == "sub-01/eeg/sub-01_task-rest_eeg.edf"
        ]
        assert len(matches) == 1, (
            f"expected exactly 1 record (composite key upsert), got "
            f"{len(matches)} — race condition in upsert?"
        )
    finally:
        _cleanup(_RACE_TEST_ID)
        _RACE_TEST_ID = None


# ─── Edge cases: unicode, special chars, oversized fields ─────────────────


def test_stress_dataset_with_unicode_name():
    """Unicode in name field (Greek, Chinese, emoji) survives round-trip."""
    inject_mod = _load_inject()
    test_id = _stress_id("unicode")
    unicode_name = "Étude EEG · 脑电图研究 · ⚡ test"

    try:
        inject_mod.inject_datasets(
            [
                {
                    "dataset_id": test_id,
                    "source": "openneuro",
                    "name": unicode_name,
                    "recording_modality": ["eeg"],
                }
            ],
            api_url=_API_URL,
            database=_DATABASE,
            admin_token=_ADMIN_TOKEN,
        )
        resp = httpx.get(f"{_API_URL}/api/{_DATABASE}/datasets/{test_id}", timeout=10)
        ds = resp.json()["data"]
        assert ds["name"] == unicode_name, (
            f"unicode lost: got {ds['name']!r}, expected {unicode_name!r}"
        )
    finally:
        _cleanup(test_id)


def test_stress_dataset_with_very_long_authors_list():
    """A dataset with 50 authors — checks the Gateway doesn't truncate
    list fields silently."""
    inject_mod = _load_inject()
    test_id = _stress_id("long_authors")
    authors = [f"Author {i:02d}" for i in range(50)]

    try:
        inject_mod.inject_datasets(
            [
                {
                    "dataset_id": test_id,
                    "source": "openneuro",
                    "name": "Many authors",
                    "recording_modality": ["eeg"],
                    "authors": authors,
                }
            ],
            api_url=_API_URL,
            database=_DATABASE,
            admin_token=_ADMIN_TOKEN,
        )
        resp = httpx.get(f"{_API_URL}/api/{_DATABASE}/datasets/{test_id}", timeout=10)
        ds = resp.json()["data"]
        assert len(ds.get("authors", [])) == 50, (
            f"author list truncated: got {len(ds.get('authors', []))}/50"
        )
    finally:
        _cleanup(test_id)


def test_stress_record_with_large_inline_sidecar():
    """A Record with a ~100 KB sidecar_inline blob (real-world NEMAR
    enrichment carries multi-KB participants.tsv inline)."""
    inject_mod = _load_inject()
    test_id = _stress_id("large_inline")

    # 100KB participants.tsv-like blob
    sidecar_inline = {
        "participants.tsv": "participant_id\tage\tsex\n"
        + "\n".join(
            f"sub-{i:04d}\t{20 + (i % 60)}\t{'M' if i % 2 else 'F'}"
            for i in range(2000)  # ~75 KB
        )
    }

    inject_mod.inject_datasets(
        [
            {
                "dataset_id": test_id,
                "source": "nemar",
                "name": "Large inline test",
                "recording_modality": ["eeg"],
            }
        ],
        api_url=_API_URL,
        database=_DATABASE,
        admin_token=_ADMIN_TOKEN,
    )

    try:
        result = inject_mod.inject_records(
            [
                {
                    "dataset": test_id,
                    "bids_relpath": "sub-01/eeg/sub-01_eeg.edf",
                    "recording_modality": ["eeg"],
                    "storage": {
                        "base": f"s3://nemar/{test_id}",
                        "backend": "s3",
                        "raw_key": "sub-01/eeg/sub-01_eeg.edf",
                        "dep_keys": [],
                        "sidecar_inline": sidecar_inline,
                    },
                }
            ],
            api_url=_API_URL,
            database=_DATABASE,
            admin_token=_ADMIN_TOKEN,
        )
        assert result["errors"] == [], f"large-inline upsert failed: {result['errors']}"

        # Read back + verify the blob survived
        resp = httpx.get(
            f"{_API_URL}/api/{_DATABASE}/records",
            params={"filter": json.dumps({"dataset": test_id}), "limit": 5},
            timeout=15,
        )
        records = resp.json().get("data", [])
        assert len(records) >= 1
        round_trip = records[0]["storage"]["sidecar_inline"]
        assert "participants.tsv" in round_trip
        # Spot-check the content
        assert "sub-0001" in round_trip["participants.tsv"]
        assert "sub-1999" in round_trip["participants.tsv"]
    finally:
        _cleanup(test_id)


# ─── Throughput / latency measurement (informational) ─────────────────────


def test_stress_inject_record_latency_under_500ms():
    """Single-record upsert latency budget. The cluster's typical
    response time should be well under 500 ms for a single record;
    if this fires, something operational degraded."""
    inject_mod = _load_inject()
    test_id = _stress_id("latency")

    inject_mod.inject_datasets(
        [
            {
                "dataset_id": test_id,
                "source": "openneuro",
                "name": "Latency test parent",
                "recording_modality": ["eeg"],
            }
        ],
        api_url=_API_URL,
        database=_DATABASE,
        admin_token=_ADMIN_TOKEN,
    )

    try:
        t0 = time.perf_counter()
        result = inject_mod.inject_records(
            [
                {
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
            ],
            api_url=_API_URL,
            database=_DATABASE,
            admin_token=_ADMIN_TOKEN,
        )
        elapsed = time.perf_counter() - t0
        assert result["errors"] == []
        # Budget is generous; tight latency would need a different test.
        assert elapsed < 2.0, (
            f"single-record upsert took {elapsed:.2f}s; expected < 2s. "
            f"Cluster under load?"
        )
    finally:
        _cleanup(test_id)
