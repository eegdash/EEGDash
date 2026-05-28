from __future__ import annotations

import json
import math
from pathlib import Path

import httpx
import pytest
import respx
from _helpers import load_inject

# ─── load_dataset / load_records / load_montages ──────────────────────────


def test_load_dataset_returns_none_for_missing_file(tmp_path: Path):
    ds_dir = tmp_path / "ds-001"
    ds_dir.mkdir()
    inject = load_inject()
    assert inject.load_dataset(ds_dir) is None


def test_load_dataset_reads_json(tmp_path: Path):
    ds_dir = tmp_path / "ds-001"
    ds_dir.mkdir()
    (ds_dir / "ds-001_dataset.json").write_text(
        json.dumps({"dataset_id": "ds-001", "name": "Test"})
    )
    inject = load_inject()
    out = inject.load_dataset(ds_dir)
    assert out["dataset_id"] == "ds-001"
    assert out["name"] == "Test"


@pytest.mark.parametrize(
    ("file_name", "content", "expected_len"),
    [
        pytest.param(
            "ds-001_records.json",
            {
                "dataset_id": "ds-001",
                "records": [
                    {"dataset": "ds-001", "bids_relpath": "f1.edf"},
                    {"dataset": "ds-001", "bids_relpath": "f2.edf"},
                ],
            },
            2,
            id="new_schema_envelope",
        ),
        pytest.param(
            "ds-001_records.json",
            [{"dataset": "ds-001", "bids_relpath": "f1.edf"}],
            1,
            id="bare_list",
        ),
        pytest.param(
            None,
            None,
            0,
            id="no_file",
        ),
        pytest.param(
            "ds-001_core.json",
            [{"dataset": "ds-001", "bids_relpath": "f1.edf"}],
            1,
            id="legacy_core_json_fallback",
        ),
        pytest.param(
            "ds-001_records.json",
            {"dataset_id": "ds-001", "garbage": True},
            0,
            id="malformed_returns_empty",
        ),
    ],
)
def test_load_records(tmp_path: Path, file_name, content, expected_len):
    ds_dir = tmp_path / "ds-001"
    ds_dir.mkdir()
    if file_name is not None:
        (ds_dir / file_name).write_text(json.dumps(content))
    inject = load_inject()
    records = inject.load_records(ds_dir)
    assert len(records) == expected_len


@pytest.mark.parametrize(
    ("content", "expected_len"),
    [
        pytest.param(
            {"montages": [{"hash": "abc"}, {"hash": "def"}]},
            2,
            id="envelope",
        ),
        pytest.param(
            [{"hash": "abc"}],
            1,
            id="bare_list",
        ),
        pytest.param(
            None,
            0,
            id="missing_file",
        ),
        pytest.param(
            {"montages": None},
            0,
            id="null_payload",
        ),
    ],
)
def test_load_montages(tmp_path: Path, content, expected_len):
    ds_dir = tmp_path / "ds-001"
    ds_dir.mkdir()
    if expected_len != 0 or content == {"montages": None}:
        (ds_dir / "ds-001_montages.json").write_text(json.dumps(content))
    inject = load_inject()
    montages = inject.load_montages(ds_dir)
    assert len(montages) == expected_len


# ─── _flatten_entities ─────────────────────────────────────────────────────


def test_flatten_entities_lifts_nested_keys():
    inject = load_inject()
    rec = {
        "dataset": "ds-001",
        "entities": {"subject": "01", "task": "rest", "session": "01", "run": "1"},
    }
    out = inject._flatten_entities(rec)
    assert out["subject"] == "01"
    assert out["task"] == "rest"
    assert out["session"] == "01"
    assert out["run"] == "1"
    assert "entities" not in out


def test_flatten_entities_top_level_wins_over_entity():
    inject = load_inject()
    rec = {
        "dataset": "ds-001",
        "subject": "01",  # top-level
        "entities": {"subject": "1", "task": "rest"},  # nested
    }
    out = inject._flatten_entities(rec)
    assert out["subject"] == "01"
    assert out["task"] == "rest"


def test_flatten_entities_no_entities_passes_through():
    inject = load_inject()
    rec = {"dataset": "ds-001", "subject": "01"}
    out = inject._flatten_entities(rec)
    assert out == rec


def test_flatten_entities_does_not_mutate_input():
    inject = load_inject()
    rec = {
        "dataset": "ds-001",
        "entities": {"subject": "01"},
    }
    original = rec.copy()
    inject._flatten_entities(rec)
    assert rec == original


# ─── _sanitize_for_json (NaN/Inf handling) ────────────────────────────────


@pytest.mark.parametrize(
    "bad_float",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive_inf"),
        pytest.param(float("-inf"), id="negative_inf"),
    ],
)
def test_sanitize_for_json_replaces_non_finite_float_with_none(bad_float):
    inject = load_inject()
    out = inject._sanitize_for_json({"field": bad_float})
    assert out["field"] is None


def test_sanitize_for_json_preserves_finite_floats():
    inject = load_inject()
    out = inject._sanitize_for_json({"sfreq": 500.0, "x": -3.14})
    assert out["sfreq"] == 500.0
    assert out["x"] == -3.14


def test_sanitize_for_json_recurses_into_nested_dicts():
    inject = load_inject()
    out = inject._sanitize_for_json(
        {"outer": {"inner_nan": math.nan, "inner_str": "shallow"}}
    )
    assert out["outer"]["inner_nan"] is None
    assert out["outer"]["inner_str"] == "shallow"


def test_sanitize_for_json_recurses_into_lists():
    inject = load_inject()
    out = inject._sanitize_for_json([1.0, math.nan, 3.0, float("inf")])
    assert out == [1.0, None, 3.0, None]


def test_sanitize_for_json_passes_through_non_float_primitives():
    inject = load_inject()
    assert inject._sanitize_for_json(42) == 42
    assert inject._sanitize_for_json("hello") == "hello"
    assert inject._sanitize_for_json(None) is None
    assert inject._sanitize_for_json(True) is True
    assert inject._sanitize_for_json(b"raw") == b"raw"


# ─── _ensure_fingerprint ──────────────────────────────────────────────────


def test_ensure_fingerprint_preserves_existing():
    inject = load_inject()
    ds = {"dataset_id": "ds-001", "ingestion_fingerprint": "existing-hash"}
    out = inject._ensure_fingerprint("ds-001", ds, [])
    assert out["ingestion_fingerprint"] == "existing-hash"


def test_ensure_fingerprint_creates_from_records():
    inject = load_inject()
    ds = {"dataset_id": "ds-001", "source": "openneuro"}
    records = [
        {"dataset": "ds-001", "bids_relpath": "f1.edf"},
        {"dataset": "ds-001", "bids_relpath": "f2.edf"},
    ]
    out = inject._ensure_fingerprint("ds-001", ds, records)
    assert "ingestion_fingerprint" in out
    assert isinstance(out["ingestion_fingerprint"], str)
    assert len(out["ingestion_fingerprint"]) > 0


def test_ensure_fingerprint_synthesises_dataset_when_none():
    inject = load_inject()
    out = inject._ensure_fingerprint(
        "ds-001",
        None,
        [{"dataset": "ds-001", "bids_relpath": "f1.edf"}],
    )
    assert out["dataset_id"] == "ds-001"
    assert "ingestion_fingerprint" in out


def test_ensure_fingerprint_idempotent():
    inject = load_inject()
    ds = {"dataset_id": "ds-001", "source": "openneuro"}
    records = [{"dataset": "ds-001", "bids_relpath": "f1.edf"}]
    out1 = inject._ensure_fingerprint("ds-001", ds.copy(), records)
    out2 = inject._ensure_fingerprint("ds-001", ds.copy(), records)
    assert out1["ingestion_fingerprint"] == out2["ingestion_fingerprint"]


# ─── fetch_existing_dataset ───────────────────────────────────────────────


@respx.mock
def test_fetch_existing_returns_dict_on_200():
    inject = load_inject()
    respx.get("https://api.example.com/api/eegdash_dev/datasets/ds-001").mock(
        return_value=httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "dataset_id": "ds-001",
                    "ingestion_fingerprint": "abc123",
                },
            },
        )
    )
    out = inject.fetch_existing_dataset(
        "https://api.example.com", "eegdash_dev", "ds-001"
    )
    assert out is not None
    assert out.get("ingestion_fingerprint") == "abc123"


@respx.mock
def test_fetch_existing_returns_none_on_404():
    inject = load_inject()
    respx.get("https://api.example.com/api/eegdash_dev/datasets/ds-missing").mock(
        return_value=httpx.Response(404)
    )
    out = inject.fetch_existing_dataset(
        "https://api.example.com", "eegdash_dev", "ds-missing"
    )
    assert out is None


@respx.mock
def test_fetch_existing_returns_none_on_network_error():
    inject = load_inject()
    respx.get("https://api.example.com/api/eegdash_dev/datasets/ds-001").mock(
        side_effect=httpx.ConnectError("network down")
    )
    out = inject.fetch_existing_dataset(
        "https://api.example.com", "eegdash_dev", "ds-001"
    )
    assert out is None


# ─── filter_changed_datasets ──────────────────────────────────────────────


@respx.mock
def test_filter_skips_unchanged_by_fingerprint():
    inject = load_inject()

    fp = "abc123def456"
    respx.get("https://api.example.com/api/eegdash_dev/datasets/ds-001").mock(
        return_value=httpx.Response(200, json={"data": {"ingestion_fingerprint": fp}})
    )

    datasets_by_id = {
        "ds-001": {
            "dataset_id": "ds-001",
            "ingestion_fingerprint": fp,
            "source": "openneuro",
        }
    }
    records_by_id = {"ds-001": []}

    changed, skipped = inject.filter_changed_datasets(
        ["ds-001"],
        datasets_by_id,
        records_by_id,
        "https://api.example.com",
        "eegdash_dev",
    )
    assert skipped == ["ds-001"]
    assert changed == []


@respx.mock
def test_filter_marks_changed_when_fingerprint_differs():
    inject = load_inject()

    respx.get("https://api.example.com/api/eegdash_dev/datasets/ds-001").mock(
        return_value=httpx.Response(
            200, json={"data": {"ingestion_fingerprint": "OLD_FP"}}
        )
    )

    datasets_by_id = {
        "ds-001": {
            "dataset_id": "ds-001",
            "ingestion_fingerprint": "NEW_FP",
            "source": "openneuro",
        }
    }
    records_by_id = {"ds-001": []}

    changed, skipped = inject.filter_changed_datasets(
        ["ds-001"],
        datasets_by_id,
        records_by_id,
        "https://api.example.com",
        "eegdash_dev",
    )
    assert changed == ["ds-001"]
    assert skipped == []


@respx.mock
def test_filter_marks_new_dataset_as_changed():
    inject = load_inject()

    respx.get("https://api.example.com/api/eegdash_dev/datasets/ds-new").mock(
        return_value=httpx.Response(404)
    )

    datasets_by_id = {
        "ds-new": {
            "dataset_id": "ds-new",
            "ingestion_fingerprint": "fresh",
            "source": "openneuro",
        }
    }
    records_by_id = {"ds-new": []}

    changed, skipped = inject.filter_changed_datasets(
        ["ds-new"],
        datasets_by_id,
        records_by_id,
        "https://api.example.com",
        "eegdash_dev",
    )
    assert changed == ["ds-new"]
    assert skipped == []


# ─── find_digested_datasets ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("setup_fn", "expected_names"),
    [
        pytest.param(
            lambda tmp: [
                (tmp / ds_id).mkdir()
                or (tmp / ds_id / f"{ds_id}_dataset.json").write_text("{}")
                for ds_id in ("ds-001", "ds-002", "ds-003")
            ],
            ["ds-001", "ds-002", "ds-003"],
            id="walks_input_dir",
        ),
        pytest.param(
            lambda tmp: [
                (tmp / "real").mkdir()
                or (tmp / "real" / "real_dataset.json").write_text("{}"),
                (tmp / "incomplete").mkdir(),
            ],
            ["real"],
            id="ignores_subdirs_without_dataset_json",
        ),
        pytest.param(
            lambda tmp: None,
            [],
            id="empty_input",
        ),
    ],
)
def test_find_digested_datasets(tmp_path: Path, setup_fn, expected_names):
    setup_fn(tmp_path)
    inject = load_inject()
    found = inject.find_digested_datasets(tmp_path)
    assert sorted(d.name for d in found) == sorted(expected_names)
