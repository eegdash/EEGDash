"""Unit tests for 5_inject.py helpers (C6.2 follow-up).

C6.2 covered the network-facing inject functions. This file covers
the pure helpers + filtering layer: load_dataset / load_records /
load_montages, _flatten_entities, _sanitize_for_json,
_ensure_fingerprint, filter_changed_datasets, find_digested_datasets.

No live cluster needed — uses tmp_path + respx for the
filter_changed_datasets fetch.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import respx

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))


def _load_inject():
    spec = importlib.util.spec_from_file_location(
        "_inject_helpers_target", _INGEST_DIR / "5_inject.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── load_dataset / load_records / load_montages ──────────────────────────


def test_load_dataset_returns_none_for_missing_file(tmp_path: Path):
    ds_dir = tmp_path / "ds-001"
    ds_dir.mkdir()
    inject = _load_inject()
    assert inject.load_dataset(ds_dir) is None


def test_load_dataset_reads_json(tmp_path: Path):
    ds_dir = tmp_path / "ds-001"
    ds_dir.mkdir()
    (ds_dir / "ds-001_dataset.json").write_text(
        json.dumps({"dataset_id": "ds-001", "name": "Test"})
    )
    inject = _load_inject()
    out = inject.load_dataset(ds_dir)
    assert out["dataset_id"] == "ds-001"
    assert out["name"] == "Test"


def test_load_records_reads_new_schema_records_key(tmp_path: Path):
    """New schema: ``{records: [...]}`` envelope."""
    ds_dir = tmp_path / "ds-001"
    ds_dir.mkdir()
    payload = {
        "dataset_id": "ds-001",
        "records": [
            {"dataset": "ds-001", "bids_relpath": "f1.edf"},
            {"dataset": "ds-001", "bids_relpath": "f2.edf"},
        ],
    }
    (ds_dir / "ds-001_records.json").write_text(json.dumps(payload))
    inject = _load_inject()
    records = inject.load_records(ds_dir)
    assert len(records) == 2


def test_load_records_reads_bare_list(tmp_path: Path):
    """Some pipelines write a bare JSON list (no envelope)."""
    ds_dir = tmp_path / "ds-001"
    ds_dir.mkdir()
    (ds_dir / "ds-001_records.json").write_text(
        json.dumps([{"dataset": "ds-001", "bids_relpath": "f1.edf"}])
    )
    inject = _load_inject()
    records = inject.load_records(ds_dir)
    assert len(records) == 1


def test_load_records_returns_empty_for_no_file(tmp_path: Path):
    ds_dir = tmp_path / "ds-001"
    ds_dir.mkdir()
    inject = _load_inject()
    assert inject.load_records(ds_dir) == []


def test_load_records_legacy_core_json_fallback(tmp_path: Path):
    """Falls back to ``<id>_core.json`` if ``_records.json`` is missing."""
    ds_dir = tmp_path / "ds-001"
    ds_dir.mkdir()
    (ds_dir / "ds-001_core.json").write_text(
        json.dumps([{"dataset": "ds-001", "bids_relpath": "f1.edf"}])
    )
    inject = _load_inject()
    records = inject.load_records(ds_dir)
    assert len(records) == 1


def test_load_records_malformed_returns_empty(tmp_path: Path):
    """A JSON object without ``records`` and not a list → empty result."""
    ds_dir = tmp_path / "ds-001"
    ds_dir.mkdir()
    (ds_dir / "ds-001_records.json").write_text(
        json.dumps({"dataset_id": "ds-001", "garbage": True})
    )
    inject = _load_inject()
    assert inject.load_records(ds_dir) == []


def test_load_montages_envelope(tmp_path: Path):
    ds_dir = tmp_path / "ds-001"
    ds_dir.mkdir()
    (ds_dir / "ds-001_montages.json").write_text(
        json.dumps({"montages": [{"hash": "abc"}, {"hash": "def"}]})
    )
    inject = _load_inject()
    montages = inject.load_montages(ds_dir)
    assert len(montages) == 2


def test_load_montages_bare_list(tmp_path: Path):
    ds_dir = tmp_path / "ds-001"
    ds_dir.mkdir()
    (ds_dir / "ds-001_montages.json").write_text(json.dumps([{"hash": "abc"}]))
    inject = _load_inject()
    montages = inject.load_montages(ds_dir)
    assert len(montages) == 1


def test_load_montages_missing_file_returns_empty(tmp_path: Path):
    ds_dir = tmp_path / "ds-001"
    ds_dir.mkdir()
    inject = _load_inject()
    assert inject.load_montages(ds_dir) == []


def test_load_montages_envelope_with_null_montages(tmp_path: Path):
    """``{"montages": null}`` → empty list (not None)."""
    ds_dir = tmp_path / "ds-001"
    ds_dir.mkdir()
    (ds_dir / "ds-001_montages.json").write_text(json.dumps({"montages": None}))
    inject = _load_inject()
    assert inject.load_montages(ds_dir) == []


# ─── _flatten_entities ─────────────────────────────────────────────────────


def test_flatten_entities_lifts_nested_keys():
    """Subject/task/session/run lifted from entities dict to top level."""
    inject = _load_inject()
    rec = {
        "dataset": "ds-001",
        "entities": {"subject": "01", "task": "rest", "session": "01", "run": "1"},
    }
    out = inject._flatten_entities(rec)
    assert out["subject"] == "01"
    assert out["task"] == "rest"
    assert out["session"] == "01"
    assert out["run"] == "1"
    # entities dict removed
    assert "entities" not in out


def test_flatten_entities_top_level_wins_over_entity():
    """If subject exists at top level AND in entities, top-level wins.

    Pins the documented conflict-resolution contract — preserves
    explicitly-set digester output."""
    inject = _load_inject()
    rec = {
        "dataset": "ds-001",
        "subject": "01",  # top-level
        "entities": {"subject": "1", "task": "rest"},  # nested
    }
    out = inject._flatten_entities(rec)
    assert out["subject"] == "01", "top-level subject should win"
    assert out["task"] == "rest"


def test_flatten_entities_no_entities_passes_through():
    """A record without ``entities`` is returned unchanged."""
    inject = _load_inject()
    rec = {"dataset": "ds-001", "subject": "01"}
    out = inject._flatten_entities(rec)
    assert out == rec


def test_flatten_entities_does_not_mutate_input():
    """The input dict is NOT mutated; flatten returns a new dict."""
    inject = _load_inject()
    rec = {
        "dataset": "ds-001",
        "entities": {"subject": "01"},
    }
    original = rec.copy()
    inject._flatten_entities(rec)
    assert rec == original, "input was mutated"


# ─── _sanitize_for_json (NaN/Inf handling) ────────────────────────────────


def test_sanitize_for_json_replaces_nan_with_none():
    """NaN is not valid JSON; the digest can produce it via failed
    floats. Sanitiser swaps it for None."""
    import math

    inject = _load_inject()
    out = inject._sanitize_for_json({"sampling_frequency": math.nan})
    assert out["sampling_frequency"] is None


def test_sanitize_for_json_replaces_positive_inf_with_none():
    inject = _load_inject()
    out = inject._sanitize_for_json({"duration": float("inf")})
    assert out["duration"] is None


def test_sanitize_for_json_replaces_negative_inf_with_none():
    inject = _load_inject()
    out = inject._sanitize_for_json({"value": float("-inf")})
    assert out["value"] is None


def test_sanitize_for_json_preserves_finite_floats():
    """Real numbers pass through untouched (no precision loss)."""
    inject = _load_inject()
    out = inject._sanitize_for_json({"sfreq": 500.0, "x": -3.14})
    assert out["sfreq"] == 500.0
    assert out["x"] == -3.14


def test_sanitize_for_json_recurses_into_nested_dicts():
    """A NaN deeply nested in a sub-dict still gets replaced."""
    import math

    inject = _load_inject()
    out = inject._sanitize_for_json(
        {"outer": {"inner_nan": math.nan, "inner_str": "shallow"}}
    )
    assert out["outer"]["inner_nan"] is None
    assert out["outer"]["inner_str"] == "shallow"


def test_sanitize_for_json_recurses_into_lists():
    """A NaN inside a list is replaced; other items pass through."""
    import math

    inject = _load_inject()
    out = inject._sanitize_for_json([1.0, math.nan, 3.0, float("inf")])
    assert out == [1.0, None, 3.0, None]


def test_sanitize_for_json_passes_through_non_float_primitives():
    """Strings, ints, bools, None — all unchanged. Bytes pass through
    too (the sanitiser doesn't decode them — that's caller's job)."""
    inject = _load_inject()
    assert inject._sanitize_for_json(42) == 42
    assert inject._sanitize_for_json("hello") == "hello"
    assert inject._sanitize_for_json(None) is None
    assert inject._sanitize_for_json(True) is True
    # bytes pass through — pin the actual behaviour (not decoded here)
    assert inject._sanitize_for_json(b"raw") == b"raw"


# ─── _ensure_fingerprint ──────────────────────────────────────────────────


def test_ensure_fingerprint_preserves_existing():
    """If dataset already has ingestion_fingerprint, leave it."""
    inject = _load_inject()
    ds = {"dataset_id": "ds-001", "ingestion_fingerprint": "existing-hash"}
    out = inject._ensure_fingerprint("ds-001", ds, [])
    assert out["ingestion_fingerprint"] == "existing-hash"


def test_ensure_fingerprint_creates_from_records():
    """Missing fingerprint + non-empty records → derive."""
    inject = _load_inject()
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
    """If dataset is None, build a stub with dataset_id."""
    inject = _load_inject()
    out = inject._ensure_fingerprint(
        "ds-001",
        None,
        [{"dataset": "ds-001", "bids_relpath": "f1.edf"}],
    )
    assert out["dataset_id"] == "ds-001"
    assert "ingestion_fingerprint" in out


def test_ensure_fingerprint_idempotent():
    """Re-running ensure_fingerprint twice → same hash."""
    inject = _load_inject()
    ds = {"dataset_id": "ds-001", "source": "openneuro"}
    records = [{"dataset": "ds-001", "bids_relpath": "f1.edf"}]
    out1 = inject._ensure_fingerprint("ds-001", ds.copy(), records)
    out2 = inject._ensure_fingerprint("ds-001", ds.copy(), records)
    assert out1["ingestion_fingerprint"] == out2["ingestion_fingerprint"]


# ─── fetch_existing_dataset ───────────────────────────────────────────────


@respx.mock
def test_fetch_existing_returns_dict_on_200():
    inject = _load_inject()
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
    inject = _load_inject()
    respx.get("https://api.example.com/api/eegdash_dev/datasets/ds-missing").mock(
        return_value=httpx.Response(404)
    )
    out = inject.fetch_existing_dataset(
        "https://api.example.com", "eegdash_dev", "ds-missing"
    )
    assert out is None


@respx.mock
def test_fetch_existing_returns_none_on_network_error():
    """A network exception → None (not a crash)."""
    inject = _load_inject()
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
    """If the API returns the same fingerprint, the dataset is skipped."""
    inject = _load_inject()

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
    """Different fingerprints → dataset is changed → injected."""
    inject = _load_inject()

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
    """Dataset not yet in API → always changed (inject it)."""
    inject = _load_inject()

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


def test_find_digested_datasets_walks_input_dir(tmp_path: Path):
    """Each sub-directory containing ``<id>_dataset.json`` is one
    dataset directory."""
    inject = _load_inject()
    for ds_id in ("ds-001", "ds-002", "ds-003"):
        sub = tmp_path / ds_id
        sub.mkdir()
        (sub / f"{ds_id}_dataset.json").write_text("{}")

    found = inject.find_digested_datasets(tmp_path)
    found_names = sorted(d.name for d in found)
    assert found_names == ["ds-001", "ds-002", "ds-003"]


def test_find_digested_datasets_ignores_subdirs_without_dataset_json(
    tmp_path: Path,
):
    """A subdirectory without the marker file is skipped."""
    inject = _load_inject()
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "real_dataset.json").write_text("{}")
    (tmp_path / "incomplete").mkdir()  # no _dataset.json
    found = inject.find_digested_datasets(tmp_path)
    assert sorted(d.name for d in found) == ["real"]


def test_find_digested_datasets_empty_input(tmp_path: Path):
    inject = _load_inject()
    assert inject.find_digested_datasets(tmp_path) == []
