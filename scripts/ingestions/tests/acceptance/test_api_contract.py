"""Layer 4 — API contract acceptance against mongodb-eegdash-server/api/main.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# Path to the API module, sibling of the eegdash repo.
#   parents[0]=acceptance, [1]=tests, [2]=ingestions, [3]=scripts,
#   [4]=eegdash repo root → then sibling ``mongodb-eegdash-server/api``.
_API_DIR = Path(__file__).resolve().parents[4] / "mongodb-eegdash-server" / "api"


@pytest.fixture(scope="session")
def api_module():
    """Lazy-load ``mongodb-eegdash-server/api/main.py``; skip if not checked out."""
    if not _API_DIR.exists():
        pytest.skip(
            f"API repo not found at {_API_DIR}. "
            f"Clone mongodb-eegdash-server as a sibling of this repo "
            f"to enable Layer 4 acceptance."
        )

    if str(_API_DIR) not in sys.path:
        sys.path.insert(0, str(_API_DIR))

    spec = importlib.util.spec_from_file_location(
        "api_main_under_test", _API_DIR / "main.py"
    )
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not load {_API_DIR / 'main.py'}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as exc:
        pytest.skip(f"API module has unmet imports (likely beanie/motor): {exc}")
    except Exception as exc:
        pytest.skip(f"API module raised at import time: {exc!r}")
    return mod


# ─── Pydantic models on the API side ──────────────────────────────────────


def test_every_record_validates_against_api_RecordModel(snapshot_outputs, api_module):
    """Producer Records must round-trip through the API's ``RecordModel`` (skips if absent)."""
    RecordModel = getattr(api_module, "RecordModel", None)
    if RecordModel is None:
        pytest.skip("API module does not expose RecordModel.")
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        for idx, record in enumerate(payload["records"]["records"]):
            try:
                RecordModel.model_validate(record)
            except ValidationError as exc:
                failures.append(
                    f"{fixture_name}[{idx}] ({record.get('bids_relpath', '?')}): {exc}"
                )
    assert not failures, "API RecordModel rejected producer output:\n" + "\n".join(
        failures
    )


def test_every_dataset_validates_against_api_DatasetModel(snapshot_outputs, api_module):
    """Producer Datasets must validate against the API's ``DatasetModel`` (skips if absent)."""
    DatasetModel = getattr(api_module, "DatasetModel", None)
    if DatasetModel is None:
        pytest.skip("API module does not expose DatasetModel.")
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        try:
            DatasetModel.model_validate(payload["dataset"])
        except ValidationError as exc:
            failures.append(f"{fixture_name}: {exc}")
    assert not failures, "API DatasetModel rejected producer output:\n" + "\n".join(
        failures
    )


# ─── Beanie Document index acceptance ─────────────────────────────────────

_API_INDEXED_RECORD_TOP_FIELDS: tuple[str, ...] = (
    "data_name",
    "dataset",
    "bids_relpath",
)
_API_INDEXED_RECORD_ENTITY_FIELDS: tuple[str, ...] = ("subject",)


def _record_has_field(record: dict, field: str) -> bool:
    """Return True if ``field`` is reachable at top level OR in ``entities``."""
    if field in record:
        return True
    entities = record.get("entities")
    if isinstance(entities, dict) and field in entities:
        return True
    return False


def test_every_record_has_api_indexed_fields(snapshot_outputs):
    """All API-indexed fields are present (top level or in ``entities``) on every Record."""
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        for idx, record in enumerate(payload["records"]["records"]):
            for field in _API_INDEXED_RECORD_TOP_FIELDS:
                if field not in record:
                    failures.append(
                        f"{fixture_name}[{idx}] ({record.get('bids_relpath', '?')}): "
                        f"missing top-level API-indexed field {field!r}"
                    )
            for field in _API_INDEXED_RECORD_ENTITY_FIELDS:
                if not _record_has_field(record, field):
                    failures.append(
                        f"{fixture_name}[{idx}] ({record.get('bids_relpath', '?')}): "
                        f"missing API-indexed entity field {field!r} "
                        f"(checked top level + entities)"
                    )
    assert not failures, "API index drift:\n" + "\n".join(failures)


def test_record_dataset_field_matches_outer_dataset_id(snapshot_outputs):
    """Record.dataset matches its parent Dataset.dataset_id (compound-index join key)."""
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        ds_id = payload["dataset"].get("dataset_id")
        for idx, record in enumerate(payload["records"]["records"]):
            if record.get("dataset") != ds_id:
                failures.append(
                    f"{fixture_name}[{idx}]: record.dataset={record.get('dataset')!r} "
                    f"vs dataset.dataset_id={ds_id!r}"
                )
    assert not failures, "Record/Dataset join key drift:\n" + "\n".join(failures)


def test_bids_relpath_is_unique_within_dataset(snapshot_outputs):
    """``bids_relpath`` is unique within each dataset (bulk-upsert key)."""
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        seen: set[str] = set()
        for idx, record in enumerate(payload["records"]["records"]):
            rel = record.get("bids_relpath")
            if rel is None:
                continue
            if rel in seen:
                failures.append(
                    f"{fixture_name}[{idx}]: duplicate bids_relpath={rel!r}"
                )
            seen.add(rel)
    assert not failures, "Bulk-upsert key collision:\n" + "\n".join(failures)
