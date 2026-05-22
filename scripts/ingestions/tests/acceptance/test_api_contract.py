"""Layer 4 — API contract acceptance.

The cluster's API in ``mongodb-eegdash-server/api/main.py`` defines
the consumer's effective contract. This layer imports the API's
Pydantic models from that file and validates every snapshot
fixture's records / datasets against them.

Distinct from Layer 2 (producer-side models in ``eegdash.schemas``)
because the two repos have independently-evolving models with the
same class names. Layer 2 catches producer drift; Layer 4 catches
producer/consumer drift.

The API module also defines a Beanie ``Record(Document)`` with
indexes on specific fields — we don't instantiate Beanie (needs a
MongoDB connection) but we DO assert the indexed fields exist on
every Record.
"""

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
    """Lazy-load ``mongodb-eegdash-server/api/main.py`` for its Pydantic models.

    Skips the layer entirely when the API repo isn't checked out as
    a sibling — CI without it shouldn't fail this gate.
    """
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
    except Exception as exc:  # pragma: no cover - defensive
        pytest.skip(f"API module raised at import time: {exc!r}")
    return mod


# ─── Pydantic models on the API side ──────────────────────────────────────


def test_every_record_validates_against_api_RecordModel(snapshot_outputs, api_module):
    """The API's ``RecordModel`` (api/main.py:138) is the response-shape
    Pydantic model. Producer output must round-trip through it."""
    RecordModel = getattr(api_module, "RecordModel", None)
    if RecordModel is None:
        pytest.skip(
            "API module does not expose RecordModel (response-shape "
            "Pydantic model). Layer 4 RecordModel check skipped."
        )
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
    """The API's ``DatasetModel`` (api/main.py:158) is the response-shape
    Pydantic model for Dataset queries. Producer output must validate."""
    DatasetModel = getattr(api_module, "DatasetModel", None)
    if DatasetModel is None:
        pytest.skip(
            "API module does not expose DatasetModel (response-shape "
            "Pydantic model). Layer 4 DatasetModel check skipped."
        )
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


# The API's Beanie Document (api/main.py:142) declares indexes on:
#   - data_name
#   - dataset
#   - (dataset, subject) compound
#   - (dataset, bids_relpath) compound — the canonical upsert key
# If any of these fields go missing from a Record, the upsert path
# breaks. Encode them here.
#
# NOTE on ``subject``: the digest stage emits ``subject`` nested inside
# ``entities``; the inject stage (``5_inject._flatten_entities``) lifts
# it to the top level just before bulk-upsert. We check BOTH locations
# so the snapshot fixtures (pre-flatten) and the live cluster docs
# (post-flatten) both satisfy the contract.
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
    """Every indexed field on the API's Beanie Record document must be
    reachable (either top level or in ``entities``) so the bulk-upsert
    key works. ``subject`` is lifted from ``entities.subject`` at
    inject time by ``5_inject._flatten_entities``."""
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
    """The Record's ``dataset`` field MUST equal the parent Dataset's
    ``dataset_id``. This is the compound-index canonical join key."""
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
    """``(dataset, bids_relpath)`` is the bulk-upsert canonical key.
    Duplicate ``bids_relpath`` within a dataset would silently overwrite
    on inject — surface it here."""
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
