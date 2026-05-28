"""Layer 2 — Pydantic schema acceptance against ``eegdash.schemas``."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from eegdash.schemas import (
    DatasetModel,
    EntitiesModel,
    RecordModel,
    StorageModel,
)

# ─── Records ──────────────────────────────────────────────────────────────


def test_every_record_validates_against_RecordModel(snapshot_outputs):
    """Every Record in every snapshot fixture validates against RecordModel."""
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        for idx, record in enumerate(payload["records"]["records"]):
            try:
                RecordModel.model_validate(record)
            except ValidationError as exc:
                failures.append(
                    f"{fixture_name}[{idx}] ({record.get('bids_relpath', '?')}): {exc}"
                )
    assert not failures, "RecordModel rejected:\n" + "\n".join(failures)


def test_every_record_storage_validates_against_StorageModel(snapshot_outputs):
    """The nested ``storage`` block has its own Pydantic model."""
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        for idx, record in enumerate(payload["records"]["records"]):
            storage = record.get("storage")
            if storage is None:
                failures.append(f"{fixture_name}[{idx}]: missing storage block")
                continue
            try:
                StorageModel.model_validate(storage)
            except ValidationError as exc:
                failures.append(f"{fixture_name}[{idx}].storage: {exc}")
    assert not failures, "StorageModel rejected:\n" + "\n".join(failures)


def test_every_record_entities_validates_against_EntitiesModel(snapshot_outputs):
    """The ``entities`` block, when present, validates against EntitiesModel."""
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        for idx, record in enumerate(payload["records"]["records"]):
            entities = record.get("entities")
            if entities is None:
                continue
            try:
                EntitiesModel.model_validate(entities)
            except ValidationError as exc:
                failures.append(f"{fixture_name}[{idx}].entities: {exc}")
    assert not failures, "EntitiesModel rejected:\n" + "\n".join(failures)


# ─── Datasets ─────────────────────────────────────────────────────────────


def test_every_dataset_validates_against_DatasetModel(snapshot_outputs):
    """Every Dataset doc must validate against DatasetModel."""
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        try:
            DatasetModel.model_validate(payload["dataset"])
        except ValidationError as exc:
            failures.append(f"{fixture_name}: {exc}")
    assert not failures, "DatasetModel rejected:\n" + "\n".join(failures)


# ─── Montages (no Pydantic model — JSON-shape acceptance) ─────────────────

_REQUIRED_MONTAGE_FIELDS: tuple[str, ...] = (
    "hash",
    "n_sensors",
    "sensors",
    "modality",
    "first_seen",
    "representative_dataset",
)


def _skip_if_no_montages(snapshot_outputs: dict[str, Any]) -> None:
    """Skip montage tests when all fixtures produce zero montages."""

    total = sum(
        len(payload["montages"]["montages"]) for payload in snapshot_outputs.values()
    )
    if total == 0:
        pytest.skip("All snapshot fixtures emit zero montages.")


def test_every_montage_has_required_fields(snapshot_outputs):
    """Every montage doc carries the required JSON fields."""
    _skip_if_no_montages(snapshot_outputs)
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        montages: list[dict[str, Any]] = payload["montages"]["montages"]
        for idx, doc in enumerate(montages):
            for field in _REQUIRED_MONTAGE_FIELDS:
                if field not in doc:
                    failures.append(f"{fixture_name}.montages[{idx}]: missing {field}")
    assert not failures, "Montage shape check failed:\n" + "\n".join(failures)


def test_every_montage_hash_is_present_and_unique(snapshot_outputs):
    """Every montage has a non-empty ``hash`` and all hashes are unique per dataset."""
    _skip_if_no_montages(snapshot_outputs)
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        montages: list[dict[str, Any]] = payload["montages"]["montages"]
        seen: set[str] = set()
        for idx, doc in enumerate(montages):
            embedded = doc.get("hash")
            if not embedded:
                failures.append(
                    f"{fixture_name}.montages[{idx}]: empty hash={embedded!r}"
                )
                continue
            if embedded in seen:
                failures.append(
                    f"{fixture_name}.montages[{idx}]: duplicate hash={embedded!r}"
                )
            seen.add(embedded)
    assert not failures, "Montage hash drift:\n" + "\n".join(failures)


def test_every_record_montage_hash_references_an_existing_montage(
    snapshot_outputs,
) -> None:
    """Every Record.montage_hash references an existing montage in the same dataset."""
    _skip_if_no_montages(snapshot_outputs)
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        montages: list[dict[str, Any]] = payload["montages"]["montages"]
        known_hashes = {doc.get("hash") for doc in montages if doc.get("hash")}
        for idx, record in enumerate(payload["records"]["records"]):
            ref = record.get("montage_hash")
            if ref is None:
                continue  # not every record has a montage (e.g. behavioural)
            if ref not in known_hashes:
                failures.append(
                    f"{fixture_name}[{idx}] ({record.get('bids_relpath', '?')}): "
                    f"montage_hash={ref!r} not in known hashes {sorted(known_hashes)}"
                )
    assert not failures, "Record→Montage join drift:\n" + "\n".join(failures)
