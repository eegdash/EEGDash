"""Layer 2 — Pydantic schema acceptance.

For every Record / Dataset / Montage in every snapshot fixture,
validate it against the producer-side Pydantic model in
``eegdash.schemas``. If a refactor renames a field or changes a
type, this layer catches it without needing a byte-snapshot refresh.
"""

from __future__ import annotations

from typing import Any

from eegdash.schemas import (
    DatasetModel,
    EntitiesModel,
    RecordModel,
    StorageModel,
)
from pydantic import ValidationError

# ─── Records ──────────────────────────────────────────────────────────────


def test_every_record_validates_against_RecordModel(snapshot_outputs):
    """Every Record in every snapshot fixture must validate.

    Catches: field renames, type changes (str → int), missing required
    fields, extra fields that the model doesn't expect (RecordModel
    uses ``extra="allow"`` so this last one is a soft check).
    """
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
    """The ``entities`` block (subject / session / task / run / acq) is
    optional but, when present, must satisfy ``EntitiesModel``."""
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        for idx, record in enumerate(payload["records"]["records"]):
            entities = record.get("entities")
            if entities is None:
                continue  # not all records carry entities (manifest-only path)
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
    "montage_hash",
    "n_channels",
    "channels",
    "first_seen",
    "representative_dataset",
)


def _skip_if_no_montages(snapshot_outputs: dict[str, Any]) -> None:
    """Skip-guard for the montage tests so vacuous passes are visible.

    Today both snapshot fixtures (`ds_snapshot_vhdr`, `ds_snapshot_manifest`)
    emit zero montages, so the iteration body of these tests never
    executes — silent vacuous pass. The skip converts that into a
    visible SKIPPED in pytest output, surfacing the coverage gap so
    a future fixture-bearing-montages task can replace it.
    """
    import pytest

    total = sum(
        len(payload["montages"]["montages"]) for payload in snapshot_outputs.values()
    )
    if total == 0:
        pytest.skip(
            "All snapshot fixtures emit zero montages — montage acceptance "
            "is not exercised. Add a fixture with a non-empty montage set "
            "(e.g. a real EEG dataset with electrodes.tsv) to re-engage."
        )


def test_every_montage_has_required_fields(snapshot_outputs):
    """Montages don't yet have a Pydantic model; assert the JSON keys
    we know the API consumer indexes on are present.

    Note: ``_montages.json`` serialises the in-memory ``dict[hash, doc]``
    as a flat list of docs (see ``record_enumerator.write_dataset_outputs``);
    the hash key is preserved inside each doc as ``montage_hash``.
    """
    _skip_if_no_montages(snapshot_outputs)
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        montages: list[dict[str, Any]] = payload["montages"]["montages"]
        for idx, doc in enumerate(montages):
            for field in _REQUIRED_MONTAGE_FIELDS:
                if field not in doc:
                    failures.append(f"{fixture_name}.montages[{idx}]: missing {field}")
    assert not failures, "Montage shape check failed:\n" + "\n".join(failures)


def test_every_montage_hash_keys_its_own_doc(snapshot_outputs):
    """The in-memory montages structure is ``dict[hash, doc]`` (the hash
    is the key); the JSON serialisation flattens that to a list of docs
    via ``list(result.montages.values())``. The dict→list-preserving
    invariant is that every doc carries a non-empty ``montage_hash``
    and all hashes within a dataset are unique (no key collisions on
    the original dict)."""
    _skip_if_no_montages(snapshot_outputs)
    failures: list[str] = []
    for fixture_name, payload in snapshot_outputs.items():
        montages: list[dict[str, Any]] = payload["montages"]["montages"]
        seen: set[str] = set()
        for idx, doc in enumerate(montages):
            embedded = doc.get("montage_hash")
            if not embedded:
                failures.append(
                    f"{fixture_name}.montages[{idx}]: empty montage_hash={embedded!r}"
                )
                continue
            if embedded in seen:
                failures.append(
                    f"{fixture_name}.montages[{idx}]: duplicate "
                    f"montage_hash={embedded!r} — original dict invariant broken"
                )
            seen.add(embedded)
    assert not failures, "Montage hash-key drift:\n" + "\n".join(failures)
