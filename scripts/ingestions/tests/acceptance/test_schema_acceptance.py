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


# Verified against the producer (_montage.py:_extract_tsv_layout) and
# the live ds_snapshot_eeg_montage fixture. The doc carries `hash`
# (content-derived; this is what Record.montage_hash REFERENCES — the
# `montage_hash` name only appears on the Record side), `n_sensors`,
# and `sensors` (electrode array). Provenance fields stamped by
# `_attach_montage_to_record`: `first_seen`, `representative_dataset`.
# Modality identifier (`eeg` / `meg` / `ieeg` / `fnirs`) lets the
# consumer pick the right downstream renderer.
_REQUIRED_MONTAGE_FIELDS: tuple[str, ...] = (
    "hash",
    "n_sensors",
    "sensors",
    "modality",
    "first_seen",
    "representative_dataset",
)


def _skip_if_no_montages(snapshot_outputs: dict[str, Any]) -> None:
    """Skip-guard for the montage tests so vacuous passes are visible.

    The ``ds_snapshot_eeg_montage`` fixture produces a 19-electrode
    10-20 layout montage, so this skip should NOT fire today — kept
    as a defence so a future fixture cleanup that accidentally
    removes the electrodes.tsv would surface as SKIPPED rather than
    silent vacuous pass.
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

    ``_montages.json`` serialises the in-memory ``dict[hash, doc]`` as
    a flat list via ``record_enumerator.write_dataset_outputs``. Field
    names below match the producer in ``_montage.py``: ``hash`` (NOT
    ``montage_hash`` — that's the REFERENCING field on Records),
    ``n_sensors``, ``sensors``, ``modality``, plus the provenance
    stamps from ``_attach_montage_to_record``.
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


def test_every_montage_hash_is_present_and_unique(snapshot_outputs):
    """The in-memory montages structure is ``dict[hash, doc]`` (the hash
    is the dict key); the JSON serialisation flattens that to a list of
    docs via ``list(result.montages.values())``. The dict → list-
    preserving invariant: every doc carries a non-empty ``hash`` and
    all hashes within a dataset are unique (no key collisions on the
    original dict — Python dicts can't have duplicate keys, so a
    duplicate in the list means the producer is bypassing the dict).
    """
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
                    f"{fixture_name}.montages[{idx}]: duplicate "
                    f"hash={embedded!r} — original dict invariant broken"
                )
            seen.add(embedded)
    assert not failures, "Montage hash drift:\n" + "\n".join(failures)


def test_every_record_montage_hash_references_an_existing_montage(
    snapshot_outputs,
) -> None:
    """The producer/consumer join: every Record's ``montage_hash`` (when
    set) must reference an existing ``montages[i].hash`` in the same
    dataset. A drift in either side breaks the API's montage-detail
    lookup endpoint.

    Skipped when no montages exist in any fixture — same gate as the
    other montage acceptance tests.
    """
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
                    f"montage_hash={ref!r} not in known hashes "
                    f"{sorted(known_hashes)}"
                )
    assert not failures, "Record→Montage join drift:\n" + "\n".join(failures)
