# Authors: The EEGDash contributors.
# License: BSD-3-Clause

"""Record schema helpers (v2) and backward-compatible adapters.

This module defines a minimal "schema v2" view of EEGDash records and provides
helpers to adapt legacy (v1) records into v2-shaped records in-memory.

The adapter is intentionally conservative:
- It preserves all original keys from the input record.
- It adds v2 keys (`schema_version`, `record_id`, `variant`, `bids_relpath`, ...).
- It does not mutate the legacy `bidspath` field (used by older code paths).

The goal is to enable incremental refactors (resolver, loader, offline discovery)
without requiring a DB migration first.
"""

from __future__ import annotations

import uuid
from pathlib import PurePosixPath
from typing import Any, Literal, Mapping, TypedDict


DEFAULT_OPENNEURO_BUCKET = "s3://openneuro.org"

# Stable namespace for deterministic UUIDv5 record IDs.
_RECORD_ID_NAMESPACE = uuid.UUID("4e5c2ad6-85c8-4f8c-8c2c-3a0cb11b1f6a")


class StorageV2(TypedDict):
    backend: Literal["s3"]
    base: str
    raw_key: str
    dep_keys: list[str]


class CacheV2(TypedDict):
    dataset_subdir: str
    raw_relpath: str
    dep_relpaths: list[str]


class RecordV2(TypedDict, total=False):
    schema_version: int
    record_id: str
    variant: str
    dataset: str
    data_name: str
    bids_relpath: str
    datatype: str
    suffix: str
    extension: str
    entities: dict[str, Any]
    entities_mne: dict[str, Any]
    storage: StorageV2
    cache: CacheV2


def _strip_dataset_prefix(path: str, dataset: str) -> str:
    p = PurePosixPath(str(path))
    if p.parts and p.parts[0] == dataset:
        return PurePosixPath(*p.parts[1:]).as_posix()
    return p.as_posix()


def _sanitize_index_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return value if value.isdigit() else None
    return None


def _canonicalize_entity_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if value.startswith("sub-"):
            return value[4:]
        return value
    if isinstance(value, int):
        return str(value)
    return str(value)


def infer_variant(*, variant: str | None, s3_bucket: str | None) -> str:
    """Infer a variant for legacy records when it is not explicitly provided."""
    if isinstance(variant, str) and variant.strip():
        return variant.strip()

    if not s3_bucket or "openneuro.org" in str(s3_bucket).lower():
        return "openneuro_raw"

    bucket = str(s3_bucket).lower()
    if "neurips25" in bucket and "bdf" in bucket:
        return "challenge_l100_bdf_mini" if "mini" in bucket else "challenge_l100_bdf"

    return "custom"


def dataset_subdir_for_variant(dataset: str, variant: str) -> str:
    """Compute a cache dataset subdir for a given dataset + variant."""
    v = str(variant).lower()
    suffixes: list[str] = []
    if "bdf" in v:
        suffixes.append("bdf")
    if "mini" in v:
        suffixes.append("mini")
    return f"{dataset}-{'-'.join(suffixes)}" if suffixes else dataset


def adapt_record_v1_to_v2(
    record: Mapping[str, Any],
    *,
    s3_bucket: str | None = None,
    variant: str | None = None,
) -> RecordV2:
    """Adapt a legacy record (v1) to a schema-v2 shaped record (in-memory)."""
    dataset = str(record.get("dataset") or "").strip()
    if not dataset:
        raise ValueError("record must include a non-empty 'dataset'")

    data_name = str(record.get("data_name") or "").strip()
    if not data_name:
        # Keep stable legacy behavior: dataset + basename if available
        bidspath = record.get("bidspath")
        fname = PurePosixPath(str(bidspath)).name if bidspath else ""
        data_name = f"{dataset}_{fname}" if fname else dataset

    # Prefer explicit v2 field if present; otherwise derive from legacy bidspath.
    bids_relpath = str(record.get("bids_relpath") or "").strip()
    if not bids_relpath:
        bidspath = record.get("bidspath")
        if not bidspath:
            raise ValueError("record must include 'bidspath' or 'bids_relpath'")
        bids_relpath = _strip_dataset_prefix(str(bidspath), dataset)

    resolved_variant = infer_variant(variant=variant, s3_bucket=s3_bucket)

    # Apply legacy compatibility rule: challenge buckets store BDF, while legacy
    # records may still reference EEGLAB .set filenames.
    if "bdf" in resolved_variant.lower() and bids_relpath.endswith(".set"):
        bids_relpath = bids_relpath[:-4] + ".bdf"

    datatype = str(record.get("modality") or record.get("datatype") or "eeg").lower()
    suffix = str(record.get("suffix") or datatype)
    extension = PurePosixPath(bids_relpath).suffix

    entities = {
        "subject": _canonicalize_entity_value(record.get("subject")),
        "session": _canonicalize_entity_value(record.get("session")),
        "task": _canonicalize_entity_value(record.get("task")),
        "run": _canonicalize_entity_value(record.get("run")),
    }
    entities_mne = dict(entities)
    entities_mne["run"] = _sanitize_index_value(record.get("run"))

    record_id = str(record.get("record_id") or "").strip()
    if not record_id:
        record_id = str(
            uuid.uuid5(
                _RECORD_ID_NAMESPACE,
                f"{dataset}|{resolved_variant}|{bids_relpath}",
            )
        )

    storage_base = None
    if resolved_variant == "openneuro_raw":
        storage_base = f"{DEFAULT_OPENNEURO_BUCKET.rstrip('/')}/{dataset}"
    else:
        storage_base = (s3_bucket or "").rstrip("/")
        if not storage_base:
            storage_base = f"{DEFAULT_OPENNEURO_BUCKET.rstrip('/')}/{dataset}"

    dep_keys: list[str] = []
    deps = record.get("bidsdependencies") or []
    if isinstance(deps, (list, tuple)):
        raw_path = PurePosixPath(bids_relpath)
        raw_dir = raw_path.parent
        raw_stem = raw_path.stem
        base = (
            raw_stem[: -len(f"_{suffix}")]
            if raw_stem.endswith(f"_{suffix}")
            else raw_stem
        )

        task = entities.get("task")
        keep_names: set[str] = {
            "dataset_description.json",
            "participants.tsv",
            "participants.json",
            f"{base}_channels.tsv",
            f"{base}_channels.tsv.gz",
            f"{base}_events.tsv",
            f"{base}_events.tsv.gz",
            f"{raw_stem}.json",
            f"{raw_stem}.json.gz",
            f"{base}_electrodes.tsv",
            f"{base}_electrodes.tsv.gz",
            f"{base}_coordsystem.json",
        }
        if task:
            keep_names.add(f"task-{task}_events.json")
            keep_names.add(f"task-{task}_{suffix}.json")

        same_stem_sidecar_exts = {".eeg", ".vmrk", ".fdt"}

        for dep in deps:
            if dep is None:
                continue
            dep_key = _strip_dataset_prefix(str(dep), dataset)
            if "bdf" in resolved_variant.lower() and dep_key.endswith(".set"):
                dep_key = dep_key[:-4] + ".bdf"
            dep_path = PurePosixPath(dep_key)
            if dep_key == bids_relpath:
                dep_keys.append(dep_key)
                continue
            if dep_path.name in keep_names:
                dep_keys.append(dep_key)
                continue
            if (
                dep_path.parent == raw_dir
                and dep_path.stem == raw_stem
                and dep_path.suffix.lower() in same_stem_sidecar_exts
            ):
                dep_keys.append(dep_key)

    dataset_subdir = dataset_subdir_for_variant(dataset, resolved_variant)

    out: dict[str, Any] = dict(record)
    out.update(
        {
            "schema_version": 2,
            "record_id": record_id,
            "variant": resolved_variant,
            "dataset": dataset,
            "data_name": data_name,
            "bids_relpath": bids_relpath,
            "datatype": datatype,
            "suffix": suffix,
            "extension": extension,
            "entities": entities,
            "entities_mne": entities_mne,
            "storage": {
                "backend": "s3",
                "base": storage_base,
                "raw_key": bids_relpath,
                "dep_keys": dep_keys,
            },
            "cache": {
                "dataset_subdir": dataset_subdir,
                "raw_relpath": bids_relpath,
                "dep_relpaths": dep_keys,
            },
        }
    )
    return out  # type: ignore[return-value]


__all__ = [
    "CacheV2",
    "RecordV2",
    "StorageV2",
    "adapt_record_v1_to_v2",
    "dataset_subdir_for_variant",
    "infer_variant",
]
