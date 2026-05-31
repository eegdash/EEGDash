"""BIDS filesystem digest Seam: synthesize records by walking a local BIDS tree.

Extracted from ``3_digest.py``. For datasets present on disk, enumerates neuro data
files, builds one record each (montage attach + telemetry), and assembles the
``EnumerationResult``. Depends only on leaf helpers (``_record_extractor``,
``_dataset_metadata``, ``_montage``, ``_bids_path``, ``_fingerprint``,
``digest_telemetry``, ``source_adapter``) plus ``record_enumerator``'s
``EnumerationResult`` — never on ``3_digest``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from _bids_path import is_neuro_data_file
from _dataset_metadata import extract_dataset_metadata
from _fingerprint import fingerprint_from_files
from _montage import extract_layout
from _record_extractor import extract_record
from digest_telemetry import TelemetryEvent, get_emitter
from record_enumerator import EnumerationResult
from source_adapter import SourceAdapter

logger = logging.getLogger(__name__)

__all__ = ["_enumerate_via_bids"]


def _attach_montage_to_record(
    record: dict[str, Any],
    bids_file: Any,
    dataset_dir: Path,
    montages: dict[str, dict[str, Any]],
    dataset_id: str,
    digested_at: str,
    montage_cache: dict[tuple[str, int], tuple[str, dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Extract layout, stamp montage_hash on record, and populate montages; MEG layouts cached by (dataset_id, nchans)."""
    record_datatype = (record.get("datatype") or "").lower()
    errors: list[dict[str, Any]] = []

    cache_key: tuple[str, int] | None = None
    record_nchans = record.get("nchans")
    if (
        montage_cache is not None
        and record_datatype == "meg"
        and isinstance(record_nchans, int)
        and record_nchans > 0
    ):
        cache_key = (dataset_id, record_nchans)
        cached = montage_cache.get(cache_key)
        if cached is not None:
            cached_hash, cached_doc = cached
            record["montage_hash"] = cached_hash
            if cached_hash not in montages:
                montages[cached_hash] = cached_doc
            return errors

    try:
        layout_result = extract_layout(
            Path(str(bids_file)), dataset_dir, datatype=record_datatype
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        record["montage_hash"] = None
        errors.append(
            {"file": str(bids_file), "error": f"layout extraction failed: {exc}"}
        )
        return errors

    if layout_result is None:
        record["montage_hash"] = None
        return errors

    h, doc = layout_result
    record["montage_hash"] = h
    if h not in montages:
        doc["first_seen"] = digested_at
        doc["representative_dataset"] = dataset_id
        subject_entity = record.get("entities", {}).get("subject")
        if subject_entity:
            doc["representative_subject"] = f"sub-{subject_entity}"
        montages[h] = doc

    if cache_key is not None:
        montage_cache[cache_key] = (h, montages[h])

    return errors


def _build_one_record_from_bids(
    bids_dataset: Any,
    bids_file: Any,
    dataset_id: str,
    source: str,
    source_adapter: SourceAdapter,
    digested_at: str,
    dataset_dir: Path,
    montages: dict[str, dict[str, Any]],
    montage_cache: dict[tuple[str, int], tuple[str, dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Extract one Record, attach its montage, emit telemetry; returns (record_or_None, errors)."""
    errors: list[dict[str, Any]] = []
    try:
        record = extract_record(
            bids_dataset,
            bids_file,
            dataset_id,
            source,
            digested_at,
            source_adapter=source_adapter,
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        errors.append({"file": str(bids_file), "error": str(e)})
        get_emitter().emit(
            TelemetryEvent(
                event_kind="record_failed",
                dataset_id=dataset_id,
                record_id=str(bids_file),
                payload={"bids_file": str(bids_file), "error": str(e)},
            )
        )
        return None, errors

    if any("Split FIF" in issue for issue in record.get("_data_integrity_issues", [])):
        errors.append(
            {
                "file": str(bids_file),
                "error": "Split FIF without continuation files — skipped",
            }
        )
        get_emitter().emit(
            TelemetryEvent(
                event_kind="record_failed",
                dataset_id=dataset_id,
                record_id=str(bids_file),
                payload={
                    "bids_file": str(bids_file),
                    "error": "Split FIF without continuation files — skipped",
                },
            )
        )
        return None, errors

    errors.extend(
        _attach_montage_to_record(
            record,
            bids_file,
            dataset_dir,
            montages,
            dataset_id,
            digested_at,
            montage_cache=montage_cache,
        )
    )

    get_emitter().emit(
        TelemetryEvent(
            event_kind="record_built",
            dataset_id=dataset_id,
            record_id=record.get("bids_relpath"),
            payload={
                "bids_relpath": record.get("bids_relpath"),
                "datatype": record.get("datatype"),
                "sampling_frequency": record.get("sampling_frequency"),
                "nchans": record.get("nchans"),
                "ntimes": record.get("ntimes"),
                "duration_seconds": record.get("duration_seconds"),
                "metadata_provenance": record.get("_metadata_provenance"),
            },
        )
    )
    return record, errors


def _enumerate_via_bids(
    dataset_dir: Path,
    dataset_id: str,
    source: str,
    source_adapter: SourceAdapter,
    digested_at: str,
    bids_dataset,
    manifest_data: dict | None = None,
) -> EnumerationResult:
    """Walk the BIDS filesystem and build an EnumerationResult."""
    files = bids_dataset.get_files()
    if not files:
        return EnumerationResult(
            dataset_meta={"dataset_id": dataset_id, "source": source},
            digest_method="bids_filesystem",
        )

    try:
        dataset_meta = extract_dataset_metadata(
            bids_dataset,
            dataset_id,
            source,
            digested_at,
            metadata=manifest_data,
            source_adapter=source_adapter,
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        dataset_meta = {
            "dataset_id": dataset_id,
            "source": source,
            "error": str(e),
        }

    try:
        file_paths = [Path(str(f)) for f in files]
        fingerprint = fingerprint_from_files(
            dataset_id, source, file_paths, dataset_dir
        )
        dataset_meta["ingestion_fingerprint"] = fingerprint
    except (OSError, ValueError, KeyError):
        pass

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    montages: dict[str, dict[str, Any]] = {}

    # MEG layouts are device-defined and identical for the same nchans; cache to avoid re-extraction.
    meg_montage_cache: dict[tuple[str, int], tuple[str, dict[str, Any]]] = {}

    for bids_file in files:
        if not is_neuro_data_file(str(bids_file)):
            continue
        record, per_file_errors = _build_one_record_from_bids(
            bids_dataset=bids_dataset,
            bids_file=bids_file,
            dataset_id=dataset_id,
            source=source,
            source_adapter=source_adapter,
            digested_at=digested_at,
            dataset_dir=dataset_dir,
            montages=montages,
            montage_cache=meg_montage_cache,
        )
        errors.extend(per_file_errors)
        if record is not None:
            records.append(record)

    return EnumerationResult(
        dataset_meta=dataset_meta,
        records=records,
        errors=errors,
        montages=montages,
        digest_method="bids_filesystem",
    )
