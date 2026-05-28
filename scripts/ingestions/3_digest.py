#!/usr/bin/env python3
"""Digest BIDS datasets and generate JSON records for MongoDB.

Produces one Dataset doc (discovery/filtering) and one Record doc per file (loading metadata).
See ``python 3_digest.py --help``.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

# Avoid numba cache issues by setting cache dir before importing MNE.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".cache") / "numba"))

from eegdash.dataset.io import _repair_participants_tsv_ids

logger = logging.getLogger(__name__)

from _bids_digest import (  # noqa: F401 — re-export (canary + tests + back-compat)
    _attach_montage_to_record,
    _build_one_record_from_bids,
    _enumerate_via_bids,
)
from _bids_path import (
    is_neuro_data_file,  # noqa: F401 — re-export for tests (digest.<fn>)
    parse_bids_entities_from_path,  # noqa: F401 — re-export for tests (digest.<fn>)
    strip_dataset_prefix,  # noqa: F401 — re-export for tests (digest.<fn>)
)
from _constants import EXCLUDED_DATASETS
from _dataset_metadata import (  # noqa: F401 — re-export (canary + tests + back-compat)
    _build_global_storage_info,
    _extract_dataset_description_extras,
    _read_bids_readme,
    _read_participants_demographics,
    extract_dataset_metadata,
)
from _digest_config import load_digest_config_from_argv
from _digest_runner import process_datasets_with_watchdog
from _manifest_digest import (  # noqa: F401 — re-export (canary + tests + back-compat)
    _build_bids_data_zip_records,
    _build_ctf_ds_records,
    _build_regular_manifest_record,
    _build_standalone_zip_content_records,
    _build_subject_zip_record,
    _build_zip_extracted_records,
    _collect_bids_entities_from_paths,
    _determine_manifest_storage_base,
    _enumerate_via_manifest,
    _fetch_subject_count_via_http,
    _is_bids_data_zip,
)
from _metadata_cascade import (  # noqa: F401 — re-export for back-compat
    CascadeContext,
    MetadataCascade,
    _parse_fif_with_mne,
    extract_sfreq_nchans_from_channels_tsv,
    extract_sfreq_nchans_from_modality_sidecar,
    sum_bids_channel_counts,
)
from _record_extractor import (  # noqa: F401 — re-export (canary + tests + back-compat)
    _build_dep_keys,
    _clamp_metadata_extremes,
    _extract_bids_sidecar_fields,
    _extract_channel_status_counts,
    _extract_technical_metadata,
    _file_exists_or_symlink,
    extract_record,
    validate_companion_files,
)
from _source_id import (
    _reconcile_source,
    _source_from_dataset_id,
)
from digest_telemetry import TelemetryEvent, auto_configure_from_env, get_emitter
from record_enumerator import (
    EnumerationResult,
    ManifestEnumerator,
    RecordEnumerator,
    get_record_enumerator,
    write_dataset_outputs,
)

auto_configure_from_env()

from source_adapter import SourceAdapter, get_source_adapter


def detect_source(dataset_dir: Path) -> str:
    """Detect source from manifest.json or dataset structure."""
    dataset_id = dataset_dir.name
    manifest_path = dataset_dir / "manifest.json"
    manifest_src: str | None = None
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifest_src = manifest.get("source")
        except (OSError, json.JSONDecodeError, ValueError, KeyError):
            manifest_src = None

    return _reconcile_source(manifest_src, dataset_id, context="detect_source")


_PROV_MNE_BIDS = "mne_bids"
_PROV_MODALITY_SIDECAR = "modality_sidecar"
_PROV_CHANNELS_TSV = "channels_tsv"
_PROV_BINARY_PARSER = "binary_parser"
_PROV_MNE_FALLBACK = "mne_fallback"

_METADATA_FIELDS: tuple[str, ...] = (
    "sampling_frequency",
    "nchans",
    "ntimes",
    "ch_names",
)


def _empty_provenance() -> dict[str, str | None]:
    """A fresh provenance dict — all fields unattributed."""
    return {field: None for field in _METADATA_FIELDS}


def _stamp_provenance(
    provenance: dict[str, str | None],
    source: str,
    *,
    field: str,
    old_value: Any,
    new_value: Any,
) -> None:
    """Set provenance[field] = source if this step was the first to fill it."""
    if old_value is None and new_value is not None and provenance[field] is None:
        provenance[field] = source


def _emit_dataset_finished(dataset_id: str, summary: dict[str, Any]) -> None:
    """Emit a dataset_finished telemetry event."""
    get_emitter().emit(
        TelemetryEvent(
            event_kind="dataset_finished",
            dataset_id=dataset_id,
            payload={
                "status": summary.get("status"),
                "record_count": summary.get("record_count"),
                "error_count": summary.get("error_count"),
                "digest_method": summary.get("digest_method"),
                "integrity_issues_count": summary.get("integrity_issues_count"),
                "montage_count": summary.get("montage_count"),
                "total_files": summary.get("total_files"),
            },
        )
    )


def _run_enumerator_with_manifest_fallback(
    enumerator: RecordEnumerator,
    *,
    dataset_id: str,
    dataset_dir: Path,
    source: str,
    source_adapter: SourceAdapter,
    digested_at: str,
    has_manifest: bool,
) -> tuple[EnumerationResult | None, dict[str, Any] | None]:
    """Run enumerator.enumerate() with ManifestEnumerator fallback; returns (result, None) or (None, summary)."""
    try:
        result = enumerator.enumerate()
    except (
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
        PermissionError,
    ) as exc:
        if has_manifest and not isinstance(enumerator, ManifestEnumerator):
            logger.info(
                "BIDS load failed for %s (%s: %s); falling back to manifest path",
                dataset_id,
                type(exc).__name__,
                exc,
            )
            fallback = ManifestEnumerator(
                dataset_id, dataset_dir, source, source_adapter, digested_at
            )
            try:
                return fallback.enumerate(), None
            except Exception as fb_exc:  # noqa: BLE001
                logger.warning(
                    "Manifest fallback raised %s for %s after BIDS %s: %s",
                    type(fb_exc).__name__,
                    dataset_id,
                    type(exc).__name__,
                    fb_exc,
                )
                return None, {
                    "status": "error",
                    "dataset_id": dataset_id,
                    "error": (
                        f"BIDS path raised {type(exc).__name__}: {exc}; "
                        f"manifest fallback also raised "
                        f"{type(fb_exc).__name__}: {fb_exc}"
                    ),
                }
        return None, {
            "status": "error",
            "dataset_id": dataset_id,
            "error": f"Failed to load BIDS dataset: {exc}",
        }

    if (
        not result.records
        and has_manifest
        and not isinstance(enumerator, ManifestEnumerator)
    ):
        logger.info(
            "BIDS path produced 0 records for %s (errors=%d); "
            "falling back to manifest path",
            dataset_id,
            len(result.errors),
        )
        fallback = ManifestEnumerator(
            dataset_id, dataset_dir, source, source_adapter, digested_at
        )
        try:
            fb_result = fallback.enumerate()
        except Exception as fb_exc:  # noqa: BLE001
            logger.warning(
                "Manifest fallback raised %s for %s (BIDS produced 0 records): %s",
                type(fb_exc).__name__,
                dataset_id,
                fb_exc,
            )
            return None, {
                "status": "error",
                "dataset_id": dataset_id,
                "error": (
                    f"BIDS path returned no records; manifest fallback "
                    f"raised {type(fb_exc).__name__}: {fb_exc}"
                ),
                "errors": result.errors,
            }
        if result.errors:
            fb_result.errors = list(fb_result.errors or []) + list(result.errors)
        return fb_result, None

    return result, None


def _summarise_empty_or_error(
    dataset_id: str,
    result: EnumerationResult,
) -> dict[str, Any]:
    """Return an "empty" or "error" summary dict when no records were produced."""
    structural_errors = [
        e for e in result.errors if e.get("status") not in (None, "skipped", "warning")
    ]
    if structural_errors:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "error": "No records extracted",
            "errors": result.errors,
        }

    total_files = result.total_files
    if total_files == 0:
        reason = "no files in manifest"
    elif total_files is not None:
        reason = "no records extracted"
    else:
        reason = "no neurophysiology files found"
    empty_summary: dict[str, Any] = {
        "status": "empty",
        "dataset_id": dataset_id,
        "reason": reason,
    }
    if result.errors:
        empty_summary["errors"] = result.errors
    return empty_summary


def _check_dataset_skip_conditions(
    dataset_id: str,
    dataset_dir: Path,
    dataset_output_dir: Path,
) -> dict[str, Any] | None:
    """Return a skip summary if output exists or input dir is missing, else None."""
    if dataset_output_dir.exists():
        return {
            "status": "skipped",
            "dataset_id": dataset_id,
            "reason": "already digested",
        }
    if not dataset_dir.exists():
        return {
            "status": "skipped",
            "dataset_id": dataset_id,
            "reason": "directory not found",
        }
    return None


def digest_dataset(
    dataset_id: str,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Digest a single dataset; write dataset/records/montages/summary JSON and return the summary."""
    dataset_dir = input_dir / dataset_id
    dataset_output_dir = output_dir / dataset_id

    skip = _check_dataset_skip_conditions(dataset_id, dataset_dir, dataset_output_dir)
    if skip is not None:
        return skip

    source = detect_source(dataset_dir)
    digested_at = datetime.now(timezone.utc).isoformat()
    _repair_participants_tsv_ids(dataset_dir)
    source_adapter = get_source_adapter(source, dataset_id, dataset_dir)

    get_emitter().emit(
        TelemetryEvent(
            event_kind="dataset_started",
            dataset_id=dataset_id,
            payload={"source": source, "dataset_dir": str(dataset_dir)},
        )
    )

    summary: dict[str, Any] | None = None
    try:
        has_manifest = (dataset_dir / "manifest.json").exists()
        enumerator = get_record_enumerator(
            dataset_id, dataset_dir, source, source_adapter, digested_at
        )

        result, fallback_summary = _run_enumerator_with_manifest_fallback(
            enumerator,
            dataset_id=dataset_id,
            dataset_dir=dataset_dir,
            source=source,
            source_adapter=source_adapter,
            digested_at=digested_at,
            has_manifest=has_manifest,
        )
        if fallback_summary is not None:
            summary = fallback_summary
            return fallback_summary

        assert (
            result is not None
        )  # exactly one of (result, fallback_summary) is non-None

        if not result.records:
            summary = _summarise_empty_or_error(dataset_id, result)
            return summary

        summary = write_dataset_outputs(
            dataset_output_dir,
            result,
            dataset_id=dataset_id,
            source=source,
            digested_at=digested_at,
            total_files=result.total_files,
        )
        return summary
    finally:
        if summary is None:
            summary = {
                "status": "error",
                "dataset_id": dataset_id,
                "error": "digest_dataset raised an unhandled exception",
            }
        _emit_dataset_finished(dataset_id, summary)


def find_datasets(input_dir: Path, datasets: list[str] | None = None) -> list[str]:
    """Return dataset IDs from input_dir that have a manifest or dataset_description."""
    if datasets:
        return datasets

    found = []
    for d in input_dir.iterdir():
        if (
            d.is_dir()
            and d.name not in ("__pycache__", ".git")
            and d.name not in EXCLUDED_DATASETS
        ):
            if (d / "manifest.json").exists() or (
                d / "dataset_description.json"
            ).exists():
                found.append(d.name)

    return sorted(found)


def _dataset_boundary_profile(dataset_id: str, input_dir: Path) -> str:
    """Return a structural profile string for stall-boundary diagnostics."""
    dataset_dir = input_dir / dataset_id
    if not dataset_dir.exists():
        return f"{dataset_id}: missing directory"

    manifest_path = dataset_dir / "manifest.json"
    description_path = dataset_dir / "dataset_description.json"
    parts = [
        f"{dataset_id}:",
        f"pattern_source={_source_from_dataset_id(dataset_id)}",
        f"manifest={manifest_path.exists()}",
        f"dataset_description={description_path.exists()}",
    ]

    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifest_source = manifest.get("source")
            files = manifest.get("files", [])
            zip_contents = manifest.get("zip_contents", [])
            files_count = len(files) if isinstance(files, list) else "n/a"
            zip_contents_count = (
                len(zip_contents) if isinstance(zip_contents, list) else "n/a"
            )
            parts.extend(
                [
                    f"manifest_source={manifest_source!r}",
                    f"manifest_files={files_count}",
                    f"zip_contents={zip_contents_count}",
                ]
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            parts.append(f"manifest_error={exc}")

    try:
        root_entries = list(dataset_dir.iterdir())
        root_dirs = sum(1 for p in root_entries if p.is_dir())
        root_files = sum(1 for p in root_entries if p.is_file() or p.is_symlink())
        parts.extend([f"root_dirs={root_dirs}", f"root_files={root_files}"])
    except (OSError, PermissionError) as exc:
        parts.append(f"root_scan_error={exc}")

    return " ".join(parts)


def print_stall_boundary_diagnostics(dataset_ids: list[str], input_dir: Path) -> None:
    """Print structural profiles around the #287 deterministic stall boundary."""
    boundary_indices = [285, 286]  # 0-based: completed #286 and next #287.
    present = [idx for idx in boundary_indices if idx < len(dataset_ids)]
    if not present:
        return

    print("Stall-boundary diagnostics:")
    for idx in present:
        profile = _dataset_boundary_profile(dataset_ids[idx], input_dir)
        print(f"  #{idx + 1} (0-based {idx}): {profile}")


def main():
    try:
        cfg = load_digest_config_from_argv()
    except ValidationError as exc:
        print("Config error(s):", file=sys.stderr)
        for err in exc.errors():
            field = ".".join(str(p) for p in err.get("loc", []))
            print(f"  {field}: {err.get('msg')}", file=sys.stderr)
        return 1

    dataset_ids = find_datasets(cfg.input, cfg.datasets)
    if cfg.limit:
        dataset_ids = dataset_ids[: cfg.limit]

    print(f"Found {len(dataset_ids)} datasets to digest")
    print(f"Workers: {cfg.workers}")
    print(f"Dataset timeout: {cfg.dataset_timeout:g}s")
    print_stall_boundary_diagnostics(dataset_ids, cfg.input)
    print("=" * 60)

    cfg.output.mkdir(parents=True, exist_ok=True)

    results, stats = process_datasets_with_watchdog(
        dataset_ids,
        cfg.input,
        cfg.output,
        workers=cfg.workers,
        dataset_timeout=cfg.dataset_timeout,
        digest_fn=digest_dataset,
    )

    batch_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_datasets": len(dataset_ids),
        "stats": stats,
        "total_records": sum(
            r.get("record_count", 0) for r in results if r.get("status") == "success"
        ),
    }

    batch_summary_path = cfg.output / "BATCH_SUMMARY.json"
    with open(batch_summary_path, "w") as f:
        json.dump(batch_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("DIGESTION SUMMARY")
    print("=" * 60)
    print(f"  Success:  {stats['success']}")
    print(f"  Skipped:  {stats['skipped']}")
    print(f"  Empty:    {stats['empty']}")
    print(f"  Error:    {stats['error']}")
    print(f"\nTotal records: {batch_summary['total_records']}")
    print(f"Batch summary: {batch_summary_path}")
    print("=" * 60)

    return 0 if stats["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
