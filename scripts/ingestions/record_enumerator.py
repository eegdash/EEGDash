"""RecordEnumerator — unify the two digest algorithms behind one Seam.

Two Adapters (:class:`BIDSFilesystemEnumerator`, :class:`ManifestEnumerator`)
share a common abstract base and return type. The factory
:func:`get_record_enumerator` selects the right one and manages the
BIDS-to-manifest fallback so the orchestrator never branches on algorithm.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path, PurePath
from typing import Any

from eegdash.dataset.bids_dataset import EEGBIDSDataset
from source_adapter import SourceAdapter

logger = logging.getLogger(__name__)


@dataclass
class EnumerationResult:
    """Shared return type for both :class:`BIDSFilesystemEnumerator` and
    :class:`ManifestEnumerator`.

    ``montages`` is empty for the manifest path (no filesystem to read
    ``electrodes.tsv`` from). ``total_files`` is ``None`` for the BIDS
    path so the summary field is omitted there.
    """

    dataset_meta: dict[str, Any]
    records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    montages: dict[str, dict[str, Any]] = field(default_factory=dict)
    digest_method: str = "bids_filesystem"
    total_files: int | None = None


class RecordEnumerator(ABC):
    """Abstract base — produce all Records for one Dataset (one instance per dataset)."""

    def __init__(
        self,
        dataset_id: str,
        dataset_dir: Path,
        source: str,
        source_adapter: SourceAdapter,
        digested_at: str,
    ) -> None:
        self.dataset_id = dataset_id
        self.dataset_dir = dataset_dir
        self.source = source
        self.source_adapter = source_adapter
        self.digested_at = digested_at

    @abstractmethod
    def enumerate(self) -> EnumerationResult:
        """Build the Dataset doc + Records + montages.

        Per-file failures go in :attr:`EnumerationResult.errors`; programmer
        errors propagate normally.
        """


def get_record_enumerator(
    dataset_id: str,
    dataset_dir: Path,
    source: str,
    source_adapter: SourceAdapter,
    digested_at: str,
) -> RecordEnumerator:
    """Return the appropriate :class:`RecordEnumerator` for this Dataset.

    Tries :class:`BIDSFilesystemEnumerator` when recording files are present;
    falls back to :class:`ManifestEnumerator` on BIDS construction failure or
    when only a ``manifest.json`` exists.
    """
    has_manifest = (dataset_dir / "manifest.json").exists()
    has_actual_files = _has_actual_recording_files(dataset_dir)

    # Case 2: no actual files on disk → manifest path is the only option.
    if has_manifest and not has_actual_files:
        return ManifestEnumerator(
            dataset_id, dataset_dir, source, source_adapter, digested_at
        )

    # Case 1: try BIDS filesystem; on construction failure, fall back.
    if has_actual_files:
        try:
            return BIDSFilesystemEnumerator(
                dataset_id, dataset_dir, source, source_adapter, digested_at
            )
        except (
            OSError,
            ValueError,
            KeyError,
            FileNotFoundError,
            PermissionError,
        ) as exc:
            if has_manifest:
                logger.info(
                    "BIDS load failed for %s (%s); falling back to manifest path",
                    dataset_id,
                    exc,
                )
                return ManifestEnumerator(
                    dataset_id, dataset_dir, source, source_adapter, digested_at
                )
            raise

    # Case 3: nothing viable — return manifest if present so enumerate() produces
    # an empty+error result; else let BIDSFilesystemEnumerator try and surface the error.
    if has_manifest:
        return ManifestEnumerator(
            dataset_id, dataset_dir, source, source_adapter, digested_at
        )
    return BIDSFilesystemEnumerator(
        dataset_id, dataset_dir, source, source_adapter, digested_at
    )


def _has_actual_recording_files(dataset_dir: Path) -> bool:
    """Return True if any recording file or directory exists under ``dataset_dir``.

    Symlinks count — git-annex pointers are present from the pipeline's perspective.
    """
    canonical_exts = {
        ".set",
        ".edf",
        ".bdf",
        ".vhdr",
        ".fif",
        ".cnt",
        ".snirf",
        ".mefd",
    }
    for entry in dataset_dir.rglob("*"):
        if entry.is_file() or entry.is_symlink():
            if entry.suffix in canonical_exts:
                return True
        if entry.is_dir() and entry.suffix in {".ds", ".mefd"}:
            return True
    return False


class BIDSFilesystemEnumerator(RecordEnumerator):
    """Walk a BIDS filesystem via ``EEGBIDSDataset`` (OpenNeuro, NEMAR path)."""

    def enumerate(self) -> EnumerationResult:
        """Load ``EEGBIDSDataset`` and run the BIDS-filesystem algorithm.

        Raises ``OSError`` / ``ValueError`` / ``KeyError`` / ``FileNotFoundError`` /
        ``PermissionError`` on a non-viable BIDS root; the factory catches and falls back.
        """
        bids_dataset = EEGBIDSDataset(
            data_dir=str(self.dataset_dir),
            dataset=self.dataset_id,
            allow_symlinks=True,
        )
        digest_mod = _load_digest_module()
        return digest_mod._enumerate_via_bids(
            self.dataset_dir,
            self.dataset_id,
            self.source,
            self.source_adapter,
            self.digested_at,
            bids_dataset,
            manifest_data=_load_manifest_data(self.dataset_dir),
        )


class ManifestEnumerator(RecordEnumerator):
    """Walk a flat ``manifest["files"]`` list (API-only Sources or BIDS fallback).

    Used for Zenodo, Figshare, OSF, SciDB, DataRN, and any BIDS clone
    that is malformed or empty but has a ``manifest.json``.
    """

    def enumerate(self) -> EnumerationResult:
        """Run the manifest-only algorithm via the shared helper."""
        manifest, load_summary = load_manifest_or_summary(
            self.dataset_dir, self.dataset_id
        )
        if manifest is None:
            assert load_summary is not None  # invariant of load_manifest_or_summary
            return EnumerationResult(
                dataset_meta={"dataset_id": self.dataset_id, "source": self.source},
                # Explicit 0 (not None) so callers can rely on manifest path
                # always carrying a non-None total_files.
                total_files=0,
                errors=[
                    {
                        "file": self.dataset_id,
                        **load_summary,
                    }
                ],
                digest_method="manifest_only",
            )
        digest_mod = _load_digest_module()
        result, total_files = digest_mod._enumerate_via_manifest(
            self.dataset_id, manifest, self.digested_at
        )
        result.total_files = total_files
        return result


_DIGEST_MODULE_CACHE: Any = None


def _load_digest_module() -> Any:
    """Lazy-load ``3_digest.py`` via importlib (digit prefix blocks normal import), cached."""
    global _DIGEST_MODULE_CACHE
    if _DIGEST_MODULE_CACHE is not None:
        return _DIGEST_MODULE_CACHE
    spec = spec_from_file_location(
        "_record_enumerator_digest_target",
        Path(__file__).parent / "3_digest.py",
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError("could not load 3_digest.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    _DIGEST_MODULE_CACHE = mod
    return mod


def load_manifest_or_summary(
    dataset_dir: Path, dataset_id: str
) -> tuple[dict | None, dict | None]:
    """Read ``manifest.json`` and return ``(manifest, None)`` on success or
    ``(None, summary)`` on failure with distinct ``reason`` fields for missing,
    corrupt, and permission-denied cases.
    """
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        return None, {
            "status": "skipped",
            "dataset_id": dataset_id,
            "reason": "manifest.json not found",
        }
    try:
        with open(manifest_path) as fh:
            return json.load(fh), None
    except json.JSONDecodeError as exc:
        logger.warning("Manifest at %s is corrupt: %s", manifest_path, exc)
        return None, {
            "status": "error",
            "dataset_id": dataset_id,
            "error": f"Failed to load manifest: {exc}",
        }
    except (OSError, ValueError) as exc:
        logger.warning("Manifest at %s could not be read: %s", manifest_path, exc)
        return None, {
            "status": "error",
            "dataset_id": dataset_id,
            "error": f"Failed to read manifest: {exc}",
        }


def _load_manifest_data(dataset_dir: Path) -> dict | None:
    """Return the manifest dict, or ``None`` on absence or failure (failures are logged)."""
    manifest, _summary = load_manifest_or_summary(dataset_dir, dataset_dir.name)
    return manifest


def _json_default_serializer(obj: Any) -> Any:
    """JSON ``default=`` for ``Path``, ``datetime``/``date``, and numpy scalars.

    Falls through to ``str()`` so an unexpected type never crashes the pipeline.
    """
    if isinstance(obj, PurePath):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    # numpy scalars / common third-party types
    if hasattr(obj, "item"):
        try:
            return obj.item()  # numpy scalar -> Python primitive
        except (ValueError, TypeError):
            pass
    return str(obj)


def write_dataset_outputs(
    dataset_output_dir: Path,
    result: EnumerationResult,
    dataset_id: str,
    source: str,
    digested_at: str,
    *,
    total_files: int | None = None,
) -> dict[str, Any]:
    """Write ``_dataset.json``, ``_records.json``, ``_montages.json``, and
    ``_summary.json`` into ``dataset_output_dir``; return the summary dict.

    ``_montages.json`` is written for every Adapter path (empty dict included).
    The summary always carries ``digest_method``, ``integrity_issues_count``,
    and ``montage_count``.
    """
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_output_dir / f"{dataset_id}_dataset.json"
    with open(dataset_path, "w") as fh:
        json.dump(
            dict(result.dataset_meta),
            fh,
            indent=2,
            default=_json_default_serializer,
        )

    records_with_issues = [
        r for r in result.records if r.get("_has_missing_files", False)
    ]
    integrity_issues_count = len(records_with_issues)
    if records_with_issues:
        authors = result.dataset_meta.get("authors", [])
        contact_info = result.dataset_meta.get("contact_info")
        source_url = result.dataset_meta.get("external_links", {}).get("source_url")
        for rec in records_with_issues:
            if authors:
                rec["_dataset_authors"] = authors
            if contact_info:
                rec["_dataset_contact"] = contact_info
            if source_url:
                rec["_source_url"] = source_url
        logger.warning(
            "Dataset %s has %d record(s) with missing companion files",
            dataset_id,
            integrity_issues_count,
        )
        for rec in records_with_issues:
            issues = rec.get("_data_integrity_issues", [])
            logger.warning("  - %s: %s", rec.get("bids_relpath"), "; ".join(issues))

    records_path = dataset_output_dir / f"{dataset_id}_records.json"
    records_data = {
        "dataset": dataset_id,
        "source": source,
        "digested_at": digested_at,
        "record_count": len(result.records),
        "records_with_integrity_issues": integrity_issues_count,
        "records": result.records,
    }
    with open(records_path, "w") as fh:
        json.dump(records_data, fh, indent=2, default=_json_default_serializer)

    montages_path = dataset_output_dir / f"{dataset_id}_montages.json"
    montages_data = {
        "dataset": dataset_id,
        "source": source,
        "digested_at": digested_at,
        "montage_count": len(result.montages),
        "montages": list(result.montages.values()),
    }
    with open(montages_path, "w") as fh:
        json.dump(montages_data, fh, indent=2, default=_json_default_serializer)

    summary: dict[str, Any] = {
        "status": "success" if result.records else "no_neuro_files",
        "dataset_id": dataset_id,
        "source": source,
        "record_count": len(result.records),
        "error_count": len(result.errors),
        "integrity_issues_count": integrity_issues_count,
        "montage_count": len(result.montages),
        "dataset_file": str(dataset_path),
        "records_file": str(records_path),
        "montages_file": str(montages_path),
        "digest_method": result.digest_method,
    }
    if total_files is not None:
        summary["total_files"] = total_files

    summary_path = dataset_output_dir / f"{dataset_id}_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    return summary


__all__ = [
    "BIDSFilesystemEnumerator",
    "EnumerationResult",
    "ManifestEnumerator",
    "RecordEnumerator",
    "get_record_enumerator",
    "write_dataset_outputs",
]
