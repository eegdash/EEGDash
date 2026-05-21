"""RecordEnumerator Module — unify the two digest algorithms behind one Seam.

Background
----------

The ingestion pipeline used to have two top-level functions that produced
the same output shape (Dataset doc + Records list + montages) from two
very different inputs:

* ``digest_dataset`` (3_digest.py) — walks a BIDS filesystem via
  ``mne_bids.EEGBIDSDataset``. Used by OpenNeuro and NEMAR (the
  production paths).
* ``digest_from_manifest`` (3_digest.py) — walks a flat
  ``manifest["files"]`` list. Used for API-only sources (Zenodo,
  Figshare, OSF, SciDB) where the files don't exist on local disk.

The two functions were 327 + 647 = 974 LOC of *parallel* implementation
with an implicit fallback graph between them (3 separate places in
``digest_dataset`` would call ``digest_from_manifest`` when something
went wrong with the BIDS path). Cross-function bug fixes routinely
missed one of the two paths — Phase 9 audit-1 F1 was originally fixed
in `digest_dataset` only.

This Module names the Seam they were both at: **how do we enumerate
Records for a Dataset?** Two Adapters implement it
(:class:`BIDSFilesystemEnumerator`, :class:`ManifestEnumerator`); a
factory (:func:`get_record_enumerator`) handles the fallback graph.
The orchestrator picks the algorithm in one place via the factory.

Design
------

- :class:`EnumerationResult` is the shared return type. Both Adapters
  produce ``(dataset_meta, records, errors, montages)`` plus a
  ``digest_method`` string for the summary.
- :class:`RecordEnumerator` is the abstract base. ``__init__`` takes
  per-Dataset state; ``enumerate()`` returns the result.
- :class:`BIDSFilesystemEnumerator` wraps the BIDS-filesystem
  algorithm (the production path for OpenNeuro and NEMAR).
- :class:`ManifestEnumerator` wraps the manifest-only algorithm
  (Zenodo, Figshare, OSF, SciDB, and the fallback path for any BIDS
  clone that can't be parsed).
- :func:`get_record_enumerator` is the factory: tries BIDSFilesystem
  first when ``has_actual_files`` is True, falls back to Manifest when
  that fails or when no files are present at all.

The factory's fallback rules are documented inline; they mirror the
3 fallback sites that used to live in ``digest_dataset``:

1. No actual binary files on disk → Manifest
2. ``EEGBIDSDataset`` construction fails → Manifest (if manifest.json
   exists; else surface the error)
3. ``bids_dataset.get_files()`` returns empty → Manifest (if manifest
   exists; else "empty" status)

See Also
--------
- ``source_adapter.py`` — per-Source ingest behaviour (storage, URLs)
- ``ROBUSTNESS/PROGRESS-4.md`` — the design grill that produced this
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from source_adapter import SourceAdapter

logger = logging.getLogger(__name__)


@dataclass
class EnumerationResult:
    """Output of :meth:`RecordEnumerator.enumerate`.

    Carries everything the orchestrator needs to write the per-Dataset
    JSON files (``_dataset.json`` / ``_records.json`` /
    ``_montages.json`` / ``_summary.json``). Both Adapters produce this
    shape; the orchestrator never branches on which one was used.

    Attributes
    ----------
    dataset_meta : dict
        Dataset document (``create_dataset`` output, with
        ``ingestion_fingerprint`` already attached).
    records : list[dict]
        Per-file Record documents (``create_record`` output, each with
        the per-Source storage + any annotation_events).
    errors : list[dict]
        Recoverable per-file failures. Each entry has at least
        ``{"file": str, "error": str}``.
    montages : dict[str, dict]
        Mapping of ``montage_hash -> montage_doc`` for layout
        deduplication across datasets. Empty for ManifestEnumerator
        (no filesystem to read electrodes.tsv from).
    digest_method : str
        Either ``"bids_filesystem"`` or ``"manifest_only"`` — surfaced
        in the summary for operational observability.
    """

    dataset_meta: dict[str, Any]
    records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    montages: dict[str, dict[str, Any]] = field(default_factory=dict)
    digest_method: str = "bids_filesystem"


class RecordEnumerator(ABC):
    """Abstract base class — produce all Records for one Dataset.

    Each concrete Adapter encapsulates one algorithm (BIDS-fs or
    manifest-only). The per-Dataset state lives as instance attributes;
    each Adapter is instantiated once per ``digest_dataset`` call.

    Parameters
    ----------
    dataset_id : str
        Dataset accession (e.g. ``"ds002893"``).
    dataset_dir : Path
        Absolute path to the cloned dataset's root (the directory that
        contains ``sub-*/`` and optionally ``manifest.json``).
    source : str
        Source identifier (``"openneuro"``, ``"nemar"``, etc.).
    source_adapter : SourceAdapter
        Per-Source ingest behaviour (storage addressing, URL builders,
        annex / inline prefetch). Same instance threaded into
        ``extract_record`` and ``extract_dataset_metadata``.
    digested_at : str
        ISO 8601 timestamp; identical across the dataset_meta and every
        Record produced by this Enumerator.
    """

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
        """Build the Dataset doc + Records + montages for this Dataset.

        Returns
        -------
        EnumerationResult
            The Dataset metadata, the list of Records, accumulated
            errors, deduplicated montages, and the digest_method label.

        Notes
        -----
        Implementations may emit logger warnings but must NOT raise on
        recoverable per-file failures (those go in
        :attr:`EnumerationResult.errors`). Programmer errors propagate
        per ROBUSTNESS/PROGRESS-3 (Phase 9 F1).
        """


# ─── Factory ──────────────────────────────────────────────────────────


def get_record_enumerator(
    dataset_id: str,
    dataset_dir: Path,
    source: str,
    source_adapter: SourceAdapter,
    digested_at: str,
) -> RecordEnumerator:
    """Pick a :class:`RecordEnumerator` for this Dataset.

    Fallback rules (mirror the 3 sites that used to be in
    ``digest_dataset``):

    1. If the dataset directory contains real binary recording files,
       try :class:`BIDSFilesystemEnumerator` first. If its constructor
       raises (BIDS structure malformed, ``EEGBIDSDataset`` rejects it,
       no files surfaced by mne_bids), fall back to
       :class:`ManifestEnumerator` when ``manifest.json`` exists.
    2. If the dataset directory contains only a ``manifest.json`` and
       no actual recording files (the API-only Source path), go
       straight to :class:`ManifestEnumerator`.
    3. If neither path is viable, return whichever Enumerator best
       represents what's on disk and let its ``enumerate`` produce an
       empty result with errors — the orchestrator decides the status.

    Parameters
    ----------
    dataset_id : str
    dataset_dir : Path
    source : str
    source_adapter : SourceAdapter
    digested_at : str

    Returns
    -------
    RecordEnumerator
        Either a :class:`BIDSFilesystemEnumerator` or a
        :class:`ManifestEnumerator`, ready to call ``enumerate()``.
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

    # Case 3: nothing viable. Return manifest enumerator if possible
    # (its enumerate() will produce an empty + error result); else
    # let BIDSFilesystemEnumerator try and fail.
    if has_manifest:
        return ManifestEnumerator(
            dataset_id, dataset_dir, source, source_adapter, digested_at
        )
    return BIDSFilesystemEnumerator(
        dataset_id, dataset_dir, source, source_adapter, digested_at
    )


def _has_actual_recording_files(dataset_dir: Path) -> bool:
    """Detect any real recording files (binaries or annex symlinks) under root.

    Looks for canonical extensions (``.set``, ``.edf``, ``.bdf``,
    ``.vhdr``, ``.fif``, ``.cnt``, ``.snirf``, ``.mefd``) anywhere in
    the tree, plus CTF ``.ds`` and MEF3 ``.mefd`` directories. Symlinks
    count as real files (git-annex pointers are still "present" from
    the digest pipeline's perspective; the actual binary fetch happens
    later in the runtime).
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


# ─── Concrete Adapters (implementations land in stage 2) ──────────────


class BIDSFilesystemEnumerator(RecordEnumerator):
    """Walk a BIDS filesystem via ``mne_bids.EEGBIDSDataset``.

    The production path for OpenNeuro and NEMAR. The constructor loads
    the EEGBIDSDataset (may raise; the factory catches and falls back
    to :class:`ManifestEnumerator` when ``manifest.json`` exists).

    Implementation note (stage-1 stub): the actual ``enumerate()`` body
    is wired in stage 2 — it delegates to the body that used to live
    in ``3_digest.py:digest_dataset``. This class exists in stage 1 so
    the factory + tests can be built first.
    """

    def __init__(
        self,
        dataset_id: str,
        dataset_dir: Path,
        source: str,
        source_adapter: SourceAdapter,
        digested_at: str,
    ) -> None:
        super().__init__(dataset_id, dataset_dir, source, source_adapter, digested_at)

    def enumerate(self) -> EnumerationResult:
        # Wired in stage 2.
        raise NotImplementedError(
            "BIDSFilesystemEnumerator.enumerate is wired in stage 2"
        )


class ManifestEnumerator(RecordEnumerator):
    """Walk a flat ``manifest["files"]`` list (API-only Sources).

    The fallback path. Used when:
    - the Source is API-only (Zenodo, Figshare, OSF, SciDB, DataRN), or
    - the BIDS clone is malformed / empty and ``manifest.json`` exists.

    Implementation note (stage-1 stub): the actual ``enumerate()`` body
    is wired in stage 2 — it delegates to the body that used to live
    in ``3_digest.py:digest_from_manifest``.
    """

    def enumerate(self) -> EnumerationResult:
        # Wired in stage 2.
        raise NotImplementedError("ManifestEnumerator.enumerate is wired in stage 2")


# ─── Shared output writer ─────────────────────────────────────────────


def _json_default_serializer(obj: Any) -> Any:
    """JSON ``default=`` for values that the stdlib encoder doesn't handle.

    Handles the small set of types that show up in Dataset / Record
    documents: ``Path`` (stringified), ``datetime`` (isoformat), and
    numpy scalars (Python primitive). Anything else falls through to
    a ``str()`` representation rather than crashing — the digest
    pipeline must finish even if one Record carries an unexpected
    type from a third-party extension.

    Mirrors the inline ``_json_serializer`` that used to live in
    ``3_digest.py``. Hoisted here so both Adapter writes use the same
    rule.
    """
    from datetime import date, datetime
    from pathlib import PurePath

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
    """Write the 4 per-Dataset JSON files; return the summary dict.

    Shared between both Adapter paths. Before this helper existed, the
    JSON-write block was duplicated inline in ``digest_dataset`` and
    ``digest_from_manifest`` with subtle drift between them — see
    ``ROBUSTNESS/STAGE-2-PLAN.md`` for the catalogue.

    Files written:
      - ``<dataset_id>_dataset.json``   the Dataset document
      - ``<dataset_id>_records.json``   the Records list + count
      - ``<dataset_id>_montages.json``  deduplicated montage hashes
      - ``<dataset_id>_summary.json``   the per-Dataset summary

    Parameters
    ----------
    dataset_output_dir : Path
        Directory to write the four JSON files into. Created if missing.
    result : EnumerationResult
        Output of :meth:`RecordEnumerator.enumerate`.
    dataset_id, source, digested_at : str
        Identity fields stamped into every output file.
    total_files : int, optional
        Total file count from the input (e.g. the manifest's full
        ``files`` length) — surfaced in the summary. Only the manifest
        Adapter populates this; the BIDS Adapter leaves it as None.

    Returns
    -------
    dict
        The summary dictionary (also persisted to ``_summary.json``).
        Same shape as both legacy functions used to return.

    Notes
    -----
    Behaviour changes vs the previous duplicated code:

    1. ``_montages.json`` is now written for **every** Adapter path,
       even when the montages dict is empty. Previously the manifest
       path skipped this file; downstream tooling could no longer
       assume it exists. The empty file format is
       ``{"montage_count": 0, "montages": [], ...}``.
    2. The summary now always carries ``digest_method``, surfaced from
       :attr:`EnumerationResult.digest_method`. Previously only the
       manifest path set it ("manifest_only"); the BIDS path silently
       omitted the field. Now both populate it.
    3. The summary's ``integrity_issues_count`` and ``montage_count``
       are always present (zero for the manifest path). Previously
       only the BIDS path included them.

    These are minor additions to existing fields — not type changes —
    so downstream consumers that read by key name will continue to work.
    """
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset document ──────────────────────────────────────────────
    dataset_path = dataset_output_dir / f"{dataset_id}_dataset.json"
    with open(dataset_path, "w") as fh:
        json.dump(
            dict(result.dataset_meta),
            fh,
            indent=2,
            default=_json_default_serializer,
        )

    # ── Integrity-issue enrichment (BIDS path uses this; manifest
    # path will simply find no records with the marker flag).
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
        # Log per-issue warning (was inlined in digest_dataset)
        logger.warning(
            "Dataset %s has %d record(s) with missing companion files",
            dataset_id,
            integrity_issues_count,
        )
        for rec in records_with_issues:
            issues = rec.get("_data_integrity_issues", [])
            logger.warning("  - %s: %s", rec.get("bids_relpath"), "; ".join(issues))

    # ── Records document ──────────────────────────────────────────────
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

    # ── Montages document (always written, even when empty) ───────────
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

    # ── Summary ───────────────────────────────────────────────────────
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
