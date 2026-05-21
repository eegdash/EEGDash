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


__all__ = [
    "BIDSFilesystemEnumerator",
    "EnumerationResult",
    "ManifestEnumerator",
    "RecordEnumerator",
    "get_record_enumerator",
]
