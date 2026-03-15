# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Data utilities and dataset classes for EEG data handling.

This module provides core dataset classes for working with EEG data in the EEGDash ecosystem,
including classes for individual recordings and collections of datasets. It integrates with
braindecode for machine learning workflows and handles data loading from both local and remote sources.
"""

import configparser
import re
import shutil
from contextlib import contextmanager
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any
from unittest.mock import patch

import mne.io.ctf.info as ctf_info
import mne_bids
from mne.io import BaseRaw
from mne_bids import BIDSPath

from braindecode.datasets.base import RawDataset

from .. import downloader
from ..const import MODALITY_ALIASES
from ..logging import logger
from ..schemas import validate_record
from .bids_dataset import _COMPANION_FILES
from .exceptions import DataIntegrityError
from .io import (
    _ANNEX_KEY_RE,
    _convert_time_with_numeric_dash,
    _ensure_coordsystem_symlink,
    _fix_negative_annotation_durations,
    _generate_coordsystem_json,
    _generate_vhdr_from_metadata,
    _generate_vhdr_from_sibling,
    _generate_vmrk_stub,
    _is_annex_placeholder,
    _load_raw_direct,
    _load_raw_eeglab_alleeg,
    _load_raw_eeglab_fallback,
    _load_raw_from_eeglab_epochs,
    _load_raw_snirf_fallback,
    _repair_channels_tsv,
    _repair_channels_tsv_duplicates,
    _repair_eeglab_fdt,
    _repair_events_tsv_nan_samples,
    _repair_participants_tsv_ids,
    _repair_scans_tsv_timestamps,
    _repair_snirf_bids_metadata,
    _repair_tsv_decimal_separators,
    _repair_tsv_encoding,
    _repair_tsv_na_whitespace,
    _repair_vhdr_missing_markerfile,
    _repair_vhdr_pointers,
)

# Error messages indicating unrecoverable data corruption — the source
# file itself is broken and no amount of BIDS-metadata repair will help.
_UNRECOVERABLE_PATTERNS = [
    "Bad EDF",
    "invalid literal for int",
    # NumPy / SciPy array errors from corrupt MAT / EEGLAB files
    "could not read bytes",
    "buffer is too small for requested array",
    "buffer size must be",
    "iteration over a 0-d array",
    "cannot reshape array",
    "setting an array element with a sequence",
    # EEGLAB reader: invalid file extension check
    "EEGLAB file extension",
    # EEGLAB reader errors from non-standard .set structures
    "Allowed values",
    "has no attribute",
    # MNE data read: sample count mismatch (corrupt/truncated data file)
    "Incorrect number of samples",
    # Hardware / format-level issues that no metadata repair can fix
    "incorrect number of samples",
    # Note: "only supports reading continuous" (SNIRF TD-NIRS) was removed
    # from unrecoverable — we handle it via _load_raw_snirf_fallback.
]

_SPLIT_FIF_MISSING_RE = re.compile(
    r"Split raw file detected but next file (?P<path>.+?) does not exist"
)
_SPLIT_ENTITY_RE = re.compile(r"_split-(?P<num>\d+)(?=_)")
_SPLIT_PART_RE = re.compile(r"-(?P<num>\d+)(?=\.fif$)", re.IGNORECASE)


def _clamp_negative_annotation_durations(raw: BaseRaw) -> None:
    """Clamp any negative annotation durations to zero on a loaded Raw.

    Some datasets (especially EEGLAB ``.set`` files) produce annotations
    whose computed duration is negative.  MNE asserts
    ``(self.duration >= 0).all()`` inside ``Annotations.crop``, which
    crashes downstream operations.  This helper fixes them in-place
    after loading so subsequent processing is safe.
    """
    import numpy as np

    annots = raw.annotations
    if annots is None or len(annots) == 0:
        return
    mask = annots.duration < 0
    if np.any(mask):
        n_neg = int(np.sum(mask))
        logger.warning("Clamping %d annotation(s) with negative duration to 0.", n_neg)
        annots.duration[mask] = 0.0


@contextmanager
def _dummy_filelock(*args, **kwargs):
    """No-op context manager used as a stand-in for ``filelock.FileLock``."""
    yield


@contextmanager
def _noop_filelock():
    """Replace ``filelock.FileLock`` with a no-op so mne-bids can read
    sidecar JSON files in read-only directories without creating ``.lock``
    files.
    """
    with patch("filelock.FileLock", _dummy_filelock):
        yield


def _parse_split_fif_missing_path(message: str) -> Path | None:
    """Extract the missing continuation path from an MNE split FIF error."""
    match = _SPLIT_FIF_MISSING_RE.search(message)
    if match is None:
        return None
    return Path(match.group("path").strip().strip("'\""))


def _increment_name_match(name: str, match: re.Match[str]) -> str:
    """Increment the numeric portion captured by ``match`` in ``name``."""
    width = len(match.group("num"))
    next_num = int(match.group("num")) + 1
    return f"{name[: match.start('num')]}{next_num:0{width}d}{name[match.end('num') :]}"


def _iter_split_fif_candidates(
    current_key: str,
    current_path: Path,
    expected_path: Path | None,
) -> list[tuple[str, Path]]:
    """Return candidate remote/local paths for the next FIF continuation."""
    key_path = PurePosixPath(current_key)
    current_name = key_path.name
    candidate_names: list[str] = []

    if expected_path is not None and expected_path.suffix.lower() == ".fif":
        candidate_names.append(expected_path.name)

    if match := _SPLIT_PART_RE.search(current_name):
        candidate_names.append(_increment_name_match(current_name, match))
    else:
        if match := _SPLIT_ENTITY_RE.search(current_name):
            candidate_names.append(_increment_name_match(current_name, match))

        if current_name.lower().endswith(".fif"):
            candidate_names.append(f"{current_name[:-4]}-1{current_name[-4:]}")

    candidates: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for candidate_name in candidate_names:
        if candidate_name == current_name or candidate_name in seen:
            continue
        seen.add(candidate_name)
        candidates.append(
            (
                str(key_path.parent / candidate_name),
                current_path.with_name(candidate_name),
            )
        )
    return candidates


class EEGDashRaw(RawDataset):
    """A single EEG recording dataset.

    Represents a single EEG recording, typically hosted on a remote server (like AWS S3)
    and cached locally upon first access. This class is a subclass of
    :class:`braindecode.datasets.BaseDataset` and can be used with braindecode's
    preprocessing and training pipelines.

    Parameters
    ----------
    record : dict
        A v2 record containing all metadata and storage information.
        Must have schema_version=2 and include storage.base (no default bucket).
    cache_dir : str
        The local directory where the data will be cached.
    on_error : str, default "raise"
        How to handle :class:`DataIntegrityError` when accessing ``.raw``:

        - ``"raise"`` (default): propagate the exception.
        - ``"warn"``: log the error as a warning and set ``.raw`` to ``None``.
        - ``"skip"``: silently set ``.raw`` to ``None``.
    **kwargs
        Additional keyword arguments passed to the
        :class:`braindecode.datasets.BaseDataset` constructor.

    Raises
    ------
    ValueError
        If the record is not a valid v2 record or is missing required fields.

    """

    def __init__(
        self,
        record: dict[str, Any],
        cache_dir: str,
        **kwargs,
    ):
        self._on_error = kwargs.pop("on_error", "raise")
        self._skipped = False
        self._integrity_error = None
        super().__init__(None, **kwargs)
        self.cache_dir = Path(cache_dir)

        # Validate record
        errors = validate_record(record)
        if errors:
            raise ValueError(f"Invalid record: {errors}")

        self.record = record

        # Derive local cache paths from record fields (portable - no absolute paths stored)
        storage = self.record.get("storage", {})
        dataset_id = self.record["dataset"]
        bids_relpath = self.record["bids_relpath"]
        dep_keys = storage.get("dep_keys", [])

        # Robust root resolution: check if a folder with the dataset_id exists,
        # or if there's a unique folder that "matches" the dataset (e.g. ds0001mini)
        self.bids_root = self.cache_dir / dataset_id

        self.filecache = self.bids_root / bids_relpath
        self._dep_paths = [self.bids_root / p for p in dep_keys]

        # Build remote URIs based on storage backend
        backend = storage.get("backend")
        base = storage.get("base", "").rstrip("/")
        raw_key = storage.get("raw_key", "")
        dep_keys = storage.get("dep_keys") or []

        if backend in ("s3", "https") and base and raw_key:
            self._raw_uri = f"{base}/{raw_key}"
            self._dep_uris = [f"{base}/{k}" for k in dep_keys]
        elif backend == "local" and base:
            # Local backend: data already exists at storage.base
            local_base = Path(base)
            self.bids_root = local_base
            self.filecache = local_base / raw_key if raw_key else self.filecache
            self._dep_paths = (
                [local_base / k for k in dep_keys] if dep_keys else self._dep_paths
            )
            self._raw_uri = None
            self._dep_uris = []
        else:
            self._raw_uri = None
            self._dep_uris = []

        if not self.bids_root.exists() and self._raw_uri:
            self.bids_root.mkdir(parents=True, exist_ok=True)

        # Public-ish attribute used in tests; now reflects the actual remote URI.
        self.s3file = self._raw_uri

        entities_mne = self.record.get("entities_mne") or {}

        # Fallback: parse acquisition from bids_relpath for records
        # that predate the acquisition field in entities_mne
        acq_val = entities_mne.get("acquisition")
        if acq_val is None:
            _acq_match = re.search(r"acq-([^_/]+)", self.record.get("bids_relpath", ""))
            if _acq_match:
                acq_val = _acq_match.group(1)

        self.bidspath = BIDSPath(
            root=self.bids_root,
            datatype=MODALITY_ALIASES.get(
                self.record.get("datatype", "eeg"), self.record.get("datatype", "eeg")
            ),
            suffix=MODALITY_ALIASES.get(
                self.record.get("suffix", "eeg"), self.record.get("suffix", "eeg")
            ),
            extension=self.record.get("extension", self.filecache.suffix),
            subject=entities_mne.get("subject"),
            session=entities_mne.get("session"),
            task=entities_mne.get("task"),
            run=entities_mne.get("run"),
            acquisition=acq_val,
            split=entities_mne.get("split"),
            check=False,
        )

        self._raw = None

    def _has_non_numeric_run(self) -> bool:
        """Return True when the original BIDS run entity exists but is non-numeric.

        MNE-BIDS requires numeric run values.  Non-numeric values like
        ``"5H"`` cause ``BIDSPath`` to fail, so we detect them here to
        trigger a direct-loading fallback.
        """
        entities = self.record.get("entities") or {}
        run = entities.get("run")
        return run is not None and not str(run).isdigit()

    def _resolve_annex_key_uri(self, uri: str) -> str | None:
        """Return a BIDS-named URI when *uri* contains a git-annex key filename.

        Git-annex stores files under hash-based names like
        ``MD5E-s11657--7a519e74754041a678931b7b7d72f0ab.vhdr``.  When the
        ingestion pipeline records these as S3 keys the download will fail
        because OpenNeuro stores objects under their BIDS names.  This
        method detects such keys and derives the correct URI from
        ``bids_relpath``.
        """
        filename = PurePosixPath(uri).name
        if not _ANNEX_KEY_RE.match(filename):
            return None
        bids_filename = PurePosixPath(self.record["bids_relpath"]).name
        return uri.rsplit("/", 1)[0] + "/" + bids_filename

    def _download_required_files(self) -> None:
        if self._raw_uri is not None:
            filesystem = downloader.get_s3_filesystem()

            # Download deps first (sidecars, companions), then raw.
            # skip_missing=True because dep_keys may include companion files
            # that don't exist on S3 (e.g., .fdt listed but never uploaded).
            downloader.download_files(
                list(zip(self._dep_uris, self._dep_paths, strict=False)),
                filesystem=filesystem,
                skip_existing=True,
                skip_missing=True,
            )
            try:
                downloader.download_s3_file(
                    self._raw_uri, self.filecache, filesystem=filesystem
                )
            except FileNotFoundError:
                # If the URI contains a git-annex key, try the BIDS-named
                # alternative before giving up.
                resolved_uri = self._resolve_annex_key_uri(self._raw_uri)
                if resolved_uri:
                    logger.info(
                        "Raw URI %s contains git-annex key; trying BIDS name: %s",
                        self._raw_uri,
                        resolved_uri,
                    )
                    try:
                        downloader.download_s3_file(
                            resolved_uri,
                            self.filecache,
                            filesystem=filesystem,
                        )
                    except FileNotFoundError:
                        raise DataIntegrityError(
                            message=(
                                "Primary data file not found on S3 "
                                "(tried annex key and BIDS name): "
                                f"{self._raw_uri}"
                            ),
                            record=self.record,
                            issues=[
                                f"Missing S3 file: {self._raw_uri}",
                                f"Also tried BIDS name: {resolved_uri}",
                            ],
                        )
                else:
                    raise DataIntegrityError(
                        message=(f"Primary data file not found on S3: {self._raw_uri}"),
                        record=self.record,
                        issues=[f"Missing S3 file: {self._raw_uri}"],
                    )

            # Auto-discover and download companion files (.fdt, .eeg, .vmrk)
            # that may not have been included in dep_keys.
            self._download_companion_files(filesystem)

        # CTF MEG .ds directories require internal files (.meg4, .res4, etc.)
        if self.filecache and self.filecache.suffix == ".ds":
            self._ensure_ctf_directory_complete()

        # Always set filenames (important for local datasets)
        self.filenames = [self.filecache]

    def _download_companion_files(self, filesystem) -> None:
        """Download companion files for formats that require them.

        Some EEG file formats store data across multiple files (e.g.,
        EEGLAB ``.set`` + ``.fdt``, BrainVision ``.vhdr`` + ``.eeg`` +
        ``.vmrk``).  When the ingestion pipeline does not list these in
        ``dep_keys``, they are never fetched.  This method inspects
        ``_COMPANION_FILES`` and attempts to download any missing
        companions from S3.
        """
        suffix = self.filecache.suffix.lower()
        companions = _COMPANION_FILES.get(suffix)
        if not companions:
            return

        # Use BIDS-named base URI when the raw URI has a git-annex key,
        # so companion URIs also resolve to real S3 objects.
        base_uri = self._raw_uri
        if _ANNEX_KEY_RE.match(PurePosixPath(self._raw_uri).name):
            bids_filename = PurePosixPath(self.record["bids_relpath"]).name
            base_uri = self._raw_uri.rsplit("/", 1)[0] + "/" + bids_filename

        for ext in companions:
            local_path = self.filecache.with_suffix(ext)
            if local_path.exists():
                continue

            # Derive S3 URI by replacing the primary file's extension
            companion_uri = base_uri.rsplit(".", 1)[0] + ext

            try:
                downloader.download_s3_file(
                    companion_uri, local_path, filesystem=filesystem
                )
            except FileNotFoundError:
                # For .set files, the .fdt may have a non-BIDS name embedded
                # in the header (e.g. "1673.s.1hzHighpass.fdt").
                if suffix == ".set" and ext == ".fdt":
                    self._download_embedded_fdt(filesystem)
                else:
                    logger.warning(
                        "Companion file %s not found on S3 for %s",
                        ext,
                        base_uri,
                    )
                continue

    def _download_embedded_fdt(self, filesystem) -> None:
        """Download a non-BIDS-named ``.fdt`` referenced inside a ``.set`` header.

        Some EEGLAB ``.set`` files store data in ``.fdt`` files whose names
        do not follow BIDS conventions (e.g. ``1673.s.1hzHighpass.fdt``
        instead of ``sub-1673_task-Baseline_eeg.fdt``).  This parses the
        ``.set`` MATLAB header to discover the actual filename and
        downloads it from the same S3 directory.
        """
        try:
            import scipy.io

            mat = scipy.io.loadmat(
                str(self.filecache), struct_as_record=False, squeeze_me=True
            )
            eeg = mat.get("EEG")
            if eeg is None or not hasattr(eeg, "datfile") or not eeg.datfile:
                return
            datfile = eeg.datfile
            if hasattr(datfile, "item"):
                datfile = datfile.item()
            datfile = str(datfile)
        except Exception:
            return

        local_path = self.filecache.parent / datfile
        if local_path.exists():
            return

        base_uri = self._raw_uri.rsplit("/", 1)[0]
        fdt_uri = f"{base_uri}/{datfile}"
        try:
            downloader.download_s3_file(fdt_uri, local_path, filesystem=filesystem)
            logger.info("Downloaded embedded .fdt companion: %s", datfile)
        except FileNotFoundError:
            logger.warning(
                "Embedded .fdt %s not found on S3 for %s",
                datfile,
                self._raw_uri,
            )

    def _ensure_ctf_directory_complete(self) -> None:
        """Verify a CTF ``.ds`` directory has the required internal files.

        CTF MEG datasets are stored as directories containing ``.meg4``
        (data), ``.res4`` (header), and other supporting files.  When
        downloading from S3 the individual dependency keys may not
        include all internal files.  This method checks for the minimum
        required files and, if missing, attempts a recursive download of
        the entire ``.ds`` S3 prefix.
        """
        ds_dir = self.filecache
        if not ds_dir.exists():
            ds_dir.mkdir(parents=True, exist_ok=True)

        required_exts = (".meg4", ".res4")
        missing = [ext for ext in required_exts if not list(ds_dir.glob(f"*{ext}"))]

        if not missing:
            return

        # Attempt recursive S3 download of the full .ds prefix
        if self._raw_uri is not None:
            logger.info(
                "CTF .ds directory missing %s — attempting recursive S3 download.",
                ", ".join(missing),
            )
            try:
                filesystem = downloader.get_s3_filesystem()
                # Strip protocol prefix for s3fs operations
                s3_prefix = re.sub(r"^s3://", "", self._raw_uri)
                remote_files = filesystem.ls(s3_prefix, detail=False)
                pairs = []
                for remote in remote_files:
                    fname = Path(remote).name
                    local = ds_dir / fname
                    pairs.append((f"s3://{remote}", local))
                if pairs:
                    downloader.download_files(
                        pairs, filesystem=filesystem, skip_existing=True
                    )
            except Exception as e:
                logger.warning("Recursive CTF download failed: %s", e)

        # Re-check after download attempt
        still_missing = [
            ext for ext in required_exts if not list(ds_dir.glob(f"*{ext}"))
        ]
        if still_missing:
            raise DataIntegrityError(
                message=(
                    f"CTF .ds directory incomplete — missing {', '.join(still_missing)} "
                    f"in {ds_dir}"
                ),
                record=self.record,
                issues=[f"Missing required CTF file(s): {', '.join(still_missing)}"],
            )

    def _ensure_raw(self) -> None:
        """Ensure the raw data file and its dependencies are cached locally.

        Warns if the record has known data integrity issues but continues loading.
        """
        # Check for data integrity issues and warn (but don't block loading)
        if self.record.get("_has_missing_files"):
            DataIntegrityError.warn_from_record(self.record)

        self._download_required_files()

        # Helper: Fix participants.tsv ID padding to match sub-* folder names
        if self.bids_root and self.bids_root.exists():
            _repair_participants_tsv_ids(self.bids_root)

        # Helper: Fix MNE-BIDS strictness regarding coordsystem.json location
        if self.filecache and self.filecache.parent.exists():
            _ensure_coordsystem_symlink(self.filecache.parent)
            _repair_tsv_encoding(self.filecache.parent)
            _repair_tsv_na_whitespace(self.filecache.parent)
            _repair_tsv_decimal_separators(self.filecache.parent)
            _repair_scans_tsv_timestamps(self.filecache.parent)
            _repair_events_tsv_nan_samples(self.filecache.parent)
            _repair_channels_tsv(self.filecache.parent)

        # Helper: Handle VHDR files - generate if missing, repair if broken
        if self.filecache and self.filecache.suffix == ".vhdr":
            # Early exit: when the data file (.eeg) is a git-annex
            # placeholder (content never fetched), no header repair can
            # help — the binary data is simply not available.
            eeg_path = self.filecache.with_suffix(".eeg")
            if _is_annex_placeholder(eeg_path) or (
                eeg_path.exists() and eeg_path.stat().st_size == 0
            ):
                raise DataIntegrityError(
                    message=(
                        f"BrainVision data file is a git-annex placeholder "
                        f"(content unavailable): {eeg_path.name}"
                    ),
                    record=self.record,
                    issues=[
                        f"Data file {eeg_path.name} contains placeholder "
                        f"content instead of actual EEG data",
                        "The git-annex content was likely never fetched or "
                        "has been dropped from this clone",
                    ],
                )

            # Generate VMRK stub first so pointer repair can find the target
            vmrk_path = self.filecache.with_suffix(".vmrk")
            if not vmrk_path.exists():
                _generate_vmrk_stub(vmrk_path, self.filecache.name)

            if not self.filecache.exists():
                # Generate VHDR from database metadata if file is missing
                if not _generate_vhdr_from_metadata(self.filecache, self.record):
                    _generate_vhdr_from_sibling(self.filecache)
            else:
                # Check if the VHDR contains required sections; if completely
                # corrupted, regenerate before trying pointer repairs.
                try:
                    vhdr_text = self.filecache.read_text(encoding="utf-8")
                except Exception:
                    vhdr_text = ""
                if (
                    "[Common Infos]" not in vhdr_text
                    and "[Common infos]" not in vhdr_text
                ):
                    logger.warning(
                        "VHDR file %s is corrupted, regenerating from metadata.",
                        self.filecache.name,
                    )
                    if not _generate_vhdr_from_metadata(self.filecache, self.record):
                        _generate_vhdr_from_sibling(self.filecache)

                # Auto-Repair broken VHDR pointers (common in OpenNeuro exports)
                _repair_vhdr_pointers(self.filecache)

        # Repair .set header: fix .fdt filename mismatches and truncated data
        if self.filecache and self.filecache.suffix.lower() == ".set":
            _repair_eeglab_fdt(self.filecache)

        if self._raw is None:
            try:
                self._raw = self._load_raw()
            except DataIntegrityError as e:
                e.log_warning()
                raise
            except Exception as e:
                logger.error(
                    f"Error reading {self.bidspath}: {e}. Try `rm -rf {self.bids_root}`"
                )
                raise

            # Clamp any negative annotation durations to 0.  Some files
            # (especially EEGLAB .set) produce annotations with negative
            # durations that crash downstream MNE operations.
            _clamp_negative_annotation_durations(self._raw)

            # Validate that data is actually readable (catches corrupt/truncated
            # data files that MNE only discovers during lazy segment reads).
            # Read from start and end to catch truncated files.
            try:
                n = self._raw.n_times
                if n > 0:
                    self._raw.get_data(start=0, stop=min(1, n))
                    if n > 1:
                        self._raw.get_data(start=n - 1, stop=n)
            except Exception as e:
                self._raw = None
                raise DataIntegrityError(
                    message=f"Data file unreadable: {e}",
                    record=self.record,
                    issues=[str(e)],
                ) from e

    def _read_raw_bids(self, extra_params: dict | None = None) -> BaseRaw:
        """Call ``mne_bids.read_raw_bids`` with the standard arguments.

        Uses a no-op file lock to avoid ``PermissionError`` when the dataset
        directory is read-only (e.g. shared cluster storage where mne-bids
        cannot create ``.json.lock`` files).
        """
        with _noop_filelock():
            return mne_bids.read_raw_bids(
                bids_path=self.bidspath,
                extra_params=extra_params,
                verbose="ERROR",
                on_ch_mismatch="rename",
            )

    def _download_split_fif_continuation(
        self,
        *,
        current_key: str,
        current_path: Path,
        error_message: str,
        attempted_keys: set[str],
    ) -> tuple[str, Path] | None:
        """Download the next split FIF continuation file, if it exists."""
        storage = self.record.get("storage", {})
        backend = storage.get("backend")
        base = str(storage.get("base", "")).rstrip("/")
        if backend not in ("s3", "https") or not base:
            return None

        expected_path = _parse_split_fif_missing_path(error_message)
        candidates = _iter_split_fif_candidates(
            current_key=current_key,
            current_path=current_path,
            expected_path=expected_path,
        )
        if not candidates:
            return None

        filesystem = downloader.get_s3_filesystem()
        failures: list[str] = []
        for next_key, next_path in candidates:
            if next_key in attempted_keys:
                continue
            attempted_keys.add(next_key)
            next_uri = downloader.get_s3path(base, next_key)
            try:
                logger.info(
                    "Split FIF continuation missing; attempting download %s -> %s",
                    next_uri,
                    next_path,
                )
                downloader.download_s3_file(
                    next_uri,
                    next_path,
                    filesystem=filesystem,
                )
                # When the primary file is resolved through git-annex, MNE
                # expects the continuation at the annex path, not the BIDS
                # path.  Place a copy there so MNE can find it on retry.
                if (
                    expected_path is not None
                    and expected_path.resolve() != next_path.resolve()
                ):
                    try:
                        expected_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(next_path, expected_path)
                        logger.info(
                            "Copied continuation to expected path %s",
                            expected_path,
                        )
                    except Exception as copy_err:
                        logger.warning(
                            "Failed to copy continuation to expected path %s: %s",
                            expected_path,
                            copy_err,
                        )
                return next_key, next_path
            except Exception as error:
                failures.append(f"{next_uri}: {error}")

        if failures:
            logger.warning(
                "Unable to download split FIF continuation for %s. Tried %s",
                current_key,
                "; ".join(failures),
            )
        return None

    def _load_raw(self) -> BaseRaw:
        """Load raw data, preferring MNE-BIDS if BIDSPath resolves.

        Applies on-the-fly fixes and retries for known failure modes:

        - **Missing or wrong-key ``coordsystem.json``**: mne-bids raises
          ``RuntimeError`` or ``KeyError`` when the file is absent or has keys
          for the wrong datatype (e.g. ``EEGCoordinateSystem`` in iEEG data).
          We (re-)generate a minimal ``coordsystem.json`` with the correct
          datatype-specific keys and retry.
        - **SNIRF** metadata issues: regenerates ``channels.tsv`` /
          ``scans.tsv`` and retries.
        - **EEGLAB char-encoding** (``.set`` files with non-UTF char fields):
          ``scipy.io.loadmat`` raises ``TypeError: buffer is too small``.
          Retries with ``uint16_codec='latin-1'`` via ``extra_params``.
        - **Unrecoverable corruption** (Bad EDF, empty MEG data, corrupt
          MAT/EEGLAB files with array errors, etc.): attempts EEGLAB fallback
          for ``.set`` files and direct MNE reader before raising
          ``DataIntegrityError``.
        - **Invalid scans.tsv timestamps** (seconds >= 60, NaN): repairs
          the scans.tsv and retries.
        - **participants.tsv subject mismatches**: repairs ``participant_id``
          values to match ``sub-*`` folder names and retries.
        - **events.tsv rows with NaN onset/sample**: drops broken rows and
          retries, then falls back to direct MNE loading if needed.
        - **Split FIF continuations missing locally**: derives the next split
          key, downloads it from remote storage, and retries.
        - **Invalid BIDS entity characters** (hyphens in task, etc.):
          falls back to direct MNE reader.
        - **Non-numeric ``run`` entity** (e.g. ``run-5H``): MNE-BIDS rejects
          non-integer run values. Falls back to direct MNE reader for any
          file format.
        - **CTF "Illegal date"** (numeric dash dates like 14-10-1925): patches
          MNE's CTF date parser to try %d-%m-%Y and retries.
        - **Empty/malformed channels.tsv** (KeyError: 'name'): removes empty
          files or renames the first column to 'name' and retries.
        - **Duplicate channel names in channels.tsv**: deduplicates by appending
          ``-0``, ``-1``, … suffixes and retries.
        - **Missing sidecars** (channels.tsv, events.tsv absent from S3):
          falls back to direct MNE reader.
        - **Split FIF** (record points to split-02+ file): direct MNE reader
          loads with ``on_split_missing="warn"`` so partial data is returned.
        - **Bad record metadata** (e.g. ``.json`` extension, missing ``task``
          entity): attempts direct MNE reader before raising
          ``DataIntegrityError``.
        """
        current_split_key = self.record.get("storage", {}).get("raw_key")
        current_split_path = self.filecache
        attempted_split_keys: set[str] = set()

        while True:
            try:
                return self._read_raw_bids()
            except NotImplementedError as first_error:
                msg = str(first_error)
                if (
                    "ALLEEG" in msg
                    and self.filecache
                    and self.filecache.suffix.lower() == ".set"
                ):
                    logger.info(
                        "EEGLAB file contains ALLEEG array; loading first dataset (bypassing MNE-BIDS)."
                    )
                    try:
                        return _load_raw_eeglab_alleeg(self.filecache)
                    except Exception as fallback_error:
                        raise fallback_error from first_error
                raise
            except RuntimeError as first_error:
                if "coordsystem.json" in str(first_error):
                    return self._retry_with_generated_coordsystem(first_error)
                if "Illegal date" in str(first_error):
                    return self._retry_with_ctf_date_patch(first_error)

                msg = str(first_error)

                # SNIRF channel mismatch — fall back to direct reader.
                # mne-bids sometimes fails even with on_ch_mismatch="rename"
                # when the channels.tsv naming convention diverges significantly
                # from the SNIRF internal channel names.
                if (
                    "Channel mismatch" in msg
                    and self.filecache
                    and self.filecache.suffix.lower() == ".snirf"
                ):
                    logger.warning(
                        "SNIRF channel mismatch — falling back to direct reader."
                    )
                    try:
                        return _load_raw_direct(self.filecache)
                    except Exception as fallback_error:
                        raise DataIntegrityError(
                            message=f"SNIRF channel mismatch and direct reader failed: {fallback_error}",
                            record=self.record,
                            issues=[str(fallback_error)],
                        ) from first_error

                # CTF mandatory HPI coil-kind mismatch
                if "mandatory HPI" in msg and self.filecache:
                    return self._retry_with_ctf_hpi_fix(first_error)

                # Projector channel type conflict during MNE-BIDS channel
                # renaming — the direct reader bypasses this.
                if "in projector" in msg and self.filecache:
                    return self._retry_without_projectors(first_error)

                # CTF trial-size mismatch — the meg4 file is truncated
                # (fewer samples than res4 header declares).  Patch
                # MNE's _get_sample_info to treat available data as a
                # single continuous block.
                if "even multiple of the trial size" in msg and self.filecache:
                    logger.warning(
                        "CTF trial size mismatch — retrying with "
                        "truncation-tolerant sample info."
                    )
                    try:
                        return self._retry_with_ctf_truncated_fix(first_error)
                    except DataIntegrityError:
                        raise
                    except Exception as fallback_error:
                        raise DataIntegrityError(
                            message=f"CTF trial size mismatch and all fallbacks failed: {fallback_error}",
                            record=self.record,
                            issues=[str(fallback_error)],
                        ) from first_error

                # SNIRF TD-NIRS: MNE rejects dataType 301 but the raw
                # time-series is usable via h5py direct read.
                if (
                    "only supports reading continuous" in msg
                    and self.filecache
                    and self.filecache.suffix.lower() == ".snirf"
                ):
                    logger.warning(
                        "Unsupported SNIRF data type — loading via h5py fallback."
                    )
                    try:
                        return _load_raw_snirf_fallback(self.filecache)
                    except Exception as fallback_error:
                        raise DataIntegrityError(
                            message=f"SNIRF h5py fallback failed: {fallback_error}",
                            record=self.record,
                            issues=[str(fallback_error)],
                        ) from first_error

                # Unrecoverable patterns in RuntimeError
                if any(p in msg for p in _UNRECOVERABLE_PATTERNS):
                    # For .set files, try manual EEGLAB parser before giving up
                    if self.filecache and self.filecache.suffix.lower() == ".set":
                        try:
                            return _load_raw_eeglab_fallback(
                                self.filecache, bids_root=self.bids_root
                            )
                        except Exception:
                            pass

                    # For supported formats, try direct MNE reader (bypasses BIDS)
                    if self.filecache:
                        try:
                            return _load_raw_direct(self.filecache)
                        except Exception:
                            pass

                    raise DataIntegrityError(
                        message=f"Cannot read data file: {msg}",
                        record=self.record,
                        issues=[msg],
                    ) from first_error

                # Bad record metadata (e.g. .json extension, missing task entity)
                if "must contain" in msg and "task" in msg:
                    if self.filecache:
                        logger.warning(
                            "Missing 'task' entity in BIDSPath — "
                            "falling back to direct reader."
                        )
                        try:
                            return _load_raw_direct(self.filecache)
                        except Exception as fallback_error:
                            raise DataIntegrityError(
                                message=f"Bad record metadata and direct reader failed: {msg}",
                                record=self.record,
                                issues=[msg, str(fallback_error)],
                            ) from first_error
                    raise DataIntegrityError(
                        message=f"Bad record metadata: {msg}",
                        record=self.record,
                        issues=[msg],
                    ) from first_error
                raise
            except (
                TypeError,
                ValueError,
                OSError,
                KeyError,
                AttributeError,
            ) as first_error:
                msg = str(first_error)

                if (
                    current_split_key
                    and current_split_path
                    and ("Split raw file detected but next file" in msg)
                ):
                    split_download = self._download_split_fif_continuation(
                        current_key=current_split_key,
                        current_path=current_split_path,
                        error_message=msg,
                        attempted_keys=attempted_split_keys,
                    )
                    if split_download is not None:
                        current_split_key, current_split_path = split_download
                        continue

                    # All download attempts failed — fall back to direct
                    # reader which uses on_split_missing="warn" for .fif
                    # files, loading only the available splits.
                    if self.filecache:
                        logger.warning(
                            "Split FIF continuation download failed — "
                            "falling back to direct reader."
                        )
                        try:
                            return _load_raw_direct(self.filecache)
                        except Exception as fallback_error:
                            raise DataIntegrityError(
                                message=(
                                    f"Split FIF continuation missing and "
                                    f"direct reader failed: {fallback_error}"
                                ),
                                record=self.record,
                                issues=[msg, str(fallback_error)],
                            ) from first_error

                # Non-UTF-8 encoding in TSV sidecar files (e.g. µ in Latin-1)
                if isinstance(first_error, UnicodeDecodeError) and self.filecache:
                    data_dir = self.filecache.parent
                    if _repair_tsv_encoding(data_dir):
                        logger.info("Repaired non-UTF-8 TSV encoding, retrying load...")
                        try:
                            return self._read_raw_bids()
                        except Exception as retry_error:
                            raise retry_error from first_error

                # CTF "no valid samples" — MNE misinterprets a zeroed
                # system clock (SCLK01) channel as empty data (OSError).
                # Retry with system_clock="ignore" to bypass the SCLK check.
                if (
                    isinstance(first_error, OSError)
                    and ("no valid samples" in msg or "Could not find any data" in msg)
                    and self.filecache
                    and self.filecache.suffix.lower() == ".ds"
                ):
                    import mne

                    logger.warning(
                        "CTF no valid samples — retrying with system_clock='ignore'."
                    )
                    try:
                        return mne.io.read_raw_ctf(
                            str(self.filecache),
                            system_clock="ignore",
                            preload=False,
                            verbose="ERROR",
                        )
                    except Exception as fallback_error:
                        raise DataIntegrityError(
                            message=f"CTF no valid samples (system_clock=ignore also failed): {fallback_error}",
                            record=self.record,
                            issues=[str(fallback_error)],
                        ) from first_error

                # FIFFV_COIL_NONE KeyError from MNE-BIDS montage setting —
                # the data is readable, just the montage lookup fails.
                # Also catches re-raised errors from _retry_with_ctf_hpi_fix.
                if (
                    isinstance(first_error, KeyError)
                    and ("FIFFV_COIL_NONE" in msg or first_error.args == (0,))
                    and self.filecache
                ):
                    logger.warning(
                        "FIFFV_COIL_NONE montage error — falling back to direct reader."
                    )
                    try:
                        return _load_raw_direct(self.filecache)
                    except Exception as fallback_error:
                        raise DataIntegrityError(
                            message=f"FIFFV_COIL_NONE and direct reader failed: {fallback_error}",
                            record=self.record,
                            issues=[str(fallback_error)],
                        ) from first_error

                # Wrong keys in coordsystem.json (e.g., EEGCoordinateSystem for iEEG data)
                if isinstance(first_error, KeyError) and "CoordinateSystem" in msg:
                    return self._retry_with_generated_coordsystem(first_error)

                # Malformed channels.tsv (empty or missing 'name' column)
                if isinstance(first_error, KeyError) and first_error.args == ("name",):
                    if self.filecache and _repair_channels_tsv(self.filecache.parent):
                        logger.info("Repaired malformed channels.tsv, retrying load...")
                        try:
                            return self._read_raw_bids()
                        except Exception as retry_error:
                            raise retry_error from first_error

                # Duplicate channel names in channels.tsv
                if (
                    isinstance(first_error, ValueError)
                    and "not unique" in msg
                    and "renaming" in msg
                    and self.filecache
                ):
                    if _repair_channels_tsv_duplicates(self.filecache.parent):
                        logger.info(
                            "Deduplicated channel names in channels.tsv, "
                            "retrying load..."
                        )
                        try:
                            return self._read_raw_bids()
                        except Exception as retry_error:
                            raise retry_error from first_error

                # EEGLAB epoch files: trials > 1 → use read_epochs_eeglab
                if (
                    isinstance(first_error, TypeError)
                    and "number of trials is" in msg
                    and self.filecache
                    and self.filecache.suffix.lower() == ".set"
                ):
                    logger.info(
                        "EEGLAB file contains epochs (trials > 1), "
                        "loading via read_epochs_eeglab."
                    )
                    try:
                        return _load_raw_from_eeglab_epochs(self.filecache)
                    except Exception as fallback_error:
                        raise DataIntegrityError(
                            message=f"Cannot read epoched EEGLAB file: {fallback_error}",
                            record=self.record,
                            issues=[str(fallback_error)],
                        ) from first_error

                # EEGLAB .set files with non-UTF char fields: scipy.io.loadmat
                # crashes with "buffer is too small" in read_char.  Retry with
                # uint16_codec='latin-1' before giving up.
                if (
                    isinstance(first_error, TypeError)
                    and "buffer is too small" in msg
                    and self.filecache
                    and self.filecache.suffix.lower() == ".set"
                ):
                    logger.info(
                        "EEGLAB char-encoding error, retrying with "
                        "uint16_codec='latin-1'..."
                    )
                    try:
                        return self._read_raw_bids(
                            extra_params={"uint16_codec": "latin-1"}
                        )
                    except Exception as retry_error:
                        raise retry_error from first_error

                # Unrecoverable data corruption (bad EDF, empty MEG, corrupt MAT,
                # or any TypeError/AttributeError from array/parsing failures
                # in scipy/numpy)
                if any(p in msg for p in _UNRECOVERABLE_PATTERNS) or isinstance(
                    first_error, (TypeError, AttributeError)
                ):
                    # Try EEGLAB fallback for .set files before giving up
                    if self.filecache and self.filecache.suffix.lower() == ".set":
                        try:
                            return _load_raw_eeglab_fallback(
                                self.filecache, bids_root=self.bids_root
                            )
                        except Exception:
                            pass  # Fall through to DataIntegrityError

                    raise DataIntegrityError(
                        message=f"Cannot read data file: {msg}",
                        record=self.record,
                        issues=[msg],
                    ) from first_error

                # Projector channels not found in data — bypass MNE-BIDS
                # channel renaming and remove projectors from the raw object.
                if "projector channels not found" in msg and self.filecache:
                    return self._retry_without_projectors(first_error)

                # Invalid timestamp in scans.tsv (seconds >= 60, NaN, etc.)
                if "second must be" in msg or "does not match format" in msg:
                    if self.filecache and _repair_scans_tsv_timestamps(
                        self.filecache.parent
                    ):
                        logger.info("Repaired scans.tsv timestamps, retrying load...")
                        try:
                            return self._read_raw_bids()
                        except Exception as retry_error:
                            raise retry_error from first_error

                # scans.tsv path mismatch (EEG file not listed in scans.tsv)
                if (
                    "is not in list" in msg
                    and ("Did you mean" in msg or "PosixPath" in msg)
                    and self.filecache
                ):
                    logger.warning(
                        "scans.tsv path mismatch — falling back to direct reader."
                    )
                    try:
                        return _load_raw_direct(self.filecache)
                    except Exception as fallback_error:
                        raise fallback_error from first_error

                # Subject/session ID mismatch between folders and participants.tsv
                if "is not in list" in msg and self.bids_root:
                    if _repair_participants_tsv_ids(self.bids_root):
                        logger.info(
                            "Repaired participants.tsv ID padding, retrying load..."
                        )
                        try:
                            return self._read_raw_bids()
                        except Exception as retry_error:
                            raise retry_error from first_error

                # Whitespace-padded n/a in TSV fields (float('n/a      ') fails)
                if (
                    "could not convert string to float" in msg
                    and "n/a" in msg
                    and self.filecache
                ):
                    if _repair_tsv_na_whitespace(self.filecache.parent):
                        logger.info(
                            "Repaired n/a whitespace in TSV files, retrying load..."
                        )
                        try:
                            return self._read_raw_bids()
                        except Exception as retry_error:
                            raise retry_error from first_error

                # NaN onset/sample in events.tsv (mne-bids tries int(NaN))
                if "cannot convert float NaN to integer" in msg and self.filecache:
                    if _repair_events_tsv_nan_samples(self.filecache.parent):
                        logger.info("Repaired events.tsv NaN samples, retrying load...")
                        try:
                            return self._read_raw_bids()
                        except Exception as retry_error:
                            raise retry_error from first_error

                # Invalid BIDS entity characters (hyphens/underscores in task, etc.)
                if "Unallowed" in msg and self.filecache:
                    try:
                        return _load_raw_direct(self.filecache)
                    except Exception as fallback_error:
                        raise fallback_error from first_error

                # Non-numeric run (ValueError from MNE-BIDS entity validation)
                if self.filecache and self._has_non_numeric_run():
                    logger.warning(
                        "MNE-BIDS failed for file with non-numeric run, "
                        "falling back to direct MNE reader."
                    )
                    try:
                        return _load_raw_direct(self.filecache)
                    except Exception as fallback_error:
                        raise fallback_error from first_error

                # Missing sidecar or data file (FileNotFoundError is a subclass
                # of OSError) — bypass MNE-BIDS and load directly with MNE.
                if isinstance(first_error, FileNotFoundError) and self.filecache:
                    logger.warning(
                        "MNE-BIDS failed due to missing file (%s), "
                        "falling back to direct MNE reader.",
                        msg,
                    )
                    try:
                        return _load_raw_direct(self.filecache)
                    except FileNotFoundError as fallback_error:
                        # Both MNE-BIDS and direct reader can't find a
                        # required file (e.g. .fdt missing from S3).
                        raise DataIntegrityError(
                            message=f"Required data file missing: {fallback_error}",
                            record=self.record,
                            issues=[str(fallback_error)],
                        ) from first_error
                    except Exception as fallback_error:
                        raise fallback_error from first_error

                raise
            except AssertionError as first_error:
                # Annotation assertion errors (negative duration) — try
                # loading with the duration-clamping monkey-patch first;
                # only hide events.tsv as a last resort.
                try:
                    with _fix_negative_annotation_durations():
                        return self._read_raw_bids()
                except Exception:
                    pass
                # Still failing — hide the events file and retry.
                if self.filecache and self.filecache.parent.exists():
                    return self._retry_without_events_tsv(first_error)
                raise
            except Exception as first_error:
                # Missing MarkerFile entry in VHDR [Common Infos]
                if (
                    isinstance(first_error, configparser.NoOptionError)
                    and "markerfile" in str(first_error).lower()
                    and self.filecache
                    and self.filecache.suffix.lower() == ".vhdr"
                ):
                    if _repair_vhdr_missing_markerfile(self.filecache):
                        try:
                            return self._read_raw_bids()
                        except Exception as retry_error:
                            raise retry_error from first_error

                # Corrupted VHDR file (missing required sections)
                if isinstance(first_error, configparser.NoSectionError):
                    raise DataIntegrityError(
                        message=f"Corrupted header file: {first_error}",
                        record=self.record,
                        issues=[str(first_error)],
                    ) from first_error

                # EDF annotation encoding error — retry with latin1
                if "encoding='latin1'" in str(first_error) and self.filecache:
                    logger.info(
                        "EDF annotation encoding error, retrying with encoding='latin1'..."
                    )
                    try:
                        return self._read_raw_bids(extra_params={"encoding": "latin1"})
                    except Exception as retry_error:
                        raise retry_error from first_error

                # IndexError from scans.tsv handling (empty file list) —
                # fall back to direct reader
                if isinstance(first_error, IndexError) and self.filecache:
                    logger.warning(
                        "IndexError during BIDS loading (likely malformed scans.tsv) "
                        "— falling back to direct reader."
                    )
                    try:
                        return _load_raw_direct(self.filecache)
                    except Exception as fallback_error:
                        raise fallback_error from first_error

                # For SNIRF files, try to fix and retry
                if self.filecache and self.filecache.suffix.lower() == ".snirf":
                    logger.warning(
                        "Initial load failed for SNIRF file, attempting to fix BIDS metadata..."
                    )
                    if _repair_snirf_bids_metadata(self.filecache, self.record):
                        try:
                            return self._read_raw_bids()
                        except Exception as retry_error:
                            logger.error(f"Retry also failed: {retry_error}")
                            raise retry_error from first_error

                # Not a fixable case - re-raise original error
                raise

    def _retry_with_generated_coordsystem(self, first_error: Exception) -> BaseRaw:
        """Generate a minimal coordsystem.json on-the-fly and retry loading.

        This handles the case where mne-bids raises ``RuntimeError`` for
        missing ``coordsystem.json`` — particularly strict for iEEG where
        coordinates are always required.
        """
        data_dir = self.filecache.parent if self.filecache else None
        if data_dir is None or not data_dir.exists():
            raise first_error

        datatype = data_dir.name  # eeg, ieeg, meg
        electrodes_files = list(data_dir.glob("*_electrodes.tsv"))
        if not electrodes_files:
            raise first_error

        logger.warning(
            f"Missing coordsystem.json for {datatype} data, "
            "generating minimal one and retrying..."
        )
        if _generate_coordsystem_json(electrodes_files[0], datatype=datatype):
            try:
                return self._read_raw_bids()
            except Exception as retry_error:
                logger.error(
                    f"Retry after coordsystem generation also failed: {retry_error}"
                )
                raise retry_error from first_error

        # Generation failed (e.g. read-only directory) — fall back to direct reader
        if self.filecache:
            logger.warning(
                "coordsystem.json generation failed — falling back to direct reader."
            )
            try:
                return _load_raw_direct(self.filecache)
            except Exception as fallback_error:
                raise fallback_error from first_error
        raise first_error

    def _retry_with_ctf_date_patch(self, first_error: Exception) -> BaseRaw:
        """Retry CTF read after patching MNE to accept numeric dash dates (e.g. 14-10-1925)."""
        orig = ctf_info._convert_time
        try:
            ctf_info._convert_time = partial(_convert_time_with_numeric_dash, orig=orig)
            return self._read_raw_bids()
        finally:
            ctf_info._convert_time = orig

    def _retry_with_ctf_hpi_fix(self, first_error: Exception) -> BaseRaw:
        """Retry CTF read after extending the HPI coil-kind dictionary.

        Some CTF ``.hc`` files use ``"Nasion"`` / ``"LPA"`` / ``"RPA"``
        instead of the lowercase ``"nasion"`` / ``"left ear"`` /
        ``"right ear"`` that MNE expects.  This patches the lookup dict
        to accept both forms.
        """
        import mne.io.ctf.hc as ctf_hc
        from mne.io.ctf.constants import CTF

        orig_dict = ctf_hc._kind_dict
        patched = dict(orig_dict)
        patched.update(
            {
                "Nasion": CTF.CTFV_COIL_NAS,
                "LPA": CTF.CTFV_COIL_LPA,
                "RPA": CTF.CTFV_COIL_RPA,
            }
        )
        try:
            ctf_hc._kind_dict = patched
            logger.warning(
                "CTF HPI coil-kind mismatch — retrying with extended kind dict."
            )
            try:
                return self._read_raw_bids()
            except Exception:
                pass  # MNE-BIDS failed; try direct reader with patch still active

            # Direct reader while HPI patch is still active
            if self.filecache:
                logger.warning(
                    "CTF HPI fix via MNE-BIDS failed — "
                    "falling back to direct reader (with HPI patch)."
                )
                try:
                    return _load_raw_direct(self.filecache)
                except Exception as fallback_error:
                    raise DataIntegrityError(
                        message=f"CTF HPI/coil error and direct reader failed: {fallback_error}",
                        record=self.record,
                        issues=[str(first_error), str(fallback_error)],
                    ) from first_error
            raise first_error
        finally:
            ctf_hc._kind_dict = orig_dict

    def _retry_with_ctf_truncated_fix(self, first_error: Exception) -> BaseRaw:
        """Retry CTF read with a tolerant sample-info parser.

        When a ``.meg4`` file is truncated (fewer complete sample rows than
        the ``.res4`` header declares), MNE raises ``RuntimeError("The
        number of samples is not an even multiple of the trial size")``.
        This method patches ``mne.io.ctf.ctf._get_sample_info`` to treat
        the available data as a single continuous block.
        """
        import os
        from unittest.mock import patch as _patch

        import mne.io.ctf.ctf as _ctf_mod

        _CTF_HEADER = 8  # "MEG41CP\x00"
        _orig_fn = _ctf_mod._get_sample_info

        def _tolerant_get_sample_info(fname, res4, system_clock):
            st_size = os.path.getsize(fname)
            nchan = res4["nchan"]
            data_bytes = st_size - _CTF_HEADER
            trial_bytes = 4 * res4["nsamp"] * nchan
            if trial_bytes > 0 and data_bytes % trial_bytes != 0:
                n_samp_tot = data_bytes // (4 * nchan)
                logger.warning(
                    "CTF meg4 truncated: expected %d samples/trial, "
                    "using %d total samples as 1 block.",
                    res4["nsamp"],
                    n_samp_tot,
                )
                return dict(
                    n_samp=n_samp_tot,
                    n_samp_tot=n_samp_tot,
                    block_size=n_samp_tot,
                    res4_nsamp=n_samp_tot,
                    n_chan=nchan,
                )
            return _orig_fn(fname, res4, system_clock)

        with _patch.object(_ctf_mod, "_get_sample_info", _tolerant_get_sample_info):
            import mne

            return mne.io.read_raw_ctf(
                str(self.filecache),
                system_clock="ignore",
                preload=False,
                verbose="ERROR",
            )

    def _retry_without_projectors(self, first_error: Exception) -> BaseRaw:
        """Fall back to a direct reader and strip projectors.

        MNE-BIDS channel renaming can cause a mismatch between projector
        channel names and the renamed data channels, triggering
        ``ValueError: projector channels not found in data``.  Loading
        via the format-specific reader bypasses the rename and lets us
        delete the offending projectors safely.
        """
        logger.warning(
            "Projector channel mismatch — retrying with direct reader "
            "and removing projectors."
        )
        try:
            raw = _load_raw_direct(self.filecache)
        except Exception as fallback_error:
            raise fallback_error from first_error
        if raw.info.get("projs"):
            raw.del_proj()
        return raw

    def _retry_without_events_tsv(self, first_error: Exception) -> BaseRaw:
        """Temporarily hide ``*_events.tsv`` and retry loading.

        Some datasets have malformed events files that cause
        ``AssertionError`` inside MNE-BIDS annotation handling.  The
        underlying data is fine — only the event markers are broken.
        """
        data_dir = self.filecache.parent
        events_files = list(data_dir.glob("*_events.tsv"))
        if not events_files:
            raise first_error

        hidden = []
        try:
            for ef in events_files:
                dest = ef.with_suffix(".tsv._hidden")
                ef.rename(dest)
                hidden.append((dest, ef))
            logger.warning(
                "Hiding %d events.tsv file(s) and retrying load.", len(hidden)
            )
            return self._read_raw_bids()
        except Exception as retry_error:
            raise retry_error from first_error
        finally:
            for src, dst in hidden:
                if src.exists():
                    src.rename(dst)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self._skipped:
            return 0
        if self._raw is None:
            ntimes = self.record.get("ntimes")
            if ntimes is not None:
                return int(ntimes)
            try:
                self._ensure_raw()
            except Exception as e:
                # If we can't load the raw data (corrupted file, etc.),
                # return 0 to mark this dataset as invalid
                logger.warning(
                    f"Could not load raw data for {self.bidspath}, "
                    f"marking as invalid (length=0). Error: {e}"
                )
                return 0
        return len(self._raw)

    @property
    def raw(self) -> BaseRaw | None:
        """The MNE Raw object for this recording.

        Accessing this property triggers the download and caching of the data
        if it has not been accessed before.

        Returns ``None`` when ``on_error`` is ``"warn"`` or ``"skip"`` and
        the record could not be loaded due to a
        :class:`~eegdash.dataset.exceptions.DataIntegrityError`.

        Returns
        -------
        mne.io.BaseRaw | None
            The loaded MNE Raw object, or ``None`` for skipped records.

        """
        if self._raw is None and not self._skipped:
            try:
                self._ensure_raw()
            except DataIntegrityError as e:
                if self._on_error == "raise":
                    raise
                self._skipped = True
                self._integrity_error = e
                if self._on_error == "warn":
                    e.log_warning()
        return self._raw

    @raw.setter
    def raw(self, raw: BaseRaw):
        self._raw = raw


__all__ = ["EEGDashRaw"]
