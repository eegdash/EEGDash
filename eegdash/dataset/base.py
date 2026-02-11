# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Data utilities and dataset classes for EEG data handling.

This module provides core dataset classes for working with EEG data in the EEGDash ecosystem,
including classes for individual recordings and collections of datasets. It integrates with
braindecode for machine learning workflows and handles data loading from both local and remote sources.
"""

from pathlib import Path
from typing import Any

import mne_bids
from mne.io import BaseRaw
from mne_bids import BIDSPath

from braindecode.datasets.base import RawDataset

from .. import downloader
from ..const import MODALITY_ALIASES
from ..logging import logger
from ..schemas import validate_record
from .exceptions import DataIntegrityError, UnsupportedDataError
from .io import (
    _ensure_coordsystem_symlink,
    _generate_vhdr_from_metadata,
    _generate_vmrk_stub,
    _load_epoched_eeglab_as_raw,
    _load_raw_direct,
    _load_set_via_scipy,
    _repair_ctf_eeg_position_file,
    _repair_electrodes_tsv,
    _repair_events_tsv_na_duration,
    _repair_snirf_bids_metadata,
    _repair_tsv_decimal_separators,
    _repair_tsv_encoding,
    _repair_tsv_na_values,
    _repair_vhdr_missing_markerfile,
    _repair_vhdr_pointers,
)

# Pattern-based unrecoverable errors → (patterns, reason, description)
_UNRECOVERABLE_ERRORS = [
    (
        ["buffer is too small", "could not read bytes", "truncated file"],
        "corrupted_file",
        "Data file is corrupted or truncated",
    ),
    (
        ["type code"],
        "unsupported_format_variant",
        "Unsupported data format variant",
    ),
    (
        ["inhomogeneous"],
        "inhomogeneous_data",
        "Malformed data structure",
    ),
]


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

        # Sanitize non-numeric run entities (e.g., "5H") before creating BIDSPath
        # because mne_bids rejects non-integer run values even with check=False
        run = entities_mne.get("run")
        if run is not None and not str(run).isdigit():
            logger.info(f"Sanitizing non-numeric run entity: '{run}' → None")
            run = None

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
            run=run,
            check=False,
        )

        self._raw = None

    def _download_required_files(self) -> None:
        if self._raw_uri is not None:
            filesystem = downloader.get_s3_filesystem()

            # Download deps first (sidecars, companions), then raw.
            downloader.download_files(
                list(zip(self._dep_uris, self._dep_paths, strict=False)),
                filesystem=filesystem,
                skip_existing=True,
            )
            downloader.download_s3_file(
                self._raw_uri, self.filecache, filesystem=filesystem
            )

        # Always set filenames (important for local datasets)
        self.filenames = [self.filecache]

    def _ensure_raw(self) -> None:
        """Ensure the raw data file and its dependencies are cached locally.

        Warns if the record has known data integrity issues but continues loading.
        """
        # Check for data integrity issues and warn (but don't block loading)
        if self.record.get("_has_missing_files"):
            DataIntegrityError.warn_from_record(self.record)

        self._download_required_files()

        # Apply directory-level repairs (coordsystem, TSV encoding/values, etc.)
        if self.filecache and self.filecache.parent.exists():
            for repair_fn in (
                _ensure_coordsystem_symlink,
                _repair_tsv_encoding,
                _repair_electrodes_tsv,
                _repair_tsv_decimal_separators,
                _repair_tsv_na_values,
                _repair_events_tsv_na_duration,
            ):
                repair_fn(self.filecache.parent)

        # Helper: Repair CTF .ds internal files (e.g., .eeg with n/a)
        if (
            self.filecache
            and self.filecache.suffix == ".ds"
            and self.filecache.is_dir()
        ):
            _repair_ctf_eeg_position_file(self.filecache)

        # Helper: Handle VHDR files - generate if missing, repair if broken
        if self.filecache and self.filecache.suffix == ".vhdr":
            if not self.filecache.exists():
                # Generate VHDR from database metadata if file is missing
                _generate_vhdr_from_metadata(self.filecache, self.record)
            else:
                # Auto-Repair broken VHDR pointers (common in OpenNeuro exports)
                _repair_vhdr_pointers(self.filecache)
                # Fix missing MarkerFile entry (causes KeyError: 'markerfile')
                _repair_vhdr_missing_markerfile(self.filecache)

            # Also generate VMRK stub if missing (common issue with some datasets)
            vmrk_path = self.filecache.with_suffix(".vmrk")
            if not vmrk_path.exists():
                _generate_vmrk_stub(vmrk_path, self.filecache.name)

        if self._raw is None:
            try:
                self._raw = self._load_raw()
            except Exception as e:
                logger.error(
                    f"Error reading {self.bidspath}: {e}. Try `rm -rf {self.bids_root}`"
                )
                raise

    def _raise_unsupported(self, msg, reason, cause):
        """Raise UnsupportedDataError with record context."""
        raise UnsupportedDataError(msg, record=self.record, reason=reason) from cause

    def _load_raw(self) -> BaseRaw:
        """Load raw data with cascading recovery strategies for known failures."""
        try:
            return mne_bids.read_raw_bids(
                bids_path=self.bidspath, verbose="ERROR", on_ch_mismatch="rename"
            )
        except Exception as first_error:
            error_msg = str(first_error)
            ext = self.filecache.suffix.lower() if self.filecache else ""
            dataset_id = self.record.get("dataset", "unknown")

            # EEGLAB epoched files
            if ext == ".set" and "number of trials" in error_msg.lower():
                logger.warning(
                    f"[{dataset_id}] Epoched EEGLAB file detected, "
                    f"converting to continuous..."
                )
                try:
                    return _load_epoched_eeglab_as_raw(self.filecache)
                except Exception as epoch_error:
                    logger.error(
                        f"[{dataset_id}] Epoched conversion failed: {epoch_error}"
                    )
                    self._raise_unsupported(
                        f"Cannot load epoched EEGLAB file: {epoch_error}",
                        "epoched_eeglab_conversion_failed",
                        first_error,
                    )

            # SNIRF repair logic
            if ext == ".snirf":
                logger.warning(
                    f"[{dataset_id}] Initial load failed for SNIRF file, "
                    f"attempting to fix BIDS metadata..."
                )
                if _repair_snirf_bids_metadata(self.filecache, self.record):
                    try:
                        return mne_bids.read_raw_bids(
                            bids_path=self.bidspath,
                            verbose="ERROR",
                            on_ch_mismatch="rename",
                        )
                    except Exception as retry_error:
                        logger.error(
                            f"[{dataset_id}] SNIRF retry failed: {retry_error}"
                        )

                if any(
                    s in error_msg
                    for s in ["0-d array", "type code", "truncated file", "truncated"]
                ):
                    self._raise_unsupported(
                        f"Cannot load SNIRF file '{self.filecache.name}': {error_msg}",
                        "unsupported_snirf_format",
                        first_error,
                    )
                raise

            # EEGLAB extension mismatch
            if ext == ".set" and "EEGLAB file extension" in error_msg:
                logger.warning(
                    f"[{dataset_id}] EEGLAB extension error, trying scipy loader..."
                )
                try:
                    return _load_set_via_scipy(self.filecache)
                except Exception as scipy_error:
                    logger.error(
                        f"[{dataset_id}] Scipy EEGLAB read failed: {scipy_error}"
                    )
                    self._raise_unsupported(
                        f"Cannot load EEGLAB file: {scipy_error}",
                        "eeglab_extension_mismatch",
                        first_error,
                    )

            # Channel type conflict in projectors
            if "Cannot change channel type" in error_msg:
                logger.warning(
                    f"[{dataset_id}] Channel type conflict, "
                    f"retrying with on_ch_mismatch='ignore'..."
                )
                try:
                    return mne_bids.read_raw_bids(
                        bids_path=self.bidspath,
                        verbose="ERROR",
                        on_ch_mismatch="ignore",
                    )
                except Exception as ignore_error:
                    logger.error(
                        f"[{dataset_id}] Retry with ignore also failed: {ignore_error}"
                    )

            # Direct reader fallback for FIF/MEG validation errors
            if any(
                s in error_msg
                for s in [
                    "Illegal date",
                    "FIFFV_COIL",
                    "cannot reshape array",
                    "HPI",
                    "device-coordinate",
                    "Cannot change channel type",
                ]
            ):
                logger.warning(
                    f"[{dataset_id}] BIDS validation error, trying direct reader..."
                )
                try:
                    return _load_raw_direct(self.filecache, allow_maxshield=True)
                except Exception as direct_error:
                    logger.error(
                        f"[{dataset_id}] Direct reader also failed: {direct_error}"
                    )
                    self._raise_unsupported(
                        f"Cannot load file '{self.filecache.name}': {direct_error}",
                        "fif_validation_error",
                        first_error,
                    )

            # Data-driven unrecoverable error dispatch
            for patterns, reason, desc in _UNRECOVERABLE_ERRORS:
                if any(s in error_msg for s in patterns):
                    self._raise_unsupported(f"{desc}: {error_msg}", reason, first_error)

            # Missing companion files (CTF .ds, etc.)
            if any(s in error_msg.lower() for s in ["not found", "could not find"]):
                raise DataIntegrityError(
                    message=f"Missing companion files: {error_msg}",
                    record=self.record,
                    issues=[error_msg],
                )

            # BIDS path mismatch (e.g., eeg vs func)
            if "is not in list" in error_msg and "Did you mean" in error_msg:
                self._raise_unsupported(
                    f"BIDS path mismatch — record metadata may be incorrect: {error_msg}",
                    "bids_path_mismatch",
                    first_error,
                )

            # Catch-all for AssertionErrors
            if isinstance(first_error, AssertionError):
                self._raise_unsupported(
                    f"Assertion failed during loading of "
                    f"'{self.filecache.name}': {error_msg}",
                    "assertion_error",
                    first_error,
                )

            raise

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
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
    def raw(self) -> BaseRaw:
        """The MNE Raw object for this recording.

        Accessing this property triggers the download and caching of the data
        if it has not been accessed before.

        Returns
        -------
        mne.io.BaseRaw
            The loaded MNE Raw object.

        """
        if self._raw is None:
            self._ensure_raw()
        return self._raw

    @raw.setter
    def raw(self, raw: BaseRaw):
        self._raw = raw


__all__ = ["EEGDashRaw"]
