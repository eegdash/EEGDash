# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Data utilities and dataset classes for EEG data handling.

This module provides core dataset classes for working with EEG data in the EEGDash ecosystem,
including classes for individual recordings and collections of datasets. It integrates with
braindecode for machine learning workflows and handles data loading from both local and remote sources.
"""

import re
from functools import partial
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
from .exceptions import DataIntegrityError
from .io import (
    _convert_time_with_numeric_dash,
    _ensure_coordsystem_symlink,
    _generate_coordsystem_json,
    _generate_vhdr_from_metadata,
    _generate_vmrk_stub,
    _load_raw_direct,
    _repair_events_tsv_nan_samples,
    _repair_participants_tsv_ids,
    _repair_scans_tsv_timestamps,
    _repair_snirf_bids_metadata,
    _repair_tsv_decimal_separators,
    _repair_tsv_encoding,
    _repair_vhdr_pointers,
)

# Error messages indicating unrecoverable data corruption — the source
# file itself is broken and no amount of BIDS-metadata repair will help.
_UNRECOVERABLE_PATTERNS = [
    "Bad EDF",
    "invalid literal for int",
    "Could not find any data",
    "no valid samples",
    # NumPy / SciPy array errors from corrupt MAT / EEGLAB files
    "buffer is too small for requested array",
    "buffer size must be",
    "iteration over a 0-d array",
    "cannot reshape array",
    "setting an array element with a sequence",
    # EEGLAB reader errors from non-standard .set structures
    "allowed values",
    "has no attribute",
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

        # CTF MEG .ds directories require internal files (.meg4, .res4, etc.)
        if self.filecache and self.filecache.suffix == ".ds":
            self._ensure_ctf_directory_complete()

        # Always set filenames (important for local datasets)
        self.filenames = [self.filecache]

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
            _repair_tsv_decimal_separators(self.filecache.parent)
            _repair_scans_tsv_timestamps(self.filecache.parent)
            _repair_events_tsv_nan_samples(self.filecache.parent)

        # Helper: Handle VHDR files - generate if missing, repair if broken
        if self.filecache and self.filecache.suffix == ".vhdr":
            # Generate VMRK stub first so pointer repair can find the target
            vmrk_path = self.filecache.with_suffix(".vmrk")
            if not vmrk_path.exists():
                _generate_vmrk_stub(vmrk_path, self.filecache.name)

            if not self.filecache.exists():
                # Generate VHDR from database metadata if file is missing
                _generate_vhdr_from_metadata(self.filecache, self.record)
            else:
                # Auto-Repair broken VHDR pointers (common in OpenNeuro exports)
                _repair_vhdr_pointers(self.filecache)

        if self._raw is None:
            try:
                self._raw = self._load_raw()
            except Exception as e:
                logger.error(
                    f"Error reading {self.bidspath}: {e}. Try `rm -rf {self.bids_root}`"
                )
                raise

    def _read_raw_bids(self) -> BaseRaw:
        """Call ``mne_bids.read_raw_bids`` with the standard arguments."""
        return mne_bids.read_raw_bids(
            bids_path=self.bidspath, verbose="ERROR", on_ch_mismatch="rename"
        )

    def _load_raw(self) -> BaseRaw:
        """Load raw data, preferring MNE-BIDS if BIDSPath resolves.

        Applies on-the-fly fixes and retries for known failure modes:

        - **iEEG** missing ``coordsystem.json``: mne-bids raises
          ``RuntimeError`` because coordinates are mandatory for intracranial
          data. We generate a minimal ``coordsystem.json`` with the correct
          ``iEEGCoordinateSystem`` keys and retry.
        - **SNIRF** metadata issues: regenerates ``channels.tsv`` /
          ``scans.tsv`` and retries.
        - **Unrecoverable corruption** (Bad EDF, empty MEG data, corrupt
          MAT/EEGLAB files with array errors, etc.): raises
          ``DataIntegrityError`` for clean error reporting.
        - **Invalid scans.tsv timestamps** (seconds >= 60, NaN): repairs
          the scans.tsv and retries.
        - **participants.tsv subject mismatches**: repairs ``participant_id``
          values to match ``sub-*`` folder names and retries.
        - **events.tsv rows with NaN onset/sample**: drops broken rows and
          retries, then falls back to direct MNE loading if needed.
        - **Invalid BIDS entity characters** (hyphens in task, etc.):
          falls back to direct MNE reader.
        - **CTF "Illegal date"** (numeric dash dates like 14-10-1925): patches
          MNE's CTF date parser to try %d-%m-%Y and retries.
        """
        try:
            return self._read_raw_bids()
        except RuntimeError as first_error:
            if "coordsystem.json is REQUIRED" in str(first_error):
                return self._retry_with_generated_coordsystem(first_error)
            if "Illegal date" in str(first_error):
                return self._retry_with_ctf_date_patch(first_error)
            raise
        except (TypeError, ValueError, OSError, AttributeError) as first_error:
            msg = str(first_error)

            # Unrecoverable data corruption (bad EDF, empty MEG, corrupt MAT,
            # or any TypeError from array/parsing failures in scipy/numpy)
            if any(p in msg for p in _UNRECOVERABLE_PATTERNS) or isinstance(
                first_error, (TypeError, AttributeError)
            ):
                raise DataIntegrityError(
                    message=f"Cannot read data file: {msg}",
                    record=self.record,
                    issues=[msg],
                ) from first_error

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

            # VHDR with non-numeric run (ValueError from MNE-BIDS entity validation)
            if (
                self.filecache
                and self.filecache.suffix.lower() == ".vhdr"
                and self._has_non_numeric_run()
            ):
                logger.warning(
                    "MNE-BIDS failed for VHDR with non-numeric run, "
                    "falling back to direct MNE reader."
                )
                try:
                    return _load_raw_direct(self.filecache)
                except Exception as fallback_error:
                    raise fallback_error from first_error

            raise
        except Exception as first_error:
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
        raise first_error

    def _retry_with_ctf_date_patch(self, first_error: Exception) -> BaseRaw:
        """Retry CTF read after patching MNE to accept numeric dash dates (e.g. 14-10-1925)."""
        import mne.io.ctf.info as ctf_info

        orig = ctf_info._convert_time
        try:
            ctf_info._convert_time = partial(_convert_time_with_numeric_dash, orig=orig)
            return self._read_raw_bids()
        finally:
            ctf_info._convert_time = orig

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
