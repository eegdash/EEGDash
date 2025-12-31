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
from ..logging import logger
from ..records import validate_record


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
        if not self.bids_root.exists():
            # Try to find a folder that starts with dataset_id if it's not a common name like 'ds'
            potential = [d for d in self.cache_dir.glob(f"{dataset_id}*") if d.is_dir()]
            if len(potential) == 1:
                self.bids_root = potential[0]
            elif (self.cache_dir / dataset_id).exists():
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

        self.bidspath = BIDSPath(
            root=self.bids_root,
            datatype=self.record.get("datatype", "eeg"),
            suffix=self.record.get("suffix", "eeg"),
            extension=self.record.get("extension", self.filecache.suffix),
            subject=entities_mne.get("subject"),
            session=entities_mne.get("session"),
            task=entities_mne.get("task"),
            run=entities_mne.get("run"),
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
        """Ensure the raw data file and its dependencies are cached locally."""
        self._download_required_files()
        if self._raw is None:
            try:
                self._raw = self._load_raw()
            except Exception as e:
                logger.error(
                    f"Error reading {self.bidspath}: {e}. Try `rm -rf {self.bids_root}`"
                )
                raise

    def _load_raw(self) -> BaseRaw:
        """Load raw data, preferring MNE-BIDS if BIDSPath resolves."""
        # MNE-BIDS handles sidecars automatically
        return mne_bids.read_raw_bids(bids_path=self.bidspath, verbose="ERROR")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self._raw is None:
            # v2 records may not have ntimes/sampling_frequency - use .get() for safety
            ntimes = self.record.get("ntimes")
            sfreq = self.record.get("sampling_frequency")
            if ntimes is None or sfreq is None:
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
            else:
                # FIXME: this is a bit strange and should definitely not change as a side effect
                #  of accessing the data (which it will, since ntimes is the actual length but rounded down)
                return int(ntimes * sfreq)
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
