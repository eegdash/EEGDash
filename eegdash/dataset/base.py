# Authors: The EEGDash contributors.
# License: GNU General Public License
# Copyright the EEGDash contributors.

"""Data utilities and dataset classes for EEG data handling.

This module provides core dataset classes for working with EEG data in the EEGDash ecosystem,
including classes for individual recordings and collections of datasets. It integrates with
braindecode for machine learning workflows and handles data loading from both local and remote sources.
"""

import io
import os
import traceback
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any

import mne
import mne_bids
from mne._fiff.utils import _read_segments_file
from mne.io import BaseRaw
from mne_bids import BIDSPath

from braindecode.datasets import BaseDataset

from .. import downloader
from ..bids_eeg_metadata import enrich_from_participants
from ..logging import logger
from ..paths import get_default_cache_dir


class EEGDashBaseDataset(BaseDataset):
    """A single EEG recording dataset.

    Represents a single EEG recording, typically hosted on a remote server (like AWS S3)
    and cached locally upon first access. This class is a subclass of
    :class:`braindecode.datasets.BaseDataset` and can be used with braindecode's
    preprocessing and training pipelines.

    Parameters
    ----------
    record : dict
        A fully resolved metadata record for the data to load.
    cache_dir : str
        The local directory where the data will be cached.
    s3_bucket : str, optional
        The S3 bucket to download data from. If not provided, defaults to the
        OpenNeuro bucket.
    **kwargs
        Additional keyword arguments passed to the
        :class:`braindecode.datasets.BaseDataset` constructor.

    """

    _AWS_BUCKET = "s3://openneuro.org"

    def __init__(
        self,
        record: dict[str, Any],
        cache_dir: str,
        s3_bucket: str | None = None,
        **kwargs,
    ):
        super().__init__(None, **kwargs)
        self.record = record
        self.cache_dir = Path(cache_dir)
        self.bids_kwargs = self._get_raw_bids_args()

        if s3_bucket:
            self.s3_bucket = s3_bucket
            self.s3_open_neuro = False
        else:
            self.s3_bucket = self._AWS_BUCKET
            self.s3_open_neuro = True

        # Compute a dataset folder name under cache_dir that encodes preprocessing
        # (e.g., bdf, mini) to avoid overlapping with the original dataset cache.
        self.dataset_folder = record.get("dataset", "")
        # TODO: remove this hack when competition is over
        if s3_bucket:
            suffixes: list[str] = []
            bucket_lower = str(s3_bucket).lower()
            if "bdf" in bucket_lower:
                suffixes.append("bdf")
            if "mini" in bucket_lower:
                suffixes.append("mini")
            if suffixes:
                self.dataset_folder = f"{self.dataset_folder}-{'-'.join(suffixes)}"

        # Place files under the dataset-specific folder (with suffix if any)
        rel = Path(record["bidspath"])  # usually starts with dataset id
        if rel.parts and rel.parts[0] == record.get("dataset"):
            rel = Path(self.dataset_folder, *rel.parts[1:])
        else:
            rel = Path(self.dataset_folder) / rel
        self.filecache = self.cache_dir / rel
        self.bids_root = self.cache_dir / self.dataset_folder

        self.bidspath = BIDSPath(
            root=self.bids_root,
            datatype="eeg",
            suffix="eeg",
            **self.bids_kwargs,
        )

        self.s3file = downloader.get_s3path(self.s3_bucket, record["bidspath"])
        self.bids_dependencies = record["bidsdependencies"]
        self.bids_dependencies_original = record["bidsdependencies"]
        # TODO: removing temporary fix for BIDS dependencies path
        # when the competition is over and dataset is digested properly
        if not self.s3_open_neuro:
            self.bids_dependencies = [
                dep.split("/", 1)[1] for dep in self.bids_dependencies
            ]

        self._raw = None

    def _get_raw_bids_args(self) -> dict[str, Any]:
        """Extract BIDS-related arguments from the metadata record."""
        desired_fields = ["subject", "session", "task", "run"]
        return {k: self.record[k] for k in desired_fields if self.record[k]}

    def _ensure_raw(self) -> None:
        """Ensure the raw data file and its dependencies are cached locally."""
        # TO-DO: remove this once is fixed on the our side
        # for the competition
        if not self.s3_open_neuro:
            self.bidspath = self.bidspath.update(extension=".bdf")
            self.filecache = self.filecache.with_suffix(".bdf")

        if not os.path.exists(self.filecache):  # not preload
            if self.bids_dependencies:
                downloader.download_dependencies(
                    s3_bucket=self.s3_bucket,
                    bids_dependencies=self.bids_dependencies,
                    bids_dependencies_original=self.bids_dependencies_original,
                    cache_dir=self.cache_dir,
                    dataset_folder=self.dataset_folder,
                    record=self.record,
                    s3_open_neuro=self.s3_open_neuro,
                )
            self.filecache = downloader.download_s3_file(
                self.s3file, self.filecache, self.s3_open_neuro
            )
            self.filenames = [self.filecache]
        if self._raw is None:
            try:
                # mne-bids can emit noisy warnings to stderr; keep user logs clean
                _stderr_buffer = io.StringIO()
                with redirect_stderr(_stderr_buffer):
                    self._raw = mne_bids.read_raw_bids(
                        bids_path=self.bidspath, verbose="ERROR"
                    )
                # Enrich Raw.info and description with participants.tsv extras
                enrich_from_participants(
                    self.bids_root, self.bidspath, self._raw, self.description
                )

            except Exception as e:
                logger.error(
                    f"Error while reading BIDS file: {self.bidspath}\n"
                    "This may be due to a missing or corrupted file.\n"
                    "Please check the file and try again.\n"
                    "Usually erasing the local cache and re-downloading helps.\n"
                    f"`rm {self.bidspath}`"
                )
                logger.error(f"Exception: {e}")
                logger.error(traceback.format_exc())
                raise e

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self._raw is None:
            if (
                self.record["ntimes"] is None
                or self.record["sampling_frequency"] is None
            ):
                self._ensure_raw()
            else:
                # FIXME: this is a bit strange and should definitely not change as a side effect
                #  of accessing the data (which it will, since ntimes is the actual length but rounded down)
                return int(self.record["ntimes"] * self.record["sampling_frequency"])
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


class EEGDashBaseRaw(BaseRaw):
    """MNE BaseRaw wrapper for automatic S3 data fetching.

    This class extends :class:`mne.io.BaseRaw` to automatically fetch data
    from an S3 bucket and cache it locally when data is first accessed.
    It is intended for internal use within the EEGDash ecosystem.

    Parameters
    ----------
    input_fname : str
        The path to the file on the S3 bucket (relative to the bucket root).
    metadata : dict
        The metadata record for the recording, containing information like
        sampling frequency, channel names, etc.
    preload : bool, default False
        If True, preload the data into memory.
    cache_dir : str, optional
        Local directory for caching data. If None, a default directory is used.
    bids_dependencies : list of str, default []
        A list of BIDS metadata files to download alongside the main recording.
    verbose : str, int, or None, default None
        The MNE verbosity level.

    See Also
    --------
    mne.io.Raw : The base class for Raw objects in MNE.

    """

    _AWS_BUCKET = "s3://openneuro.org"

    def __init__(
        self,
        input_fname: str,
        metadata: dict[str, Any],
        preload: bool = False,
        *,
        cache_dir: str | None = None,
        bids_dependencies: list[str] = [],
        verbose: Any = None,
    ):
        # Create a simple RawArray
        sfreq = metadata["sfreq"]  # Sampling frequency
        n_times = metadata["n_times"]
        ch_names = metadata["ch_names"]
        ch_types = []
        for ch in metadata["ch_types"]:
            chtype = ch.lower()
            if chtype == "heog" or chtype == "veog":
                chtype = "eog"
            ch_types.append(chtype)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        self.s3file = downloader.get_s3path(self._AWS_BUCKET, input_fname)
        self.cache_dir = Path(cache_dir) if cache_dir else get_default_cache_dir()
        self.filecache = self.cache_dir / input_fname
        self.bids_dependencies = bids_dependencies

        if preload and not os.path.exists(self.filecache):
            self.filecache = downloader.download_s3_file(
                self.s3file, self.filecache, self.s3_open_neuro
            )
            self.filenames = [self.filecache]
            preload = self.filecache

        super().__init__(
            info,
            preload,
            last_samps=[n_times - 1],
            orig_format="single",
            verbose=verbose,
        )

    def _read_segment(
        self, start=0, stop=None, sel=None, data_buffer=None, *, verbose=None
    ):
        """Read a segment of data, downloading if necessary."""
        if not os.path.exists(self.filecache):  # not preload
            if self.bids_dependencies:  # this is use only to sidecars for now
                downloader.download_dependencies(
                    s3_bucket=self._AWS_BUCKET,
                    bids_dependencies=self.bids_dependencies,
                    bids_dependencies_original=None,
                    cache_dir=self.cache_dir,
                    dataset_folder=self.filecache,
                    record={},
                    s3_open_neuro=self.s3_open_neuro,
                )
            self.filecache = downloader.download_s3_file(
                self.s3file, self.filecache, self.s3_open_neuro
            )
            self.filenames = [self.filecache]
        else:  # not preload and file is not cached
            self.filenames = [self.filecache]
        return super()._read_segment(start, stop, sel, data_buffer, verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of data from a local file."""
        _read_segments_file(self, data, idx, fi, start, stop, cals, mult, dtype="<f4")


__all__ = ["EEGDashBaseDataset", "EEGDashBaseRaw"]
