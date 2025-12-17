# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Data utilities and dataset classes for EEG data handling.

This module provides core dataset classes for working with EEG data in the EEGDash ecosystem,
including classes for individual recordings and collections of datasets. It integrates with
braindecode for machine learning workflows and handles data loading from both local and remote sources.
"""

import io
import os
import traceback
from contextlib import contextmanager, nullcontext, redirect_stderr
from pathlib import Path
from typing import Any

import mne
import mne_bids
from mne._fiff.utils import _read_segments_file
from mne.io import BaseRaw
from mne_bids import BIDSPath
from mne_bids.config import reader as _mne_bids_reader

from braindecode.datasets.base import BaseDataset

from .. import downloader
from ..bids_eeg_metadata import enrich_from_participants
from ..logging import logger
from ..paths import get_default_cache_dir
from ..records import adapt_record_v1_to_v2
from ..resolver import resolve_record


@contextmanager
def _patch_mne_eeglab_missing_chanlocs():
    """Patch MNE's EEGLAB reader to tolerate missing ``chanlocs``.

    Some OpenNeuro EEGLAB ``.set`` files omit the ``EEG.chanlocs`` field and rely
    on BIDS sidecars (e.g., ``*_channels.tsv``) for channel metadata. MNE's
    reader currently errors early when ``chanlocs`` is absent.
    """
    try:
        import mne.io.eeglab.eeglab as _mne_eeglab
    except Exception:
        yield
        return

    orig = getattr(_mne_eeglab, "_check_load_mat", None)
    if orig is None:
        yield
        return

    def _patched(fname, uint16_codec, *, preload=False):
        eeg = orig(fname, uint16_codec, preload=preload)
        if not hasattr(eeg, "chanlocs"):
            eeg.chanlocs = []
        return eeg

    _mne_eeglab._check_load_mat = _patched
    try:
        yield
    finally:
        _mne_eeglab._check_load_mat = orig


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
        self.cache_dir = Path(cache_dir)
        self.s3_bucket = s3_bucket or self._AWS_BUCKET
        # Normalize legacy records to a v2-shaped record at the boundary.
        self.record = (
            record
            if record.get("schema_version") == 2
            else adapt_record_v1_to_v2(record, s3_bucket=self.s3_bucket)
        )

        resolved = resolve_record(self.record, cache_dir=self.cache_dir)
        self.bids_root = resolved.bids_root
        self.filecache = resolved.raw_path
        self._dep_paths = resolved.dep_paths
        self._raw_uri = resolved.raw_uri
        self._dep_uris = resolved.dep_uris

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
        if self._raw_uri is None:
            return
        filesystem = downloader.get_s3_filesystem()

        # Download deps first (sidecars, companions), then raw.
        downloader.download_files(
            list(zip(self._dep_uris, self._dep_paths, strict=False)),
            filesystem=filesystem,
            skip_existing=True,
        )
        downloader.download_s3_file(self._raw_uri, self.filecache, filesystem=filesystem)
        self.filenames = [self.filecache]

    def _ensure_raw(self) -> None:
        """Ensure the raw data file and its dependencies are cached locally."""
        self._download_required_files()
        if self._raw is None:
            try:
                patch_ctx = (
                    _patch_mne_eeglab_missing_chanlocs()
                    if self.filecache.suffix.lower() == ".set"
                    else nullcontext()
                )
                # Prefer MNE-BIDS when the BIDSPath actually resolves to the cached file.
                use_bids = self.bidspath.fpath.exists()
                with patch_ctx:
                    if use_bids:
                        # mne-bids can emit noisy warnings to stderr; keep user logs clean
                        _stderr_buffer = io.StringIO()
                        with redirect_stderr(_stderr_buffer):
                            self._raw = mne_bids.read_raw_bids(
                                bids_path=self.bidspath, verbose="ERROR"
                            )
                    else:
                        self._raw = self._read_raw_direct()
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

    def _read_raw_direct(self) -> BaseRaw:
        """Read a cached raw file directly and apply BIDS sidecars where possible."""
        ext = self.filecache.suffix
        read_func = _mne_bids_reader.get(ext)
        if read_func is None:
            raise RuntimeError(f"No MNE-BIDS reader registered for extension: {ext}")

        _stderr_buffer = io.StringIO()
        with redirect_stderr(_stderr_buffer):
            raw = read_func(str(self.filecache), preload=False, verbose="ERROR")

        try:
            import mne_bids.read as _mb_read

            raw_dir = self.filecache.parent
            stem = self.filecache.stem
            suffix = str(self.record.get("suffix") or "eeg")
            base = stem[: -len(f"_{suffix}")] if stem.endswith(f"_{suffix}") else stem

            # Prefer non-suffix variants first (BIDS convention), then fall back.
            events_candidates = [
                raw_dir / f"{base}_events.tsv",
                raw_dir / f"{stem}_events.tsv",
            ]
            for p in events_candidates:
                if p.exists():
                    _mb_read._handle_events_reading(str(p), raw)
                    break

            channels_candidates = [
                raw_dir / f"{base}_channels.tsv",
                raw_dir / f"{stem}_channels.tsv",
                raw_dir / "channels.tsv",
            ]
            for p in channels_candidates:
                if p.exists():
                    _mb_read._handle_channels_reading(str(p), raw)
                    break
        except Exception:
            pass

        return raw

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self._raw is None:
            if (
                self.record["ntimes"] is None
                or self.record["sampling_frequency"] is None
            ):
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
        bids_dependencies: list[str] | None = None,
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
        if bids_dependencies is None:
            bids_dependencies = []
        self.bids_dependencies = bids_dependencies

        if preload and not os.path.exists(self.filecache):
            self.filecache = downloader.download_s3_file(self.s3file, self.filecache)
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
            if self.bids_dependencies:
                deps = [
                    (downloader.get_s3path(self._AWS_BUCKET, dep), self.cache_dir / dep)
                    for dep in self.bids_dependencies
                ]
                downloader.download_files(deps)
            self.filecache = downloader.download_s3_file(self.s3file, self.filecache)
            self.filenames = [self.filecache]
        else:  # not preload and file is not cached
            self.filenames = [self.filecache]
        return super()._read_segment(start, stop, sel, data_buffer, verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of data from a local file."""
        _read_segments_file(self, data, idx, fi, start, stop, cals, mult, dtype="<f4")


__all__ = ["EEGDashBaseDataset", "EEGDashBaseRaw"]
