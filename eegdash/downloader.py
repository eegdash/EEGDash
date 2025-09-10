import logging
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import mne
import numpy as np
import s3fs
import xarray as xr
from fsspec.callbacks import TqdmCallback

logger = logging.getLogger(__name__)


def get_s3_filesystem():
    """Returns an S3FileSystem object."""
    return s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-east-2"})


def get_s3path(s3_bucket: str, filepath: str) -> str:
    """Helper to form an AWS S3 URI for the given relative filepath."""
    return f"{s3_bucket}/{filepath}"


def download_s3_file(s3_path: str, local_path: Path, s3_open_neuro: bool):
    """Download function that gets the raw EEG data from S3."""
    filesystem = get_s3_filesystem()
    if not s3_open_neuro:
        s3_path = re.sub(r"(^|/)ds\d{6}/", r"\1", s3_path, count=1)
        # TODO: remove this hack when competition is over
        if s3_path.endswith(".set"):
            s3_path = s3_path[:-4] + ".bdf"
            local_path = local_path.with_suffix(".bdf")

    local_path.parent.mkdir(parents=True, exist_ok=True)
    _filesystem_get(filesystem=filesystem, s3path=s3_path, filepath=local_path)

    return local_path


def download_dependencies(
    s3_bucket: str,
    bids_dependencies: list[str],
    bids_dependencies_original: list[str],
    cache_dir: Path,
    dataset_folder: Path,
    record: dict[str, Any],
    s3_open_neuro: bool,
):
    """Download all BIDS dependency files from S3 and cache them locally."""
    filesystem = get_s3_filesystem()
    for i, dep in enumerate(bids_dependencies):
        if not s3_open_neuro:
            if dep.endswith(".set"):
                dep = dep[:-4] + ".bdf"

        s3path = get_s3path(s3_bucket, dep)
        if not s3_open_neuro:
            dep = bids_dependencies_original[i]

        dep_path = Path(dep)
        if dep_path.parts and dep_path.parts[0] == record.get("dataset"):
            dep_local = Path(dataset_folder, *dep_path.parts[1:])
        else:
            dep_local = Path(dataset_folder) / dep_path
        filepath = cache_dir / dep_local
        if not s3_open_neuro:
            if filepath.suffix == ".set":
                filepath = filepath.with_suffix(".bdf")

        if not filepath.exists():
            filepath.parent.mkdir(parents=True, exist_ok=True)
            _filesystem_get(filesystem=filesystem, s3path=s3path, filepath=filepath)


def _filesystem_get(filesystem: s3fs.S3FileSystem, s3path: str, filepath: Path):
    """Helper to download a file from S3 with a progress bar."""
    info = filesystem.info(s3path)
    size = info.get("size") or info.get("Size")

    callback = TqdmCallback(
        size=size,
        tqdm_kwargs=dict(
            desc=f"Downloading {Path(s3path).name}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
            leave=True,
            mininterval=0.2,
            smoothing=0.1,
            miniters=1,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]",
        ),
    )
    filesystem.get(s3path, str(filepath), callback=callback)
    return filepath


def load_eeg_from_s3(s3path: str):
    """Load EEG data from an S3 URI into an ``xarray.DataArray``.

    Preserves the original filename, downloads sidecar files when applicable
    (e.g., ``.fdt`` for EEGLAB, ``.vmrk``/``.eeg`` for BrainVision), and uses
    MNE's direct readers.

    Parameters
    ----------
    s3path : str
        An S3 URI (should start with "s3://").

    Returns
    -------
    xr.DataArray
        EEG data with dimensions ``("channel", "time")``.

    Raises
    ------
    ValueError
        If the file extension is unsupported.

    """
    filesystem = get_s3_filesystem()
    # choose a temp dir so sidecars can be colocated
    with tempfile.TemporaryDirectory() as tmpdir:
        # Derive local filenames from the S3 key to keep base name consistent
        s3_key = urlsplit(s3path).path  # e.g., "/dsXXXX/sub-.../..._eeg.set"
        basename = Path(s3_key).name
        ext = Path(basename).suffix.lower()
        local_main = Path(tmpdir) / basename

        # Download main file
        with (
            filesystem.open(s3path, mode="rb") as fsrc,
            open(local_main, "wb") as fdst,
        ):
            fdst.write(fsrc.read())

        # Determine and fetch any required sidecars
        sidecars: list[str] = []
        if ext == ".set":  # EEGLAB
            sidecars = [".fdt"]
        elif ext == ".vhdr":  # BrainVision
            sidecars = [".vmrk", ".eeg", ".dat", ".raw"]

        for sc_ext in sidecars:
            sc_key = s3_key[: -len(ext)] + sc_ext
            sc_uri = f"s3://{urlsplit(s3path).netloc}{sc_key}"
            try:
                # If sidecar exists, download next to the main file
                info = filesystem.info(sc_uri)
                if info:
                    sc_local = Path(tmpdir) / Path(sc_key).name
                    with (
                        filesystem.open(sc_uri, mode="rb") as fsrc,
                        open(sc_local, "wb") as fdst,
                    ):
                        fdst.write(fsrc.read())
            except Exception:
                # Sidecar not present; skip silently
                pass

        # Read using appropriate MNE reader
        raw = mne.io.read_raw(str(local_main), preload=True, verbose=False)

        data = raw.get_data()
        fs = raw.info["sfreq"]
        max_time = data.shape[1] / fs
        time_steps = np.linspace(0, max_time, data.shape[1]).squeeze()
        channel_names = raw.ch_names

        return xr.DataArray(
            data=data,
            dims=["channel", "time"],
            coords={"time": time_steps, "channel": channel_names},
        )
