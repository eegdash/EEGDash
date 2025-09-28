# Authors: The EEGDash contributors.
# License: GNU General Public License
# Copyright the EEGDash contributors.

"""File downloading utilities for EEG data from cloud storage.

This module provides functions for downloading EEG data files and BIDS dependencies from
AWS S3 storage, with support for caching and progress tracking. It handles the communication
between the EEGDash metadata database and the actual EEG data stored in the cloud.
"""

import re
from pathlib import Path
from typing import Any

import s3fs
from fsspec.callbacks import TqdmCallback


def get_s3_filesystem() -> s3fs.S3FileSystem:
    """Get an anonymous S3 filesystem object.

    Initializes and returns an ``s3fs.S3FileSystem`` for anonymous access
    to public S3 buckets, configured for the 'us-east-2' region.

    Returns
    -------
    s3fs.S3FileSystem
        An S3 filesystem object.
    """
    return s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-east-2"})


def get_s3path(s3_bucket: str, filepath: str) -> str:
    """Construct an S3 URI from a bucket and file path.

    Parameters
    ----------
    s3_bucket : str
        The S3 bucket name (e.g., "s3://my-bucket").
    filepath : str
        The path to the file within the bucket.

    Returns
    -------
    str
        The full S3 URI (e.g., "s3://my-bucket/path/to/file").
    """
    return f"{s3_bucket}/{filepath}"


def download_s3_file(s3_path: str, local_path: Path, s3_open_neuro: bool) -> Path:
    """Download a single file from S3 to a local path.

    Handles the download of a raw EEG data file from an S3 bucket, caching it
    at the specified local path. Creates parent directories if they do not exist.

    Parameters
    ----------
    s3_path : str
        The full S3 URI of the file to download.
    local_path : pathlib.Path
        The local file path where the downloaded file will be saved.
    s3_open_neuro : bool
        A flag indicating if the S3 bucket is the OpenNeuro main bucket, which
        may affect path handling.

    Returns
    -------
    pathlib.Path
        The local path to the downloaded file.
    """
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
) -> None:
    """Download all BIDS dependency files from S3.

    Iterates through a list of BIDS dependency files, downloads each from the
    specified S3 bucket, and caches them in the appropriate local directory
    structure.

    Parameters
    ----------
    s3_bucket : str
        The S3 bucket to download from.
    bids_dependencies : list of str
        A list of dependency file paths relative to the S3 bucket root.
    bids_dependencies_original : list of str
        The original dependency paths, used for resolving local cache paths.
    cache_dir : pathlib.Path
        The root directory for caching.
    dataset_folder : pathlib.Path
        The specific folder for the dataset within the cache directory.
    record : dict
        The metadata record for the main data file, used to resolve paths.
    s3_open_neuro : bool
        Flag for OpenNeuro-specific path handling.
    """
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


def _filesystem_get(filesystem: s3fs.S3FileSystem, s3path: str, filepath: Path) -> Path:
    """Perform the file download using fsspec with a progress bar.

    Internal helper function that wraps the ``filesystem.get`` call to include
    a TQDM progress bar.

    Parameters
    ----------
    filesystem : s3fs.S3FileSystem
        The filesystem object to use for the download.
    s3path : str
        The full S3 URI of the source file.
    filepath : pathlib.Path
        The local destination path.

    Returns
    -------
    pathlib.Path
        The local path to the downloaded file.
    """
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


__all__ = [
    "download_s3_file",
    "download_dependencies",
    "get_s3path",
    "get_s3_filesystem",
]