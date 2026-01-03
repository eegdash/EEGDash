# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""File downloading utilities for EEG data from cloud storage.

This module provides functions for downloading EEG data files and BIDS dependencies from
AWS S3 storage, with support for caching and progress tracking. It handles the communication
between the EEGDash metadata database and the actual EEG data stored in the cloud.
"""

from pathlib import Path
from typing import Iterable, Sequence

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
    s3_bucket = str(s3_bucket).rstrip("/")
    filepath = str(filepath).lstrip("/")
    return f"{s3_bucket}/{filepath}" if filepath else s3_bucket


def download_s3_file(
    s3_path: str, local_path: Path, *, filesystem: s3fs.S3FileSystem | None = None
) -> Path:
    """Download a single file from S3 to a local path.

    Handles the download of a raw EEG data file from an S3 bucket, caching it
    at the specified local path. Creates parent directories if they do not exist.

    Parameters
    ----------
    s3_path : str
        The full S3 URI of the file to download.
    local_path : pathlib.Path
        The local file path where the downloaded file will be saved.
    filesystem : s3fs.S3FileSystem | None
        Optional pre-created filesystem to reuse across multiple downloads.

    Returns
    -------
    pathlib.Path
        The local path to the downloaded file.

    """
    filesystem = filesystem or get_s3_filesystem()
    local_path.parent.mkdir(parents=True, exist_ok=True)

    remote_size = _remote_size(filesystem, s3_path)
    if local_path.exists():
        if remote_size is None:
            return local_path
        if local_path.stat().st_size == remote_size:
            return local_path
        local_path.unlink(missing_ok=True)

    _filesystem_get(
        filesystem=filesystem, s3path=s3_path, filepath=local_path, size=remote_size
    )
    if remote_size is not None and local_path.stat().st_size != remote_size:
        local_path.unlink(missing_ok=True)
        raise OSError(
            f"Incomplete download for {s3_path} -> {local_path} "
            f"(expected {remote_size} bytes)."
        )

    return local_path


def download_files(
    files: Sequence[tuple[str, Path]] | Iterable[tuple[str, Path]],
    *,
    filesystem: s3fs.S3FileSystem | None = None,
    skip_existing: bool = True,
) -> list[Path]:
    """Download multiple S3 URIs to local destinations.

    Parameters
    ----------
    files : iterable of (str, Path)
        Pairs of (S3 URI, local destination path).
    filesystem : s3fs.S3FileSystem | None
        Optional pre-created filesystem to reuse across multiple downloads.
    skip_existing : bool
        If True, do not download files that already exist locally.

    """
    filesystem = filesystem or get_s3_filesystem()
    downloaded: list[Path] = []
    for uri, dest in files:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        remote_size = _remote_size(filesystem, uri)
        if dest.exists():
            if skip_existing:
                if remote_size is None or dest.stat().st_size == remote_size:
                    continue
            dest.unlink(missing_ok=True)

        _filesystem_get(
            filesystem=filesystem, s3path=uri, filepath=dest, size=remote_size
        )
        if remote_size is not None and dest.stat().st_size != remote_size:
            dest.unlink(missing_ok=True)
            raise OSError(
                f"Incomplete download for {uri} -> {dest} (expected {remote_size} bytes)."
            )

        downloaded.append(dest)
    return downloaded


def _remote_size(filesystem: s3fs.S3FileSystem, s3path: str) -> int | None:
    try:
        info = filesystem.info(s3path)
    except Exception:
        return None
    size = info.get("size") or info.get("Size")
    if size is None:
        return None
    try:
        return int(size)
    except Exception:
        return None


def _filesystem_get(
    filesystem: s3fs.S3FileSystem,
    s3path: str,
    filepath: Path,
    *,
    size: int | None = None,
) -> Path:
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
    "download_files",
    "get_s3path",
    "get_s3_filesystem",
]
