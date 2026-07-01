# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""File downloading utilities for EEG data from cloud storage.

This module provides functions for downloading EEG data files and BIDS dependencies from
AWS S3 storage, with support for caching and progress tracking. It handles the communication
between the EEGDash metadata database and the actual EEG data stored in the cloud.

It talks to S3 through a plain synchronous ``boto3`` client rather than ``s3fs``.
``s3fs`` transitively pulls in ``aiobotocore``, which hard-pins ``botocore`` to a
narrow range and makes any environment that also needs ``boto3`` (moabb, awscli,
neuralbench, …) effectively unsolvable. All access here is anonymous, read-only,
and limited to a handful of operations (HEAD, GET, LIST), so ``boto3`` covers it
directly. See https://github.com/eegdash/EEGDash/issues/397.
"""

import re
from pathlib import Path
from typing import Iterable, Sequence

import boto3
import rich.progress
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from rich.console import Console
from tqdm.auto import tqdm

from .logging import logger


def _split_s3_uri(uri: str) -> tuple[str, str]:
    """Split an ``s3://bucket/key`` (or ``bucket/key``) string into (bucket, key)."""
    without_scheme = re.sub(r"^s3://", "", str(uri)).lstrip("/")
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


class S3Client:
    """Minimal, anonymous S3 accessor with an ``s3fs``-compatible surface.

    Exposes just the operations EEGDash needs — :meth:`info`, :meth:`get_file`,
    :meth:`ls`, :meth:`du` — backed by a synchronous ``boto3`` client configured
    for unsigned (anonymous) access to public buckets. Paths may be given as
    ``s3://bucket/key`` or ``bucket/key``.
    """

    def __init__(self, *, region: str = "us-east-2", max_concurrency: int = 20):
        self._client = boto3.client(
            "s3",
            region_name=region,
            config=Config(
                signature_version=UNSIGNED,
                max_pool_connections=max_concurrency,
                retries={"max_attempts": 5, "mode": "standard"},
            ),
        )

    def info(self, path: str) -> dict:
        """Return object metadata (``{"size": <bytes>}``) via a HEAD request."""
        bucket, key = _split_s3_uri(path)
        resp = self._client.head_object(Bucket=bucket, Key=key)
        return {"size": resp["ContentLength"]}

    def get_file(self, rpath: str, lpath: str, *, callback=None) -> None:
        """Download a single object with one unsigned GET (no HEAD, no LIST).

        Streams the body to disk, feeding each chunk's byte count to
        ``callback`` (a ``callback(nbytes)`` callable) for progress. A missing
        key is normalised to :class:`FileNotFoundError` so callers can treat it
        the way they did under ``s3fs``.
        """
        bucket, key = _split_s3_uri(rpath)
        try:
            body = self._client.get_object(Bucket=bucket, Key=key)["Body"]
            with open(lpath, "wb") as fh:
                for chunk in body.iter_chunks(chunk_size=1024 * 1024):
                    fh.write(chunk)
                    if callback is not None:
                        callback(len(chunk))
        except ClientError as e:
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = str(e.response.get("Error", {}).get("Code"))
            if status == 404 or code in ("NoSuchKey", "NoSuchBucket"):
                raise FileNotFoundError(rpath) from e
            raise

    def _iter_objects(self, path: str):
        """Yield each object dict under a prefix via paginated ListObjectsV2."""
        bucket, key = _split_s3_uri(path)
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=key):
            yield from page.get("Contents", [])

    def ls(self, path: str, detail: bool = False):
        """List objects under a prefix, returning ``bucket/key`` strings."""
        bucket, _ = _split_s3_uri(path)
        names = [f"{bucket}/{obj['Key']}" for obj in self._iter_objects(path)]
        if detail:
            return [{"name": n, "type": "file"} for n in names]
        return names

    def du(self, path: str) -> int:
        """Sum the size in bytes of every object under a prefix."""
        return sum(obj["Size"] for obj in self._iter_objects(path))


def get_s3_filesystem(
    *,
    max_concurrency: int = 20,
    region: str = "us-east-2",
) -> S3Client:
    """Get an anonymous S3 accessor for public buckets.

    Parameters
    ----------
    max_concurrency : int
        Size of the underlying connection pool (default 20).
    region : str
        AWS region for the S3 endpoint (default ``"us-east-2"``).

    Returns
    -------
    S3Client
        An anonymous S3 accessor with an ``s3fs``-compatible surface.

    """
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
    return S3Client(region=region, max_concurrency=max_concurrency)


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
    s3_path: str, local_path: Path, *, filesystem: S3Client | None = None
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
    filesystem : S3Client | None
        Optional pre-created accessor to reuse across multiple downloads.

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
    filesystem: S3Client | None = None,
    skip_existing: bool = True,
    skip_missing: bool = False,
) -> list[Path]:
    """Download multiple S3 URIs to local destinations.

    Parameters
    ----------
    files : iterable of (str, Path)
        Pairs of (S3 URI, local destination path).
    filesystem : S3Client | None
        Optional pre-created accessor to reuse across multiple downloads.
    skip_existing : bool
        If True, do not download files that already exist locally.
    skip_missing : bool
        If True, skip files that do not exist on S3 instead of raising.

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

        try:
            _filesystem_get(
                filesystem=filesystem, s3path=uri, filepath=dest, size=remote_size
            )
        except FileNotFoundError:
            if skip_missing:
                logger.warning("File not found on S3, skipping: %s", uri)
                continue
            raise

        if remote_size is not None and dest.stat().st_size != remote_size:
            dest.unlink(missing_ok=True)
            raise OSError(
                f"Incomplete download for {uri} -> {dest} (expected {remote_size} bytes)."
            )

        downloaded.append(dest)
    return downloaded


def _remote_size(filesystem: S3Client, s3path: str) -> int | None:
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


class RichCallback:
    """Progress callback rendering with Rich; callable for ``boto3`` transfers."""

    def __init__(self, size: int | None = None, description: str = ""):
        self.progress = rich.progress.Progress(
            rich.progress.TextColumn("[bold blue]{task.description}"),
            rich.progress.BarColumn(bar_width=None),
            rich.progress.TaskProgressColumn(),
            "•",
            rich.progress.DownloadColumn(),
            "•",
            rich.progress.TransferSpeedColumn(),
            "•",
            rich.progress.TimeRemainingColumn(),
        )
        self.task_id = self.progress.add_task(description, total=size)
        self.progress.start()

    def set_size(self, size):
        self.progress.update(self.task_id, total=size)

    def relative_update(self, inc=1):
        self.progress.update(self.task_id, advance=inc)

    def __call__(self, inc):
        # boto3 invokes the callback with the number of bytes just transferred.
        self.relative_update(inc)

    def close(self):
        self.progress.stop()


class TqdmCallback:
    """Progress callback rendering with tqdm; callable for ``boto3`` transfers."""

    def __init__(self, size: int | None = None, description: str = ""):
        self.bar = tqdm(
            total=size,
            desc=description,
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
        )

    def __call__(self, inc):
        self.bar.update(inc)

    def close(self):
        self.bar.close()


def _filesystem_get(
    filesystem: S3Client,
    s3path: str,
    filepath: Path,
    *,
    size: int | None = None,
) -> Path:
    """Perform the file download with a progress bar (Rich if a TTY, else tqdm).

    Parameters
    ----------
    filesystem : S3Client
        The accessor to use for the download.
    s3path : str
        The full S3 URI of the source file.
    filepath : pathlib.Path
        The local destination path.

    Returns
    -------
    pathlib.Path
        The local path to the downloaded file.

    """
    # Show the BIDS-named local destination, not the SHA-keyed S3 object,
    # so the progress bar matches what the user expects on disk.
    filename = Path(filepath).name
    description = f"Downloading {filename}"

    # Check if we should use Rich
    use_rich = False
    try:
        # Check if console is available and interactive-ish
        console = Console()
        if console.is_terminal:  # or some other heuristic if needed
            use_rich = True
    except Exception:
        pass

    if use_rich:
        callback = RichCallback(size=size, description=description)
    else:
        callback = TqdmCallback(size=size, description=description)

    try:
        # NEMAR denies anonymous ListBucket but allows GetObject. ``get_file``
        # issues a single GetObject — no HeadObject, no ListObjectsV2 — so it
        # works on those buckets. All callers here pass single object URIs; the
        # CTF directory case in base.py pre-expands via ``S3Client.ls`` into
        # per-file pairs.
        filesystem.get_file(s3path, str(filepath), callback=callback)
    finally:
        # Ensure callback is closed properly (important for Rich to clean up display)
        if hasattr(callback, "close"):
            callback.close()

    return filepath


__all__ = [
    "download_s3_file",
    "download_files",
    "get_s3path",
    "get_s3_filesystem",
    "S3Client",
]
