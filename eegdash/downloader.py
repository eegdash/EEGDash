# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""File downloading utilities for EEG data from cloud storage.

This module provides functions for downloading EEG data files and BIDS dependencies from
AWS S3 storage, with support for caching and progress tracking. It handles the communication
between the EEGDash metadata database and the actual EEG data stored in the cloud.

The module automatically uses s5cmd for faster downloads when available, falling back to
s3fs if s5cmd is not installed. All existing functionality (progress bars, size checking,
error handling) is preserved regardless of the backend used.
"""

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterable, Sequence

import rich.progress
import s3fs
from fsspec.callbacks import Callback, TqdmCallback
from rich.console import Console


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


def _s5cmd_available() -> bool:
    """Check if s5cmd is available in the system PATH."""
    return shutil.which("s5cmd") is not None


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
    """Get remote file size, trying s5cmd first, then falling back to s3fs."""
    # Try s5cmd first if available
    if _s5cmd_available():
        try:
            # s5cmd ls with --json returns JSON with size information
            result = subprocess.run(
                ["s5cmd", "--no-sign-request", "ls", "--json", s3path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Parse JSON output (s5cmd outputs one JSON object per line)
                for line in result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    try:
                        ls_info = json.loads(line)
                        # s5cmd ls JSON format: {"key": "...", "size": 123, ...}
                        size = ls_info.get("size")
                        if size is not None:
                            return int(size)
                    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
                        continue
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    # Fall back to s3fs
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


class RichCallback(Callback):
    """FSSpec callback using Rich Progress."""

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

    def close(self):
        self.progress.stop()


def _filesystem_get(
    filesystem: s3fs.S3FileSystem,
    s3path: str,
    filepath: Path,
    *,
    size: int | None = None,
) -> Path:
    """Perform the file download using s5cmd (if available) or fsspec with a progress bar.

    Internal helper function that attempts to use s5cmd for faster downloads,
    falling back to s3fs if s5cmd is not available. Includes a progress bar
    (Rich if available/console, else TQDM).

    Parameters
    ----------
    filesystem : s3fs.S3FileSystem
        The filesystem object to use for the download (used as fallback).
    s3path : str
        The full S3 URI of the source file.
    filepath : pathlib.Path
        The local destination path.

    Returns
    -------
    pathlib.Path
        The local path to the downloaded file.

    """
    filename = Path(s3path).name
    description = f"Downloading {filename}"

    # Try s5cmd first if available
    if _s5cmd_available():
        try:
            return _s5cmd_get(s3path, filepath, size=size, description=description)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError, ValueError):
            # Fall through to s3fs if s5cmd fails
            pass

    # Fall back to s3fs
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
        callback = TqdmCallback(
            size=size,
            tqdm_kwargs=dict(
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
            ),
        )

    try:
        filesystem.get(s3path, str(filepath), callback=callback)
    finally:
        # Ensure callback is closed properly (important for Rich to clean up display)
        if hasattr(callback, "close"):
            callback.close()

    return filepath


def _s5cmd_get(
    s3path: str, filepath: Path, *, size: int | None = None, description: str = ""
) -> Path:
    """Download file using s5cmd with progress tracking.

    Parameters
    ----------
    s3path : str
        The full S3 URI of the source file.
    filepath : pathlib.Path
        The local destination path.
    size : int | None
        Expected file size for progress tracking.
    description : str
        Description for progress bar.

    Returns
    -------
    pathlib.Path
        The local path to the downloaded file.

    Raises
    ------
    OSError
        If the download fails or is incomplete.

    """
    # Check if we should use Rich
    use_rich = False
    try:
        console = Console()
        if console.is_terminal:
            use_rich = True
    except Exception:
        pass

    if use_rich:
        progress = rich.progress.Progress(
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
        task_id = progress.add_task(description, total=size)
        progress.start()
    else:
        from tqdm import tqdm

        progress = tqdm(
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

    # Track download progress by monitoring file size
    download_complete = threading.Event()
    last_size = 0

    def monitor_progress():
        """Monitor file size and update progress bar."""
        nonlocal last_size
        while not download_complete.is_set():
            if filepath.exists():
                try:
                    current_size = filepath.stat().st_size
                    if current_size > last_size:
                        if use_rich:
                            progress.update(task_id, completed=current_size)
                        else:
                            progress.n = current_size
                            progress.refresh()
                        last_size = current_size
                except (OSError, FileNotFoundError):
                    pass
            time.sleep(0.1)

    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
    monitor_thread.start()

    try:
        # Run s5cmd download
        process = subprocess.run(
            [
                "s5cmd",
                "--no-sign-request",
                "cp",
                s3path,
                str(filepath),
            ],
            capture_output=True,
            text=True,
            timeout=None,
        )

        download_complete.set()

        if process.returncode != 0:
            error_msg = process.stderr or process.stdout or "Unknown error"
            raise OSError(
                f"s5cmd download failed with return code {process.returncode}: {error_msg}"
            )

        # Final update to ensure progress bar shows 100%
        if filepath.exists():
            final_size = filepath.stat().st_size
            if use_rich:
                progress.update(task_id, completed=final_size)
            else:
                progress.n = final_size
                progress.refresh()

    except subprocess.TimeoutExpired:
        download_complete.set()
        raise OSError(f"s5cmd download timed out for {s3path}")
    except FileNotFoundError:
        download_complete.set()
        raise
    finally:
        download_complete.set()
        monitor_thread.join(timeout=1.0)
        if use_rich:
            progress.stop()
        else:
            progress.close()

    return filepath


__all__ = [
    "download_s3_file",
    "download_files",
    "get_s3path",
    "get_s3_filesystem",
]
