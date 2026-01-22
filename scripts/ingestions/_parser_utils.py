"""Common utilities for parser modules.

This module provides shared functionality for validating file paths,
reading files with encoding fallback, and fetching from S3 for
git-annex symlinks.
"""

from __future__ import annotations

import logging
import re
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote

logger = logging.getLogger(__name__)


def is_broken_symlink(path: Path) -> bool:
    """Check if a path is a broken symlink (e.g., git-annex).

    Parameters
    ----------
    path : Path
        Path to check.

    Returns
    -------
    bool
        True if the path is a symlink but the target doesn't exist.

    """
    try:
        # is_symlink() returns True even for broken symlinks
        # exists() returns False for broken symlinks
        return path.is_symlink() and not path.exists()
    except (OSError, RuntimeError):
        return False


def extract_openneuro_info(path: Path) -> tuple[str, str] | None:
    """Extract OpenNeuro dataset ID and relative path from a file path.

    Looks for patterns like:
    - .../data/cloned/ds001234/sub-01/eeg/file.vhdr
    - .../ds001234/sub-01/eeg/file.vhdr

    Parameters
    ----------
    path : Path
        Path to the file.

    Returns
    -------
    tuple[str, str] | None
        Tuple of (dataset_id, relative_path) if found, None otherwise.
        relative_path is the path within the dataset (e.g., "sub-01/eeg/file.vhdr")

    """
    path_str = str(path.resolve() if not is_broken_symlink(path) else path.absolute())

    # Match OpenNeuro dataset ID pattern (ds followed by one or more digits)
    # Supports variable length IDs: ds000117, ds00117, ds0001234, etc.
    match = re.search(r"[/\\](ds\d+)[/\\](.+)$", path_str)
    if match:
        dataset_id = match.group(1)
        relative_path = match.group(2).replace("\\", "/")  # Normalize path separators
        return dataset_id, relative_path

    return None


def build_s3_url(dataset_id: str, relative_path: str, source: str = "openneuro") -> str:
    """Build S3 URL for a file in a dataset.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g., "ds001234").
    relative_path : str
        Path within the dataset (e.g., "sub-01/eeg/file.vhdr").
    source : str
        Data source. Currently only "openneuro" is supported.

    Returns
    -------
    str
        S3 URL for the file.

    Raises
    ------
    ValueError
        If the source is not supported.

    """
    if source == "openneuro":
        # URL-encode the path, preserving forward slashes
        encoded_path = quote(relative_path, safe="/")
        return f"https://s3.amazonaws.com/openneuro.org/{dataset_id}/{encoded_path}"
    else:
        raise ValueError(f"Unsupported source for S3 URL: {source}")


def fetch_from_s3(
    url: str,
    timeout: float = 30.0,
    encodings: tuple[str, ...] = ("utf-8", "latin-1", "cp1252"),
) -> str | None:
    """Fetch text content from an S3 URL.

    Parameters
    ----------
    url : str
        S3 URL to fetch.
    timeout : float
        Request timeout in seconds.
    encodings : tuple[str, ...]
        Encodings to try when decoding the response.

    Returns
    -------
    str | None
        File content as string, or None if fetch fails.

    """
    logger.debug("Fetching from S3: %s", url)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            content_bytes = response.read()

            # Try each encoding
            for encoding in encodings:
                try:
                    content = content_bytes.decode(encoding)
                    logger.debug(
                        "Successfully fetched %d bytes from S3", len(content_bytes)
                    )
                    return content
                except UnicodeDecodeError:
                    continue

            logger.warning("Failed to decode S3 content with any encoding: %s", url)
            return None
    except HTTPError as e:
        logger.debug("HTTP error fetching from S3: %s - %s", url, e)
        return None
    except URLError as e:
        logger.debug("URL error fetching from S3: %s - %s", url, e)
        return None
    except (TimeoutError, OSError) as e:
        logger.debug("Network error fetching from S3: %s - %s", url, e)
        return None


def validate_file_path(path: Path) -> bool:
    """Check if a file path exists and is readable.

    Handles broken symlinks (e.g., git-annex) by resolving the path
    and checking if the target exists.

    Parameters
    ----------
    path : Path
        Path to validate.

    Returns
    -------
    bool
        True if the file exists and is readable, False otherwise.

    Notes
    -----
    For broken git-annex symlinks, this returns False. Use
    `read_with_encoding_fallback` which will attempt S3 fallback
    for OpenNeuro datasets.

    """
    if not path.exists():
        return False

    # Handle broken symlinks (git-annex)
    try:
        resolved = path.resolve()
        if not resolved.exists():
            return False
    except (OSError, RuntimeError):
        return False

    return True


def read_with_encoding_fallback(
    path: Path,
    encodings: tuple[str, ...] = ("utf-8", "latin-1", "cp1252"),
    s3_fallback: bool = True,
) -> str | None:
    """Read a file with encoding fallback and optional S3 fallback.

    Tries multiple encodings to read a text file. If the file is a broken
    git-annex symlink and belongs to an OpenNeuro dataset, attempts to
    fetch the content from S3.

    Parameters
    ----------
    path : Path
        Path to the file to read.
    encodings : tuple[str, ...]
        Tuple of encodings to try, in order.
    s3_fallback : bool
        If True, attempt to fetch from S3 for broken git-annex symlinks
        in OpenNeuro datasets.

    Returns
    -------
    str | None
        File content as string, or None if all methods fail.

    """
    # First, try to read the file directly
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, OSError):
            continue

    # If direct read failed and S3 fallback is enabled, check if it's a broken symlink
    if s3_fallback and is_broken_symlink(path):
        info = extract_openneuro_info(path)
        if info:
            dataset_id, relative_path = info
            s3_url = build_s3_url(dataset_id, relative_path)
            logger.info(
                "Using S3 fallback for git-annex symlink: %s -> %s",
                path.name,
                s3_url,
            )
            return fetch_from_s3(s3_url, encodings=encodings)
        else:
            logger.debug(
                "Broken symlink is not an OpenNeuro dataset, cannot use S3 fallback: %s",
                path,
            )

    return None
