"""Common utilities for parser modules.

This module provides shared functionality for validating file paths
and reading files with encoding fallback.
"""

from __future__ import annotations

from pathlib import Path


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
    path: Path, encodings: tuple[str, ...] = ("utf-8", "latin-1", "cp1252")
) -> str | None:
    """Read a file with encoding fallback.

    Tries multiple encodings to read a text file, returning the content
    on first success.

    Parameters
    ----------
    path : Path
        Path to the file to read.
    encodings : tuple[str, ...]
        Tuple of encodings to try, in order.

    Returns
    -------
    str | None
        File content as string, or None if all encodings fail.

    """
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, OSError):
            continue

    return None
