"""Resolve a data file's byte size WITHOUT reading its contents.

Order of cheapness:
1. git-annex pointer/symlink key (``MD5E-s<size>--…``) — zero bytes read.
2. ``os.stat`` of a real present file.
Returns ``None`` when neither works (missing file, unresolvable pointer).

This is the single seam the file-size arithmetic tier uses so the
``n_times = data_bytes / (nchans × dtype_bytes)`` computation never
fetches signal — on a shallow clone the ``.eeg``/``.edf`` is an annex
pointer and its size comes straight from the key.
"""

from __future__ import annotations

import os
from pathlib import Path

from _file_utils import parse_annex_size

# A git-annex pointer file is tiny; never read more than this as a "pointer".
_ANNEX_POINTER_MAX_SIZE = 256


def _annex_size_from_symlink(path: Path) -> int | None:
    """Size from a (possibly broken) git-annex symlink target."""
    try:
        if not path.is_symlink():
            return None
        target = os.readlink(path)
    except OSError:
        return None
    return parse_annex_size(str(target))


def _annex_size_from_pointer_text(path: Path) -> int | None:
    """Size from a git-annex *pointer file* (content is the annex key)."""
    try:
        if not path.is_file() or path.stat().st_size > _ANNEX_POINTER_MAX_SIZE:
            return None
        text = path.read_text(errors="ignore")
    except OSError:
        return None
    if "/annex/" not in text and "MD5E-s" not in text and "SHA256E-s" not in text:
        return None
    return parse_annex_size(text)


def data_file_size(path: Path | str) -> int | None:
    """Best-effort byte size of *path* without reading signal data."""
    path = Path(path)
    size = _annex_size_from_symlink(path)
    if size is not None and size > 0:
        return size
    try:
        if path.is_file():
            real = path.stat().st_size
            # A tiny "real" file may itself be an annex pointer.
            if real <= _ANNEX_POINTER_MAX_SIZE:
                ptr = _annex_size_from_pointer_text(path)
                if ptr is not None and ptr > 0:
                    return ptr
            return real
    except OSError:
        return None
    return None


__all__ = ["data_file_size"]
