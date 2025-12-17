# Authors: The EEGDash contributors.
# License: BSD-3-Clause

"""Resolve EEGDash records to remote URIs and local cache paths.

This module centralizes the mapping from a *record* to:
- remote download URIs (e.g., S3)
- local cache destinations

The intent is to keep storage/layout rules in one place so downloader + loader
code stays simple and does not contain bucket-specific hacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


def _join_uri(base: str, key: str) -> str:
    base = str(base).rstrip("/")
    key = str(key).lstrip("/")
    return f"{base}/{key}" if key else base


def _as_posix(path: str) -> str:
    return PurePosixPath(str(path)).as_posix()


@dataclass(frozen=True, slots=True)
class ResolvedRecord:
    bids_root: Path
    raw_path: Path
    dep_paths: list[Path]
    raw_uri: str | None
    dep_uris: list[str]


def resolve_record(record: Mapping[str, Any], *, cache_dir: str | Path) -> ResolvedRecord:
    """Resolve a (v2-shaped) record to concrete local paths and remote URIs."""
    cache_dir_path = Path(cache_dir)

    cache = record.get("cache") or {}
    storage = record.get("storage") or {}

    dataset_subdir = cache.get("dataset_subdir")
    raw_relpath = cache.get("raw_relpath")
    dep_relpaths = cache.get("dep_relpaths") or []

    if not dataset_subdir or not raw_relpath:
        raise ValueError("record missing required v2 fields: cache.dataset_subdir/raw_relpath")

    bids_root = cache_dir_path / str(dataset_subdir)
    raw_path = bids_root / _as_posix(raw_relpath)
    dep_paths = [bids_root / _as_posix(p) for p in dep_relpaths]

    raw_uri = None
    dep_uris: list[str] = []

    if storage:
        backend = storage.get("backend")
        base = storage.get("base")
        raw_key = storage.get("raw_key")
        dep_keys = storage.get("dep_keys") or []

        if backend == "s3" and base and raw_key:
            raw_uri = _join_uri(str(base), _as_posix(raw_key))
            dep_uris = [_join_uri(str(base), _as_posix(k)) for k in dep_keys]

    return ResolvedRecord(
        bids_root=bids_root,
        raw_path=raw_path,
        dep_paths=dep_paths,
        raw_uri=raw_uri,
        dep_uris=dep_uris,
    )


__all__ = ["ResolvedRecord", "resolve_record"]

