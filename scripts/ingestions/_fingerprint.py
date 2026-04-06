"""Fingerprint helpers for incremental ingestion."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def _stable_str(value: str) -> str:
    return value.replace("\0", "")


def _hash_entries(entries: Iterable[tuple[str, int]], seed: str = "") -> str:
    hasher = hashlib.sha256()
    if seed:
        hasher.update(seed.encode("utf-8", errors="replace"))
        hasher.update(b"\0")
    for path, size in sorted(entries):
        hasher.update(_stable_str(path).encode("utf-8", errors="replace"))
        hasher.update(b"\0")
        hasher.update(str(int(size)).encode("ascii", errors="ignore"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def fingerprint_from_manifest(dataset_id: str, source: str, manifest: dict) -> str:
    """Compute a fingerprint from a manifest.json payload."""
    entries: list[tuple[str, int]] = []

    files = manifest.get("files", [])
    for f in files:
        path = f.get("path") or f.get("name", "")
        size = int(f.get("size", 0) or 0)
        if path:
            entries.append((path, size))

        for inner in f.get("_zip_contents", []):
            inner_path = inner.get("path") or inner.get("name", "")
            inner_size = int(inner.get("size", 0) or 0)
            if inner_path:
                entries.append((f"{path}::{inner_path}", inner_size))

    seed = f"{dataset_id}|{source}"
    return _hash_entries(entries, seed=seed)


def fingerprint_from_files(
    dataset_id: str, source: str, files: Iterable[Path], root: Path
) -> str:
    """Compute a fingerprint from local file paths and sizes."""
    entries: list[tuple[str, int]] = []
    for file_path in files:
        try:
            rel_path = str(file_path.relative_to(root))
        except ValueError:
            rel_path = str(file_path)
        size = 0
        try:
            if file_path.is_file():
                size = file_path.stat().st_size
        except OSError:
            size = 0
        entries.append((rel_path, size))

    seed = f"{dataset_id}|{source}"
    return _hash_entries(entries, seed=seed)


def fingerprint_from_records(dataset_id: str, source: str, records: list[dict]) -> str:
    """Compute a fingerprint from record entries.

    Uses the file path and size to generate a stable hash. The path is derived from:
    1. storage.raw_key (preferred - set by create_record)
    2. bids_relpath (fallback - the canonical field)

    Note: storage.raw_key == bids_relpath when records are created via create_record(),
    so the fallback mainly handles legacy or manually created records.
    """
    entries: list[tuple[str, int]] = []
    for record in records:
        storage = record.get("storage", {}) if isinstance(record, dict) else {}
        # Prefer storage.raw_key, fall back to bids_relpath (both should have same value)
        key = storage.get("raw_key") or record.get("bids_relpath") or ""
        size = int(record.get("size_bytes", 0) or 0)
        if key:
            entries.append((key, size))

    seed = f"{dataset_id}|{source}"
    return _hash_entries(entries, seed=seed)
