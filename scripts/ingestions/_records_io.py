"""Locate and read a digested ``<id>_records.json``, plain or gzipped.

GitHub rejects files over 100 MB, so the digest workflow gzips any records
file above ~90 MB before committing it to eegdash-dataset-listings
(``<id>_records.json.gz``). Every reader goes through here so both forms
are accepted.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


def records_path(dataset_dir: Path, dataset_id: str) -> Path | None:
    """Existing records file for ``dataset_id`` (plain preferred), else ``None``."""
    for name in (f"{dataset_id}_records.json", f"{dataset_id}_records.json.gz"):
        path = Path(dataset_dir) / name
        if path.exists():
            return path
    return None


def load_json(path: Path) -> Any:
    """``json.load`` that transparently handles a ``.gz`` file."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, encoding="utf-8") as f:
        return json.load(f)
