"""Shared fixtures for acceptance tests.

The acceptance suite loads outputs from the snapshot fixtures
(``tests/fixtures/digest_snapshots/outputs/``) and validates them
against three layers:

* Layer 2 — producer-side Pydantic models (``eegdash.schemas``).
* Layer 3 — re-running ``digest_dataset`` against the input twice
  yields byte-identical output.
* Layer 4 — API-side Beanie document
  (``mongodb-eegdash-server/api/main.py``).

This conftest exposes the loader fixtures every layer reuses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# Path layout:
#   scripts/ingestions/tests/acceptance/conftest.py  ← this file
#   scripts/ingestions/                              ← INGEST_DIR
#   scripts/ingestions/tests/fixtures/digest_snapshots/outputs/
_INGEST_DIR = Path(__file__).resolve().parent.parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

_SNAPSHOT_OUTPUTS = _INGEST_DIR / "tests" / "fixtures" / "digest_snapshots" / "outputs"


def _load_dataset_outputs(dataset_dir: Path) -> dict[str, Any] | None:
    """Load the four JSON files emitted by ``write_dataset_outputs``.

    Returns ``None`` if any of the four files are missing — the caller
    can then skip the fixture (e.g. if a future snapshot only emits
    three files because the dataset has no montages).
    """
    dataset_id = dataset_dir.name
    out: dict[str, Any] = {}
    for kind in ("dataset", "records", "montages", "summary"):
        path = dataset_dir / f"{dataset_id}_{kind}.json"
        if not path.exists():
            return None
        out[kind] = json.loads(path.read_text())
    return out


@pytest.fixture(scope="session")
def snapshot_outputs() -> dict[str, dict[str, Any]]:
    """Load every snapshot fixture's outputs.

    Returned dict is keyed by fixture name (e.g. ``"ds_snapshot_vhdr"``);
    each value has keys ``dataset`` / ``records`` / ``montages`` / ``summary``.

    The Layer-3 idempotency test produces its own outputs in a tmp
    dir; the snapshot outputs here are the *committed* baselines that
    Layers 2 + 4 validate.
    """
    if not _SNAPSHOT_OUTPUTS.exists():
        pytest.skip(
            f"Snapshot output dir missing: {_SNAPSHOT_OUTPUTS}. "
            f"Run `pytest tests/test_digest_snapshot.py` to generate, "
            f"or recover via tests/fixtures/digest_snapshots/README.md."
        )
    out: dict[str, dict[str, Any]] = {}
    for child in _SNAPSHOT_OUTPUTS.iterdir():
        if not child.is_dir():
            continue
        loaded = _load_dataset_outputs(child)
        if loaded is not None:
            out[child.name] = loaded
    if not out:
        pytest.skip(f"No complete snapshot outputs found under {_SNAPSHOT_OUTPUTS}")
    return out


@pytest.fixture(scope="session")
def ingest_dir() -> Path:
    """Absolute path to ``scripts/ingestions/``."""
    return _INGEST_DIR
