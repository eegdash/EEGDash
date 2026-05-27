"""Tests for the within-dataset MEG montage cache (Task 1 — Perf sprint).

The cache exploits the domain invariant: within a single MEG dataset,
records that share `nchans` share device → share layout. First record
extracts; subsequent records reuse the cached (hash, doc) without
re-running extract_layout or its network calls.

Cache MUST NOT leak across datasets — the cache key includes dataset_id.
Non-MEG records bypass the cache entirely (their layouts come from
electrodes.tsv, not the FIF header — sidecar reads are cheap and
not the bottleneck).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))


@pytest.fixture(scope="module")
def digest() -> ModuleType:
    """Load 3_digest.py via importlib (numeric filename)."""
    spec = importlib.util.spec_from_file_location(
        "digest_under_test", _INGEST_DIR / "3_digest.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _meg_record(nchans: int, name: str = "sub-01_run-01_meg.fif") -> dict:
    return {
        "datatype": "meg",
        "nchans": nchans,
        "bids_relpath": name,
    }


def test_first_meg_record_calls_extract_layout(
    digest: ModuleType, tmp_path: Path
) -> None:
    """First MEG record with a given nchans triggers extract_layout."""
    record = _meg_record(306)
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        return_value=("hash-306-A", {"system": "neuromag306"}),
    ) as mocked:
        errors = digest._attach_montage_to_record(
            record,
            tmp_path / "sub-01_meg.fif",
            tmp_path,
            montages,
            "ds-meg-001",
            "2026-05-22T00:00:00+00:00",
            montage_cache=cache,
        )

    assert errors == []
    assert mocked.call_count == 1
    assert record["montage_hash"] == "hash-306-A"
    assert "hash-306-A" in montages
    assert cache[("ds-meg-001", 306)] == (
        "hash-306-A",
        {
            "system": "neuromag306",
            "first_seen": "2026-05-22T00:00:00+00:00",
            "representative_dataset": "ds-meg-001",
        },
    )


def test_second_meg_record_same_nchans_reuses_cache(
    digest: ModuleType, tmp_path: Path
) -> None:
    """Second record with same (dataset, nchans) MUST NOT call extract_layout."""
    record_a = _meg_record(306, "sub-01_run-01_meg.fif")
    record_b = _meg_record(306, "sub-01_run-02_meg.fif")
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        return_value=("hash-306-A", {"system": "neuromag306"}),
    ) as mocked:
        digest._attach_montage_to_record(
            record_a,
            tmp_path / "a.fif",
            tmp_path,
            montages,
            "ds-meg-001",
            "2026-05-22T00:00:00+00:00",
            montage_cache=cache,
        )
        digest._attach_montage_to_record(
            record_b,
            tmp_path / "b.fif",
            tmp_path,
            montages,
            "ds-meg-001",
            "2026-05-22T00:00:00+00:00",
            montage_cache=cache,
        )

    # extract_layout called ONCE despite two records.
    assert mocked.call_count == 1
    assert record_a["montage_hash"] == "hash-306-A"
    assert record_b["montage_hash"] == "hash-306-A"


def test_different_nchans_skips_cache(digest: ModuleType, tmp_path: Path) -> None:
    """Two records with different nchans → two extract_layout calls."""
    record_a = _meg_record(306)
    record_b = _meg_record(204)  # different device / channel count
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        side_effect=[
            ("hash-306-A", {"system": "neuromag306"}),
            ("hash-204-B", {"system": "ctf204"}),
        ],
    ) as mocked:
        digest._attach_montage_to_record(
            record_a,
            tmp_path / "a.fif",
            tmp_path,
            montages,
            "ds-meg-001",
            "now",
            montage_cache=cache,
        )
        digest._attach_montage_to_record(
            record_b,
            tmp_path / "b.fif",
            tmp_path,
            montages,
            "ds-meg-001",
            "now",
            montage_cache=cache,
        )

    assert mocked.call_count == 2
    assert record_a["montage_hash"] == "hash-306-A"
    assert record_b["montage_hash"] == "hash-204-B"


def test_cache_does_not_leak_across_datasets(
    digest: ModuleType, tmp_path: Path
) -> None:
    """Same nchans, different dataset_id → cache miss (different cache keys)."""
    record_a = _meg_record(306)
    record_b = _meg_record(306)
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        side_effect=[
            ("hash-306-A", {"system": "neuromag306"}),
            (
                "hash-306-A",
                {"system": "neuromag306"},
            ),  # different doc still hashes same
        ],
    ) as mocked:
        digest._attach_montage_to_record(
            record_a,
            tmp_path / "a.fif",
            tmp_path,
            montages,
            "ds-meg-001",
            "now",
            montage_cache=cache,
        )
        digest._attach_montage_to_record(
            record_b,
            tmp_path / "b.fif",
            tmp_path,
            montages,
            "ds-meg-002",
            "now",
            montage_cache=cache,
        )

    # extract_layout called once per dataset_id, even with same nchans.
    assert mocked.call_count == 2


def test_non_meg_record_bypasses_cache(digest: ModuleType, tmp_path: Path) -> None:
    """EEG records still call extract_layout per-file — the cache only
    helps MEG where the device check is well-defined."""
    record_a = {"datatype": "eeg", "nchans": 64, "bids_relpath": "a.vhdr"}
    record_b = {"datatype": "eeg", "nchans": 64, "bids_relpath": "b.vhdr"}
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        side_effect=[
            ("hash-eeg-a", {"layout": "ten-twenty"}),
            ("hash-eeg-b", {"layout": "ten-twenty"}),
        ],
    ) as mocked:
        digest._attach_montage_to_record(
            record_a,
            tmp_path / "a.vhdr",
            tmp_path,
            montages,
            "ds-eeg-001",
            "now",
            montage_cache=cache,
        )
        digest._attach_montage_to_record(
            record_b,
            tmp_path / "b.vhdr",
            tmp_path,
            montages,
            "ds-eeg-001",
            "now",
            montage_cache=cache,
        )

    # Per-file extraction — cache MUST NOT have hijacked the EEG path.
    assert mocked.call_count == 2
    assert cache == {}  # no MEG entries; EEG bypasses


def test_missing_nchans_skips_cache(digest: ModuleType, tmp_path: Path) -> None:
    """A MEG record with no nchans (missing metadata) must NOT be cached
    — without a channel count there's no safe key."""
    record = {"datatype": "meg", "bids_relpath": "broken.fif"}  # no nchans
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        return_value=("hash-x", {"layout": "x"}),
    ) as mocked:
        digest._attach_montage_to_record(
            record,
            tmp_path / "broken.fif",
            tmp_path,
            montages,
            "ds-x",
            "now",
            montage_cache=cache,
        )

    assert mocked.call_count == 1
    assert cache == {}  # no key without nchans
    assert record["montage_hash"] == "hash-x"


def test_extract_layout_returning_none_is_not_cached(
    digest: ModuleType, tmp_path: Path
) -> None:
    """If extract_layout returns None (no montage available), don't
    cache the absence — next record gets another chance."""
    record_a = _meg_record(306)
    record_b = _meg_record(306)
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        side_effect=[None, ("hash-306-A", {"system": "neuromag306"})],
    ) as mocked:
        digest._attach_montage_to_record(
            record_a,
            tmp_path / "a.fif",
            tmp_path,
            montages,
            "ds-meg-001",
            "now",
            montage_cache=cache,
        )
        digest._attach_montage_to_record(
            record_b,
            tmp_path / "b.fif",
            tmp_path,
            montages,
            "ds-meg-001",
            "now",
            montage_cache=cache,
        )

    # Second call still went through extract_layout because the first
    # returned None (cache only stores positive results).
    assert mocked.call_count == 2
    assert record_a["montage_hash"] is None
    assert record_b["montage_hash"] == "hash-306-A"
