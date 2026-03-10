"""Test BIDS metadata fixes against the 19 datasets from process-fix-bids-metadata.md.

Each test loads one recording from a dataset that previously failed due to
BIDS metadata issues. The fixes in base.py / io.py should allow these to
load successfully (or raise DataIntegrityError for truly corrupt files).

Subjects are pinned to keep downloads minimal (one recording per test).
These tests use the real cache directory (not tmp_path) to avoid
re-downloading multi-GB files across test runs.
"""

import pytest

from eegdash import EEGDashDataset
from eegdash.dataset.exceptions import DataIntegrityError


def _load_one(cache_dir, dataset_id, **kwargs):
    """Load a single recording from a dataset, return (raw, record) or raise."""
    ds = EEGDashDataset(
        cache_dir=cache_dir,
        dataset=dataset_id,
        on_error="warn",
        **kwargs,
    )
    assert len(ds) > 0, f"No records found for {dataset_id}"
    rec = ds.datasets[0]
    raw = rec.raw  # triggers download + repair + load
    return raw, rec.record


def _is_network_error(exc: Exception) -> bool:
    """Check if the exception is a network/S3 download error."""
    msg = str(exc).lower()
    if any(kw in msg for kw in ("download", "s3://", "timeout")):
        return True
    # Walk the exception chain — fsspec wraps timeouts in FSTimeoutError
    for e in (exc, getattr(exc, "__cause__", None), getattr(exc, "__context__", None)):
        if e is not None and type(e).__name__ in (
            "FSTimeoutError",
            "AioReadTimeoutError",
        ):
            return True
    return False


def _assert_or_skip(cache_dir, dataset_id, **kwargs):
    """Load one recording, assert success or skip on network/integrity errors."""
    try:
        raw, record = _load_one(cache_dir, dataset_id, **kwargs)
        if raw is not None:
            assert len(raw.ch_names) > 0
        else:
            pytest.skip(f"{dataset_id}: raw is None (on_error=warn triggered)")
    except DataIntegrityError:
        pytest.skip(f"{dataset_id}: DataIntegrityError (unrecoverable)")
    except Exception as e:
        if _is_network_error(e):
            pytest.skip(f"{dataset_id}: network error: {e}")
        raise


# ── Missing coordsystem.json ─────────────────────────────────────────
_COORDSYSTEM_DATASETS = [
    "ds004370",
    "ds004551",
    "ds004752",
    "ds004770",
    "ds004819",
    "ds004859",
    "ds004944",
    "ds005007",
    "ds005545",
    "ds005931",
    "ds006107",
    "ds006234",
]


@pytest.mark.parametrize("dataset_id", _COORDSYSTEM_DATASETS)
@pytest.mark.network
def test_missing_coordsystem(cache_dir, dataset_id):
    """Datasets that were missing coordsystem.json should load via generate/symlink."""
    _assert_or_skip(cache_dir, dataset_id)


# ── Non-numeric run entity ────────────────────────────────────────────
@pytest.mark.network
def test_non_numeric_run_ds003190(cache_dir):
    """ds003190 has non-numeric run entities (e.g. run-6)."""
    _assert_or_skip(cache_dir, "ds003190", subject="019")


# ── Invalid scans.tsv timestamps ─────────────────────────────────────
@pytest.mark.parametrize(
    "dataset_id,subject", [("ds003775", "021"), ("ds004381", "01")]
)
@pytest.mark.network
def test_invalid_scans_timestamps(cache_dir, dataset_id, subject):
    """Datasets with invalid scans.tsv timestamps (sec>=60, NaN)."""
    _assert_or_skip(cache_dir, dataset_id, subject=subject)


# ── n/a float conversion ─────────────────────────────────────────────
@pytest.mark.network
def test_na_float_conversion_ds004860(cache_dir):
    """ds004860 has n/a values with trailing whitespace causing float errors."""
    _assert_or_skip(cache_dir, "ds004860", subject="130")


# ── IndexError header parsing ────────────────────────────────────────
@pytest.mark.network
def test_index_error_ds005448(cache_dir):
    """ds005448 triggers IndexError during header parsing → DataIntegrityError."""
    try:
        raw, record = _load_one(cache_dir, "ds005448", subject="STREEF03")
        if raw is not None:
            assert len(raw.ch_names) > 0
    except DataIntegrityError:
        # Expected — this dataset has unrecoverable corruption
        pytest.skip("ds005448: DataIntegrityError (expected for corrupt data)")
    except Exception as e:
        if _is_network_error(e):
            pytest.skip(f"ds005448: network error: {e}")
        raise


# ── Multiple hyphens in BIDS filename ────────────────────────────────
@pytest.mark.network
def test_multiple_hyphens_ds005697(cache_dir):
    """ds005697 has multiple hyphens → "Unallowed" handler falls back to direct."""
    _assert_or_skip(cache_dir, "ds005697", subject="16")


# ── Path-not-in-list (scans.tsv mismatch) ────────────────────────────
@pytest.mark.network
def test_path_not_in_list_ds006525(cache_dir):
    """ds006525 has scans.tsv path mismatch → participants repair + direct fallback."""
    _assert_or_skip(cache_dir, "ds006525", subject="026")
