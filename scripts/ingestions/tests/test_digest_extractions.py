"""Unit tests for the Phase-8 helper extractions from ``3_digest.py``.

Tests two helpers extracted as the first pass of the digest mega-
function decomposition:

- ``sum_bids_channel_counts(sidecar_data)`` — sum every BIDS
  ``*ChannelCount`` field present.
- ``strip_dataset_prefix(bids_relpath, dataset_id)`` — strip the
  ``<dataset_id>/`` leading directory.

Both helpers were inlined inside the 521-LOC ``extract_record``
before Phase 8.

The characterisation tests in ``test_digest_helpers.py`` continue to
pass through the same module — together they confirm the
decomposition didn't drift observable behaviour.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_DIGEST_PATH = Path(__file__).parent.parent / "3_digest.py"


@pytest.fixture(scope="module")
def digest() -> ModuleType:
    """Load ``3_digest.py`` despite its digit-prefixed filename."""
    spec = importlib.util.spec_from_file_location("digest_under_test", _DIGEST_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── sum_bids_channel_counts ───────────────────────────────────────────────


def test_sum_channel_counts_eeg_only(digest: ModuleType) -> None:
    """Single-type sidecar (just EEGChannelCount)."""
    assert digest.sum_bids_channel_counts({"EEGChannelCount": 64}) == 64


def test_sum_channel_counts_meg_with_misc(digest: ModuleType) -> None:
    """MEG + miscellaneous channels — both contribute."""
    sidecar = {
        "MEGChannelCount": 306,
        "MiscChannelCount": 8,
        "TriggerChannelCount": 4,
    }
    assert digest.sum_bids_channel_counts(sidecar) == 318


def test_sum_channel_counts_ieeg_seeg(digest: ModuleType) -> None:
    """iEEG sidecars use iEEGChannelCount + SEEGChannelCount."""
    sidecar = {"iEEGChannelCount": 80, "SEEGChannelCount": 30, "EOGChannelCount": 1}
    assert digest.sum_bids_channel_counts(sidecar) == 111


def test_sum_channel_counts_all_fields(digest: ModuleType) -> None:
    """All 12 known fields contribute to the sum if present."""
    every_field = {
        "MEGChannelCount": 1,
        "EEGChannelCount": 2,
        "EOGChannelCount": 3,
        "ECGChannelCount": 4,
        "EMGChannelCount": 5,
        "MiscChannelCount": 6,
        "TriggerChannelCount": 7,
        "iEEGChannelCount": 8,
        "SEEGChannelCount": 9,
        "ECOGChannelCount": 10,
        "NIRSChannelCount": 11,
        "ACCELChannelCount": 12,
    }
    assert digest.sum_bids_channel_counts(every_field) == sum(range(1, 13))


def test_sum_channel_counts_returns_none_when_all_zero(digest: ModuleType) -> None:
    """All-zero / all-absent sidecar returns None, not 0.

    Distinguishing 'no info' from '0 channels' lets the caller decide
    whether to fall back to channels.tsv.
    """
    assert digest.sum_bids_channel_counts({}) is None
    assert digest.sum_bids_channel_counts({"EEGChannelCount": 0}) is None


def test_sum_channel_counts_handles_none_values(digest: ModuleType) -> None:
    """A field explicitly set to None is treated as 0 (not a crash)."""
    sidecar = {"EEGChannelCount": 32, "MEGChannelCount": None}
    assert digest.sum_bids_channel_counts(sidecar) == 32


def test_sum_channel_counts_ignores_unknown_fields(digest: ModuleType) -> None:
    """Unknown BIDS fields do not affect the sum."""
    sidecar = {"EEGChannelCount": 16, "SomeUnknownChannelCount": 999}
    assert digest.sum_bids_channel_counts(sidecar) == 16


# ─── strip_dataset_prefix ──────────────────────────────────────────────────


def test_strip_dataset_prefix_basic(digest: ModuleType) -> None:
    """Standard case: path begins with <dataset_id>/."""
    assert (
        digest.strip_dataset_prefix(
            "ds002893/sub-001/eeg/sub-001_task-rest_eeg.set", "ds002893"
        )
        == "sub-001/eeg/sub-001_task-rest_eeg.set"
    )


def test_strip_dataset_prefix_no_prefix(digest: ModuleType) -> None:
    """Path without the dataset prefix passes through unchanged."""
    assert (
        digest.strip_dataset_prefix("sub-001/eeg/sub-001_eeg.set", "ds002893")
        == "sub-001/eeg/sub-001_eeg.set"
    )


def test_strip_dataset_prefix_partial_match_not_stripped(digest: ModuleType) -> None:
    """Partial dataset-id match (no trailing slash) is NOT stripped."""
    # 'ds00' is a prefix of 'ds002893' but isn't followed by /.
    assert (
        digest.strip_dataset_prefix("ds002893/sub-01/eeg/x.edf", "ds00")
        == "ds002893/sub-01/eeg/x.edf"
    )


def test_strip_dataset_prefix_dataset_id_in_middle_of_path(
    digest: ModuleType,
) -> None:
    """Dataset id appearing mid-path is NOT stripped (only leading prefix)."""
    assert (
        digest.strip_dataset_prefix("sub-01/ds002893/eeg/x.edf", "ds002893")
        == "sub-01/ds002893/eeg/x.edf"
    )


def test_strip_dataset_prefix_empty_path(digest: ModuleType) -> None:
    """Empty path returns empty (no crash)."""
    assert digest.strip_dataset_prefix("", "ds002893") == ""


def test_strip_dataset_prefix_only_dataset_id(digest: ModuleType) -> None:
    """A path that IS the dataset id (no trailing /) is unchanged."""
    # 'ds002893' alone doesn't match the 'ds002893/' prefix.
    assert digest.strip_dataset_prefix("ds002893", "ds002893") == "ds002893"


def test_strip_dataset_prefix_with_trailing_slash_only(digest: ModuleType) -> None:
    """``<dataset_id>/`` with nothing after returns empty string."""
    assert digest.strip_dataset_prefix("ds002893/", "ds002893") == ""
