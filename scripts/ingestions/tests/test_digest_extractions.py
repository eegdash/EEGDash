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
from _helpers import INGEST_DIR as _INGEST_DIR

_DIGEST_PATH = _INGEST_DIR / "3_digest.py"


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


# ─── extract_sfreq_nchans_from_modality_sidecar ───────────────────────────


def _make_bids_tree(tmp_path: Path) -> Path:
    """Create a minimal BIDS tree under ``tmp_path`` and return its root."""
    root = tmp_path / "ds_test"
    eeg_dir = root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    return root


def test_modality_sidecar_short_circuits_when_both_populated(
    digest: ModuleType, tmp_path: Path
) -> None:
    """If caller already has sfreq and nchans, no filesystem walk happens."""
    root = _make_bids_tree(tmp_path)
    bids_file = root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set"
    bids_file.touch()
    # No sidecar exists, so a walk would return (None, None).
    sfreq, nchans = digest.extract_sfreq_nchans_from_modality_sidecar(
        bids_file, root, sampling_frequency=250.0, nchans=32
    )
    assert sfreq == 250.0
    assert nchans == 32


def test_modality_sidecar_reads_eeg_json(digest: ModuleType, tmp_path: Path) -> None:
    """Walks finds adjacent ``*_eeg.json`` and reads SamplingFrequency + counts."""
    root = _make_bids_tree(tmp_path)
    bids_file = root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set"
    bids_file.touch()
    sidecar = bids_file.with_suffix("").with_name("sub-01_task-rest_eeg.json")
    sidecar.write_text(
        '{"SamplingFrequency": 500.0, "EEGChannelCount": 64, "EOGChannelCount": 2}'
    )
    sfreq, nchans = digest.extract_sfreq_nchans_from_modality_sidecar(
        bids_file, root, sampling_frequency=None, nchans=None
    )
    assert sfreq == 500.0
    assert nchans == 66


def test_modality_sidecar_inheritance_walks_up(
    digest: ModuleType, tmp_path: Path
) -> None:
    """BIDS inheritance: a task-level sidecar at the dataset root applies."""
    root = _make_bids_tree(tmp_path)
    bids_file = root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set"
    bids_file.touch()
    # Place the sidecar at the dataset root with only the task entity.
    root_sidecar = root / "task-rest_eeg.json"
    root_sidecar.write_text('{"SamplingFrequency": 1000.0, "EEGChannelCount": 128}')
    sfreq, nchans = digest.extract_sfreq_nchans_from_modality_sidecar(
        bids_file, root, sampling_frequency=None, nchans=None
    )
    assert sfreq == 1000.0
    assert nchans == 128


def test_modality_sidecar_returns_none_when_no_sidecar(
    digest: ModuleType, tmp_path: Path
) -> None:
    """No sidecar anywhere → return caller's (None, None) unchanged."""
    root = _make_bids_tree(tmp_path)
    bids_file = root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set"
    bids_file.touch()
    sfreq, nchans = digest.extract_sfreq_nchans_from_modality_sidecar(
        bids_file, root, sampling_frequency=None, nchans=None
    )
    assert sfreq is None
    assert nchans is None


def test_modality_sidecar_skips_malformed_json(
    digest: ModuleType, tmp_path: Path
) -> None:
    """A malformed sidecar must not crash — caller gets None back."""
    root = _make_bids_tree(tmp_path)
    bids_file = root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set"
    bids_file.touch()
    sidecar = bids_file.with_suffix("").with_name("sub-01_task-rest_eeg.json")
    sidecar.write_text("{not a valid JSON}")
    sfreq, nchans = digest.extract_sfreq_nchans_from_modality_sidecar(
        bids_file, root, sampling_frequency=None, nchans=None
    )
    assert sfreq is None
    assert nchans is None


def test_modality_sidecar_preserves_existing_sfreq(
    digest: ModuleType, tmp_path: Path
) -> None:
    """If caller already has sfreq but not nchans, only nchans is filled."""
    root = _make_bids_tree(tmp_path)
    bids_file = root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set"
    bids_file.touch()
    sidecar = bids_file.with_suffix("").with_name("sub-01_task-rest_eeg.json")
    sidecar.write_text('{"SamplingFrequency": 999.0, "EEGChannelCount": 64}')
    sfreq, nchans = digest.extract_sfreq_nchans_from_modality_sidecar(
        bids_file, root, sampling_frequency=250.0, nchans=None
    )
    # Caller's sfreq was preserved; sidecar's value was NOT overwritten.
    assert sfreq == 250.0
    assert nchans == 64


# ─── extract_sfreq_nchans_from_channels_tsv ───────────────────────────────


def test_channels_tsv_reads_row_count(digest: ModuleType, tmp_path: Path) -> None:
    """nchans is derived from the TSV's row count when caller has none."""
    root = _make_bids_tree(tmp_path)
    bids_file = root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set"
    bids_file.touch()
    channels_tsv = bids_file.with_suffix("").with_name("sub-01_task-rest_channels.tsv")
    channels_tsv.write_text("name\ttype\nFp1\tEEG\nFp2\tEEG\nCz\tEEG\n")
    sfreq, nchans = digest.extract_sfreq_nchans_from_channels_tsv(
        bids_file, root, sampling_frequency=None, nchans=None
    )
    assert nchans == 3
    assert sfreq is None  # no sampling_frequency column


def test_channels_tsv_reads_sampling_frequency_column(
    digest: ModuleType, tmp_path: Path
) -> None:
    """When the TSV has a ``sampling_frequency`` column, first positive
    value is picked."""
    root = _make_bids_tree(tmp_path)
    bids_file = root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set"
    bids_file.touch()
    channels_tsv = bids_file.with_suffix("").with_name("sub-01_task-rest_channels.tsv")
    channels_tsv.write_text(
        "name\ttype\tsampling_frequency\nFp1\tEEG\t512\nFp2\tEEG\t512\n"
    )
    sfreq, nchans = digest.extract_sfreq_nchans_from_channels_tsv(
        bids_file, root, sampling_frequency=None, nchans=None
    )
    assert sfreq == 512.0
    assert nchans == 2


def test_channels_tsv_accepts_pascal_case_column(
    digest: ModuleType, tmp_path: Path
) -> None:
    """The PascalCase column name ``SamplingFrequency`` is also accepted."""
    root = _make_bids_tree(tmp_path)
    bids_file = root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set"
    bids_file.touch()
    channels_tsv = bids_file.with_suffix("").with_name("sub-01_task-rest_channels.tsv")
    channels_tsv.write_text("name\tSamplingFrequency\nFp1\t256\n")
    sfreq, _ = digest.extract_sfreq_nchans_from_channels_tsv(
        bids_file, root, sampling_frequency=None, nchans=None
    )
    assert sfreq == 256.0


def test_channels_tsv_skips_malformed(digest: ModuleType, tmp_path: Path) -> None:
    """A malformed TSV must not crash."""
    root = _make_bids_tree(tmp_path)
    bids_file = root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set"
    bids_file.touch()
    channels_tsv = bids_file.with_suffix("").with_name("sub-01_task-rest_channels.tsv")
    channels_tsv.write_bytes(b"\xff\xfe garbage \x00\x01")
    sfreq, nchans = digest.extract_sfreq_nchans_from_channels_tsv(
        bids_file, root, sampling_frequency=None, nchans=None
    )
    # The function should return None, None on garbage rather than raise.
    assert sfreq is None
    assert nchans is None


def test_channels_tsv_short_circuits_when_both_populated(
    digest: ModuleType, tmp_path: Path
) -> None:
    """Caller-supplied values pass through untouched."""
    root = _make_bids_tree(tmp_path)
    bids_file = root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set"
    bids_file.touch()
    # Even if a TSV would exist, the function shouldn't look.
    channels_tsv = bids_file.with_suffix("").with_name("sub-01_task-rest_channels.tsv")
    channels_tsv.write_text("name\nFp1\nFp2\n")  # would give nchans=2
    sfreq, nchans = digest.extract_sfreq_nchans_from_channels_tsv(
        bids_file, root, sampling_frequency=128.0, nchans=99
    )
    assert sfreq == 128.0
    assert nchans == 99
