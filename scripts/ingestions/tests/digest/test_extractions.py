from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from _helpers import INGEST_DIR as _INGEST_DIR

_DIGEST_PATH = _INGEST_DIR / "3_digest.py"


@pytest.fixture(scope="module")
def digest() -> ModuleType:
    """Load ``3_digest.py`` by file path (digit-prefixed, not importable by name)."""
    spec = importlib.util.spec_from_file_location("digest_under_test", _DIGEST_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── sum_bids_channel_counts ───────────────────────────────────────────────


def test_sum_channel_counts_eeg_only(digest: ModuleType) -> None:
    assert digest.sum_bids_channel_counts({"EEGChannelCount": 64}) == 64


def test_sum_channel_counts_meg_with_misc(digest: ModuleType) -> None:
    sidecar = {
        "MEGChannelCount": 306,
        "MiscChannelCount": 8,
        "TriggerChannelCount": 4,
    }
    assert digest.sum_bids_channel_counts(sidecar) == 318


def test_sum_channel_counts_ieeg_seeg(digest: ModuleType) -> None:
    sidecar = {"iEEGChannelCount": 80, "SEEGChannelCount": 30, "EOGChannelCount": 1}
    assert digest.sum_bids_channel_counts(sidecar) == 111


def test_sum_channel_counts_all_fields(digest: ModuleType) -> None:
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
    """Returns None (not 0) to distinguish 'no info' from '0 channels'."""
    assert digest.sum_bids_channel_counts({}) is None
    assert digest.sum_bids_channel_counts({"EEGChannelCount": 0}) is None


def test_sum_channel_counts_handles_none_values(digest: ModuleType) -> None:
    sidecar = {"EEGChannelCount": 32, "MEGChannelCount": None}
    assert digest.sum_bids_channel_counts(sidecar) == 32


def test_sum_channel_counts_ignores_unknown_fields(digest: ModuleType) -> None:
    sidecar = {"EEGChannelCount": 16, "SomeUnknownChannelCount": 999}
    assert digest.sum_bids_channel_counts(sidecar) == 16


# ─── strip_dataset_prefix ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "dataset_id", "expected"),
    [
        pytest.param(
            "ds002893/sub-001/eeg/sub-001_task-rest_eeg.set",
            "ds002893",
            "sub-001/eeg/sub-001_task-rest_eeg.set",
            id="leading_prefix_stripped",
        ),
        pytest.param(
            "sub-001/eeg/sub-001_eeg.set",
            "ds002893",
            "sub-001/eeg/sub-001_eeg.set",
            id="no_prefix_passthrough",
        ),
        pytest.param(
            "ds002893/sub-01/eeg/x.edf",
            "ds00",
            "ds002893/sub-01/eeg/x.edf",
            id="partial_prefix_not_stripped",
        ),
        pytest.param(
            "sub-01/ds002893/eeg/x.edf",
            "ds002893",
            "sub-01/ds002893/eeg/x.edf",
            id="mid_path_id_not_stripped",
        ),
        pytest.param("", "ds002893", "", id="empty_path"),
        pytest.param("ds002893", "ds002893", "ds002893", id="exact_id_no_slash"),
        pytest.param("ds002893/", "ds002893", "", id="exact_id_with_trailing_slash"),
    ],
)
def test_strip_dataset_prefix(
    digest: ModuleType, path: str, dataset_id: str, expected: str
) -> None:
    assert digest.strip_dataset_prefix(path, dataset_id) == expected


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
    """Skips filesystem walk when caller already supplies both values."""
    root = _make_bids_tree(tmp_path)
    bids_file = root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set"
    bids_file.touch()
    sfreq, nchans = digest.extract_sfreq_nchans_from_modality_sidecar(
        bids_file, root, sampling_frequency=250.0, nchans=32
    )
    assert sfreq == 250.0
    assert nchans == 32


def test_modality_sidecar_reads_eeg_json(digest: ModuleType, tmp_path: Path) -> None:
    """Reads SamplingFrequency and channel counts from an adjacent ``*_eeg.json``."""
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
    """A malformed sidecar must not crash."""
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
    """Caller's sfreq is preserved; only missing nchans is filled from the sidecar."""
    root = _make_bids_tree(tmp_path)
    bids_file = root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set"
    bids_file.touch()
    sidecar = bids_file.with_suffix("").with_name("sub-01_task-rest_eeg.json")
    sidecar.write_text('{"SamplingFrequency": 999.0, "EEGChannelCount": 64}')
    sfreq, nchans = digest.extract_sfreq_nchans_from_modality_sidecar(
        bids_file, root, sampling_frequency=250.0, nchans=None
    )
    assert sfreq == 250.0
    assert nchans == 64


# ─── extract_sfreq_nchans_from_channels_tsv ───────────────────────────────


def test_channels_tsv_reads_row_count(digest: ModuleType, tmp_path: Path) -> None:
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
    """First positive value in the ``sampling_frequency`` column is used."""
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
    assert sfreq is None
    assert nchans is None


def test_channels_tsv_short_circuits_when_both_populated(
    digest: ModuleType, tmp_path: Path
) -> None:
    root = _make_bids_tree(tmp_path)
    bids_file = root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set"
    bids_file.touch()
    # Even if a TSV exists, the function shouldn't look when both values are already set.
    channels_tsv = bids_file.with_suffix("").with_name("sub-01_task-rest_channels.tsv")
    channels_tsv.write_text("name\nFp1\nFp2\n")
    sfreq, nchans = digest.extract_sfreq_nchans_from_channels_tsv(
        bids_file, root, sampling_frequency=128.0, nchans=99
    )
    assert sfreq == 128.0
    assert nchans == 99
