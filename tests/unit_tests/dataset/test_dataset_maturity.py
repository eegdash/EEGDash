from unittest.mock import patch

import pytest

from eegdash.dataset.bids_dataset import EEGBIDSDataset
from eegdash.dataset.dataset import EEGDashDataset


@pytest.fixture
def mock_record():
    return {
        "dataset": "dsTest",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
        "bidspath": "dsTest/sub-01/eeg/sub-01_task-rest_eeg.set",
        "storage": {"backend": "local", "base": "/tmp/dsTest"},
        "entities_mne": {"subject": "01", "task": "rest"},
    }


def test_eegdashdataset_init_errors(tmp_path):
    # Missing dataset argument
    with pytest.raises(ValueError, match="You must provide a 'dataset' argument"):
        EEGDashDataset(cache_dir=tmp_path, records=[])

    # Records missing dataset key
    records = [{"bids_relpath": "foo", "bidspath": "foo"}]
    with pytest.raises(ValueError, match="You must provide a 'dataset' argument"):
        EEGDashDataset(cache_dir=tmp_path, records=records)

    # No records, no query, no data_dir
    # Set download=True to bypass offline check. This will trigger "No datasets found" because query matches nothing in mock.
    with pytest.raises(ValueError, match="No datasets found matching the query"):
        EEGDashDataset(cache_dir=tmp_path, dataset="dsTest", download=True)

    # Offline mode but dir missing
    with pytest.raises(ValueError, match="Offline mode is enabled, but local data_dir"):
        EEGDashDataset(cache_dir=tmp_path, dataset="dsMissing", download=False)


def test_competition_warning(tmp_path, capsys):
    # Mock map to trigger warning
    with patch(
        "eegdash.dataset.dataset.RELEASE_TO_OPENNEURO_DATASET_MAP", {"R1": "dsComp"}
    ):
        records = [
            {
                "dataset": "dsComp",
                "bids_relpath": "f.set",
                "bidspath": "dsComp/f.set",
                "storage": {"backend": "local", "base": str(tmp_path)},
                "entities_mne": {"subject": "01", "task": "rest"},
            }
        ]
        (tmp_path / "dsComp").mkdir()

        # Should print warning to console
        EEGDashDataset(cache_dir=tmp_path, records=records, download=False)
        # We can't easily capture Rich console output via capsys standard,
        # but we can verify code execution path didn't crash.
        # Alternatively check logger if exception occurred (which logic does catch)


def test_normalize_records_dedupe(tmp_path):
    # Add entities_mne to satisfy EEGDashRaw and BIDSPath validation
    # Add ntimes/sfreq to bypass _ensure_raw() -> load() in __len__
    entities = {"subject": "01", "task": "rest"}
    common = {
        "storage": {"backend": "local", "base": str(tmp_path)},
        "entities_mne": entities,
        "ntimes": 100,
        "sampling_frequency": 100,
    }

    r1 = {"dataset": "ds1", "bids_relpath": "file1", "bidspath": "ds1/file1", **common}
    r2 = {"dataset": "ds1", "bids_relpath": "file1", "bidspath": "ds1/file1", **common}
    r3 = {"dataset": "ds1", "bids_relpath": "file2", "bidspath": "ds1/file2", **common}

    ds = EEGDashDataset(
        cache_dir=tmp_path,
        dataset="ds1",
        records=[r1, r2, r3],
        download=False,
        _dedupe_records=True,
    )
    assert len(ds.datasets) == 2


def test_download_all_logic(tmp_path, mock_record):
    # Setup dataset with remote URI
    mock_record["storage"] = {
        "backend": "https",
        "base": "https://example.com",
        "raw_key": "data.set",
    }
    ds = EEGDashDataset(cache_dir=tmp_path, dataset="dsTest", records=[mock_record])

    # Mock downloader
    with patch("eegdash.downloader.download_files") as mock_dl:
        with patch("eegdash.downloader.download_s3_file") as mock_s3:
            ds.download_all()
            # Should be called because file doesn't exist
            assert mock_s3.called or mock_dl.called

    # Test download=False
    ds.download = False
    with patch("eegdash.downloader.download_files") as mock_dl:
        ds.download_all()
        assert not mock_dl.called


def test_bids_dataset_init_validation(tmp_path):
    # Data dir must exist
    with pytest.raises(ValueError, match="data_dir must be specified and must exist"):
        EEGBIDSDataset(data_dir=tmp_path / "nonexistent")

    # Dir name must match dataset
    dummy_dir = tmp_path / "dsWrongName"
    dummy_dir.mkdir()
    with pytest.raises(
        AssertionError, match="BIDS directory 'dsWrongName' does not correspond"
    ):
        EEGBIDSDataset(data_dir=dummy_dir, dataset="dsCorrect")


def test_bids_filename_parsing(tmp_path):
    # Setup dummy structure
    ds_dir = tmp_path / "dsParse"
    ds_dir.mkdir()
    (ds_dir / "sub-01" / "eeg").mkdir(parents=True)

    # Tricky filename: task-foo_run-1
    # Regular parsing
    f1 = ds_dir / "sub-01/eeg/sub-01_task-rest_run-1_eeg.set"
    f1.touch()

    # Weird task name edge case from code comment: task-ECONrun-1
    f2 = ds_dir / "sub-01/eeg/sub-01_task-ECONrun-1_eeg.set"
    f2.touch()

    ds = EEGBIDSDataset(data_dir=ds_dir, dataset="dsParse")

    # Verify extraction
    bp1 = ds._get_bids_path_from_file(str(f1))
    assert bp1.task == "rest"
    assert bp1.run == "1"

    bp2 = ds._get_bids_path_from_file(str(f2))
    assert bp2.task == "ECON"  # Should extract 'ECON' despite missing run separator


def test_bids_path_outside_root(tmp_path):
    ds_dir = tmp_path / "dsRoot"
    ds_dir.mkdir()
    # Satisfy init checks (needs at least one file)
    (ds_dir / "sub-01" / "eeg").mkdir(parents=True)
    (ds_dir / "sub-01/eeg/f.set").touch()

    ds = EEGBIDSDataset(data_dir=ds_dir, dataset="dsRoot")

    # File outside
    outside = tmp_path / "other.set"
    res = ds.get_relative_bidspath(outside)
    assert res == "dsRoot/other.set"  # Fallback behavior


def test_channel_labels_fallback(tmp_path):
    ds_dir = tmp_path / "dsChans"
    ds_dir.mkdir()
    (ds_dir / "sub-01" / "eeg").mkdir(parents=True)

    data_file = ds_dir / "sub-01/eeg/sub-01_task-rest_eeg.set"
    data_file.touch()

    # Case 1: No channels.tsv
    ds = EEGBIDSDataset(data_dir=ds_dir, dataset="dsChans")
    with pytest.raises(FileNotFoundError, match="No channels.tsv found"):
        ds.channel_labels(str(data_file))

    # Case 2: Specific *_channels.tsv
    # Create sub-01_task-rest_channels.tsv
    (ds_dir / "sub-01/eeg/sub-01_task-rest_channels.tsv").write_text(
        "name\ttype\nCz\tEEG\n"
    )

    labels = ds.channel_labels(str(data_file))
    assert labels == ["Cz"]

    types = ds.channel_types(str(data_file))
    assert types == ["EEG"]
