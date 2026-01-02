from eegdash.local_bids import _normalize_modalities, discover_local_bids_records


def test_normalize_modalities():
    assert _normalize_modalities(None) == ["eeg"]
    assert _normalize_modalities("eeg") == ["eeg"]
    assert _normalize_modalities(["eeg", "fmri"]) == ["eeg", "fmri"]
    assert _normalize_modalities("fnirs") == ["nirs"]  # Check alias


def test_discover_local_bids_records(tmp_path):
    # Create dummy structure
    # ds1/
    #   sub-01/
    #     eeg/
    #       sub-01_task-rest_eeg.set
    #       sub-01_task-rest_eeg.json
    ds_root = tmp_path / "ds1"
    sub_dir = ds_root / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)

    k = sub_dir / "sub-01_task-rest_eeg.set"
    k.touch()
    (sub_dir / "sub-01_task-rest_eeg.json").touch()

    # Filters
    filters = {"dataset": "ds1", "subject": "01", "modality": "eeg"}

    records = discover_local_bids_records(ds_root, filters)

    assert len(records) == 1
    rec = records[0]
    # Entities are nested
    assert rec["entities"]["subject"] == "01"
    assert rec["entities"]["task"] == "rest"
    assert rec["suffix"] == "eeg"
    assert rec["dataset"] == "ds1"
    # storage backend should be local
    assert rec["storage"]["backend"] == "local"


def test_discover_local_bids_records_filtering(tmp_path):
    # Setup: sub-01 (rest), sub-02 (task)
    ds_root = tmp_path / "ds2"
    (ds_root / "sub-01" / "eeg").mkdir(parents=True)
    (ds_root / "sub-02" / "eeg").mkdir(parents=True)

    (ds_root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.vhdr").touch()
    (ds_root / "sub-02" / "eeg" / "sub-02_task-task_eeg.vhdr").touch()

    # Filter for sub-01 only
    records = discover_local_bids_records(ds_root, {"dataset": "ds2", "subject": "01"})
    assert len(records) == 1
    assert records[0]["entities"]["subject"] == "01"

    # Filter for task=task
    records = discover_local_bids_records(ds_root, {"dataset": "ds2", "task": "task"})
    assert len(records) == 1
    assert records[0]["entities"]["subject"] == "02"
