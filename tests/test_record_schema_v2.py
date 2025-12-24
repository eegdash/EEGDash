from eegdash.records import create_record, validate_record


def test_create_openneuro_record():
    """Test creating a record for OpenNeuro."""
    record = create_record(
        dataset="ds000001",
        storage_base="s3://openneuro.org/ds000001",
        bids_relpath="sub-01/eeg/sub-01_task-test_run-5F_eeg.vhdr",
        subject="01",
        task="test",
        run="5F",
        dep_keys=["participants.tsv", "sub-01/eeg/sub-01_task-test_run-5F_events.tsv"],
    )

    assert record["dataset"] == "ds000001"
    assert record["bids_relpath"] == "sub-01/eeg/sub-01_task-test_run-5F_eeg.vhdr"

    # Original run is preserved, MNE-safe run is None (non-numeric)
    assert record["entities"]["run"] == "5F"
    assert record["entities_mne"]["run"] is None

    assert record["storage"]["base"] == "s3://openneuro.org/ds000001"
    assert record["storage"]["raw_key"] == record["bids_relpath"]
    assert len(record["storage"]["dep_keys"]) == 2


def test_create_challenge_record():
    """Test creating a challenge record."""
    record = create_record(
        dataset="ds005509",
        storage_base="s3://nmdatasets/NeurIPS25/R5_mini_L100_bdf",
        bids_relpath="sub-NDARAH793FBF/eeg/sub-NDARAH793FBF_task-DespicableMe_eeg.bdf",
        subject="NDARAH793FBF",
        task="DespicableMe",
        dep_keys=["dataset_description.json"],
    )

    assert record["dataset"] == "ds005509"
    assert record["bids_relpath"].endswith("_eeg.bdf")
    assert record["storage"]["base"] == "s3://nmdatasets/NeurIPS25/R5_mini_L100_bdf"


def test_validate_record_catches_missing_fields():
    """Test validation catches missing required fields."""
    errors = validate_record({})
    assert len(errors) > 0
    assert any("dataset" in e for e in errors)

    # Missing storage.base
    partial = {
        "dataset": "ds000001",
        "bids_relpath": "sub-01/eeg/sub-01_eeg.vhdr",
        "storage": {"backend": "s3", "raw_key": "test", "dep_keys": []},
    }
    errors = validate_record(partial)
    assert any("storage.base" in e for e in errors)


def test_create_record_requires_storage_base():
    """Test that create_record raises without storage_base."""
    import pytest

    with pytest.raises(ValueError, match="storage_base is required"):
        create_record(
            dataset="ds000001",
            storage_base="",
            bids_relpath="sub-01/eeg/sub-01_eeg.vhdr",
        )


def test_numeric_run_preserved():
    """Test numeric run is preserved in entities_mne."""
    record = create_record(
        dataset="ds000001",
        storage_base="s3://openneuro.org/ds000001",
        bids_relpath="sub-01/eeg/sub-01_task-rest_run-01_eeg.vhdr",
        subject="01",
        task="rest",
        run="01",
    )
    assert record["entities"]["run"] == "01"
    assert record["entities_mne"]["run"] == "01"
