from eegdash.records import (
    create_dataset,
    create_record,
    validate_dataset,
    validate_record,
)


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


def test_timestamps_auto_generated():
    """Test digested_at is automatically set."""
    record = create_record(
        dataset="ds000001",
        storage_base="s3://openneuro.org/ds000001",
        bids_relpath="sub-01/eeg/sub-01_task-rest_eeg.vhdr",
        subject="01",
    )
    assert "digested_at" in record
    assert record["digested_at"].endswith("Z")


def test_digested_at_custom_value():
    """Test digested_at with explicit value."""
    record = create_record(
        dataset="ds000001",
        storage_base="s3://openneuro.org/ds000001",
        bids_relpath="sub-01/eeg/sub-01_task-rest_eeg.vhdr",
        digested_at="2024-12-24T10:00:00Z",
    )
    assert record["digested_at"] == "2024-12-24T10:00:00Z"


def test_record_slim_no_clinical_paradigm():
    """Test Record doesn't have clinical/paradigm - those are Dataset-level."""
    record = create_record(
        dataset="ds000001",
        storage_base="s3://openneuro.org/ds000001",
        bids_relpath="sub-01/eeg/sub-01_task-rest_eeg.vhdr",
    )
    assert "clinical" not in record
    assert "paradigm" not in record
    assert "timestamps" not in record  # Flat digested_at now
    assert "digested_at" in record


def test_dataset_clinical_info():
    """Test clinical classification at Dataset level."""
    dataset = create_dataset(
        dataset_id="ds000001",
        is_clinical=True,
        clinical_purpose="epilepsy",
    )
    assert dataset["clinical"]["is_clinical"] is True
    assert dataset["clinical"]["purpose"] == "epilepsy"


def test_dataset_paradigm_info():
    """Test paradigm classification at Dataset level."""
    dataset = create_dataset(
        dataset_id="ds000001",
        paradigm_modality="visual",
        cognitive_domain="attention",
        is_10_20_system=True,
    )
    assert dataset["paradigm"]["modality"] == "visual"
    assert dataset["paradigm"]["cognitive_domain"] == "attention"
    assert dataset["paradigm"]["is_10_20_system"] is True


# =============================================================================
# Dataset Tests
# =============================================================================


def test_create_dataset_minimal():
    """Test creating a dataset with minimal fields."""
    dataset = create_dataset(dataset_id="ds001785")

    assert dataset["dataset_id"] == "ds001785"
    assert dataset["name"] == "ds001785"  # Defaults to dataset_id
    assert dataset["source"] == "openneuro"
    assert dataset["recording_modality"] == ["eeg"]
    assert "digested_at" in dataset["timestamps"]


def test_create_dataset_full():
    """Test creating a dataset with all fields from OpenNeuro."""
    dataset = create_dataset(
        dataset_id="ds001785",
        name="Evidence accumulation relates to perceptual consciousness",
        source="openneuro",
        recording_modality="eeg",
        modalities=["eeg"],
        bids_version="1.1.1",
        license="CC0",
        authors=["Michael Pereira", "Nathan Faivre"],
        funding=["Bertarelli Foundation", "Swiss National Science Foundation"],
        dataset_doi="10.18112/openneuro.ds001785.v1.1.1",
        tasks=["tactile detection"],
        sessions=["01"],
        total_files=242,
        size_bytes=26701806128,
        study_domain="Perceptual consciousness",
        subjects_count=18,
        ages=[30, 26, 22, 29, 28, 26],
        species="Human",
        dataset_modified_at="2021-04-28T15:27:09.000Z",
    )

    assert dataset["dataset_id"] == "ds001785"
    assert (
        dataset["name"] == "Evidence accumulation relates to perceptual consciousness"
    )
    assert dataset["license"] == "CC0"
    assert len(dataset["authors"]) == 2
    assert dataset["tasks"] == ["tactile detection"]
    assert dataset["total_files"] == 242
    assert dataset["demographics"]["subjects_count"] == 18
    assert dataset["demographics"]["age_min"] == 22
    assert dataset["demographics"]["age_max"] == 30
    assert dataset["demographics"]["species"] == "Human"
    assert dataset["timestamps"]["dataset_modified_at"] == "2021-04-28T15:27:09.000Z"


def test_create_dataset_ages_filters_none():
    """Test that None values in ages are filtered out."""
    dataset = create_dataset(
        dataset_id="ds000117",
        ages=[31, 25, None, 30, None, 26],
    )
    assert dataset["demographics"]["ages"] == [31, 25, 30, 26]
    assert dataset["demographics"]["age_min"] == 25
    assert dataset["demographics"]["age_max"] == 31


def test_validate_dataset():
    """Test dataset validation."""
    errors = validate_dataset({})
    assert any("dataset_id" in e for e in errors)

    errors = validate_dataset({"dataset_id": "ds001785"})
    assert len(errors) == 0


def test_create_dataset_requires_id():
    """Test that create_dataset raises without dataset_id."""
    import pytest

    with pytest.raises(ValueError, match="dataset_id is required"):
        create_dataset(dataset_id="")
