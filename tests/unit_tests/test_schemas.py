from eegdash.schemas import (
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


def test_manifest_file_model_path_or_name_priority():
    """Test ManifestFileModel.path_or_name returns path first."""
    from eegdash.schemas import ManifestFileModel

    # Line 74: path_or_name returns path
    model = ManifestFileModel(path="/path/to/file", name="name.txt")
    assert model.path_or_name() == "/path/to/file"

    # With only name
    model2 = ManifestFileModel(name="name.txt")
    assert model2.path_or_name() == "name.txt"

    # With neither
    model3 = ManifestFileModel()
    assert model3.path_or_name() == ""


def test_create_dataset_with_external_links():
    """Test create_dataset with external links."""
    from eegdash.schemas import create_dataset

    # Lines 439, 448: external_links added
    dataset = create_dataset(
        dataset_id="ds001",
        source_url="https://example.com",
        github_url="https://github.com/example",
    )
    assert "external_links" in dataset
    assert dataset["external_links"]["source_url"] == "https://example.com"


def test_create_dataset_with_repository_stats():
    """Test create_dataset with repository stats."""
    from eegdash.schemas import create_dataset

    # Lines 448: repository_stats added
    dataset = create_dataset(dataset_id="ds001", stars=100, forks=20, watchers=50)
    assert "repository_stats" in dataset
    assert dataset["repository_stats"]["stars"] == 100


def test_sanitize_run_for_mne_edge_cases():
    """Test _sanitize_run_for_mne with various inputs."""
    from eegdash.schemas import _sanitize_run_for_mne

    # Line 513: None input
    assert _sanitize_run_for_mne(None) is None

    # Line 515: integer input
    assert _sanitize_run_for_mne(1) == "1"

    # String numeric
    assert _sanitize_run_for_mne("5") == "5"

    # String non-numeric
    assert _sanitize_run_for_mne("run-01") is None

    # Empty string
    assert _sanitize_run_for_mne("") is None
    assert _sanitize_run_for_mne("  ") is None


def test_create_record_validation_errors():
    """Test create_record raises for missing required fields."""
    import pytest

    from eegdash.schemas import create_record

    # Line 588: missing required fields
    with pytest.raises(ValueError, match="dataset is required"):
        create_record(dataset="", storage_base="s3://", bids_relpath="file.vhdr")

    with pytest.raises(ValueError, match="storage_base is required"):
        create_record(dataset="ds001", storage_base="", bids_relpath="file.vhdr")

    with pytest.raises(ValueError, match="bids_relpath is required"):
        create_record(dataset="ds001", storage_base="s3://", bids_relpath="")


def test_sanitize_run_float():
    """Test _sanitize_run_for_mne with float input."""
    from eegdash.schemas import _sanitize_run_for_mne

    # Line 515: return None for non-int, non-str types
    assert _sanitize_run_for_mne(3.14) is None
    assert _sanitize_run_for_mne(["1"]) is None
    assert _sanitize_run_for_mne({"run": 1}) is None


def test_create_record_minimal():
    """Test create_record with minimal required fields."""
    from eegdash.schemas import create_record

    record = create_record(
        dataset="ds001234",
        storage_base="s3://bucket",
        bids_relpath="sub-01/eeg/sub-01_eeg.set",
    )

    assert record["dataset"] == "ds001234"
    assert "storage" in record
    assert record["storage"]["base"] == "s3://bucket"


def test_create_record_with_all_fields():
    """Test create_record with all optional fields."""
    from eegdash.schemas import create_record

    record = create_record(
        dataset="ds001234",
        storage_base="s3://bucket",
        bids_relpath="sub-01/eeg/sub-01_task-rest_eeg.set",
        subject="01",
        session="01",
        task="rest",
        run="01",
        dep_keys=["key1", "key2"],
        datatype="eeg",
        suffix="eeg",
        storage_backend="s3",
    )

    # Check entities are in the record
    assert record["entities"]["subject"] == "01"
    assert record["entities"]["session"] == "01"
    assert record["entities"]["task"] == "rest"


def test_validate_record_missing_fields():
    """Test validate_record with missing fields."""
    from eegdash.schemas import validate_record

    # Minimal invalid record
    record = {}
    errors = validate_record(record)
    assert "missing: dataset" in errors
    assert "missing: bids_relpath" in errors


def test_validate_record_missing_storage():
    """Test validate_record with missing storage."""
    from eegdash.schemas import validate_record

    record = {"dataset": "ds001234", "bids_relpath": "x", "bidspath": "y"}
    errors = validate_record(record)
    assert "missing: storage" in errors


def test_sanitize_run_for_mne():
    """Test _sanitize_run_for_mne function."""
    from eegdash.schemas import _sanitize_run_for_mne

    assert _sanitize_run_for_mne(None) is None
    assert _sanitize_run_for_mne(1) == "1"
    assert _sanitize_run_for_mne("2") == "2"
    assert _sanitize_run_for_mne("abc") is None  # non-numeric
    assert _sanitize_run_for_mne("") is None


def test_sanitize_run():
    from eegdash import schemas

    assert schemas._sanitize_run_for_mne(1) == "1"
    assert schemas._sanitize_run_for_mne("01") == "01"
    assert schemas._sanitize_run_for_mne("run") is None  # Not digit
    assert schemas._sanitize_run_for_mne(None) is None


def test_validate_record():
    from eegdash import schemas

    assert "missing: dataset" in schemas.validate_record({})
    assert "missing: storage" in schemas.validate_record(
        {"dataset": "d", "bids_relpath": "p", "bidspath": "p"}
    )
    # Need non-empty storage to bypass "missing: storage" check
    assert "missing: storage.base" in schemas.validate_record(
        {
            "dataset": "d",
            "bids_relpath": "p",
            "bidspath": "p",
            "storage": {"backend": "local"},
        }
    )


def test_create_record_validation():
    import pytest

    from eegdash import schemas

    with pytest.raises(ValueError):
        schemas.create_record(dataset="", storage_base="b", bids_relpath="p")

    rec = schemas.create_record(dataset="d", storage_base="b", bids_relpath="p", run=1)
    assert rec["entities_mne"]["run"] == "1"

    pass

    pass


def test_sanitize_run_edge_cases():
    # schemas.py 511-515
    from eegdash import schemas

    assert schemas._sanitize_run_for_mne("  ") is None
    assert schemas._sanitize_run_for_mne("01") == "01"
    assert schemas._sanitize_run_for_mne("run1") is None


def test_create_record_errors():
    # schemas.py 584-588
    import pytest

    from eegdash import schemas

    with pytest.raises(ValueError, match="dataset is required"):
        schemas.create_record(dataset="", storage_base="s3://b", bids_relpath="f.set")
    with pytest.raises(ValueError, match="storage_base is required"):
        schemas.create_record(dataset="ds", storage_base="", bids_relpath="f.set")
    with pytest.raises(ValueError, match="bids_relpath is required"):
        schemas.create_record(dataset="ds", storage_base="s3://b", bids_relpath="")


def test_manifest_model_coverage():
    # schemas.py 74
    from eegdash import schemas

    m = schemas.ManifestFileModel(path=" p ")
    assert m.path_or_name() == "p"
    m2 = schemas.ManifestFileModel(name=" n ")
    assert m2.path_or_name() == "n"
    m3 = schemas.ManifestFileModel()
    assert m3.path_or_name() == ""


def test_schemas_run_digit_coverage():
    # schemas.py 515
    from eegdash import schemas

    assert schemas._sanitize_run_for_mne("A123") is None
    assert schemas._sanitize_run_for_mne("01") == "01"


# =============================================================================
# Schema Contract Tests - Pydantic Models vs TypedDicts
# =============================================================================

import pytest

from eegdash import schemas


def _get_typeddict_fields(td_class: type) -> set[str]:
    """Extract field names from a TypedDict."""
    annotations = set()
    # Walk through MRO to get all annotations including inherited ones
    for cls in td_class.__mro__:
        if hasattr(cls, "__annotations__"):
            annotations.update(cls.__annotations__.keys())
    return annotations


def _get_pydantic_fields(model_class: type) -> set[str]:
    """Extract field names from a Pydantic model."""
    return set(model_class.model_fields.keys())


# =============================================================================
# Auto-discovered schema pairs: (PydanticModel, TypedDict)
# Add new pairs here and they will be automatically tested
# =============================================================================

PYDANTIC_TYPEDDICT_PAIRS = [
    (schemas.StorageModel, schemas.Storage),
    (schemas.EntitiesModel, schemas.Entities),
    # Note: RecordModel and DatasetModel are intentionally lighter than their
    # TypedDict counterparts (they only validate core fields for API input)
]

# These pairs have Pydantic models that are subsets of TypedDicts
# (Pydantic validates input, TypedDict represents full document)
PYDANTIC_SUBSET_PAIRS = [
    (schemas.RecordModel, schemas.Record),
    (schemas.DatasetModel, schemas.Dataset),
]

# All TypedDicts that should have documentation
DOCUMENTED_TYPEDDICTS = [
    schemas.Demographics,
    schemas.Clinical,
    schemas.Tags,
    schemas.ExternalLinks,
    schemas.Storage,
    schemas.Entities,
    schemas.Timestamps,
    schemas.RepositoryStats,
    schemas.Record,
    schemas.Dataset,
]

# Deprecated schemas that should NOT exist
REMOVED_SCHEMAS = ["Paradigm"]


class TestSchemaContract:
    """Tests to ensure Pydantic models and TypedDicts stay in sync.

    These tests use pytest.parametrize to automatically test all schema pairs.
    When adding a new Pydantic/TypedDict pair, just add it to the lists above.
    """

    @pytest.mark.parametrize(
        "pydantic_model,typeddict",
        PYDANTIC_TYPEDDICT_PAIRS,
        ids=lambda x: getattr(x, "__name__", str(x)),
    )
    def test_pydantic_typeddict_fields_match(self, pydantic_model, typeddict):
        """Verify Pydantic model fields match TypedDict fields exactly."""
        pydantic_fields = _get_pydantic_fields(pydantic_model)
        typeddict_fields = _get_typeddict_fields(typeddict)

        # Check TypedDict fields exist in Pydantic model
        missing_in_pydantic = typeddict_fields - pydantic_fields
        assert not missing_in_pydantic, (
            f"{typeddict.__name__} has fields not in {pydantic_model.__name__}: "
            f"{missing_in_pydantic}"
        )

        # Check Pydantic fields exist in TypedDict
        missing_in_typeddict = pydantic_fields - typeddict_fields
        assert not missing_in_typeddict, (
            f"{pydantic_model.__name__} has fields not in {typeddict.__name__}: "
            f"{missing_in_typeddict}"
        )

    @pytest.mark.parametrize(
        "pydantic_model,typeddict",
        PYDANTIC_SUBSET_PAIRS,
        ids=lambda x: getattr(x, "__name__", str(x)),
    )
    def test_pydantic_subset_of_typeddict(self, pydantic_model, typeddict):
        """Verify Pydantic model fields are a subset of TypedDict fields.

        Some Pydantic models (like RecordModel, DatasetModel) are intentionally
        lighter than their TypedDict counterparts - they validate API input
        while TypedDicts represent the full stored document.
        """
        pydantic_fields = _get_pydantic_fields(pydantic_model)
        typeddict_fields = _get_typeddict_fields(typeddict)

        # All Pydantic fields must exist in TypedDict
        missing_in_typeddict = pydantic_fields - typeddict_fields
        assert not missing_in_typeddict, (
            f"{pydantic_model.__name__} has fields not in {typeddict.__name__}: "
            f"{missing_in_typeddict}"
        )

    @pytest.mark.parametrize(
        "typeddict", DOCUMENTED_TYPEDDICTS, ids=lambda x: x.__name__
    )
    def test_typeddict_has_docstring(self, typeddict):
        """Verify TypedDict has a docstring with Attributes section."""
        assert typeddict.__doc__ is not None, f"{typeddict.__name__} missing docstring"
        assert "Attributes" in typeddict.__doc__, (
            f"{typeddict.__name__} docstring should have 'Attributes' section"
        )

    @pytest.mark.parametrize("schema_name", REMOVED_SCHEMAS)
    def test_deprecated_schema_removed(self, schema_name):
        """Verify deprecated schemas have been removed."""
        assert not hasattr(schemas, schema_name), (
            f"{schema_name} should be removed from schemas"
        )
        assert schema_name not in schemas.__all__, (
            f"{schema_name} should not be in __all__"
        )

    def test_create_dataset_no_deprecated_params(self):
        """Verify create_dataset doesn't accept deprecated parameters."""
        import inspect

        sig = inspect.signature(schemas.create_dataset)
        param_names = list(sig.parameters.keys())

        # Check for paradigm-related params (removed)
        paradigm_params = [p for p in param_names if "paradigm" in p.lower()]
        assert not paradigm_params, (
            f"create_dataset still has paradigm params: {paradigm_params}"
        )

    def test_all_exported_typeddicts_documented(self):
        """Verify all TypedDicts in __all__ are in DOCUMENTED_TYPEDDICTS."""
        from typing import _TypedDictMeta

        documented_names = {td.__name__ for td in DOCUMENTED_TYPEDDICTS}

        for name in schemas.__all__:
            obj = getattr(schemas, name, None)
            if obj is not None and isinstance(obj, _TypedDictMeta):
                assert name in documented_names, (
                    f"TypedDict '{name}' is exported but not in DOCUMENTED_TYPEDDICTS. "
                    f"Add it to the list in test_schemas.py"
                )
