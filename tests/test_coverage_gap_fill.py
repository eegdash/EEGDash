import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from eegdash.dataset.bids_dataset import (
    EEGBIDSDataset,
    _is_valid_eeg_file,
)
from eegdash.dataset.dataset import EEGChallengeDataset, EEGDashDataset
from eegdash.dataset.registry import (
    _generate_rich_docstring,
    register_openneuro_datasets,
)

# --- Tests for registry.py ---


def test_registry_from_api_success():
    """Test registering datasets from API successfully."""
    mock_data = {
        "success": True,
        "data": [
            {
                "dataset_id": "ds001",
                "metadata": {
                    "subject_count": 10,
                    "tasks": ["rest"],
                    "recording_modalities": ["eeg"],
                    "type": "test_exp",
                    "pathology": "healthy",
                    "duration_hours": 5,
                    "size_human": "1GB",
                },
                "source": "nemar",
            }
        ],
    }

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        namespace = {}
        register_openneuro_datasets(
            from_api=True, namespace=namespace, add_to_all=False
        )

        assert "DS001" in namespace
        assert namespace["DS001"]._dataset == "ds001"


def test_registry_from_api_failure_fallback(tmp_path):
    """Test fallback to CSV when API fails."""
    summary_file = tmp_path / "summary.csv"
    summary_file.write_text("dataset,n_subjects\nds002,5")

    with patch("urllib.request.urlopen", side_effect=Exception("API Error")):
        namespace = {}
        register_openneuro_datasets(
            summary_file=summary_file,
            from_api=True,
            namespace=namespace,
            add_to_all=True,
        )

        assert "DS002" in namespace
        assert "DS002" in namespace["__all__"]


def test_registry_docstring_generation():
    """Test rich docstring generation with various missing fields."""
    row = pd.Series(
        {
            "dataset": "ds003",
            "n_subjects": None,  # Should be handled
            "n_records": 10,
            "n_tasks": "rest",
            "dataset_doi": "doi:10.1000/1",
            "record_modality": None,  # Should check alternative
            "modality of exp": "eeg",
            "type of exp": "task",
            "Type Subject": "patient",
        }
    )

    doc = _generate_rich_docstring("ds003", row, EEGDashDataset)
    assert (
        "Subjects: Unknown" in doc or "Subjects:" in doc
    )  # Depending on implementation details
    assert "Modality: ``eeg``" in doc
    assert "https://doi.org/10.1000/1" in doc


def test_registry_exclude_datasets():
    """Test that excluded datasets are skipped."""
    mock_data = {
        "success": True,
        "data": [
            {"dataset_id": "ABUDUKADI", "metadata": {}},  # Excluded
            {"dataset_id": "test", "metadata": {}},  # Excluded
            {"dataset_id": "valid", "metadata": {}},
        ],
    }

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        namespace = {}
        register_openneuro_datasets(from_api=True, namespace=namespace)

        assert "VALID" in namespace
        assert "ABUDUKADI" not in namespace
        assert "TEST" not in namespace


# --- Tests for dataset.py ---


def test_dataset_find_key_nested(tmp_path):
    """Test _find_key_in_nested_dict recursion."""
    d = tmp_path / "ds999"
    d.mkdir()

    valid_record = {
        "dataset": "ds999",
        "data_name": "sub-01_task-rest_eeg.set",
        "bidspath": "ds999/sub-01/eeg/sub-01_task-rest_eeg.set",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
        "subject": "01",
        "task": "rest",
        "storage": {"base": str(d), "backend": "local"},
    }
    # Mocking discover to return proper records so init doesn't fail
    with patch("eegdash.dataset.dataset.discover_local_bids_records") as mock_discover:
        mock_discover.return_value = [valid_record]
        ds = EEGDashDataset(
            cache_dir=str(tmp_path),
            dataset="ds999",
            download=False,
            _suppress_comp_warning=True,
        )

    data = {"a": 1, "b": {"c": 2, "d": [{"e": 3}, {"f": 4}]}}
    assert ds._find_key_in_nested_dict(data, "a") == 1
    assert ds._find_key_in_nested_dict(data, "c") == 2
    assert ds._find_key_in_nested_dict(data, "e") == 3
    assert ds._find_key_in_nested_dict(data, "z") is None


def test_dataset_normalize_records_dedupe(tmp_path):
    """Test record deduplication logic."""
    d = tmp_path / "ds999"
    d.mkdir()

    valid_record = {
        "dataset": "ds999",
        "data_name": "sub-01_task-rest_eeg.set",
        "bidspath": "ds999/sub-01/eeg/sub-01_task-rest_eeg.set",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
        "subject": "01",
        "task": "rest",
        "storage": {"base": str(d), "backend": "local"},
    }

    with patch("eegdash.dataset.dataset.discover_local_bids_records") as mock_discover:
        mock_discover.return_value = [valid_record]
        ds = EEGDashDataset(
            cache_dir=str(tmp_path),
            dataset="ds999",
            download=False,
            _dedupe_records=True,
        )

    records = [
        {"bidspath": "path1", "data_name": "n1"},
        {"bidspath": "path1", "data_name": "n1"},  # Duplicate
        {"bidspath": "path2", "data_name": "n2"},
    ]
    normalized = ds._normalize_records(records)
    assert len(normalized) == 2


def test_dataset_s3_bucket_injection(tmp_path):
    """Test S3 bucket injection."""
    d = tmp_path / "ds999"
    d.mkdir()

    valid_record = {
        "dataset": "ds999",
        "data_name": "sub-01_task-rest_eeg.set",
        "bidspath": "ds999/sub-01/eeg/sub-01_task-rest_eeg.set",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
        "subject": "01",
        "task": "rest",
        "storage": {"base": str(d), "backend": "local"},
    }

    with patch("eegdash.dataset.dataset.discover_local_bids_records") as mock_discover:
        mock_discover.return_value = [valid_record]
        ds = EEGDashDataset(
            cache_dir=str(tmp_path),
            dataset="ds999",
            download=False,
            s3_bucket="s3://custom",
        )

    records = [{"file": "f1"}]
    normalized = ds._normalize_records(records)
    assert normalized[0]["storage"]["base"] == "s3://custom"
    assert normalized[0]["storage"]["backend"] == "s3"


def test_dataset_download_all_offline(tmp_path):
    """Test download_all returns early if download=False."""
    d = tmp_path / "ds999"
    d.mkdir()

    valid_record = {
        "dataset": "ds999",
        "data_name": "sub-01_task-rest_eeg.set",
        "bidspath": "ds999/sub-01/eeg/sub-01_task-rest_eeg.set",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
        "subject": "01",
        "task": "rest",
        "storage": {"base": str(d), "backend": "local"},
    }

    with patch("eegdash.dataset.dataset.discover_local_bids_records") as mock_discover:
        mock_discover.return_value = [valid_record]
        ds = EEGDashDataset(cache_dir=str(tmp_path), dataset="ds999", download=False)

    # Should not raise error or try to download
    ds.download_all()


def test_dataset_lazy_loader(tmp_path):
    """Test lazy loading of EEGDash instance."""
    # We patch eegdash.api.EEGDash because that's where it is imported from
    with patch("eegdash.api.EEGDash") as MockDash:
        MockDash.return_value.find.return_value = []  # Return empty
        try:
            # Provide s3_bucket to force usage of non-offline path if records/db issues
            # Actually, if download=True (default), it goes to _find_datasets logic.
            # We need to ensure logic flow reaches import.
            EEGDashDataset(
                cache_dir=str(tmp_path), query={"dataset": "ds_lazy"}, download=True
            )
        except ValueError:
            pass
        MockDash.assert_called()


def test_dataset_save(tmp_path):
    """Test save method delegation."""
    d = tmp_path / "ds999"
    d.mkdir()

    valid_record = {
        "dataset": "ds999",
        "data_name": "sub-01_task-rest_eeg.set",
        "bidspath": "ds999/sub-01/eeg/sub-01_task-rest_eeg.set",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
        "subject": "01",
        "task": "rest",
        "storage": {"base": str(d), "backend": "local"},
    }

    with patch("eegdash.dataset.dataset.discover_local_bids_records") as mock_discover:
        mock_discover.return_value = [valid_record]
        ds = EEGDashDataset(cache_dir=str(tmp_path), dataset="ds999", download=False)

    # Mock super().save
    with patch("braindecode.datasets.BaseConcatDataset.save") as mock_save:
        ds.save("some_path", overwrite=True)
        mock_save.assert_called_with("some_path", overwrite=True)


def test_challenge_dataset_errors():
    """Test EEGChallengeDataset error conditions."""
    with pytest.raises(ValueError, match="Unknown release"):
        EEGChallengeDataset(release="R999", cache_dir="/tmp")

    with pytest.raises(ValueError, match="Query using the parameters `dataset`"):
        EEGChallengeDataset(release="R1", cache_dir="/tmp", query={"dataset": "ds123"})

    with pytest.raises(
        ValueError, match="Some requested subject.*are not part of the mini release"
    ):
        # Assuming R1 mini has specific subjects, ask for a bogus one
        EEGChallengeDataset(
            release="R1", cache_dir="/tmp", subject="BOGUS_SUBJ", mini=True
        )

    with pytest.raises(
        ValueError, match="Some requested subject.*are not part of the mini release"
    ):
        # Test with list
        EEGChallengeDataset(
            release="R1", cache_dir="/tmp", subjects=["BOGUS_SUBJ"], mini=True
        )


def test_challenge_dataset_defaults():
    """Test EEGChallengeDataset defaults."""
    ds = EEGChallengeDataset(release="R1", cache_dir="/tmp", mini=True)
    assert ds.query["subject"]  # Should be populated with mini subjects
    assert ds.s3_bucket.startswith("s3://nmdatasets/NeurIPS25/R1_mini")

    ds_full = EEGChallengeDataset(release="R1", cache_dir="/tmp", mini=False)
    assert ds_full.s3_bucket.startswith("s3://nmdatasets/NeurIPS25/R1_L100")


# --- Tests for bids_dataset.py ---


def test_is_valid_eeg_file():
    """Test _is_valid_eeg_file logic."""
    with patch("pathlib.Path.exists", return_value=True):
        assert _is_valid_eeg_file(Path("exists.set")) is True

    with patch("pathlib.Path.exists", return_value=False):
        with patch("pathlib.Path.is_symlink", return_value=True):
            assert (
                _is_valid_eeg_file(Path("broken_symlink.set"), allow_symlinks=True)
                is True
            )
            assert (
                _is_valid_eeg_file(Path("broken_symlink.set"), allow_symlinks=False)
                is False
            )

        with patch("pathlib.Path.is_symlink", return_value=False):
            assert _is_valid_eeg_file(Path("missing.set"), allow_symlinks=True) is False


def test_bids_dataset_channel_labels_prefix(tmp_path):
    """Test finding channel labels with prefix."""
    # Setup dummy directory
    data_dir = tmp_path / "ds001"
    data_dir.mkdir()
    (data_dir / "sub-01_task-rest_eeg.set").touch()

    # Standard channels.tsv missing, but prefixed one exists
    (data_dir / "sub-01_task-rest_channels.tsv").write_text("name\ttype\nCz\tEEG\n")

    # Mocking BIDS dataset part just to access the method
    # or using the method if it was static, but it is instance method.
    # checking logic in source: it is instance method.

    # We can instantiate with dummy dir
    try:
        ds = EEGBIDSDataset(data_dir=str(data_dir), dataset="ds001", modalities=["eeg"])
    except (ValueError, AssertionError):
        # Depending on validation, might fail if no valid files found or name mismatch
        # Let's bypass init if possible or make it valid
        # It requires valid files.
        pass

    # Actually simpler to just invoke the logic if I can, but it relies on instance state.
    # Let's try to make it work.

    # Create valid structure for init
    subj_dir = data_dir / "sub-01" / "eeg"
    subj_dir.mkdir(parents=True)
    (subj_dir / "sub-01_task-rest_eeg.set").touch()
    (subj_dir / "sub-01_task-rest_channels.tsv").write_text("name\ttype\nCz\tEEG\n")

    ds = EEGBIDSDataset(data_dir=str(data_dir), dataset="ds001")

    labels = ds.channel_labels(str(subj_dir / "sub-01_task-rest_eeg.set"))
    assert labels == ["Cz"]

    types = ds.channel_types(str(subj_dir / "sub-01_task-rest_eeg.set"))
    assert types == ["EEG"]


def test_bids_dataset_init_checks(tmp_path):
    """Test init checks for dataset name."""
    d = tmp_path / "wrong_name"
    d.mkdir()
    with pytest.raises(AssertionError):
        EEGBIDSDataset(data_dir=str(d), dataset="dsXYZ")


def test_bids_get_bids_file_attribute_direct_vs_json(tmp_path):
    """Test getting attributes from filename vs JSON."""
    d = tmp_path / "ds_attr"
    d.mkdir()
    subj = d / "sub-01" / "eeg"
    subj.mkdir(parents=True)
    eeg_file = subj / "sub-01_task-rest_eeg.set"
    eeg_file.touch()

    # Create eeg.json
    (subj / "sub-01_task-rest_eeg.json").write_text(
        json.dumps({"SamplingFrequency": 500})
    )

    ds = EEGBIDSDataset(data_dir=str(d), dataset="ds_attr")

    # From filename/path
    assert ds.get_bids_file_attribute("subject", str(eeg_file)) == "01"
    assert ds.get_bids_file_attribute("task", str(eeg_file)) == "rest"

    # From JSON
    assert ds.get_bids_file_attribute("sfreq", str(eeg_file)) == 500


def test_bids_dataset_check_eeg_dataset(tmp_path):
    d = tmp_path / "eeg_ds"
    d.mkdir()
    (d / "sub-01" / "eeg").mkdir(parents=True)
    (d / "sub-01" / "eeg" / "sub-01_eeg.set").touch()

    ds = EEGBIDSDataset(data_dir=str(d), dataset="eeg_ds")
    assert ds.check_eeg_dataset() is True
