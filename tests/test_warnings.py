import logging
import pytest
from unittest.mock import patch
from eegdash.api import EEGDashDataset

@pytest.fixture
def mock_eegdash_find():
    """Fixture to mock the EEGDash.find method."""
    mock_records = [
        {
            "dataset": "test_ds",
            "subject": "01",
            "task": "realtask",
            "session": "1",
            "run": "1",
            "bidspath": "test_ds/sub-01/ses-1/eeg/sub-01_ses-1_task-realtask_run-1_eeg.edf",
            "bidsdependencies": [],
            "ntimes": 1000,
            "sampling_frequency": 100.0
        },
        {
            "dataset": "test_ds",
            "subject": "02",
            "task": "anothertask",
            "session": "1",
            "run": "1",
            "bidspath": "test_ds/sub-02/ses-1/eeg/sub-02_ses-1_task-anothertask_run-1_eeg.edf",
            "bidsdependencies": [],
            "ntimes": 1000,
            "sampling_frequency": 100.0
        },
    ]
    with patch("eegdash.api.EEGDash.find", return_value=mock_records) as mock_find:
        yield mock_find

def test_warning_for_nonexistent_task(mock_eegdash_find, caplog):
    """Test that a warning is logged for a nonexistent task."""
    with caplog.at_level(logging.WARNING):
        _ = EEGDashDataset(
            cache_dir="/tmp/eegdash_test_cache",
            dataset="test_ds",
            task="nonexistenttask"
        )
    assert "The following value(s) for 'task' did not match any records: ['nonexistenttask']" in caplog.text

def test_no_warning_for_existing_task(mock_eegdash_find, caplog):
    """Test that no warning is logged for an existing task."""
    with caplog.at_level(logging.WARNING):
        _ = EEGDashDataset(
            cache_dir="/tmp/eegdash_test_cache",
            dataset="test_ds",
            task="realtask"
        )
    assert "did not match any records" not in caplog.text

def test_warning_for_mixed_tasks(mock_eegdash_find, caplog):
    """Test that a warning is logged for a mix of existing and nonexistent tasks."""
    with caplog.at_level(logging.WARNING):
        _ = EEGDashDataset(
            cache_dir="/tmp/eegdash_test_cache",
            dataset="test_ds",
            task=["realtask", "nonexistenttask"]
        )
    assert "The following value(s) for 'task' did not match any records: ['nonexistenttask']" in caplog.text

def test_no_records_warning(caplog):
    """Test that a warning is logged when the query returns no records."""
    with patch("eegdash.api.EEGDash.find", return_value=[]):
        with caplog.at_level(logging.WARNING):
            # BaseConcatDataset raises AssertionError if given an empty list
            with pytest.raises(AssertionError, match="datasets should not be an empty iterable"):
                _ = EEGDashDataset(
                    cache_dir="/tmp/eegdash_test_cache",
                    dataset="test_ds",
                    subject="nonexistentsubject"
                )
        assert "No records found for the given query" in caplog.text