# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Tests for the exceptions module."""

from unittest.mock import MagicMock, patch

import pytest

from eegdash.dataset.exceptions import DataIntegrityError, EEGDashError


@pytest.fixture
def sample_record():
    """Create a sample record with integrity issues."""
    return {
        "dataset": "ds001234",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
        "_data_integrity_issues": [
            "Missing companion file: sub-01_task-rest_eeg.fdt",
            "Invalid channel count in channels.tsv",
        ],
        "_dataset_authors": ["John Doe", "Jane Smith"],
        "_dataset_contact": ["john.doe@example.com", "jane.smith@example.com"],
        "_source_url": "https://openneuro.org/datasets/ds001234",
    }


@pytest.fixture
def minimal_record():
    """Create a minimal record."""
    return {
        "dataset": "ds_minimal",
        "bids_relpath": "sub-01/eeg.set",
    }


def test_eegdash_error_is_exception():
    """Test that EEGDashError is a subclass of Exception."""
    assert issubclass(EEGDashError, Exception)


def test_eegdash_error_can_be_raised():
    """Test that EEGDashError can be raised and caught."""
    with pytest.raises(EEGDashError):
        raise EEGDashError("Test error message")


def test_eegdash_error_message():
    """Test that EEGDashError preserves message."""
    error = EEGDashError("Test error message")
    assert str(error) == "Test error message"


def test_data_integrity_error_is_eegdash_error():
    """Test that DataIntegrityError is a subclass of EEGDashError."""
    assert issubclass(DataIntegrityError, EEGDashError)


def test_data_integrity_error_init_defaults():
    """Test DataIntegrityError initialization with defaults."""
    error = DataIntegrityError("Test message")
    assert str(error) == "Test message"
    assert error.record == {}
    assert error.issues == []
    assert error.authors == []
    assert error.contact_info is None
    assert error.source_url is None


def test_data_integrity_error_init_with_all_params(sample_record):
    """Test DataIntegrityError initialization with all parameters."""
    error = DataIntegrityError(
        message="Cannot load file",
        record=sample_record,
        issues=["Issue 1", "Issue 2"],
        authors=["Author 1"],
        contact_info=["email@example.com"],
        source_url="https://example.com",
    )
    assert str(error) == "Cannot load file"
    assert error.record == sample_record
    assert error.issues == ["Issue 1", "Issue 2"]
    assert error.authors == ["Author 1"]
    assert error.contact_info == ["email@example.com"]
    assert error.source_url == "https://example.com"


def test_build_rich_output(sample_record):
    """Test _build_rich_output creates a Panel."""
    from rich.panel import Panel

    error = DataIntegrityError(
        message="Test",
        record=sample_record,
        issues=sample_record["_data_integrity_issues"],
        authors=sample_record["_dataset_authors"],
        contact_info=sample_record["_dataset_contact"],
        source_url=sample_record["_source_url"],
    )
    panel = error._build_rich_output()
    assert isinstance(panel, Panel)


def test_build_rich_output_minimal_record(minimal_record):
    """Test _build_rich_output with minimal record (no issues, authors, etc.)."""
    from rich.panel import Panel

    error = DataIntegrityError(
        message="Test",
        record=minimal_record,
    )
    panel = error._build_rich_output()
    assert isinstance(panel, Panel)


def test_build_rich_output_with_string_contact(sample_record):
    """Test _build_rich_output when contact_info is a string."""
    from rich.panel import Panel

    error = DataIntegrityError(
        message="Test",
        record=sample_record,
        issues=["Issue"],
        contact_info="single@email.com",
    )
    panel = error._build_rich_output()
    assert isinstance(panel, Panel)


def test_print_rich_default_console(sample_record):
    """Test print_rich with default console."""
    error = DataIntegrityError(
        message="Test",
        record=sample_record,
        issues=["Issue 1"],
    )
    with patch("eegdash.dataset.exceptions.Console") as MockConsole:
        mock_console = MagicMock()
        MockConsole.return_value = mock_console
        error.print_rich()
        MockConsole.assert_called_once_with(stderr=True)
        mock_console.print.assert_called_once()


def test_print_rich_custom_console(sample_record):
    """Test print_rich with custom console."""
    from rich.console import Console

    error = DataIntegrityError(
        message="Test",
        record=sample_record,
        issues=["Issue 1"],
    )
    mock_console = MagicMock(spec=Console)
    error.print_rich(console=mock_console)
    mock_console.print.assert_called_once()


def test_log_error(sample_record):
    """Test log_error method."""
    error = DataIntegrityError(
        message="Test",
        record=sample_record,
        issues=sample_record["_data_integrity_issues"],
        authors=sample_record["_dataset_authors"],
        contact_info=sample_record["_dataset_contact"],
        source_url=sample_record["_source_url"],
    )
    with patch("eegdash.dataset.exceptions.logger") as mock_logger:
        error.log_error()
        assert mock_logger.error.call_count == 3  # Main + 2 issues
        assert mock_logger.info.call_count == 3  # authors, contact, source_url


def test_log_error_minimal(minimal_record):
    """Test log_error with minimal record (no authors, contact, source_url)."""
    error = DataIntegrityError(
        message="Test",
        record=minimal_record,
        issues=["One issue"],
    )
    with patch("eegdash.dataset.exceptions.logger") as mock_logger:
        error.log_error()
        assert mock_logger.error.call_count == 2  # Main + 1 issue
        assert mock_logger.info.call_count == 0


def test_log_error_with_string_contact(sample_record):
    """Test log_error when contact_info is a string."""
    error = DataIntegrityError(
        message="Test",
        record=sample_record,
        issues=["Issue"],
        contact_info="single@email.com",
    )
    with patch("eegdash.dataset.exceptions.logger") as mock_logger:
        error.log_error()
        assert mock_logger.error.call_count >= 1


def test_log_warning(sample_record):
    """Test log_warning method."""
    error = DataIntegrityError(
        message="Test",
        record=sample_record,
        issues=sample_record["_data_integrity_issues"],
        authors=sample_record["_dataset_authors"],
        contact_info=sample_record["_dataset_contact"],
        source_url=sample_record["_source_url"],
    )
    with patch("eegdash.dataset.exceptions.logger") as mock_logger:
        error.log_warning()
        assert mock_logger.warning.call_count == 3  # Main + 2 issues
        assert mock_logger.info.call_count == 3  # authors, contact, source_url


def test_log_warning_minimal(minimal_record):
    """Test log_warning with minimal record."""
    error = DataIntegrityError(
        message="Test",
        record=minimal_record,
        issues=["One issue"],
    )
    with patch("eegdash.dataset.exceptions.logger") as mock_logger:
        error.log_warning()
        assert mock_logger.warning.call_count == 2  # Main + 1 issue
        assert mock_logger.info.call_count == 0


def test_log_warning_with_string_contact(sample_record):
    """Test log_warning when contact_info is a string."""
    error = DataIntegrityError(
        message="Test",
        record=sample_record,
        issues=["Issue"],
        contact_info="single@email.com",
    )
    with patch("eegdash.dataset.exceptions.logger") as mock_logger:
        error.log_warning()
        assert mock_logger.warning.call_count >= 1


def test_warn_from_record(sample_record):
    """Test warn_from_record class method."""
    with patch("eegdash.dataset.exceptions.logger") as mock_logger:
        DataIntegrityError.warn_from_record(sample_record)
        assert mock_logger.warning.call_count >= 1


def test_warn_from_record_minimal(minimal_record):
    """Test warn_from_record with minimal record."""
    with patch("eegdash.dataset.exceptions.logger") as mock_logger:
        DataIntegrityError.warn_from_record(minimal_record)
        assert mock_logger.warning.call_count >= 1


def test_from_record(sample_record):
    """Test from_record class method."""
    with patch("eegdash.dataset.exceptions.logger"):
        error = DataIntegrityError.from_record(sample_record)

    assert isinstance(error, DataIntegrityError)
    assert error.record == sample_record
    assert error.issues == sample_record["_data_integrity_issues"]
    assert error.authors == sample_record["_dataset_authors"]
    assert error.contact_info == sample_record["_dataset_contact"]
    assert error.source_url == sample_record["_source_url"]

    msg = str(error)
    assert "sub-01/eeg/sub-01_task-rest_eeg.set" in msg
    assert "ds001234" in msg


def test_from_record_minimal(minimal_record):
    """Test from_record with minimal record."""
    with patch("eegdash.dataset.exceptions.logger"):
        error = DataIntegrityError.from_record(minimal_record)

    assert isinstance(error, DataIntegrityError)
    assert error.issues == []
    assert error.authors == []


def test_from_record_with_string_contact(sample_record):
    """Test from_record when contact is a string."""
    sample_record["_dataset_contact"] = "single@email.com"
    with patch("eegdash.dataset.exceptions.logger"):
        error = DataIntegrityError.from_record(sample_record)

    assert isinstance(error, DataIntegrityError)
    msg = str(error)
    assert "single@email.com" in msg


def test_from_record_logs_error(sample_record):
    """Test that from_record calls log_error."""
    with patch("eegdash.dataset.exceptions.logger") as mock_logger:
        DataIntegrityError.from_record(sample_record)
        assert mock_logger.error.call_count >= 1


def test_all_exports():
    """Test __all__ exports the expected classes."""
    from eegdash.dataset import exceptions

    assert "EEGDashError" in exceptions.__all__
    assert "DataIntegrityError" in exceptions.__all__
