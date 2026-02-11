from unittest.mock import MagicMock, patch

import pytest

from eegdash.dataset.io import (
    _ensure_coordsystem_symlink,
    _find_best_matching_file,
    _generate_vhdr_from_metadata,
    _generate_vmrk_stub,
    _repair_ctf_eeg_position_file,
    _repair_electrodes_tsv,
    _repair_events_tsv_na_duration,
    _repair_tsv_decimal_separators,
    _repair_tsv_encoding,
    _repair_tsv_na_values,
    _repair_vhdr_missing_markerfile,
    _repair_vhdr_pointers,
)


@pytest.mark.parametrize(
    "func",
    [
        _repair_electrodes_tsv,
        _repair_tsv_decimal_separators,
        _repair_tsv_na_values,
        _repair_events_tsv_na_duration,
        _repair_ctf_eeg_position_file,
    ],
)
def test_repair_nonexistent_dir(tmp_path, func):
    """Test repair functions return False for non-existent directory."""
    assert func(tmp_path / "nonexistent") is False


@pytest.mark.parametrize(
    "func,filename,content",
    [
        (
            _repair_electrodes_tsv,
            "sub-01_electrodes.tsv",
            "name\tx\ty\tz\nFp1\t1.0\t2.0\t3.0\n",
        ),
        (
            _repair_tsv_decimal_separators,
            "sub-01_channels.tsv",
            "name\tsampling_frequency\nFp1\t500.0\n",
        ),
        (
            _repair_tsv_na_values,
            "sub-01_channels.tsv",
            "name\tsampling_frequency\tlow_cutoff\nFp1\t256\t0.1\n",
        ),
        (
            _repair_events_tsv_na_duration,
            "sub-01_task-rest_events.tsv",
            "onset\tduration\ttrial_type\n0.5\t0.5\tgo\n",
        ),
    ],
    ids=["electrodes", "decimal_sep", "na_values", "events_duration"],
)
def test_repair_no_change_needed(tmp_path, func, filename, content):
    """Test repair functions return False when no changes are needed."""
    (tmp_path / filename).write_text(content)
    assert func(tmp_path) is False


def test_repair_vhdr_pointers(tmp_path):
    """Test that VHDR pointers are repaired if broken but BIDS files exist."""
    eeg_dir = tmp_path

    # Create the BIDS files (what we want to point to)
    (eeg_dir / "sub-01_task-rest_eeg.eeg").touch()
    (eeg_dir / "sub-01_task-rest_eeg.vmrk").touch()

    # Create the VHDR with BAD pointers
    vhdr_path = eeg_dir / "sub-01_task-rest_eeg.vhdr"
    vhdr_content = """Brain Vision Data Exchange Header File Version 1.0
[Common Infos]
DataFile=INTERNAL_NAME.eeg
MarkerFile=INTERNAL_NAME.vmrk
"""
    vhdr_path.write_text(vhdr_content)

    # Run repair
    repaired = _repair_vhdr_pointers(vhdr_path)
    assert repaired is True


def test_repair_vhdr_no_change_needed(tmp_path):
    """Test that VHDR is untouched if pointers are valid."""
    eeg_dir = tmp_path

    (eeg_dir / "correct.eeg").touch()
    vhdr_path = eeg_dir / "test.vhdr"
    vhdr_path.write_text("DataFile=correct.eeg")

    repaired = _repair_vhdr_pointers(vhdr_path)
    assert repaired is False
    assert vhdr_path.read_text() == "DataFile=correct.eeg"


def test_ensure_coordsystem_symlink(tmp_path):
    """Test symlink creation for coordsystem.json."""
    dataset_root = tmp_path
    eeg_dir = dataset_root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)

    # Create dummy files
    (eeg_dir / "sub-01_task-rest_electrodes.tsv").touch()

    # Create coordsystem in subject root
    subject_root = dataset_root / "sub-01"
    (subject_root / "sub-01_coordsystem.json").touch()

    # Run ensure function
    _ensure_coordsystem_symlink(eeg_dir)

    # Verify symlink
    expected_link = eeg_dir / "sub-01_coordsystem.json"
    assert expected_link.exists()
    assert expected_link.is_symlink()


def test_io_error_handling(tmp_path):
    """Test error handling branches in IO module."""
    # Test 1: repair_vhdr non-existent file
    assert _repair_vhdr_pointers(tmp_path / "nonexistent.vhdr") is False

    # Test 2: repair_vhdr read error
    f = tmp_path / "bad.vhdr"
    f.touch()
    with patch("pathlib.Path.read_text", side_effect=Exception("Read error")):
        assert _repair_vhdr_pointers(f) is False

    # Test 3: ensure_symlink non-existent dir
    _ensure_coordsystem_symlink(tmp_path / "missing_dir")  # Should not raise

    # Test 4: ensure_symlink missing electrodes (early return)
    exists_dir = tmp_path / "exists"
    exists_dir.mkdir()
    _ensure_coordsystem_symlink(exists_dir)  # Should return early

    # Test 5: ensure_symlink already has coordsystem
    (exists_dir / "sub-01_electrodes.tsv").touch()
    (exists_dir / "sub-01_coordsystem.json").touch()
    _ensure_coordsystem_symlink(exists_dir)  # Should return early branch


def test_find_best_matching_file_single_candidate(tmp_path):
    """Test that single candidate is returned regardless of name."""
    (tmp_path / "actual_file.eeg").touch()

    result = _find_best_matching_file(tmp_path, "completely_different.eeg", ".eeg")
    assert result == "actual_file.eeg"


def test_find_best_matching_file_fuzzy_match(tmp_path):
    """Test fuzzy matching with similar filenames."""
    (tmp_path / "sub-01_task-rest_eeg.eeg").touch()
    (tmp_path / "sub-02_task-rest_eeg.eeg").touch()

    # Look for sub-01 with a typo
    result = _find_best_matching_file(tmp_path, "rsub-01_task-rest_eeg.eeg", ".eeg")
    assert result == "sub-01_task-rest_eeg.eeg"


def test_find_best_matching_file_no_candidates(tmp_path):
    """Test returns None when no files with extension exist."""
    result = _find_best_matching_file(tmp_path, "missing.eeg", ".eeg")
    assert result is None


@pytest.mark.parametrize(
    "vhdr_stem,bad_data,bad_vmrk,expected_data,expected_vmrk",
    [
        (
            "sub-01_task-sternberg_eeg",
            "sub-01_task-sternbeg_eeg.eeg",
            "sub-01_task-sternbeg_eeg.vmrk",
            "sub-01_task-sternberg_eeg.eeg",
            "sub-01_task-sternberg_eeg.vmrk",
        ),
        (
            "sub-16_task-rest_eeg",
            "rsub-16_task-rest_eeg.eeg",
            "sub-16_task-rest_eeg.vmrk",
            "sub-16_task-rest_eeg.eeg",
            "sub-16_task-rest_eeg.vmrk",
        ),
        (
            "sub-054_ses-00_task-rest_eeg",
            "sub-054_sub-054_date-20210505ses-00_task-rest_eeg.eeg",
            "sub-054_sub-054_date-20210505ses-00_task-rest_eeg.vmrk",
            "sub-054_ses-00_task-rest_eeg.eeg",
            "sub-054_ses-00_task-rest_eeg.vmrk",
        ),
    ],
    ids=["typo", "prefix_typo", "complex_name"],
)
def test_repair_vhdr_fuzzy_match(
    tmp_path, vhdr_stem, bad_data, bad_vmrk, expected_data, expected_vmrk
):
    """Test VHDR repair with various naming mismatches (fuzzy match)."""
    (tmp_path / f"{vhdr_stem}.eeg").touch()
    (tmp_path / f"{vhdr_stem}.vmrk").touch()

    vhdr_path = tmp_path / f"{vhdr_stem}.vhdr"
    vhdr_path.write_text(
        f"Brain Vision Data Exchange Header File Version 1.0\n"
        f"[Common Infos]\n"
        f"DataFile={bad_data}\n"
        f"MarkerFile={bad_vmrk}\n"
    )

    assert _repair_vhdr_pointers(vhdr_path) is True
    content = vhdr_path.read_text()
    assert expected_data in content
    assert expected_vmrk in content


# Tests for VHDR/VMRK generation from metadata


def test_generate_vhdr_from_metadata_success(tmp_path):
    """Test successful VHDR generation from complete metadata."""
    vhdr_path = tmp_path / "sub-01_task-rest_eeg.vhdr"
    record = {
        "ch_names": ["Fp1", "Fp2", "F3", "F4"],
        "sampling_frequency": 500,
        "nchans": 4,
    }

    assert _generate_vhdr_from_metadata(vhdr_path, record) is True

    content = vhdr_path.read_text()
    assert "NumberOfChannels=4" in content
    assert "SamplingInterval=2000" in content  # 1_000_000 / 500
    assert "Ch1=Fp1,,0.1,µV" in content
    assert (tmp_path / "sub-01_task-rest_eeg.vmrk").exists()  # VMRK stub created


@pytest.mark.parametrize(
    "record,reason",
    [
        ({"sampling_frequency": 500, "nchans": 4}, "missing ch_names"),
        ({"ch_names": ["Fp1"], "nchans": 1}, "missing sampling_frequency"),
        ({"ch_names": ["Fp1"], "sampling_frequency": 500}, "missing nchans"),
        (
            {"ch_names": ["Fp1"], "sampling_frequency": 500, "nchans": 4},
            "ch_names/nchans mismatch",
        ),
    ],
    ids=["no_ch_names", "no_sfreq", "no_nchans", "mismatch"],
)
def test_generate_vhdr_invalid_metadata(tmp_path, record, reason):
    """Test VHDR generation fails gracefully with invalid metadata."""
    vhdr_path = tmp_path / "test.vhdr"
    assert _generate_vhdr_from_metadata(vhdr_path, record) is False, reason
    assert not vhdr_path.exists()


def test_generate_vhdr_creates_parent_dirs(tmp_path):
    """Test VHDR generation creates parent directories if needed."""
    vhdr_path = tmp_path / "sub-01" / "eeg" / "test.vhdr"
    record = {"ch_names": ["Fp1", "Fp2"], "sampling_frequency": 256, "nchans": 2}

    assert _generate_vhdr_from_metadata(vhdr_path, record) is True
    assert vhdr_path.exists()


def test_generate_vhdr_does_not_overwrite_vmrk(tmp_path):
    """Test VHDR generation doesn't overwrite existing VMRK file."""
    vmrk_path = tmp_path / "test.vmrk"
    vmrk_path.write_text("Custom VMRK content")

    record = {"ch_names": ["Fp1"], "sampling_frequency": 500, "nchans": 1}
    _generate_vhdr_from_metadata(tmp_path / "test.vhdr", record)

    assert vmrk_path.read_text() == "Custom VMRK content"


def test_generate_vmrk_stub_success(tmp_path):
    """Test successful VMRK stub generation."""
    vmrk_path = tmp_path / "test.vmrk"
    assert _generate_vmrk_stub(vmrk_path, "test.vhdr") is True

    content = vmrk_path.read_text()
    assert "Brain Vision Data Exchange Marker File" in content
    assert "DataFile=test.eeg" in content


@pytest.mark.parametrize(
    "func,args",
    [
        (_generate_vmrk_stub, ("test.vhdr",)),
        (
            _generate_vhdr_from_metadata,
            ({"ch_names": ["Fp1"], "sampling_frequency": 500, "nchans": 1},),
        ),
    ],
)
def test_generate_file_write_error(tmp_path, func, args):
    """Test file generation handles write errors gracefully."""
    path = (
        tmp_path / "test.vhdr"
        if func == _generate_vhdr_from_metadata
        else tmp_path / "test.vmrk"
    )
    with patch("pathlib.Path.write_text", side_effect=Exception("Write error")):
        assert func(path, *args) is False


# Tests for TSV encoding repair


@pytest.mark.parametrize(
    "encoding,expected_repair",
    [("latin-1", True), ("cp1252", True), ("utf-8", False)],
    ids=["latin1", "cp1252", "utf8_no_repair"],
)
def test_repair_tsv_encoding(tmp_path, encoding, expected_repair):
    """Test TSV encoding repair for various encodings."""
    tsv_path = tmp_path / "channels.tsv"
    content = "name\ttype\tunits\nFp1\tEEG\tµV\n"
    tsv_path.write_bytes(content.encode(encoding))

    assert _repair_tsv_encoding(tmp_path) is expected_repair
    assert tsv_path.read_text(encoding="utf-8") == content


def test_repair_tsv_encoding_edge_cases(tmp_path):
    """Test edge cases: non-existent dir, no TSV files, multiple files."""
    assert _repair_tsv_encoding(tmp_path / "nonexistent") is False

    (tmp_path / "data.json").write_text("{}")
    assert _repair_tsv_encoding(tmp_path) is False

    (tmp_path / "participants.tsv").write_text("id\nsub-01\n", encoding="utf-8")
    (tmp_path / "channels.tsv").write_bytes("name\tunits\nFp1\tµV\n".encode("latin-1"))
    assert _repair_tsv_encoding(tmp_path) is True


# ---- Tests for _repair_electrodes_tsv ----


def test_repair_electrodes_tsv_replaces_na(tmp_path):
    """Test that n/a values in coordinate columns are replaced with 0.0."""
    tsv_path = tmp_path / "sub-01_electrodes.tsv"
    tsv_path.write_text("name\tx\ty\tz\nFp1\tn/a\tn/a\tn/a\nFp2\t1.0\t2.0\t3.0\n")

    assert _repair_electrodes_tsv(tmp_path) is True

    content = tsv_path.read_text()
    assert "n/a" not in content
    assert "Fp1\t0.0\t0.0\t0.0" in content
    assert "Fp2\t1.0\t2.0\t3.0" in content


def test_repair_electrodes_tsv_no_coord_columns(tmp_path):
    """Test returns False when no x/y/z columns exist."""
    tsv_path = tmp_path / "sub-01_electrodes.tsv"
    tsv_path.write_text("name\ttype\nFp1\tEEG\n")

    assert _repair_electrodes_tsv(tmp_path) is False


# ---- Tests for _repair_tsv_decimal_separators ----


def test_repair_tsv_decimal_separators(tmp_path):
    """Test comma-to-dot conversion in numeric fields."""
    tsv_path = tmp_path / "sub-01_electrodes.tsv"
    tsv_path.write_text("name\tx\ty\tz\nFp1\t5,004\t3,2\t1,001\n")

    assert _repair_tsv_decimal_separators(tmp_path) is True

    content = tsv_path.read_text()
    assert "5.004" in content
    assert "3.2" in content
    assert "1.001" in content


def test_repair_tsv_decimal_separators_header_untouched(tmp_path):
    """Test that header line is never modified."""
    tsv_path = tmp_path / "sub-01_channels.tsv"
    tsv_path.write_text("name\tsampling_frequency\nFp1\t5,004\n")

    _repair_tsv_decimal_separators(tmp_path)

    content = tsv_path.read_text()
    lines = content.strip().split("\n")
    assert lines[0] == "name\tsampling_frequency"


# ---- Tests for _repair_tsv_na_values ----


def test_repair_tsv_na_values(tmp_path):
    """Test n/a replacement in numeric columns of channels.tsv."""
    tsv_path = tmp_path / "sub-01_channels.tsv"
    tsv_path.write_text(
        "name\ttype\tsampling_frequency\tlow_cutoff\thigh_cutoff\n"
        "Fp1\tEEG\tn/a\tn/a\tn/a\n"
        "Fp2\tEEG\t256\t0.1\t100\n"
    )

    assert _repair_tsv_na_values(tmp_path) is True

    content = tsv_path.read_text()
    lines = content.strip().split("\n")
    # First data line should have n/a replaced with 0
    assert "n/a" not in lines[1]
    # Second data line should be untouched
    assert "256" in lines[2]


def test_repair_tsv_na_values_no_numeric_columns(tmp_path):
    """Test returns False when no target numeric columns."""
    tsv_path = tmp_path / "sub-01_channels.tsv"
    tsv_path.write_text("name\ttype\tunits\nFp1\tEEG\tuV\n")

    assert _repair_tsv_na_values(tmp_path) is False


# ---- Tests for _repair_vhdr_missing_markerfile ----


def test_repair_vhdr_missing_markerfile(tmp_path):
    """Test adding MarkerFile entry when missing."""
    vhdr_path = tmp_path / "sub-01_task-rest_eeg.vhdr"
    vhdr_path.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\n"
        "DataFile=sub-01_task-rest_eeg.eeg\n"
        "[Binary Infos]\n"
    )

    assert _repair_vhdr_missing_markerfile(vhdr_path) is True

    content = vhdr_path.read_text()
    assert "MarkerFile=sub-01_task-rest_eeg.vmrk" in content
    # VMRK stub should be generated
    assert (tmp_path / "sub-01_task-rest_eeg.vmrk").exists()


def test_repair_vhdr_missing_markerfile_already_exists(tmp_path):
    """Test no repair when MarkerFile is already present."""
    vhdr_path = tmp_path / "test.vhdr"
    vhdr_path.write_text("[Common Infos]\nDataFile=test.eeg\nMarkerFile=test.vmrk\n")

    assert _repair_vhdr_missing_markerfile(vhdr_path) is False


def test_repair_vhdr_missing_markerfile_no_common_infos(tmp_path):
    """Test returns False when [Common Infos] section is missing."""
    vhdr_path = tmp_path / "test.vhdr"
    vhdr_path.write_text("DataFile=test.eeg\n")

    assert _repair_vhdr_missing_markerfile(vhdr_path) is False


def test_repair_vhdr_missing_markerfile_nonexistent(tmp_path):
    """Test returns False for non-existent file."""
    assert _repair_vhdr_missing_markerfile(tmp_path / "nonexistent.vhdr") is False


def test_repair_vhdr_missing_markerfile_wrong_extension(tmp_path):
    """Test returns False for non-.vhdr file."""
    f = tmp_path / "test.eeg"
    f.touch()
    assert _repair_vhdr_missing_markerfile(f) is False


def test_repair_vhdr_missing_markerfile_no_datafile(tmp_path):
    """Test MarkerFile is added after [Common Infos] when DataFile is also missing."""
    vhdr_path = tmp_path / "test.vhdr"
    vhdr_path.write_text("[Common Infos]\nCodepage=UTF-8\n")

    assert _repair_vhdr_missing_markerfile(vhdr_path) is True

    content = vhdr_path.read_text()
    assert "MarkerFile=test.vmrk" in content


# ---- Tests for _load_epoched_eeglab_as_raw ----


def test_load_epoched_eeglab_as_raw():
    """Test epoched EEGLAB loading and concatenation (mocked via MNE)."""
    import numpy as np

    from eegdash.dataset.io import _load_epoched_eeglab_as_raw

    # Mock the MNE functions — strategy 1 (MNE epoch reader)
    mock_epochs = MagicMock()
    mock_epochs.get_data.return_value = np.random.randn(10, 4, 100)
    mock_epochs.info = MagicMock()

    with patch("mne.read_epochs_eeglab", return_value=mock_epochs):
        with patch("mne.io.RawArray") as MockRawArray:
            from pathlib import Path

            _load_epoched_eeglab_as_raw(Path("/fake/file.set"))

            call_args = MockRawArray.call_args
            data_arg = call_args[0][0]
            assert data_arg.shape == (4, 1000)


def test_load_epoched_eeglab_scipy_fallback():
    """Test scipy fallback when MNE epoch reader fails."""
    from eegdash.dataset.io import _load_epoched_eeglab_as_raw

    # Make MNE epoch reader fail, triggering scipy fallback
    with patch(
        "mne.read_epochs_eeglab", side_effect=IndexError("list index out of range")
    ):
        with patch("eegdash.dataset.io._load_set_via_scipy") as mock_scipy:
            mock_scipy.return_value = MagicMock()
            from pathlib import Path

            result = _load_epoched_eeglab_as_raw(Path("/fake/file.set"))
            mock_scipy.assert_called_once()
            assert result is mock_scipy.return_value


# ---- Tests for _load_raw_direct ----


def test_load_raw_direct_fif():
    """Test direct FIF loading with allow_maxshield."""
    from pathlib import Path

    from eegdash.dataset.io import _load_raw_direct

    mock_raw = MagicMock()
    with patch("mne.io.read_raw_fif", return_value=mock_raw) as mock_reader:
        result = _load_raw_direct(Path("/fake/file.fif"), allow_maxshield=True)
        assert result is mock_raw
        mock_reader.assert_called_once_with(
            "/fake/file.fif", preload=False, verbose="ERROR", allow_maxshield=True
        )


def test_load_raw_direct_set():
    """Test direct EEGLAB loading."""
    from pathlib import Path

    from eegdash.dataset.io import _load_raw_direct

    mock_raw = MagicMock()
    with patch("mne.io.read_raw_eeglab", return_value=mock_raw) as mock_reader:
        result = _load_raw_direct(Path("/fake/file.set"))
        assert result is mock_raw
        mock_reader.assert_called_once_with(
            "/fake/file.set", preload=False, verbose="ERROR"
        )


def test_load_raw_direct_unsupported_extension():
    """Test that unsupported extension raises ValueError."""
    from pathlib import Path

    from eegdash.dataset.io import _load_raw_direct

    with pytest.raises(ValueError, match="No direct reader available"):
        _load_raw_direct(Path("/fake/file.xyz"))


# ---- Tests for _repair_events_tsv_na_duration ----


def test_repair_events_tsv_na_duration(tmp_path):
    """Test replacing n/a in duration column of events.tsv."""
    tsv_path = tmp_path / "sub-01_task-rest_events.tsv"
    tsv_path.write_text(
        "onset\tduration\ttrial_type\n0.5\tn/a\tgo\n1.0\t0.5\tstop\n2.0\tn/a\tgo\n"
    )

    assert _repair_events_tsv_na_duration(tmp_path) is True

    content = tsv_path.read_text()
    lines = content.strip().split("\n")
    assert lines[0] == "onset\tduration\ttrial_type"
    assert lines[1] == "0.5\t0\tgo"
    assert lines[2] == "1.0\t0.5\tstop"
    assert lines[3] == "2.0\t0\tgo"


def test_repair_events_tsv_nan_onset_removed(tmp_path):
    """Test rows with NaN onset are removed from events.tsv."""
    tsv_path = tmp_path / "sub-01_task-rest_events.tsv"
    tsv_path.write_text(
        "onset\tduration\tsample\tvalue\n"
        "0.5\t0\t128\t1\n"
        "NaN\t0\tn/a\t2\n"
        "1.0\t0\t256\t3\n"
    )

    assert _repair_events_tsv_na_duration(tmp_path) is True

    content = tsv_path.read_text()
    lines = content.strip().split("\n")
    assert len(lines) == 3  # header + 2 valid rows (NaN row removed)
    assert "NaN" not in content


def test_repair_events_tsv_na_sample_replaced(tmp_path):
    """Test n/a in sample column is replaced with 0."""
    tsv_path = tmp_path / "sub-01_task-rest_events.tsv"
    tsv_path.write_text(
        "onset\tduration\tsample\tvalue\n0.5\tn/a\tn/a\t1\n1.0\t0.5\t256\t2\n"
    )

    assert _repair_events_tsv_na_duration(tmp_path) is True

    content = tsv_path.read_text()
    lines = content.strip().split("\n")
    assert lines[1] == "0.5\t0\t0\t1"


def test_repair_events_tsv_na_duration_no_relevant_columns(tmp_path):
    """Test returns False when no onset/duration/sample columns exist."""
    tsv_path = tmp_path / "sub-01_task-rest_events.tsv"
    tsv_path.write_text("trial_type\tvalue\ngo\t1\n")

    assert _repair_events_tsv_na_duration(tmp_path) is False


def test_repair_events_tsv_na_duration_whitespace(tmp_path):
    """Test handles n/a with surrounding whitespace."""
    tsv_path = tmp_path / "sub-01_task-rest_events.tsv"
    tsv_path.write_text("onset\tduration\ttrial_type\n0.5\t n/a \tgo\n")

    assert _repair_events_tsv_na_duration(tmp_path) is True
    content = tsv_path.read_text()
    assert "0\tgo" in content


# ---- Tests for _repair_ctf_eeg_position_file ----


@pytest.mark.parametrize("content", ["n/a", "  n/a  \n"], ids=["plain", "whitespace"])
def test_repair_ctf_eeg_position_file_na(tmp_path, content):
    """Test replacing n/a content in CTF .eeg file with empty file."""
    ds_dir = tmp_path / "test_meg.ds"
    ds_dir.mkdir()
    eeg_file = ds_dir / "test_meg.eeg"
    eeg_file.write_text(content)
    assert _repair_ctf_eeg_position_file(ds_dir) is True
    assert eeg_file.read_text() == ""


def test_repair_ctf_eeg_position_file_valid(tmp_path):
    """Test no repair when .eeg file has valid position data."""
    ds_dir = tmp_path / "test_meg.ds"
    ds_dir.mkdir()
    eeg_file = ds_dir / "test_meg.eeg"
    eeg_file.write_text("1 Nasion 10.5 20.3 15.7\n2 LPA 5.0 -20.0 10.0\n")

    assert _repair_ctf_eeg_position_file(ds_dir) is False


def test_repair_ctf_eeg_position_file_no_eeg(tmp_path):
    """Test returns False when no .eeg file exists."""
    ds_dir = tmp_path / "test_meg.ds"
    ds_dir.mkdir()

    assert _repair_ctf_eeg_position_file(ds_dir) is False


def test_repair_ctf_eeg_position_file_not_dir(tmp_path):
    """Test returns False when path is not a directory."""
    f = tmp_path / "test.ds"
    f.touch()

    assert _repair_ctf_eeg_position_file(f) is False
