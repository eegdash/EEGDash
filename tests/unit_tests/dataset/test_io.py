import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from eegdash.dataset.io import (
    _convert_time_with_numeric_dash,
    _eeglab_ch_names_from_eeg,
    _eeglab_fdt_path,
    _eeglab_get_first_eeg,
    _eeglab_load_first_eeg,
    _ensure_coordsystem_symlink,
    _find_best_matching_file,
    _generate_coordsystem_json,
    _generate_vhdr_from_metadata,
    _generate_vmrk_stub,
    _load_raw_eeglab_alleeg,
    _repair_eeglab_fdt,
    _repair_events_tsv_nan_samples,
    _repair_tsv_decimal_separators,
    _repair_tsv_encoding,
    _repair_vhdr_pointers,
)


def test_convert_time_with_numeric_dash_dd_mm_yyyy():
    """Numeric dash date 14-10-1925 (DD-MM-YYYY) is normalized to 14/10/1925 and delegated."""
    orig = MagicMock(return_value=12345.0)
    out = _convert_time_with_numeric_dash("14-10-1925", "12:00:00", orig=orig)
    assert out == 12345.0
    orig.assert_called_once_with("14/10/1925", "12:00:00")


def test_convert_time_with_numeric_dash_mm_dd_yyyy():
    """Numeric dash date 10-14-1925 (MM-DD-YYYY) is normalized and delegated."""
    orig = MagicMock(return_value=0.0)
    _convert_time_with_numeric_dash("10-14-1925", "00:00:00", orig=orig)
    orig.assert_called_once_with("14/10/1925", "00:00:00")


def test_convert_time_with_numeric_dash_iso():
    """ISO-style YYYY-MM-DD is normalized to dd/mm/yyyy and delegated."""
    orig = MagicMock(return_value=1.0)
    _convert_time_with_numeric_dash("1925-10-14", "01:00:00", orig=orig)
    orig.assert_called_once_with("14/10/1925", "01:00:00")


def test_convert_time_with_numeric_dash_fallback():
    """Unsupported date format is passed through to orig unchanged."""
    orig = MagicMock(return_value=999.0)
    out = _convert_time_with_numeric_dash("14/Oct/1925", "12:00:00", orig=orig)
    assert out == 999.0
    orig.assert_called_once_with("14/Oct/1925", "12:00:00")


def test_convert_time_with_numeric_dash_strips_whitespace():
    """Date string is stripped before parsing."""
    orig = MagicMock(return_value=0.0)
    _convert_time_with_numeric_dash("  14-10-1925  ", "00:00:00", orig=orig)
    orig.assert_called_once_with("14/10/1925", "00:00:00")


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


def test_repair_vhdr_pointers_annex_marker(tmp_path):
    """VHDR with annex-key MarkerFile should point to BIDS .vmrk."""
    eeg_dir = tmp_path

    # BIDS-named companion files
    (eeg_dir / "sub-01_task-rest_eeg.eeg").touch()
    (eeg_dir / "sub-01_task-rest_eeg.vmrk").touch()

    vhdr_path = eeg_dir / "sub-01_task-rest_eeg.vhdr"
    vhdr_content = """Brain Vision Data Exchange Header File Version 1.0
[Common Infos]
DataFile=sub-01_task-rest_eeg.eeg
MarkerFile=SHA256E-s3719--aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.vmrk
"""
    vhdr_path.write_text(vhdr_content)

    repaired = _repair_vhdr_pointers(vhdr_path)
    assert repaired is True
    text = vhdr_path.read_text()
    # MarkerFile should now reference the BIDS-named file, not the annex key
    assert "MarkerFile=sub-01_task-rest_eeg.vmrk" in text
    assert "SHA256E-s3719--" not in text


def test_repair_vhdr_pointers_annex_and_internal(tmp_path):
    """VHDR with internal DataFile and annex-key MarkerFile maps both to BIDS."""
    eeg_dir = tmp_path

    # BIDS-named companion files
    (eeg_dir / "sub-01_task-main_run-001_eeg.eeg").touch()
    (eeg_dir / "sub-01_task-main_run-001_eeg.vmrk").touch()

    vhdr_path = eeg_dir / "sub-01_task-main_run-001_eeg.vhdr"
    vhdr_content = """Brain Vision Data Exchange Header File Version 1.0
[Common Infos]
DataFile=s2_run1_08062017.eeg
MarkerFile=MD5E-s11657--7a519e74754041a678931b7b7d72f0ab.vmrk
"""
    vhdr_path.write_text(vhdr_content)

    repaired = _repair_vhdr_pointers(vhdr_path)
    assert repaired is True
    text = vhdr_path.read_text()
    # Both pointers should now use the BIDS-named companions
    assert "DataFile=sub-01_task-main_run-001_eeg.eeg" in text
    assert "MarkerFile=sub-01_task-main_run-001_eeg.vmrk" in text
    assert "s2_run1_08062017.eeg" not in text
    assert "MD5E-s11657--" not in text


def test_repair_vhdr_annex_key_no_bids_file(tmp_path):
    """Annex-key pointers rewritten to BIDS names even when target doesn't exist.

    Covers ds002158, ds003688, ds003848, ds005953 where the .vmrk was not
    downloaded and is only created as a stub *after* repair.
    """
    eeg_dir = tmp_path

    # No BIDS-named companion files exist yet
    vhdr_path = eeg_dir / "sub-01_ses-01_task-visual_run-01_ieeg.vhdr"
    vhdr_content = """Brain Vision Data Exchange Header File Version 1.0
[Common Infos]
DataFile=SHA256E-s9999--abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890.eeg
MarkerFile=SHA256E-s1808--43036bd24716b3b8a7b2c56f44360206d2872b30480859311989e9e10466598a.vmrk
"""
    vhdr_path.write_text(vhdr_content)

    repaired = _repair_vhdr_pointers(vhdr_path)
    assert repaired is True
    text = vhdr_path.read_text()
    assert "DataFile=sub-01_ses-01_task-visual_run-01_ieeg.eeg" in text
    assert "MarkerFile=sub-01_ses-01_task-visual_run-01_ieeg.vmrk" in text
    assert "SHA256E-" not in text


def test_repair_vhdr_annex_key_resolved_symlink(tmp_path):
    """Annex keys are rewritten even when the annex symlink resolves."""
    eeg_dir = tmp_path

    # Simulate a resolved annex symlink — file exists with the annex key name
    annex_eeg = "SHA256E-s5000--aabbccdd11223344aabbccdd11223344aabbccdd11223344aabbccdd11223344.eeg"
    (eeg_dir / annex_eeg).touch()

    vhdr_path = eeg_dir / "sub-02_task-rest_eeg.vhdr"
    vhdr_content = f"""Brain Vision Data Exchange Header File Version 1.0
[Common Infos]
DataFile={annex_eeg}
MarkerFile=sub-02_task-rest_eeg.vmrk
"""
    vhdr_path.write_text(vhdr_content)
    (eeg_dir / "sub-02_task-rest_eeg.vmrk").touch()

    repaired = _repair_vhdr_pointers(vhdr_path)
    assert repaired is True
    text = vhdr_path.read_text()
    assert "DataFile=sub-02_task-rest_eeg.eeg" in text
    assert "SHA256E-" not in text


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


def test_repair_vhdr_fuzzy_match_typo(tmp_path):
    """Test VHDR repair with typo in filename (fuzzy match fallback)."""
    eeg_dir = tmp_path

    # Create the actual BIDS files
    (eeg_dir / "sub-01_task-sternberg_eeg.eeg").touch()
    (eeg_dir / "sub-01_task-sternberg_eeg.vmrk").touch()

    # Create VHDR with typo ("sternbeg" instead of "sternberg")
    vhdr_path = eeg_dir / "sub-01_task-sternberg_eeg.vhdr"
    vhdr_content = """Brain Vision Data Exchange Header File Version 1.0
[Common Infos]
DataFile=sub-01_task-sternbeg_eeg.eeg
MarkerFile=sub-01_task-sternbeg_eeg.vmrk
"""
    vhdr_path.write_text(vhdr_content)

    # Run repair
    repaired = _repair_vhdr_pointers(vhdr_path)
    assert repaired is True

    # Verify content was fixed
    new_content = vhdr_path.read_text()
    assert "sub-01_task-sternberg_eeg.eeg" in new_content
    assert "sub-01_task-sternberg_eeg.vmrk" in new_content


def test_repair_vhdr_fuzzy_match_prefix_typo(tmp_path):
    """Test VHDR repair with prefix typo (e.g., 'rsub-' instead of 'sub-')."""
    eeg_dir = tmp_path

    # Create actual files
    (eeg_dir / "sub-16_task-rest_eeg.eeg").touch()
    (eeg_dir / "sub-16_task-rest_eeg.vmrk").touch()

    # Create VHDR with 'r' prefix typo
    vhdr_path = eeg_dir / "sub-16_task-rest_eeg.vhdr"
    vhdr_content = """Brain Vision Data Exchange Header File Version 1.0
[Common Infos]
DataFile=rsub-16_task-rest_eeg.eeg
MarkerFile=sub-16_task-rest_eeg.vmrk
"""
    vhdr_path.write_text(vhdr_content)

    repaired = _repair_vhdr_pointers(vhdr_path)
    assert repaired is True

    new_content = vhdr_path.read_text()
    assert "DataFile=sub-16_task-rest_eeg.eeg" in new_content


def test_repair_vhdr_complex_original_name(tmp_path):
    """Test VHDR repair when original name was completely different."""
    eeg_dir = tmp_path

    # Create BIDS files
    (eeg_dir / "sub-054_ses-00_task-rest_eeg.eeg").touch()
    (eeg_dir / "sub-054_ses-00_task-rest_eeg.vmrk").touch()

    # Create VHDR with complex original filename
    vhdr_path = eeg_dir / "sub-054_ses-00_task-rest_eeg.vhdr"
    vhdr_content = """Brain Vision Data Exchange Header File Version 1.0
[Common Infos]
DataFile=sub-054_sub-054_date-20210505ses-00_task-rest_eeg.eeg
MarkerFile=sub-054_sub-054_date-20210505ses-00_task-rest_eeg.vmrk
"""
    vhdr_path.write_text(vhdr_content)

    repaired = _repair_vhdr_pointers(vhdr_path)
    assert repaired is True

    new_content = vhdr_path.read_text()
    assert "sub-054_ses-00_task-rest_eeg.eeg" in new_content
    assert "sub-054_ses-00_task-rest_eeg.vmrk" in new_content


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


# Tests for coordsystem.json generation with datatype-specific keys


@pytest.mark.parametrize(
    "datatype,expected_prefix",
    [("eeg", "EEG"), ("ieeg", "iEEG"), ("meg", "MEG")],
    ids=["eeg", "ieeg", "meg"],
)
def test_generate_coordsystem_json_datatype_keys(tmp_path, datatype, expected_prefix):
    """Generated coordsystem.json uses correct keys per BIDS datatype."""
    electrodes = tmp_path / "sub-01_space-MNI_electrodes.tsv"
    electrodes.write_text("name\tx\ty\tz\nCh1\t0\t0\t0\n")

    assert _generate_coordsystem_json(electrodes, datatype=datatype) is True

    coordsystem = tmp_path / "sub-01_space-MNI_coordsystem.json"
    assert coordsystem.exists()
    data = json.loads(coordsystem.read_text())
    assert f"{expected_prefix}CoordinateSystem" in data
    assert data[f"{expected_prefix}CoordinateSystem"] == "MNI"
    assert f"{expected_prefix}CoordinateUnits" in data
    assert data[f"{expected_prefix}CoordinateUnits"] == "m"


def test_generate_coordsystem_json_no_space_entity(tmp_path):
    """Coordsystem defaults to 'Other' when electrodes filename has no space entity."""
    electrodes = tmp_path / "sub-01_electrodes.tsv"
    electrodes.write_text("name\tx\ty\tz\nCh1\t0\t0\t0\n")

    assert _generate_coordsystem_json(electrodes, datatype="ieeg") is True

    coordsystem = tmp_path / "sub-01_coordsystem.json"
    data = json.loads(coordsystem.read_text())
    assert data["iEEGCoordinateSystem"] == "Other"


def test_generate_coordsystem_json_default_datatype(tmp_path):
    """Default datatype is 'eeg' when not specified."""
    electrodes = tmp_path / "sub-01_electrodes.tsv"
    electrodes.write_text("name\tx\ty\tz\nCh1\t0\t0\t0\n")

    assert _generate_coordsystem_json(electrodes) is True

    coordsystem = tmp_path / "sub-01_coordsystem.json"
    data = json.loads(coordsystem.read_text())
    assert "EEGCoordinateSystem" in data


def test_ensure_coordsystem_symlink_generates_ieeg_keys(tmp_path):
    """_ensure_coordsystem_symlink infers ieeg datatype from directory name."""
    ieeg_dir = tmp_path / "sub-01" / "ieeg"
    ieeg_dir.mkdir(parents=True)

    (ieeg_dir / "sub-01_electrodes.tsv").write_text("name\tx\ty\tz\nCh1\t0\t0\t0\n")

    _ensure_coordsystem_symlink(ieeg_dir)

    coordsystem = ieeg_dir / "sub-01_coordsystem.json"
    assert coordsystem.exists()
    data = json.loads(coordsystem.read_text())
    assert "iEEGCoordinateSystem" in data
    assert "iEEGCoordinateUnits" in data


# Tests for TSV decimal separator repair


def test_repair_tsv_decimal_separators_events(tmp_path):
    """Test comma-to-dot conversion in events.tsv numeric fields."""
    tsv_path = tmp_path / "sub-01_task-sitstand_events.tsv"
    tsv_path.write_text(
        "onset\tduration\ttrial_type\tsample\n"
        "4,988\t0,5\trest\t1247\n"
        "10,512\t1,0\tmotor\t2628\n"
    )

    assert _repair_tsv_decimal_separators(tmp_path) is True

    content = tsv_path.read_text()
    assert "4.988" in content
    assert "0.5" in content
    assert "10.512" in content
    assert "1.0" in content
    # Non-numeric content preserved
    assert "rest" in content
    assert "motor" in content
    # Integer values untouched
    assert "1247" in content


def test_repair_tsv_decimal_separators_electrodes(tmp_path):
    """Test comma-to-dot conversion in electrodes.tsv."""
    tsv_path = tmp_path / "sub-01_electrodes.tsv"
    tsv_path.write_text("name\tx\ty\tz\nFp1\t5,004\t3,2\t1,001\n")

    assert _repair_tsv_decimal_separators(tmp_path) is True

    content = tsv_path.read_text()
    assert "5.004" in content
    assert "3.2" in content
    assert "1.001" in content


def test_repair_tsv_decimal_separators_no_commas(tmp_path):
    """Test no repair when file already uses dots."""
    tsv_path = tmp_path / "sub-01_task-rest_events.tsv"
    tsv_path.write_text("onset\tduration\n4.988\t0.5\n")

    assert _repair_tsv_decimal_separators(tmp_path) is False


def test_repair_tsv_decimal_separators_no_target_files(tmp_path):
    """Test no repair when directory has no matching TSV files."""
    (tmp_path / "participants.tsv").write_text("id\nsub-01\n")
    assert _repair_tsv_decimal_separators(tmp_path) is False


def test_repair_tsv_decimal_separators_nonexistent_dir(tmp_path):
    """Test returns False for nonexistent directory."""
    assert _repair_tsv_decimal_separators(tmp_path / "missing") is False


def test_repair_tsv_decimal_separators_preserves_tab_commas(tmp_path):
    """Test that tab-separated values are not confused with decimal commas."""
    tsv_path = tmp_path / "sub-01_task-rest_channels.tsv"
    tsv_path.write_text("name\ttype\tunits\nFp1\tEEG\tµV\n")

    assert _repair_tsv_decimal_separators(tmp_path) is False


# ── _repair_events_tsv_nan_samples ──


def test_repair_events_tsv_drops_nan_rows(tmp_path):
    """Rows with NaN onset should be dropped from events.tsv."""
    events = (
        "onset\tduration\tsample\tvalue\n"
        "1.5\t0.0\t384\t1\n"
        "nan\tnan\tnan\t2\n"
        "3.0\t0.0\t768\t3\n"
    )
    (tmp_path / "sub-01_task-x_events.tsv").write_text(events)

    assert _repair_events_tsv_nan_samples(tmp_path) is True

    lines = (tmp_path / "sub-01_task-x_events.tsv").read_text().splitlines()
    assert len(lines) == 3  # header + 2 valid rows
    assert "nan" not in "\t".join(lines)


def test_repair_events_tsv_no_nan(tmp_path):
    """No change when events.tsv has no NaN values."""
    events = "onset\tduration\tsample\tvalue\n1.0\t0.0\t256\t1\n"
    (tmp_path / "sub-01_task-x_events.tsv").write_text(events)

    assert _repair_events_tsv_nan_samples(tmp_path) is False


def test_repair_events_tsv_nonexistent_dir(tmp_path):
    """Returns False for nonexistent directory."""
    assert _repair_events_tsv_nan_samples(tmp_path / "missing") is False


# ── EEGLAB helpers and _repair_eeglab_fdt ──


def test_eeglab_get_first_eeg_flat():
    """Flat dict with nbchan/pnts returns the same dict."""
    mat = {"nbchan": 2, "pnts": 100, "srate": 250.0}
    out = _eeglab_get_first_eeg(mat)
    assert out is mat


def test_eeglab_get_first_eeg_eeg_key():
    """Dict with EEG key returns nested struct as dict."""
    inner = {"nbchan": 2, "pnts": 50}
    mat = {"EEG": inner}
    out = _eeglab_get_first_eeg(mat)
    assert out == inner


def test_eeglab_get_first_eeg_no_eeg_returns_none():
    """Dict without EEG/ALLEEG/nbchan returns None."""
    assert _eeglab_get_first_eeg({}) is None
    assert _eeglab_get_first_eeg({"other": 1}) is None


def test_eeglab_fdt_path_external(tmp_path):
    """External data string resolves to .fdt path."""
    set_path = tmp_path / "sub-01_task-rest_eeg.set"
    (tmp_path / "sub-01_task-rest_eeg.fdt").touch()
    eeg = {"data": "sub-01_task-rest_eeg.fdt"}
    out = _eeglab_fdt_path(set_path, eeg)
    assert out is not None
    assert out.name == "sub-01_task-rest_eeg.fdt"


def test_eeglab_fdt_path_fallback_suffix(tmp_path):
    """When data ref file missing, fallback to .set.with_suffix(.fdt)."""
    set_path = tmp_path / "file.set"
    fdt_path = tmp_path / "file.fdt"
    fdt_path.touch()
    eeg = {"data": "missing.fdt"}
    out = _eeglab_fdt_path(set_path, eeg)
    assert out == fdt_path


def test_eeglab_fdt_path_inline_returns_none():
    """Inline data (non-string) returns None."""
    set_path = Path("/tmp/file.set")
    eeg = {"data": [1.0, 2.0]}
    assert _eeglab_fdt_path(set_path, eeg) is None


def test_eeglab_ch_names_from_eeg_fallback():
    """No chanlocs yields CH1, CH2, ..."""
    eeg = {}
    assert _eeglab_ch_names_from_eeg(eeg, 3) == ["CH1", "CH2", "CH3"]


def test_eeglab_ch_names_from_eeg_labels():
    """Chanlocs with labels yield those names."""
    eeg = {"chanlocs": [{"labels": "Fp1"}, {"labels": "Fp2"}]}
    assert _eeglab_ch_names_from_eeg(eeg, 2) == ["Fp1", "Fp2"]


def test_repair_eeglab_fdt_nonexistent(tmp_path):
    """Nonexistent path or non-.set returns False."""
    assert _repair_eeglab_fdt(tmp_path / "missing.set") is False
    (tmp_path / "file.txt").touch()
    assert _repair_eeglab_fdt(tmp_path / "file.txt") is False


def test_repair_eeglab_fdt_truncated_fdt_repairs(tmp_path):
    """Truncated .fdt: repair updates .set header (pnts) and returns True."""
    pytest.importorskip("scipy")

    set_path = tmp_path / "test.set"
    fdt_path = tmp_path / "test.fdt"
    nbchan, pnts_orig = 2, 100
    # .fdt smaller than header: only 8 bytes (1 sample per channel)
    fdt_path.write_bytes(b"\x00" * 8)
    eeg = {
        "nbchan": nbchan,
        "pnts": pnts_orig,
        "srate": 250.0,
        "xmax": (pnts_orig - 1) / 250.0,
        "data": "test.fdt",
    }
    savemat(str(set_path), eeg, do_compression=False)

    result = _repair_eeglab_fdt(set_path)
    assert result is True

    mat = loadmat(
        str(set_path), squeeze_me=True, mat_dtype=False, struct_as_record=False
    )
    eeg_after = _eeglab_get_first_eeg(mat)
    assert eeg_after is not None
    assert int(eeg_after["pnts"]) == 1  # 8 / 4 / 2


def test_eeglab_get_first_eeg_alleeg():
    """Dict with ALLEEG key returns first element as dict."""
    first_eeg = {"nbchan": 1, "pnts": 10}
    mat = {"ALLEEG": [first_eeg]}
    out = _eeglab_get_first_eeg(mat)
    assert out == first_eeg


def test_eeglab_get_first_eeg_alleeg_empty_returns_none():
    """ALLEEG empty array returns None."""
    mat = {"ALLEEG": np.array([], dtype=object)}
    out = _eeglab_get_first_eeg(mat)
    assert out is None


def test_eeglab_load_first_eeg_returns_eeg_when_valid_set(tmp_path):
    """Valid .set file returns first EEG dict."""
    pytest.importorskip("scipy")

    set_path = tmp_path / "valid.set"
    eeg = {"nbchan": 2, "pnts": 10, "srate": 250.0, "data": "valid.fdt"}
    savemat(str(set_path), eeg, do_compression=False)
    out = _eeglab_load_first_eeg(set_path)
    assert out is not None
    assert int(out["nbchan"]) == 2 and int(out["pnts"]) == 10


def test_eeglab_load_first_eeg_nonexistent_returns_none(tmp_path):
    """Nonexistent .set returns None."""
    assert _eeglab_load_first_eeg(tmp_path / "missing.set") is None


def test_repair_eeglab_fdt_no_fdt_file_returns_false(tmp_path):
    """Valid .set but no .fdt file returns False."""
    pytest.importorskip("scipy")

    set_path = tmp_path / "nofdt.set"
    eeg = {"nbchan": 2, "pnts": 10, "srate": 250.0, "data": "nofdt.fdt"}
    savemat(str(set_path), eeg, do_compression=False)
    assert _repair_eeglab_fdt(set_path) is False


def test_repair_eeglab_fdt_not_truncated_returns_false(tmp_path):
    """.set + .fdt with full size: no repair, returns False."""
    pytest.importorskip("scipy")

    set_path = tmp_path / "full.set"
    fdt_path = tmp_path / "full.fdt"
    nbchan, pnts = 2, 10
    fdt_path.write_bytes(b"\x00" * (nbchan * pnts * 4))
    eeg = {"nbchan": nbchan, "pnts": pnts, "srate": 250.0, "data": "full.fdt"}
    savemat(str(set_path), eeg, do_compression=False)
    assert _repair_eeglab_fdt(set_path) is False


def test_repair_eeglab_fdt_fdt_too_small_returns_false(tmp_path):
    """.fdt too small for nbchan (actual_pnts <= 0) returns False."""
    pytest.importorskip("scipy")

    set_path = tmp_path / "tiny.set"
    fdt_path = tmp_path / "tiny.fdt"
    fdt_path.write_bytes(b"\x00" * 4)  # 4 bytes, 2 channels -> 0 samples
    eeg = {"nbchan": 2, "pnts": 100, "srate": 250.0, "data": "tiny.fdt"}
    savemat(str(set_path), eeg, do_compression=False)
    assert _repair_eeglab_fdt(set_path) is False


def test_load_raw_eeglab_alleeg_no_eeg_raises(tmp_path):
    """Nonexistent or invalid .set raises ValueError."""
    with pytest.raises(ValueError, match="No EEG or ALLEEG"):
        _load_raw_eeglab_alleeg(tmp_path / "missing.set")


def test_load_raw_eeglab_alleeg_success(tmp_path):
    """Minimal .set + .fdt loads to MNE Raw."""
    pytest.importorskip("scipy")

    set_path = tmp_path / "raw.set"
    fdt_path = tmp_path / "raw.fdt"
    nbchan, pnts = 2, 4
    fdt_path.write_bytes(b"\x00" * (nbchan * pnts * 4))
    eeg = {
        "nbchan": nbchan,
        "pnts": pnts,
        "srate": 250.0,
        "data": "raw.fdt",
        "chanlocs": [{"labels": "Fp1"}, {"labels": "Fp2"}],
    }
    savemat(str(set_path), eeg, do_compression=False)
    raw = _load_raw_eeglab_alleeg(set_path)
    assert raw is not None
    assert raw.info["nchan"] == nbchan
    assert raw.times.size == pnts


def test_load_raw_eeglab_alleeg_truncated_fdt_pads(tmp_path):
    """External .fdt smaller than header: padding branch is used."""
    pytest.importorskip("scipy")

    set_path = tmp_path / "trunc.set"
    fdt_path = tmp_path / "trunc.fdt"
    nbchan, pnts = 2, 8
    # Only 4 bytes (1 sample total) so padding is applied
    fdt_path.write_bytes(b"\x00" * 4)
    eeg = {
        "nbchan": nbchan,
        "pnts": pnts,
        "srate": 250.0,
        "data": "trunc.fdt",
        "chanlocs": [{"labels": "A"}, {"labels": "B"}],
    }
    savemat(str(set_path), eeg, do_compression=False)
    raw = _load_raw_eeglab_alleeg(set_path)
    assert raw is not None
    assert raw.info["nchan"] == nbchan
    assert raw.times.size == pnts


def test_load_raw_eeglab_alleeg_inline_data(tmp_path):
    """Inline data (array in .set) uses else branch."""
    pytest.importorskip("scipy")

    set_path = tmp_path / "inline.set"
    nbchan, pnts = 2, 4
    data = np.zeros((nbchan, pnts), order="F", dtype=np.float64)
    eeg = {
        "nbchan": nbchan,
        "pnts": pnts,
        "srate": 250.0,
        "data": data,
        "chanlocs": [{"labels": "Fp1"}, {"labels": "Fp2"}],
    }
    savemat(str(set_path), eeg, do_compression=False)
    raw = _load_raw_eeglab_alleeg(set_path)
    assert raw is not None
    assert raw.info["nchan"] == nbchan
    assert raw.times.size == pnts
