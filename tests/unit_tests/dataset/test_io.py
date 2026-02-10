from unittest.mock import patch

import pytest

from eegdash.dataset.io import (
    _ensure_coordsystem_symlink,
    _find_best_matching_file,
    _generate_vhdr_from_metadata,
    _generate_vmrk_stub,
    _repair_tsv_encoding,
    _repair_vhdr_pointers,
)


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
    (tmp_path / "channels.tsv").write_bytes("name\nFp1\n".encode("latin-1"))
    assert _repair_tsv_encoding(tmp_path) is True
