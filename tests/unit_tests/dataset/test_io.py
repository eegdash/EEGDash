from unittest.mock import patch

from eegdash.dataset.io import _ensure_coordsystem_symlink, _repair_vhdr_pointers


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
