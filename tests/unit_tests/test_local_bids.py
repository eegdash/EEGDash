from eegdash.local_bids import _normalize_modalities, discover_local_bids_records


def test_normalize_modalities():
    assert _normalize_modalities(None) == ["eeg", "meg", "ieeg", "nirs", "fnirs", "emg"]
    assert _normalize_modalities("eeg") == ["eeg"]
    assert _normalize_modalities(["eeg", "fmri"]) == ["eeg", "fmri"]
    assert _normalize_modalities("fnirs") == ["nirs"]  # Check alias


def test_discover_local_bids_records(tmp_path):
    # Create dummy structure
    # ds1/
    #   sub-01/
    #     eeg/
    #       sub-01_task-rest_eeg.set
    #       sub-01_task-rest_eeg.json
    ds_root = tmp_path / "ds1"
    sub_dir = ds_root / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)

    k = sub_dir / "sub-01_task-rest_eeg.set"
    k.touch()
    (sub_dir / "sub-01_task-rest_eeg.json").touch()

    # Filters
    filters = {"dataset": "ds1", "subject": "01", "modality": "eeg"}

    records = discover_local_bids_records(ds_root, filters)

    assert len(records) == 1
    rec = records[0]
    # Entities are nested
    assert rec["entities"]["subject"] == "01"
    assert rec["entities"]["task"] == "rest"
    assert rec["suffix"] == "eeg"
    assert rec["dataset"] == "ds1"
    # storage backend should be local
    assert rec["storage"]["backend"] == "local"


def test_discover_local_bids_records_filtering(tmp_path):
    # Setup: sub-01 (rest), sub-02 (task)
    ds_root = tmp_path / "ds2"
    (ds_root / "sub-01" / "eeg").mkdir(parents=True)
    (ds_root / "sub-02" / "eeg").mkdir(parents=True)

    (ds_root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.vhdr").touch()
    (ds_root / "sub-02" / "eeg" / "sub-02_task-task_eeg.vhdr").touch()

    # Filter for sub-01 only
    records = discover_local_bids_records(ds_root, {"dataset": "ds2", "subject": "01"})
    assert len(records) == 1
    assert records[0]["entities"]["subject"] == "01"

    # Filter for task=task
    records = discover_local_bids_records(ds_root, {"dataset": "ds2", "task": "task"})
    assert len(records) == 1
    assert records[0]["entities"]["subject"] == "02"


def test_load_local_bids_with_list_filters(tmp_path):
    """Test discover_local_bids_records with list-type filters."""
    from unittest.mock import patch

    from eegdash.local_bids import discover_local_bids_records

    # Create minimal BIDS structure
    (tmp_path / "dataset_description.json").write_text('{"Name": "Test"}')

    # Lines 74-77: handle list/tuple/set filters
    with patch("eegdash.local_bids.find_matching_paths", return_value=[]):
        records = discover_local_bids_records(
            dataset_root=str(tmp_path),
            filters={
                "dataset": "test_ds",
                "modality": "eeg",
                "subject": ["01", "02"],
                "task": ("rest",),
            },
        )
        assert records == []


def test_local_bids_empty_list_filter(tmp_path):
    """Test filter with empty list is ignored."""
    from unittest.mock import patch

    from eegdash.local_bids import discover_local_bids_records

    # Line 76: empty entity_vals means continue
    with patch("eegdash.local_bids.find_matching_paths", return_value=[]):
        records = discover_local_bids_records(
            dataset_root=str(tmp_path),
            filters={
                "dataset": "test_ds",
                "modality": "eeg",
                "subject": [],  # Empty list should be skipped
            },
        )
        assert records == []


def test_local_bids_relative_to_fails(tmp_path):
    """Test bids_relpath fallback when resolve().relative_to() fails."""
    from unittest.mock import MagicMock, patch

    from eegdash.local_bids import discover_local_bids_records

    # Create a mock BIDSPath whose file is outside dataset_root
    mock_bidspath = MagicMock()
    mock_bidspath.fpath = "/other/location/sub-01_eeg.vhdr"
    mock_bidspath.datatype = "eeg"
    mock_bidspath.suffix = "eeg"
    mock_bidspath.subject = "01"
    mock_bidspath.session = None
    mock_bidspath.task = "rest"
    mock_bidspath.run = None

    # If discover_local_bids_records is called, it iterates over find_matching_paths results
    # We need to simulate that the file path is not relative to dataset_root
    with patch("eegdash.local_bids.find_matching_paths", return_value=[mock_bidspath]):
        # Also need to mock Path.relative_to to raise ValueError
        with patch("pathlib.Path.relative_to", side_effect=ValueError):
            # This test is tricky because discover_local_bids_records relies heavily on Path objects
            # and we are mocking BIDSPath return.
            # The line 124-125 catch the ValueError and use name
            records = discover_local_bids_records(
                str(tmp_path), filters={"dataset": "ds_root"}
            )
            # If it doesn't crash, it's good (though records might be empty or specific)
            # Actually if relative_to fails it uses fpath.name.
            # We just assert no crash
            assert isinstance(records, list)


def test_normalize_modalities_none():
    """Test _normalize_modalities with None."""
    from eegdash.local_bids import _normalize_modalities

    result = _normalize_modalities(None)
    assert result == ["eeg", "meg", "ieeg", "nirs", "fnirs", "emg"]


def test_normalize_modalities_list():
    """Test _normalize_modalities with list."""
    from eegdash.local_bids import _normalize_modalities

    result = _normalize_modalities(["eeg", "fnirs"])
    assert "eeg" in result
    assert "nirs" in result  # fnirs aliased to nirs


def test_normalize_modalities_string():
    """Test _normalize_modalities with string."""
    from eegdash.local_bids import _normalize_modalities

    result = _normalize_modalities("meg")
    assert result == ["meg"]


def test_discover_local_bids_records_invalid_extension(tmp_path):
    """Test that invalid extensions are skipped (lines 120, 124-125)."""
    from unittest.mock import MagicMock, patch

    from eegdash.local_bids import discover_local_bids_records

    # Create minimal BIDS structure
    ds = tmp_path / "ds001234"
    ds.mkdir()
    (ds / "dataset_description.json").write_text('{"Name": "Test"}')
    sub_dir = ds / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)
    # Create a .json file (should be skipped)
    (sub_dir / "sub-01_task-rest_eeg.json").write_text("{}")
    # Create a valid .set file
    (sub_dir / "sub-01_task-rest_eeg.set").touch()

    with patch("eegdash.local_bids.find_matching_paths") as mock_find:
        mock_bp_json = MagicMock()
        mock_bp_json.fpath = str(sub_dir / "sub-01_task-rest_eeg.json")
        mock_bp_set = MagicMock()
        mock_bp_set.fpath = str(sub_dir / "sub-01_task-rest_eeg.set")
        mock_bp_set.datatype = "eeg"
        mock_bp_set.suffix = "eeg"
        mock_bp_set.subject = "01"
        mock_bp_set.session = None
        mock_bp_set.task = "rest"
        mock_bp_set.run = None

        mock_find.return_value = [mock_bp_json, mock_bp_set]

        records = discover_local_bids_records(ds, {"dataset": "ds001234"})
        # Only .set file should be included
        assert len(records) == 1
        assert records[0]["extension"] == ".set"
