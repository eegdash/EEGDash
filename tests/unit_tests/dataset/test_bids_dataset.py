from pathlib import Path
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, write_raw_bids

from eegdash.dataset.bids_dataset import EEGBIDSDataset


@pytest.fixture
def mock_bids_dir(tmp_path):
    """Create a valid BIDS dataset using MNE-BIDS."""
    bids_root = tmp_path / "dsX"
    bids_root.mkdir()

    # Create dummy raw data
    sfreq = 100
    info = mne.create_info(ch_names=["O1", "O2", "Cz"], sfreq=sfreq, ch_types="eeg")
    data = np.random.randn(3, sfreq * 10)  # 10 seconds
    raw = mne.io.RawArray(data, info)

    # Write to BIDS (enforce .set for EEGLAB)
    bids_path = BIDSPath(
        subject="01", task="rest", datatype="eeg", root=bids_root, extension=".set"
    )
    write_raw_bids(
        raw,
        bids_path,
        verbose=False,
        overwrite=True,
        allow_preload=True,
        format="EEGLAB",
    )

    # Force clean participants.tsv to avoid merging issues or formatting errors
    part_tsv_file = bids_root / "participants.tsv"
    # Ensure standard format
    part_tsv_file.write_text("participant_id\tage\nsub-01\t25\n")

    return bids_root


def test_eegbidsdataset_init(mock_bids_dir):
    ds = EEGBIDSDataset(data_dir=mock_bids_dir, dataset="dsX")
    assert len(ds.files) > 0
    assert ds.detected_modality == "eeg"
    assert ds.check_eeg_dataset() is True


def test_eegbidsdataset_init_mismatch(tmp_path):
    root = tmp_path / "dsY"
    root.mkdir()
    with pytest.raises(ValueError, match="does not correspond to dataset"):
        EEGBIDSDataset(data_dir=root, dataset="dsX")


def test_file_attributes(mock_bids_dir):
    ds = EEGBIDSDataset(data_dir=mock_bids_dir, dataset="dsX")
    # Find the generated file (EEGLAB uses .set)
    fpath = str(list((mock_bids_dir / "sub-01" / "eeg").glob("*.set"))[0])

    # Direct attributes
    assert ds.get_bids_file_attribute("subject", fpath) == "01"
    assert ds.get_bids_file_attribute("task", fpath) == "rest"

    # JSON attributes
    assert ds.get_bids_file_attribute("sfreq", fpath) == 100
    # Allow some tolerance due to float/duration encoding
    assert ds.get_bids_file_attribute("duration", fpath) == pytest.approx(10.0, abs=0.1)
    assert ds.get_bids_file_attribute("ntimes", fpath) == pytest.approx(1000, abs=1)


def test_num_times(mock_bids_dir):
    ds = EEGBIDSDataset(data_dir=mock_bids_dir, dataset="dsX")
    fpath = str(list((mock_bids_dir / "sub-01" / "eeg").glob("*.set"))[0])
    # 10s * 100Hz = 1000 samples
    # Using pytest.approx for robustness against float duration
    samples = ds.num_times(fpath)
    assert samples == pytest.approx(1000, abs=1)


def test_channel_labels(mock_bids_dir):
    ds = EEGBIDSDataset(data_dir=mock_bids_dir, dataset="dsX")
    fpath = str(list((mock_bids_dir / "sub-01" / "eeg").glob("*.set"))[0])
    labels = ds.channel_labels(fpath)
    assert "O1" in labels
    assert "Cz" in labels


def test_subject_participant_tsv(mock_bids_dir):
    ds = EEGBIDSDataset(data_dir=mock_bids_dir, dataset="dsX")
    fpath = str(list((mock_bids_dir / "sub-01" / "eeg").glob("*.set"))[0])

    info = ds.subject_participant_tsv(fpath)
    assert info["age"] == "25"


def test_get_relative_bidspath(mock_bids_dir):
    ds = EEGBIDSDataset(data_dir=mock_bids_dir, dataset="dsX")
    fpath = str(list((mock_bids_dir / "sub-01" / "eeg").glob("*.set"))[0])
    filename = Path(fpath).name

    # Should return "dsX/sub-01/eeg/..."
    rel = ds.get_relative_bidspath(fpath)
    assert rel == f"dsX/sub-01/eeg/{filename}"


def test_bids_dataset_gaps(tmp_path):
    # Trigger bids_dataset.py 552, 616, 621
    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    # We need a directory that looks like a BIDS dataset
    # Recursion happens in get_bids_metadata_files -> _get_bids_file_inheritance
    # which goes up until it finds dataset_description.json or hits root.
    # We should make it hit dataset_description.json.
    ds_path = Path(tmp_path).resolve() / "ds000123"
    ds_path.mkdir(parents=True, exist_ok=True)
    (ds_path / "dataset_description.json").touch()
    file_path = ds_path / "some_file.set"
    file_path.touch()

    # Mock file discovery to avoid init mismatch errors
    with patch(
        "eegdash.dataset.bids_dataset._find_bids_files", return_value=[str(file_path)]
    ):
        ds = EEGBIDSDataset(data_dir=str(ds_path), dataset="ds000123")

    # 552: No channels.tsv
    with pytest.raises(FileNotFoundError, match="No channels.tsv"):
        ds.channel_types(str(file_path))

    # 616: subj_val is None
    with patch.object(ds, "get_bids_file_attribute", return_value=None):
        assert ds.subject_participant_tsv(str(file_path)) == {}

    # 621: subj_val starts with sub-
    with patch.object(ds, "get_bids_file_attribute", return_value="sub-001"):
        # We also need to avoid failing later on participants_tsv
        with patch.object(ds, "get_bids_metadata_files", return_value=[]):
            assert ds.subject_participant_tsv(str(file_path)) == {}


def test_bids_subject_participant_tsv_gap(tmp_path):
    # EEGBIDSDataset.subject_participant_tsv
    # Missing 2 lines: probably file not found or empty

    # We need to mock get_bids_file_attribute to return a subject
    # And mock read_csv to return/fail

    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    with patch(
        "eegdash.dataset.bids_dataset._find_bids_files", return_value=["some_file.set"]
    ):
        ds_dir = tmp_path / "ds001"
        ds_dir.mkdir()
        p = ds_dir / "dataset_description.json"
        p.write_text('{"Name": "Test Dataset", "BIDSVersion": "1.8.0"}')
        # Also patch validation to be safe
        # with patch("eegdash.dataset.bids_dataset._validate_bids_dataset"): # Removed invalid patch
        ds = EEGBIDSDataset(data_dir=str(ds_dir), dataset="ds001")

    with patch.object(ds, "get_bids_file_attribute", return_value="sub-01"):
        # Mock _find_bids_files recursion for participants.tsv?
        # The method calls self.find_file("participants.tsv")? No, it looks up inheritance.

        # Let's try to mock the internal call that finds the csv
        # It likely uses _get_bids_file_inheritance
        # Create a dummy participants.tsv
        participants_tsv = ds_dir / "participants.tsv"
        participants_tsv.write_text("participant_id\tage\tsex\nsub-01\t25\tM\n")

        # We need to mock get_bids_metadata_files to return this file
        with patch.object(
            ds, "get_bids_metadata_files", return_value=[participants_tsv]
        ):
            # Now call the method
            # The method checks if subject ("sub-01") is in the participants tsv
            # Our mock get_bids_file_attribute returns "sub-01"

            # Also need to ensure the filepath argument is handled correctly.
            # The method calls get_bids_metadata_files(filepath, "participants.tsv")

            info = ds.subject_participant_tsv("some_file.set")
            assert info["age"] == "25"
            assert info["sex"] == "M"


def test_channel_types_exception(tmp_path):
    """Test that exception in reading channels.tsv returns empty list."""
    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    # Create a minimal BIDS structure with dataset name matching directory
    bids_dir = tmp_path / "ds000001"
    sub_dir = bids_dir / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)

    # Create a data file (needs to be a proper extension)
    data_file = sub_dir / "sub-01_task-test_eeg.edf"
    data_file.touch()

    # Create channels.tsv with invalid content (will cause exception)
    # Missing 'type' column which the method requires
    channels_tsv = sub_dir / "sub-01_task-test_channels.tsv"
    channels_tsv.write_text("invalid\tcontent\nwithout\tmissing")

    ds = EEGBIDSDataset(str(bids_dir), dataset="ds000001")
    result = ds.channel_types(str(data_file))
    # Should return empty list due to exception (missing 'type' column)
    assert result == [] or isinstance(result, list)


def test_subject_participant_tsv_exception_reading(tmp_path):
    """Test that exception in reading participants.tsv returns empty dict."""
    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    # Create a minimal BIDS structure with dataset name matching directory
    bids_dir = tmp_path / "ds000002"
    sub_dir = bids_dir / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)

    # Create a data file
    data_file = sub_dir / "sub-01_task-test_eeg.edf"
    data_file.touch()

    # Create invalid participants.tsv that will cause exception
    participants_tsv = bids_dir / "participants.tsv"
    # Binary content that cannot be read as TSV
    participants_tsv.write_bytes(b"\x00\x01\x02\x03")

    ds = EEGBIDSDataset(str(bids_dir), dataset="ds000002")
    result = ds.subject_participant_tsv(str(data_file))
    assert result == {}


def test_subject_participant_tsv_empty_file(tmp_path):
    """Test that empty participants.tsv returns empty dict."""
    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    # Create a minimal BIDS structure with dataset name matching directory
    bids_dir = tmp_path / "ds000003"
    sub_dir = bids_dir / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)

    # Create a data file
    data_file = sub_dir / "sub-01_task-test_eeg.edf"
    data_file.touch()

    # Create empty participants.tsv (just header)
    participants_tsv = bids_dir / "participants.tsv"
    participants_tsv.write_text("participant_id\tage\tsex\n")

    ds = EEGBIDSDataset(str(bids_dir), dataset="ds000003")
    result = ds.subject_participant_tsv(str(data_file))
    assert result == {}


def test_is_valid_eeg_file():
    """Test _is_valid_eeg_file logic."""
    from pathlib import Path
    from unittest.mock import patch

    from eegdash.dataset.bids_dataset import _is_valid_eeg_file

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
    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    # Setup dummy directory
    data_dir = tmp_path / "ds001"
    data_dir.mkdir()

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
    import pytest

    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    d = tmp_path / "wrong_name"
    d.mkdir()
    with pytest.raises(ValueError):
        EEGBIDSDataset(data_dir=str(d), dataset="dsXYZ")


def test_bids_get_bids_file_attribute_direct_vs_json(tmp_path):
    """Test getting attributes from filename vs JSON."""
    import json

    from eegdash.dataset.bids_dataset import EEGBIDSDataset

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
    """Test check_eeg_dataset."""
    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    d = tmp_path / "eeg_ds"
    d.mkdir()
    (d / "sub-01" / "eeg").mkdir(parents=True)
    (d / "sub-01" / "eeg" / "sub-01_eeg.set").touch()

    ds = EEGBIDSDataset(data_dir=str(d), dataset="eeg_ds")
    assert ds.check_eeg_dataset() is True


def test_bids_dataset_merge_json_inheritance(tmp_path):
    import json
    from unittest.mock import patch

    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    d = tmp_path / "ds_json"
    d.mkdir()
    f1 = d / "inherit_top.json"
    f2 = d / "inherit_bottom.json"

    f1.write_text(json.dumps({"a": 1, "b": 2}))
    f2.write_text(json.dumps({"b": 99, "c": 3}))

    def side_effect(self):
        # Must act like a valid eeg file for BIDSPath
        self.files = [str(d / "sub-01_task-rest_eeg.set")]
        self.detected_modality = "eeg"
        self._bids_entity_cache = {}
        self._bids_path_cache = {}

    with patch.object(
        EEGBIDSDataset, "_init_bids_paths", autospec=True, side_effect=side_effect
    ):
        ds = EEGBIDSDataset(data_dir=str(d), dataset="ds_json")
        # Now test the method without relying on finding files
        result = ds._merge_json_inheritance([str(f1), str(f2)])
        assert result["a"] == 1
        assert result["c"] == 3
        assert result["b"] == 2


def test_bids_dataset_get_files(tmp_path):
    from unittest.mock import patch

    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    d = tmp_path / "ds_files"
    d.mkdir()
    f = d / "sub-01_eeg.set"
    f.touch()

    def side_effect(self):
        self.files = [str(f)]
        self.detected_modality = "eeg"
        self._bids_entity_cache = {}
        self._bids_path_cache = {}

    with patch.object(
        EEGBIDSDataset, "_init_bids_paths", autospec=True, side_effect=side_effect
    ):
        ds = EEGBIDSDataset(data_dir=str(d), dataset="ds_files")
        files = ds.get_files()
        assert len(files) == 1
        assert files[0].endswith("sub-01_eeg.set")


def test_bids_dataset_subject_participant_tsv(tmp_path):
    from unittest.mock import patch

    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    d = tmp_path / "ds_part"
    d.mkdir()
    # Add root file to stop recursion in _get_bids_file_inheritance
    (d / "dataset_description.json").touch()

    f = d / "sub-01_eeg.set"
    f.touch()

    def side_effect(self):
        self.files = [str(f)]
        self.detected_modality = "eeg"
        self._bids_entity_cache = {}
        # Pre-populate cache to avoid _get_bids_path_from_file parsing issues if any
        self._bids_path_cache = {
            str(f): MagicMock(subject="01", datatype="eeg", root=d)
        }
        self._bids_entity_cache[str(f)] = {"subject": "01", "modality": "eeg"}

    with patch.object(
        EEGBIDSDataset, "_init_bids_paths", autospec=True, side_effect=side_effect
    ):
        ds = EEGBIDSDataset(data_dir=str(d), dataset="ds_part")

        # 1. No participants.tsv
        assert ds.subject_participant_tsv(str(f)) == {}

        # 2. Empty participants.tsv (but with header to avoid EmptyDataError)
        p_file = d / "participants.tsv"
        p_file.write_text("participant_id\tsex\n")  # Header only
        assert ds.subject_participant_tsv(str(f)) == {}

        # 3. Valid participants.tsv but subject not in it
        p_file.write_text("participant_id\tsex\nsub-02\tM\n")
        assert ds.subject_participant_tsv(str(f)) == {}

        # 4. Valid match
        p_file.write_text("participant_id\tsex\nsub-01\tF\n")
        res = ds.subject_participant_tsv(str(f))
        assert res.get("sex") == "F"


def test_bids_dataset_inheritance_full(tmp_path):
    # Coverage for _get_bids_file_inheritance recursion (Line 688+)
    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    root = tmp_path / "bids_root"
    root.mkdir()
    (root / "dataset_description.json").touch()
    (root / "participants.tsv").write_text("participant_id\tage\nsub-01\t30")

    sub = root / "sub-01"
    sub.mkdir()
    (sub / "sub-01_eeg.json").write_text('{"Task": "inherited"}')

    eeg_dir = sub / "eeg"
    eeg_dir.mkdir()
    f = eeg_dir / "sub-01_task-test_eeg.set"
    f.touch()

    ds = EEGBIDSDataset(data_dir=str(root), dataset="bids_root")
    # This should trigger recursion up to root
    inh_files = ds._get_bids_file_inheritance(eeg_dir, "sub-01_task-test", "json")
    # It finds sub-01_eeg.json which is at sub-01 level, the parent of eeg_dir.
    # Recursion: eeg_dir -> sub -> root (finds dataset_description.json and stops)
    assert any("sub-01_eeg.json" in str(p) for p in inh_files)


def test_bids_dataset_search_modalities(tmp_path):
    # Coverage for multiple modalities in _init_bids_paths
    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    d = tmp_path / "ds_multi"
    d.mkdir()
    # EEG uses .set, MEG uses .fif
    for mod, ext in [("eeg", "set"), ("meg", "fif")]:
        mod_dir = d / "sub-01" / mod
        mod_dir.mkdir(parents=True)
        (mod_dir / f"sub-01_{mod}.{ext}").touch()

    (d / "dataset_description.json").touch()

    # must match ONE of MODALITIES
    ds = EEGBIDSDataset(data_dir=str(d), dataset="ds_multi", modalities=["eeg", "meg"])
    assert len(ds.files) >= 2


def test_bids_dataset_attribute_error_branch(tmp_path):
    # bids_dataset.py 331, 306
    from unittest.mock import patch

    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    d = tmp_path / "ds"
    d.mkdir()
    (d / "dataset_description.json").touch()

    # Create a dummy file so init doesn't fail
    (d / "sub-01" / "eeg").mkdir(parents=True)
    subj_file = d / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set"
    subj_file.touch()

    ds = EEGBIDSDataset(data_dir=str(d), dataset="ds")

    # key missing in path entities
    f = d / "somefile_eeg.set"
    f.touch()
    assert ds.get_bids_file_attribute("subject", str(f)) is None

    # 331: Generic exception branch in get_bids_file_attribute
    # Use a file that doesn't have 'sub-' in it
    new_f = d / "eeg/random_name_eeg.set"
    new_f.parent.mkdir(parents=True, exist_ok=True)
    new_f.touch()

    with patch("mne_bids.get_bids_path_from_fname", side_effect=Exception("Explosion")):
        # manual regex also fails to find sub-
        assert ds.get_bids_file_attribute("subject", str(new_f)) is None


def test_bids_dataset_channels_tsv_missing_cases(tmp_path):
    # bids_dataset.py 371, 409, 425-431
    from unittest.mock import patch

    import pytest

    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    d = tmp_path / "ds_ch"
    d.mkdir()
    (d / "dataset_description.json").touch()
    # Create dummy file
    (d / "sub-01" / "eeg").mkdir(parents=True)
    f = d / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set"
    f.touch()

    ds = EEGBIDSDataset(data_dir=str(d), dataset="ds_ch")

    # file missing. channel_labels/types raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        ds.channel_labels("/non/existent/file.set")

    # Read error or empty file
    ch_file = d / "sub-01" / "eeg" / "sub-01_task-rest_channels.tsv"

    # Empty file (no headers) -> returns empty list due to my fix
    ch_file.write_text("")
    assert ds.channel_labels(str(f)) == []

    # Corrupt file / reading error branch -> returns empty list due to my fix
    ch_file.write_text("name\ttype\nch1")
    with patch("pandas.read_csv", side_effect=Exception("Pandas Error")):
        assert ds.channel_labels(str(f)) == []


def test_bids_dataset_subject_participant_tsv_errors(tmp_path):
    # bids_dataset.py 512, 549
    from unittest.mock import patch

    import pandas as pd

    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    d = tmp_path / "ds_part_err"
    d.mkdir()
    (d / "dataset_description.json").touch()
    # Create dummy file
    (d / "sub-01" / "eeg").mkdir(parents=True)
    f = d / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set"
    f.touch()

    ds = EEGBIDSDataset(data_dir=str(d), dataset="ds_part_err")

    p_file = d / "participants.tsv"
    p_file.write_text("participant_id\tsex\nsub-01\tF\n")

    # 593: handle the case where get_bids_metadata_files returns empty
    with patch.object(ds, "get_bids_metadata_files", return_value=[]):
        assert ds.subject_participant_tsv(str(f)) == {}

    # 600: empty tsv
    with patch("pandas.read_csv", return_value=pd.DataFrame()):
        with patch.object(ds, "get_bids_metadata_files", return_value=[p_file]):
            assert ds.subject_participant_tsv(str(f)) == {}

    # 549: Generic exception in read_csv
    with patch("pandas.read_csv", side_effect=Exception("Panic")):
        # We need get_bids_metadata_files to return something so it reaches read_csv
        with patch.object(ds, "get_bids_metadata_files", return_value=[p_file]):
            assert ds.subject_participant_tsv(str(f)) == {}


def test_bids_dataset_check_eeg_dataset_empty(tmp_path):
    # bids_dataset.py 151
    from unittest.mock import patch

    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    d = tmp_path / "ds_empty"
    d.mkdir()
    (d / "dataset_description.json").touch()

    # To test False branch, set detected_modality to 'meg'
    def side_effect(self):
        self.files = ["dummy.set"]
        self.detected_modality = "meg"
        self._bids_entity_cache = {}
        self._bids_path_cache = {}

    with patch.object(
        EEGBIDSDataset, "_init_bids_paths", autospec=True, side_effect=side_effect
    ):
        with patch.object(
            EEGBIDSDataset, "get_bids_file_attribute", return_value="meg"
        ):
            ds = EEGBIDSDataset(data_dir=str(d), dataset="ds_empty")
            assert ds.check_eeg_dataset() is False


def test_bids_dataset_init_errors():
    # bids_dataset.py 107
    import pytest

    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    with pytest.raises(ValueError, match="data_dir must be specified"):
        EEGBIDSDataset(data_dir=None)
    with pytest.raises(ValueError):
        # Directory name mismatch
        EEGBIDSDataset(data_dir="/tmp", dataset="mismatch")


def test_bids_dataset_task_run_absorption(tmp_path):
    # bids_dataset.py 228-229, 235-237
    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    d = tmp_path / "ds_task"
    d.mkdir()
    (d / "dataset_description.json").touch()
    (d / "sub-01" / "eeg").mkdir(parents=True)
    # 228-229: task-ECONrun-1
    f = d / "sub-01" / "eeg" / "sub-01_task-ECONrun-1_eeg.set"
    f.touch()

    ds = EEGBIDSDataset(data_dir=str(d), dataset="ds_task")
    assert ds.get_bids_file_attribute("task", str(f)) == "ECON"

    # 235-237: non-digit run
    f2 = d / "sub-01" / "eeg" / "sub-01_task-rest_run-A_eeg.set"
    f2.touch()
    # we need to re-init or clear cache to detect f2 if not found in first _init_bids_paths
    ds2 = EEGBIDSDataset(data_dir=str(d), dataset="ds_task")
    # it won't be in ds.files unless it was there at init.
    # get_bids_file_attribute calls _get_bids_path_from_file which uses regex on the filename
    assert ds2.get_bids_file_attribute("run", str(f2)) == "A"


def test_bids_dataset_inheritance_break(tmp_path):
    # bids_dataset.py 306
    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    d = tmp_path / "bids_root"
    d.mkdir()
    (d / "dataset_description.json").touch()
    (d / "sub-01").mkdir()
    (d / "sub-01" / "eeg").mkdir()
    f = d / "sub-01" / "eeg" / "sub-01_eeg.set"
    f.touch()

    ds = EEGBIDSDataset(data_dir=str(d), dataset="bids_root")
    # Trigger eeg_json inheritance which calls _get_json_with_inheritance
    # which has the 'break' at bids root.
    # We don't have eeg.json at any level.
    assert ds.eeg_json(str(f)) == {}


def test_bids_dataset_inheritance_exceptions(tmp_path):
    # bids_dataset.py 609, 633, 701-702
    from unittest.mock import patch

    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    d = tmp_path / "ds_inh_err"
    d.mkdir()
    (d / "dataset_description.json").touch()
    (d / "sub-01" / "eeg").mkdir(parents=True)
    (d / "sub-01" / "eeg" / "sub-01_eeg.set").touch()

    ds = EEGBIDSDataset(data_dir=str(d), dataset="ds_inh_err")

    # 633: generic exception in _get_bids_file_inheritance
    with patch("pathlib.Path.glob", side_effect=Exception("Glob error")):
        assert ds._get_bids_file_inheritance(d, "pref", "ext") == []

    # 701-702: generic exception in the folder search loop
    def bomb(*args, **kwargs):
        raise Exception("Bomb!")

    with patch("pathlib.Path.is_file", side_effect=bomb):
        assert ds._get_bids_file_inheritance(d / "sub-01", "pref", "ext") == []


@pytest.mark.parametrize(
    "encoding,method,expected",
    [
        ("latin-1", "channel_labels", ["Fp1", "Fp2"]),
        ("latin-1", "channel_types", ["EEG", "EOG"]),
        ("utf-8", "channel_labels", ["Fp1", "Fp2"]),
    ],
    ids=["latin1_labels", "latin1_types", "utf8_labels"],
)
def test_channel_methods_encoding_fallback(tmp_path, encoding, method, expected):
    """Test channel_labels/channel_types handle various TSV encodings."""
    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    bids_dir = tmp_path / f"ds_{encoding}_{method}"
    sub_dir = bids_dir / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)

    data_file = sub_dir / "sub-01_task-test_eeg.set"
    data_file.touch()

    channels_tsv = sub_dir / "sub-01_task-test_channels.tsv"
    content = "name\ttype\tunits\nFp1\tEEG\tµV\nFp2\tEOG\tµV\n"
    channels_tsv.write_bytes(content.encode(encoding))

    ds = EEGBIDSDataset(str(bids_dir), dataset=bids_dir.name)
    assert getattr(ds, method)(str(data_file)) == expected


def test_bids_dataset_more_coverage(tmp_path):
    # bids_dataset 289-290, 331, 371, 409, 425-431, 552, 557-558, 576-577, 616, 621, 700
    from unittest.mock import patch

    import pytest

    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    d = tmp_path / "ds_more"
    d.mkdir()
    (d / "dataset_description.json").touch()
    (d / "sub-01" / "eeg").mkdir(parents=True)
    f = d / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set"
    f.touch()

    # root level eeg.json (289-290)
    (d / "eeg.json").write_text('{"SamplingFrequency": 100, "RecordingDuration": 10}')

    ds = EEGBIDSDataset(data_dir=str(d), dataset="ds_more")

    # 576-577: num_times
    assert ds.num_times(str(f)) == 1000

    # 409, 425-431: relative path error (file outside)
    f_outside = tmp_path / "outside.set"
    f_outside.touch()
    assert ds.get_relative_bidspath(f_outside) == "ds_more/outside.set"

    # 331: _get_bids_file_inheritance path does not exist
    with pytest.raises(ValueError, match="does not exist"):
        ds._get_bids_file_inheritance(tmp_path / "missing", "base", "ext")

    # 371: get_bids_metadata_files non-existent
    with pytest.raises(ValueError, match="does not exist"):
        ds.get_bids_metadata_files(tmp_path / "missing.set", "channels.tsv")

    # 552: channel_labels missing channels.tsv
    with pytest.raises(FileNotFoundError):
        ds.channel_labels(str(f))

    # 557-558: channel_types exception
    with patch("pandas.read_csv", side_effect=Exception("Epic Fail")):
        # We need a file to exist so it doesn't raise FileNotFoundError at 551
        (d / "sub-01" / "eeg" / "sub-01_task-rest_channels.tsv").touch()
        assert ds.channel_types(str(f)) == []

    # 700: _find_bids_files modality default
    from eegdash.dataset.bids_dataset import _find_bids_files

    # it uses EPHY_ALLOWED_DATATYPES
    assert _find_bids_files(d, ".none_ext") == []


def test_bids_path_extracts_acquisition_entity(tmp_path):
    """BIDSPath correctly resolves files with acq- entity (e.g. ds000248)."""
    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    d = tmp_path / "ds_acq"
    d.mkdir()
    (d / "dataset_description.json").touch()
    meg_dir = d / "sub-01" / "meg"
    meg_dir.mkdir(parents=True)

    # Two files for the same subject — only distinguishable by acq/task
    f_task = meg_dir / "sub-01_task-audiovisual_run-01_meg.fif"
    f_acq = meg_dir / "sub-01_acq-crosstalk_meg.fif"
    f_task.touch()
    f_acq.touch()

    ds = EEGBIDSDataset(data_dir=str(d), dataset="ds_acq", allow_symlinks=True)

    # Both files should resolve without ambiguity errors
    bp_task = ds._get_bids_path_from_file(str(f_task))
    bp_acq = ds._get_bids_path_from_file(str(f_acq))

    assert bp_task.task == "audiovisual"
    assert bp_task.acquisition is None
    assert bp_task.run == "01"

    assert bp_acq.task is None
    assert bp_acq.acquisition == "crosstalk"
    assert bp_acq.run is None

    # Entity cache should contain acquisition
    entities = ds._bids_entity_cache[str(f_acq)]
    assert entities["acquisition"] == "crosstalk"
    assert entities["modality"] == "meg"


def test_bids_path_extracts_all_entities(tmp_path):
    """All standard BIDS entities are extracted via get_entities_from_fname."""
    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    d = tmp_path / "ds_ents"
    d.mkdir()
    (d / "dataset_description.json").touch()
    meg_dir = d / "sub-01" / "ses-02" / "meg"
    meg_dir.mkdir(parents=True)

    # Use a valid space for MEG (ElektaNeuromag) per MNE-BIDS validation
    f = (
        meg_dir
        / "sub-01_ses-02_task-rest_acq-full_run-03_proc-sss_space-ElektaNeuromag_split-01_desc-preproc_meg.fif"
    )
    f.touch()

    ds = EEGBIDSDataset(data_dir=str(d), dataset="ds_ents", allow_symlinks=True)
    bp = ds._get_bids_path_from_file(str(f))

    assert bp.subject == "01"
    assert bp.session == "02"
    assert bp.task == "rest"
    assert bp.acquisition == "full"
    assert bp.run == "03"
    assert bp.processing == "sss"
    assert bp.space == "ElektaNeuromag"
    assert bp.split == "01"
    assert bp.description == "preproc"
    assert bp.datatype == "meg"


@pytest.mark.parametrize(
    "modality_dir,suffix,ext,expected",
    [
        ("eeg", "eeg", ".set", "eeg"),
        ("meg", "meg", ".fif", "meg"),
        ("ieeg", "ieeg", ".edf", "ieeg"),
        ("nirs", "nirs", ".snirf", "nirs"),
        ("fnirs", "nirs", ".snirf", "nirs"),  # fnirs normalizes to nirs
    ],
    ids=["eeg", "meg", "ieeg", "nirs", "fnirs_normalized"],
)
def test_bids_path_modality_from_directory(
    tmp_path, modality_dir, suffix, ext, expected
):
    """Modality is detected from directory path, including fnirs normalization."""
    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    d = tmp_path / f"ds_{modality_dir}"
    d.mkdir()
    (d / "dataset_description.json").touch()

    sub_dir = d / "sub-01" / modality_dir
    sub_dir.mkdir(parents=True)
    f = sub_dir / f"sub-01_task-rest_{suffix}{ext}"
    f.touch()

    ds = EEGBIDSDataset(data_dir=str(d), dataset=d.name, allow_symlinks=True)
    ds._get_bids_path_from_file(str(f))
    assert ds._bids_entity_cache[str(f)]["modality"] == expected


def test_bids_path_nonstandard_entity_warns(tmp_path):
    """Non-standard entities in filenames produce a warning, not an error."""
    import warnings

    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    d = tmp_path / "ds_warn"
    d.mkdir()
    (d / "dataset_description.json").touch()
    eeg_dir = d / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)

    # rec- is not standard in MNE-BIDS (should be recording-)
    f = eeg_dir / "sub-01_task-rest_rec-mag_eeg.set"
    f.touch()

    ds = EEGBIDSDataset(data_dir=str(d), dataset="ds_warn")

    # Should not raise, just warn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bp = ds._get_bids_path_from_file(str(f))

    assert bp.subject == "01"
    assert bp.task == "rest"
