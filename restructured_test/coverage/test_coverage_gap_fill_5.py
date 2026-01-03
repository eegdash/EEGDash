from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from eegdash import schemas
from eegdash.dataset.bids_dataset import EEGBIDSDataset


def test_create_dataset_full():
    # Coverage for create_dataset (schemas.py 364-454)
    # 1. basic
    ds = schemas.create_dataset(dataset_id="ds1", name="Test DS", source="openneuro")
    assert ds["dataset_id"] == "ds1"
    assert ds["name"] == "Test DS"

    # 2. full fields
    ds_full = schemas.create_dataset(
        dataset_id="ds2",
        ages=[20, 30, 40],
        age_mean=30.0,
        species="Human",
        is_clinical=True,
        clinical_purpose="epilepsy",
        paradigm_modality="resting_state",
        cognitive_domain="memory",
        is_10_20_system=True,
        source_url="http://ds.org",
        stars=10,
        authors=["Author A"],
        funding=["Grant 1"],
        modalities=["eeg"],
    )
    assert ds_full["demographics"]["age_min"] == 20
    assert ds_full["demographics"]["age_max"] == 40
    assert ds_full["clinical"]["is_clinical"] is True
    assert ds_full["paradigm"]["modality"] == "resting_state"
    assert ds_full["external_links"]["source_url"] == "http://ds.org"
    assert ds_full["repository_stats"]["stars"] == 10

    # 3. Validation
    with pytest.raises(ValueError, match="dataset_id is required"):
        schemas.create_dataset(dataset_id="")


def test_validate_dataset():
    # schemas.py 651-656
    assert "missing: dataset_id" in schemas.validate_dataset({})
    assert not schemas.validate_dataset({"dataset_id": "ds1"})


def test_sanitize_run_edge_cases():
    # schemas.py 511-515
    assert schemas._sanitize_run_for_mne("  ") is None
    assert schemas._sanitize_run_for_mne("01") == "01"
    assert schemas._sanitize_run_for_mne("run1") is None


def test_create_record_errors():
    # schemas.py 584-588
    with pytest.raises(ValueError, match="dataset is required"):
        schemas.create_record(dataset="", storage_base="s3://b", bids_relpath="f.set")
    with pytest.raises(ValueError, match="storage_base is required"):
        schemas.create_record(dataset="ds", storage_base="", bids_relpath="f.set")
    with pytest.raises(ValueError, match="bids_relpath is required"):
        schemas.create_record(dataset="ds", storage_base="s3://b", bids_relpath="")


def test_manifest_model_coverage():
    # schemas.py 74
    m = schemas.ManifestFileModel(path=" p ")
    assert m.path_or_name() == "p"
    m2 = schemas.ManifestFileModel(name=" n ")
    assert m2.path_or_name() == "n"
    m3 = schemas.ManifestFileModel()
    assert m3.path_or_name() == ""


def test_bids_dataset_attribute_error_branch(tmp_path):
    # bids_dataset.py 331, 306
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
    with pytest.raises(ValueError, match="data_dir must be specified"):
        EEGBIDSDataset(data_dir=None)
    with pytest.raises(AssertionError):
        # Assertions for directory name mismatch
        EEGBIDSDataset(data_dir="/tmp", dataset="mismatch")


def test_bids_dataset_task_run_absorption(tmp_path):
    # bids_dataset.py 228-229, 235-237
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


def test_dataset_warning_fallback(tmp_path):
    # dataset.py 247-250 (Console fallback)
    from unittest.mock import patch

    from rich.console import Console

    from eegdash.dataset.dataset import EEGDashDataset

    data_dir = tmp_path / "ds000001"
    data_dir.mkdir()

    with patch.object(Console, "print", side_effect=Exception("Rich fail")):
        with patch("eegdash.dataset.dataset.logger.warning") as mock_log:
            with patch(
                "eegdash.dataset.dataset.RELEASE_TO_OPENNEURO_DATASET_MAP",
                {"R1": "ds000001"},
            ):
                # This will raise ValueError: No datasets found... but we check the log first
                with pytest.raises(ValueError):
                    EEGDashDataset(
                        dataset="ds000001", download=False, cache_dir=str(tmp_path)
                    )
                assert mock_log.called


def test_dataset_recursive_search(tmp_path):
    # dataset.py 446

    from eegdash.dataset.dataset import EEGDashDataset

    data = {"a": {"b_c": 1}}

    ds_dir = tmp_path / "ds"
    ds_dir.mkdir()
    (ds_dir / "dataset_description.json").touch()
    (ds_dir / "sub-01" / "eeg").mkdir(parents=True)
    (ds_dir / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set").touch()

    # In download=False mode, it calls _find_local_bids_records which calls discover_local_bids_records
    real_ds = EEGDashDataset(
        dataset="ds",
        download=False,
        _suppress_comp_warning=True,
        cache_dir=str(tmp_path),
    )
    assert real_ds._find_key_in_nested_dict(data, "b-c") == 1
    assert real_ds._find_key_in_nested_dict([{"x": 2}], "x") == 2


def test_bids_dataset_inheritance_break(tmp_path):
    # bids_dataset.py 306
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


def test_dataset_init_error_no_args(tmp_path):
    # dataset.py 216 and 332
    from eegdash.dataset.dataset import EEGDashDataset

    # Case 1: missing dataset -> 216
    with pytest.raises(ValueError, match="You must provide a 'dataset' argument"):
        EEGDashDataset(cache_dir=str(tmp_path))

    # Case 2: dataset exists but no results -> 332
    mock_client = MagicMock()
    mock_client.find.return_value = []
    with pytest.raises(ValueError, match="No datasets found matching the query"):
        EEGDashDataset(
            dataset="ds", cache_dir=str(tmp_path), eeg_dash_instance=mock_client
        )


def test_schemas_run_digit_coverage():
    # schemas.py 515
    assert schemas._sanitize_run_for_mne("A123") is None
    assert schemas._sanitize_run_for_mne("01") == "01"


def test_bids_dataset_more_coverage(tmp_path):
    # bids_dataset 289-290, 331, 371, 409, 425-431, 552, 557-558, 576-577, 616, 621, 700
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


def test_dataset_init_cache_defaults(tmp_path):
    # dataset.py 181-182, 189-192
    from eegdash.dataset.dataset import EEGDashDataset

    with patch("eegdash.api.get_client"):
        with patch(
            "eegdash.dataset.dataset.get_default_cache_dir",
            return_value=str(tmp_path / "def_cache"),
        ):
            # Ensure the data_dir for the dataset exists so it doesn't fail offline check
            (tmp_path / "def_cache" / "ds").mkdir(parents=True, exist_ok=True)
            # 181: cache_dir is None
            record = {
                "dataset": "ds",
                "bids_relpath": "f.set",
                "bidspath": "ds/f.set",
                "storage": {"base": "s3", "backend": "s3"},
                "schema_version": 2,
            }
            with patch.object(
                EEGDashDataset, "_find_local_bids_records", return_value=[record]
            ):
                EEGDashDataset(dataset="ds", cache_dir=None, download=False)
                assert (tmp_path / "def_cache").exists()  # 192 hit


def test_dataset_record_inference(tmp_path):
    # dataset.py 212
    from eegdash.dataset.dataset import EEGDashDataset

    records = [
        {
            "dataset": "ds_inf",
            "bids_relpath": "f.set",
            "bidspath": "ds/f.set",
            "storage": {"base": "s3", "backend": "s3"},
            "schema_version": 2,
        }
    ]
    with patch("eegdash.api.get_client"):
        ds = EEGDashDataset(records=records, download=False, cache_dir=str(tmp_path))
        assert ds.query["dataset"] == "ds_inf"


def test_dataset_dedupe_none_key(tmp_path):
    # dataset.py 362-363
    from eegdash.dataset.dataset import EEGDashDataset

    # record with no bids_relpath, bidspath, data_name
    record = {"schema_version": 2}
    with patch("eegdash.api.get_client"):
        # We need to mock EEGDashRaw to avoid validation error during datasets list comp
        with patch("eegdash.dataset.dataset.EEGDashRaw"):
            ds = EEGDashDataset(
                dataset="ds",
                records=[record],
                _dedupe_records=True,
                download=False,
                cache_dir=str(tmp_path),
            )
            assert len(ds.records) == 1


def test_dataset_download_all_coverage(tmp_path):
    # dataset.py 385, 390, 397
    from eegdash.dataset.dataset import EEGDashDataset

    with patch("eegdash.api.get_client"):
        # We need to mock _find_datasets to return something or it will raise ValueError
        with patch.object(EEGDashDataset, "_find_datasets", return_value=[]):
            # match with re.DOTALL implicitly or just part of it
            with pytest.raises(
                ValueError, match="No datasets found matching the query"
            ):
                EEGDashDataset(dataset="ds", download=True, cache_dir=str(tmp_path))

        # Manually create one so we can call download_all
        ds = MagicMock(spec=EEGDashDataset)
        ds.download = True
        ds.n_jobs = 1

        # 397: no targets
        ds.datasets = []
        EEGDashDataset.download_all(ds)

        # 385: default n_jobs (implicitly handled if targets present)
        mock_raw = MagicMock()
        mock_raw._raw_uri = "s3://..."
        mock_raw._dep_paths = []
        mock_raw.filecache.exists.return_value = False
        ds.datasets = [mock_raw]
        EEGDashDataset.download_all(ds, n_jobs=None)
        assert mock_raw._download_required_files.called

        # 390: _raw_uri is None
        mock_raw2 = MagicMock()
        mock_raw2._raw_uri = None
        ds.datasets = [mock_raw2]
        EEGDashDataset.download_all(ds)


def test_challenge_dataset_more_coverage(tmp_path):
    # dataset.py 664, 667-668, 716-718
    from rich.console import Console

    from eegdash.dataset.dataset import EEGChallengeDataset

    record = {
        "dataset": "EEG2025R1mini",
        "subject": "NDARAC904DMU",
        "bids_relpath": "f.set",
        "bidspath": "EEG2025R1mini/sub-1/f.set",
        "storage": {"base": "s3", "backend": "s3"},
        "schema_version": 2,
    }

    # 664: $in in query subject
    query = {"subject": {"$in": ["NDARAC904DMU"]}}
    with patch("eegdash.api.get_client") as mock_get:
        mock_client = MagicMock()
        mock_client.find.return_value = [record]
        mock_get.return_value = mock_client
        # We need release R1
        ds = EEGChallengeDataset(release="R1", cache_dir=str(tmp_path), query=query)
        assert "NDARAC904DMU" in ds.query["subject"]["$in"]

    # 667-668: qval is not None
    query2 = {"subject": "NDARAC904DMU"}
    with patch("eegdash.api.get_client") as mock_get:
        mock_client = MagicMock()
        mock_client.find.return_value = [record]
        mock_get.return_value = mock_client
        ds2 = EEGChallengeDataset(release="R1", cache_dir=str(tmp_path), query=query2)
        assert ds2.query["subject"] == "NDARAC904DMU"

    # 716-718: console fail
    with patch.object(Console, "print", side_effect=Exception("Rich dead")):
        with patch("eegdash.api.get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.find.return_value = [record]
            mock_get.return_value = mock_client
            EEGChallengeDataset(release="R2", cache_dir=str(tmp_path))
