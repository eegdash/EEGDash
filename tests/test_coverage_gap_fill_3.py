import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from eegdash import bids_metadata
from eegdash.dataset.bids_dataset import EEGBIDSDataset
from eegdash.dataset.dataset import EEGChallengeDataset
from eegdash.dataset.registry import register_openneuro_datasets

# --- EEGBIDSDataset Tests ---


def test_bids_dataset_merge_json_inheritance(tmp_path):
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


# ...


def test_challenge_dataset_subjects_logic(tmp_path):
    # Verify logic by mocking the low-level get_client used by EEGDash
    # eegdash.api imports get_client from .http_api_client

    with patch.dict(
        "eegdash.dataset.dataset.RELEASE_TO_OPENNEURO_DATASET_MAP", {"R1": "ds1"}
    ):
        with patch.dict(
            "eegdash.dataset.dataset.SUBJECT_MINI_RELEASE_MAP", {"R1": ["s1", "s2"]}
        ):
            # Patch get_client in eegdash.api to return a mock DB client
            with patch("eegdash.api.get_client") as mock_get_client:
                mock_db = mock_get_client.return_value
                # EEGDash.find calls self._client.find() and wraps it in list()
                # So we need to return an iterable (e.g. list) of records
                mock_db.find.return_value = [
                    {
                        "dataset": "ds1",
                        "data_name": "sub-01_task-rest_eeg.set",
                        "bidspath": "sub-01/eeg/sub-01_task-rest_eeg.set",
                        "storage": {
                            "backend": "s3",
                            "base": "s3://bucket",
                            "raw_key": "k",
                        },
                        "subject": "s1",
                        "session": "1",
                        "run": "1",
                        "task": "rest",
                        "modality": "eeg",
                        "suffix": "eeg",
                        "extension": ".set",
                    }
                ]

                # 1. Subjects kwarg
                EEGChallengeDataset(
                    release="R1",
                    cache_dir=str(tmp_path),
                    subjects=["s1"],
                    download=True,
                )

                # Verify db.find was called.
                # EEGDash.find() calls get_client().find(query, **kwargs)
                # Wait, EEGDash.find calls self._client.find(final_query, **find_kwargs)

                args, kwargs = mock_db.find.call_args
                # kwargs might contain limit/skip if passed? logic says:
                # return list(self._client.find(final_query, **find_kwargs))
                # query is pos arg 0 to client.find usually?
                # No, pymongo find(filter, projection, ...)
                # EEGDash.find implementation:
                # return list(self._client.find(final_query, **find_kwargs))

                query = args[0]
                assert query["subject"] == {"$in": ["s1"]}

                # 2. Subject string
                EEGChallengeDataset(
                    release="R1", cache_dir=str(tmp_path), subject="s2", download=True
                )
                args, kwargs = mock_db.find.call_args
                query = args[0]
                assert query["subject"] == "s2"

                # 3. Subject list from query
                EEGChallengeDataset(
                    release="R1",
                    cache_dir=str(tmp_path),
                    query={"subject": ["s1"]},
                    download=True,
                )
                args, kwargs = mock_db.find.call_args
                query = args[0]
                assert query["subject"]["$in"] == ["s1"]


# --- BIDS Metadata Tests ---


def test_participants_extras_from_tsv(tmp_path):
    d = tmp_path

    # 1. Row is None
    with patch("eegdash.bids_metadata.participants_row_for_subject", return_value=None):
        assert bids_metadata.participants_extras_from_tsv(d, "01") == {}

    # 2. Row exists, filtering
    row = pd.Series(
        {"participant_id": "sub-01", "age": "20", "bad_col": "n/a", "empty": ""}
    )
    with patch("eegdash.bids_metadata.participants_row_for_subject", return_value=row):
        extras = bids_metadata.participants_extras_from_tsv(d, "01")
        assert "age" in extras
        assert "bad_col" not in extras
        assert "empty" not in extras
        assert "participant_id" not in extras  # id_columns excluded


def test_enrich_from_participants(tmp_path):
    mock_raw = MagicMock()
    mock_raw.info = {}
    mock_desc = {}
    mock_bids_path = MagicMock()
    mock_bids_path.subject = "01"

    with patch("eegdash.bids_metadata.participants_extras_from_tsv") as mock_extras:
        mock_extras.return_value = {"age": "25"}

        extras = bids_metadata.enrich_from_participants(
            tmp_path, mock_bids_path, mock_raw, mock_desc
        )

        assert extras == {"age": "25"}
        assert mock_raw.info["subject_info"]["participants_extras"]["age"] == "25"
        assert mock_desc["age"] == "25"


def test_enrich_from_participants_no_subject():
    mock_bids_path = MagicMock()
    mock_bids_path.subject = None  # No subject
    assert (
        bids_metadata.enrich_from_participants("root", mock_bids_path, MagicMock(), {})
        == {}
    )


def test_check_constraint_conflict():
    # q1, q2, key
    # Both have $in, intersection empty
    q1 = {"k": {"$in": [1, 2]}}
    q2 = {"k": {"$in": [3, 4]}}
    with pytest.raises(ValueError, match="Conflicting"):
        bids_metadata._check_constraint_conflict(q1, q2, "k")

    # One scalar, one $in
    q3 = {"k": 3}
    with pytest.raises(ValueError, match="Conflicting"):
        bids_metadata._check_constraint_conflict(q1, q3, "k")

    # No conflict
    q4 = {"k": {"$in": [2, 3]}}
    bids_metadata._check_constraint_conflict(q1, q4, "k")  # Should pass


# --- Registry Tests ---


def test_registry_make_init_closure(tmp_path):
    # Testing the inner function 'make_init' logic by actually registering something
    # or just mocking the flow.
    # The coverage gap is in 'make_init.__init__' 0%.
    # This implies we need to Instantiate one of the dynamically created classes.

    mock_data = {"success": True, "data": [{"dataset_id": "ds_dyn", "metadata": {}}]}

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        ns = {}
        register_openneuro_datasets(from_api=True, namespace=ns)

        # Now instantiate it to trigger __init__
        # It calls base_class.__init__
        # We need to mock base_class or ensure it works

        # We can pass a dummy base class to register_openneuro_datasets to make it easier
        class DummyBase:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        ns2 = {}
        register_openneuro_datasets(from_api=True, namespace=ns2, base_class=DummyBase)
        DS_Class2 = ns2["DS_DYN"]

        obj = DS_Class2(cache_dir="/tmp")
        assert obj.kwargs["query"] == {"dataset": "ds_dyn"}
        assert obj.kwargs["cache_dir"] == "/tmp"
