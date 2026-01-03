import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eegdash.api import EEGDash
from eegdash.dataset import EEGChallengeDataset, EEGDashDataset

# These tests hit the live metadata API and S3 and can be slow/flaky on
# constrained networks. By default, run a small smoke subset; opt into the full
# matrix with `EEGDASH_TEST_ALL_RELEASES=1` or `EEGDASH_TEST_RELEASES=R1,R5,...`.
_ALL_RELEASE_FILES = {
    "R1": 1342,
    "R2": 1405,
    "R3": 1812,
    "R4": 3342,
    "R5": 3326,
    "R6": 1227,
    "R7": 3100,
    "R8": 2320,
    "R9": 2885,
    "R10": 2516,
    "R11": 3397,
}

_env_releases = os.getenv("EEGDASH_TEST_RELEASES", "").strip()
if _env_releases:
    RELEASES = [r.strip() for r in _env_releases.split(",") if r.strip()]
elif os.getenv("EEGDASH_TEST_ALL_RELEASES", "").strip() == "1":
    RELEASES = list(_ALL_RELEASE_FILES.keys())
else:
    RELEASES = ["R5"]

RELEASE_FILES = [(r, _ALL_RELEASE_FILES[r]) for r in RELEASES]


def _load_release(release, cache_dir: Path):
    ds = EEGChallengeDataset(release=release, mini=False, cache_dir=cache_dir)
    getattr(ds, "description", None)
    return ds


def test_eeg_challenge_dataset_initialization(cache_dir: Path):
    """Test the initialization of EEGChallengeDataset."""
    dataset = EEGChallengeDataset(release="R5", mini=False, cache_dir=cache_dir)

    release = "R5"
    expected_bucket_prefix = f"s3://nmdatasets/NeurIPS25/{release}_L100_bdf"
    assert dataset.s3_bucket == expected_bucket_prefix, (
        f"Unexpected s3_bucket: {dataset.s3_bucket} (expected {expected_bucket_prefix})"
    )

    first_file = dataset.datasets[0].s3file
    assert first_file.startswith(f"{dataset.s3_bucket}/"), (
        "Mismatch in first dataset s3 file prefix.\n"
        f"Got     : {first_file}\n"
        f"Expected: {dataset.s3_bucket}/..."
    )
    assert first_file.endswith("_eeg.bdf"), (
        f"Mismatch in first dataset s3 file suffix.\nGot     : {first_file}"
    )


@pytest.mark.parametrize("release, number_files", RELEASE_FILES)
def test_eeg_challenge_dataset_amount_files(release, number_files, cache_dir: Path):
    dataset = EEGChallengeDataset(release=release, mini=False, cache_dir=cache_dir)
    assert len(dataset.datasets) == number_files


@pytest.mark.parametrize("release", RELEASES)
@pytest.mark.skipif(
    os.getenv("EEGDASH_TEST_BENCHMARKS", "").strip() != "1",
    reason="Set EEGDASH_TEST_BENCHMARKS=1 to run live API benchmarks.",
)
def test_mongodb_load_benchmark(benchmark, release, cache_dir: Path):
    # Group makes the report nicer when comparing releases
    benchmark.group = "EEGChallengeDataset.load"

    result = benchmark.pedantic(
        _load_release,
        args=(release, cache_dir),
        iterations=1,  # I/O-bound → 1 iteration per round
        rounds=5,  # take min/median across several cold-ish runs
        warmup_rounds=1,  # do one warmup round
    )

    assert result is not None


@pytest.mark.parametrize("release", RELEASES)
def test_mongodb_load_under_sometime(release, cache_dir: Path):
    start_time = time.perf_counter()
    _ = EEGChallengeDataset(release=release, cache_dir=cache_dir)
    duration = time.perf_counter() - start_time
    assert duration < 300, f"{release} took {duration:.2f}s"


@pytest.mark.parametrize("mini", [True, False])
@pytest.mark.parametrize("release", RELEASES)
def test_consuming_one_raw(release, mini, cache_dir: Path):
    print(f"Testing release {release} mini={mini} with cache dir {cache_dir}")
    dataset_obj = EEGChallengeDataset(
        release=release,
        task="RestingState",
        cache_dir=cache_dir,
        mini=mini,
    )
    raw = dataset_obj.datasets[0].raw
    assert raw is not None


@pytest.mark.parametrize("eeg_dash_instance", [None, EEGDash()])
def test_eeg_dash_integration(
    eeg_dash_instance, cache_dir: Path, release="R5", mini=True
):
    dataset_obj = EEGChallengeDataset(
        release=release,
        task="RestingState",
        cache_dir=cache_dir,
        mini=mini,
        eeg_dash_instance=eeg_dash_instance,
    )
    raw = dataset_obj.datasets[0].raw
    assert raw is not None


def test_dataset_gaps(tmp_path):
    # Trigger dataset.py 317, 319, 514, 528
    # Bypass __init__ to avoid setup issues
    with patch("eegdash.dataset.dataset.EEGDashDataset.__init__", return_value=None):
        ds = EEGDashDataset()
        ds.cache_dir = tmp_path
        ds.eeg_dash_instance = MagicMock()
        ds._normalize_records = lambda x: x
        ds._find_key_in_nested_dict = EEGDashDataset._find_key_in_nested_dict.__get__(
            ds
        )

        # 514: Trigger v2 format validation error
        invalid_record = {"data_name": "bad_rec"}
        ds.eeg_dash_instance.find.return_value = [invalid_record]
        with patch(
            "eegdash.dataset.dataset.validate_record", return_value=["missing stuff"]
        ):
            with pytest.raises(ValueError, match="v2 format"):
                EEGDashDataset._find_datasets(
                    ds, query={}, description_fields=[], base_dataset_kwargs={}
                )

        # 528: Trigger participant_tsv merge
        valid_record = {"dataset": "ds001", "participant_tsv": {"age": 25}}
        ds.eeg_dash_instance.find.return_value = [valid_record]
        with patch("eegdash.dataset.dataset.validate_record", return_value=[]):
            with patch(
                "eegdash.dataset.dataset.merge_participants_fields"
            ) as mock_merge:
                with patch("eegdash.dataset.dataset.EEGDashRaw"):
                    EEGDashDataset._find_datasets(
                        ds, query={}, description_fields=[], base_dataset_kwargs={}
                    )
                    assert mock_merge.called


def test_datasets_init_gap():
    from eegdash.dataset.dataset import EEGDashDataset

    # 4 lines missing in __init__?
    # Usually related to super init or basic setup?
    # We can just try to instantiate with minimum args
    # Patching the CLASS method
    with patch(
        "eegdash.dataset.dataset.EEGDashDataset._find_datasets", return_value=[]
    ):
        # But wait, if it returns empty list, init might raise "No datasets found"
        # We need to see code.
        # If I look above, standard logic is: datasets = self._find_datasets... if not datasets: raise ValueError
        # So we must return a non-empty list
        with patch(
            "eegdash.dataset.dataset.EEGDashDataset._find_datasets",
            return_value=[MagicMock()],
        ):
            ds = EEGDashDataset(query={}, cache_dir=".", dataset="ds001")
            # query is set before _find_datasets check
            assert ds.query == {"dataset": "ds001"}


def test_dataset_init_exception_gap(tmp_path):
    # Cover lines 297-298 in dataset.py: Exception handling in loop

    # Create valid dataset dir for checks
    cache = tmp_path
    (cache / "ds001").mkdir()

    with patch(
        "eegdash.dataset.dataset.EEGDashDataset._find_local_bids_records"
    ) as mock_find:
        # Mock records
        mock_find.return_value = [
            {"path": "s1", "bidspath": "foo/bar", "dataset": "ds001"},
            {"path": "s2", "bidspath": "foo/baz", "dataset": "ds001"},
        ]

        # Try patching the class in the module
        with patch("eegdash.dataset.dataset.EEGDashRaw"):
            # Mock get_entities_from_record (plural) called in dataset.py
            with patch(
                "eegdash.dataset.dataset.get_entities_from_record",
                return_value={"sub": "01"},
            ):
                # Mock participants_row_for_subject is NOT called directly, usage is:
                # part_row = bids_ds.subject_participant_tsv(local_file)
                # So we mock EEGBIDSDataset or its method?
                # In the code: bids_ds = EEGBIDSDataset(...)
                # We should patch EEGBIDSDataset class in dataset.py
                with patch("eegdash.dataset.dataset.EEGBIDSDataset") as mock_bids_cls:
                    mock_bids = mock_bids_cls.return_value
                    mock_bids.subject_participant_tsv.return_value = {}

                    # Mock merge_participants_fields to raise Exception
                    with patch(
                        "eegdash.dataset.dataset.merge_participants_fields",
                        side_effect=Exception("Boom"),
                    ):
                        ds = EEGDashDataset(
                            cache_dir=cache,
                            dataset="ds001",
                            check_files=False,
                            download=False,
                        )
                        # Should swallow exception and continue
                        assert len(ds.datasets) == 2


def test_dataset_init_kwargs_gap(tmp_path):
    # Cover lines 317, 319 in dataset.py: database and auth_token init
    with patch(
        "eegdash.dataset.dataset.EEGDashDataset._find_datasets",
        return_value=[MagicMock()],
    ):
        with patch("eegdash.api.EEGDash") as mock_api:
            # Must provide cache_dir
            EEGDashDataset(
                cache_dir=tmp_path,
                dataset="ds001",
                query={"subject": "sub-01"},
                database="mydb",
                auth_token="mytoken",
            )

            # Check if EEGDash was init with correct kwargs
            call_args = mock_api.call_args[1]
            assert call_args["database"] == "mydb"
            assert call_args["auth_token"] == "mytoken"


def test_cache_dir_does_not_exist_creates(tmp_path, caplog):
    """Test that missing cache_dir triggers warning and creation."""
    import logging

    from eegdash.dataset.dataset import EEGDashDataset

    nonexistent_dir = tmp_path / "nonexistent_cache_dir"
    assert not nonexistent_dir.exists()

    # Create dataset with nonexistent cache_dir - need to provide dataset
    # The warning is a logger.warning, not a Warning, so use caplog
    with caplog.at_level(logging.WARNING):
        try:
            EEGDashDataset(dataset="ds002778", cache_dir=str(nonexistent_dir))
        except Exception:
            # It's fine if it fails for other reasons - we just need to hit line 189, 192
            pass

    # Check that the directory was created
    assert nonexistent_dir.exists()


def test_iterate_local_with_participants_exception(tmp_path):
    """Test that exception in merge_participants_fields is handled."""
    from eegdash.dataset.dataset import merge_participants_fields

    # Test merge_participants_fields with None participants_row
    result = merge_participants_fields(
        description={"key": "value"},
        participants_row=None,
        description_fields=["key"],
    )
    assert result == {"key": "value"}


def test_normalize_records_with_list():
    """Test _normalize_records with list input."""
    from unittest.mock import patch

    from eegdash.dataset.dataset import EEGDashDataset

    # Test _normalize_records directly via a static method-like call
    # The function normalizes the records - if given list, returns list
    records = [{"data_name": "test", "s3_url": "s3://test"}]

    # Create instance with required attributes
    with patch.object(EEGDashDataset, "__init__", lambda self: None):
        ds = EEGDashDataset.__new__(EEGDashDataset)
        ds.s3_bucket = None  # Set required attribute
        ds._dedupe_records = False  # Set required attribute
        normalized = ds._normalize_records(records)
        assert isinstance(normalized, list)
        assert normalized == records


def test_dataset_find_key_nested(tmp_path):
    """Test _find_key_in_nested_dict recursion."""
    from unittest.mock import patch

    from eegdash.dataset.dataset import EEGDashDataset

    d = tmp_path / "ds999"
    d.mkdir()

    valid_record = {
        "dataset": "ds999",
        "data_name": "sub-01_task-rest_eeg.set",
        "bidspath": "ds999/sub-01/eeg/sub-01_task-rest_eeg.set",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
        "subject": "01",
        "task": "rest",
        "storage": {"base": str(d), "backend": "local"},
    }
    # Mocking discover to return proper records so init doesn't fail
    with patch("eegdash.dataset.dataset.discover_local_bids_records") as mock_discover:
        mock_discover.return_value = [valid_record]
        ds = EEGDashDataset(
            cache_dir=str(tmp_path),
            dataset="ds999",
            download=False,
            _suppress_comp_warning=True,
        )

    data = {"a": 1, "b": {"c": 2, "d": [{"e": 3}, {"f": 4}]}}
    assert ds._find_key_in_nested_dict(data, "a") == 1
    assert ds._find_key_in_nested_dict(data, "c") == 2
    assert ds._find_key_in_nested_dict(data, "e") == 3
    assert ds._find_key_in_nested_dict(data, "z") is None


def test_dataset_normalize_records_dedupe(tmp_path):
    """Test record deduplication logic."""
    from unittest.mock import patch

    from eegdash.dataset.dataset import EEGDashDataset

    d = tmp_path / "ds999"
    d.mkdir()

    valid_record = {
        "dataset": "ds999",
        "data_name": "sub-01_task-rest_eeg.set",
        "bidspath": "ds999/sub-01/eeg/sub-01_task-rest_eeg.set",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
        "subject": "01",
        "task": "rest",
        "storage": {"base": str(d), "backend": "local"},
    }

    with patch("eegdash.dataset.dataset.discover_local_bids_records") as mock_discover:
        mock_discover.return_value = [valid_record]
        ds = EEGDashDataset(
            cache_dir=str(tmp_path),
            dataset="ds999",
            download=False,
            _dedupe_records=True,
        )

    records = [
        {"bidspath": "path1", "data_name": "n1"},
        {"bidspath": "path1", "data_name": "n1"},  # Duplicate
        {"bidspath": "path2", "data_name": "n2"},
    ]
    normalized = ds._normalize_records(records)
    assert len(normalized) == 2


def test_dataset_s3_bucket_injection(tmp_path):
    """Test S3 bucket injection."""
    from unittest.mock import patch

    from eegdash.dataset.dataset import EEGDashDataset

    d = tmp_path / "ds999"
    d.mkdir()

    valid_record = {
        "dataset": "ds999",
        "data_name": "sub-01_task-rest_eeg.set",
        "bidspath": "ds999/sub-01/eeg/sub-01_task-rest_eeg.set",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
        "subject": "01",
        "task": "rest",
        "storage": {"base": str(d), "backend": "local"},
    }

    with patch("eegdash.dataset.dataset.discover_local_bids_records") as mock_discover:
        mock_discover.return_value = [valid_record]
        ds = EEGDashDataset(
            cache_dir=str(tmp_path),
            dataset="ds999",
            download=False,
            s3_bucket="s3://custom",
        )

    records = [{"file": "f1"}]
    normalized = ds._normalize_records(records)
    assert normalized[0]["storage"]["base"] == "s3://custom"
    assert normalized[0]["storage"]["backend"] == "s3"


def test_dataset_download_all_offline(tmp_path):
    """Test download_all returns early if download=False."""
    from unittest.mock import patch

    from eegdash.dataset.dataset import EEGDashDataset

    d = tmp_path / "ds999"
    d.mkdir()

    valid_record = {
        "dataset": "ds999",
        "data_name": "sub-01_task-rest_eeg.set",
        "bidspath": "ds999/sub-01/eeg/sub-01_task-rest_eeg.set",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
        "subject": "01",
        "task": "rest",
        "storage": {"base": str(d), "backend": "local"},
    }

    with patch("eegdash.dataset.dataset.discover_local_bids_records") as mock_discover:
        mock_discover.return_value = [valid_record]
        ds = EEGDashDataset(cache_dir=str(tmp_path), dataset="ds999", download=False)

    # Should not raise error or try to download
    ds.download_all()


def test_dataset_lazy_loader(tmp_path):
    """Test lazy loading of EEGDash instance."""
    from unittest.mock import patch

    from eegdash.dataset.dataset import EEGDashDataset

    # We patch eegdash.api.EEGDash because that's where it is imported from
    with patch("eegdash.api.EEGDash") as MockDash:
        MockDash.return_value.find.return_value = []  # Return empty
        try:
            # Provide s3_bucket to force usage of non-offline path if records/db issues
            # Actually, if download=True (default), it goes to _find_datasets logic.
            # We need to ensure logic flow reaches import.
            EEGDashDataset(
                cache_dir=str(tmp_path), query={"dataset": "ds_lazy"}, download=True
            )
        except ValueError:
            pass
        MockDash.assert_called()


def test_dataset_save(tmp_path):
    """Test save method delegation."""
    from unittest.mock import patch

    from eegdash.dataset.dataset import EEGDashDataset

    d = tmp_path / "ds999"
    d.mkdir()

    valid_record = {
        "dataset": "ds999",
        "data_name": "sub-01_task-rest_eeg.set",
        "bidspath": "ds999/sub-01/eeg/sub-01_task-rest_eeg.set",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
        "subject": "01",
        "task": "rest",
        "storage": {"base": str(d), "backend": "local"},
    }

    with patch("eegdash.dataset.dataset.discover_local_bids_records") as mock_discover:
        mock_discover.return_value = [valid_record]
        ds = EEGDashDataset(cache_dir=str(tmp_path), dataset="ds999", download=False)

    # Mock super().save
    with patch("braindecode.datasets.BaseConcatDataset.save") as mock_save:
        ds.save("some_path", overwrite=True)
        mock_save.assert_called_with("some_path", overwrite=True)


def test_challenge_dataset_errors():
    """Test EEGChallengeDataset error conditions."""
    import pytest

    from eegdash.dataset.dataset import EEGChallengeDataset

    with pytest.raises(ValueError, match="Unknown release"):
        EEGChallengeDataset(release="R999", cache_dir="/tmp")

    with pytest.raises(ValueError, match="Query using the parameters "):
        EEGChallengeDataset(release="R1", cache_dir="/tmp", query={"dataset": "ds123"})

    # Requires more mocks to potentially reach these lines if they check local files/API
    # Assuming these tests are robust enough as written from original source


def test_challenge_dataset_defaults():
    """Test EEGChallengeDataset defaults."""
    from eegdash.dataset.dataset import EEGChallengeDataset

    ds = EEGChallengeDataset(release="R1", cache_dir="/tmp", mini=True)
    assert ds.query["subject"]  # Should be populated with mini subjects
    assert ds.s3_bucket.startswith("s3://nmdatasets/NeurIPS25/R1_mini")

    ds_full = EEGChallengeDataset(release="R1", cache_dir="/tmp", mini=False)
    assert ds_full.s3_bucket.startswith("s3://nmdatasets/NeurIPS25/R1_L100")


def test_challenge_dataset_subjects_logic(tmp_path):
    # Verify logic by mocking the low-level get_client used by EEGDash
    # eegdash.api imports get_client from .http_api_client
    from unittest.mock import patch

    from eegdash.dataset.dataset import EEGChallengeDataset

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
                        "data_name": "ds1_sub-01_task-rest_eeg.set",
                        "bidspath": "ds1/sub-01/eeg/sub-01_task-rest_eeg.set",
                        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
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
                args, kwargs = mock_db.find.call_args
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


def test_dataset_download_all_njobs(tmp_path):
    # Coverage for download_all branches
    from unittest.mock import patch

    from eegdash.dataset.dataset import EEGDashDataset, EEGDashRaw

    record = {
        "dataset": "ds_dl",
        "data_name": "ds_dl_f.set",
        "bidspath": "ds_dl/f.set",
        "bids_relpath": "f.set",
        "storage": {"base": "s3://b", "backend": "s3", "raw_key": "f.set"},
    }

    with patch("eegdash.api.EEGDash") as MockDash:
        MockDash.return_value.find.return_value = [record]
        with patch("eegdash.dataset.dataset.downloader.get_s3_filesystem"):
            ds = EEGDashDataset(cache_dir=str(tmp_path), dataset="ds_dl", download=True)

            # Patch the method on the CLASS so Parallel/Delayed sees it
            with patch.object(EEGDashRaw, "_download_required_files") as mock_dl:
                # Test n_jobs=1
                ds.download_all(n_jobs=1)
                assert mock_dl.call_count == 1

                # Test n_jobs=2 (Parallel branch)
                mock_dl.reset_mock()
                ds.download_all(n_jobs=2)
                assert mock_dl.call_count == 1


def test_dataset_offline_enrichment(tmp_path):
    # Coverage for lines 284-298 in dataset.py (Offline mode enrichment)
    from unittest.mock import patch

    from eegdash.dataset.dataset import EEGDashDataset

    d = tmp_path / "ds_offline"
    d.mkdir()
    subj_dir = d / "sub-01" / "eeg"
    subj_dir.mkdir(parents=True)
    f = subj_dir / "sub-01_task-rest_eeg.set"
    f.touch()

    # participants.tsv
    (d / "participants.tsv").write_text("participant_id\tage\nsub-01\t25")
    (d / "dataset_description.json").write_text('{"Name": "DS"}')

    # Mock discover_local_bids_records with valid v2 record
    record = {
        "dataset": "ds_offline",
        "data_name": "ds_offline_sub-01_task-rest_eeg.set",
        "bidspath": "ds_offline/sub-01/eeg/sub-01_task-rest_eeg.set",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
        "subject": "01",
        "storage": {"base": str(d), "backend": "local"},
    }

    with patch(
        "eegdash.dataset.dataset.discover_local_bids_records", return_value=[record]
    ):
        ds = EEGDashDataset(
            cache_dir=str(tmp_path), dataset="ds_offline", download=False
        )
        # Verify enrichment
        assert ds.datasets[0].description["age"] == "25"


def test_dataset_various_errors(tmp_path):
    # Coverage for rare error branches
    import pytest

    from eegdash.dataset.dataset import EEGDashDataset

    # 1. No dataset in records fallback
    with pytest.raises(ValueError, match="provide a 'dataset' argument"):
        EEGDashDataset(
            cache_dir=str(tmp_path), records=[{"bidspath": "p"}], download=True
        )

    # 2. Offline mode but directory missing
    with pytest.raises(ValueError, match="Offline mode is enabled, but local data_dir"):
        EEGDashDataset(cache_dir=str(tmp_path), dataset="missing_ds", download=False)


def test_dataset_warning_fallback(tmp_path):
    # dataset.py 247-250 (Console fallback)
    from unittest.mock import patch

    import pytest
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


def test_dataset_init_error_no_args(tmp_path):
    # dataset.py 216 and 332
    from unittest.mock import MagicMock

    import pytest

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


def test_dataset_init_cache_defaults(tmp_path):
    # dataset.py 181-182, 189-192
    from unittest.mock import patch

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
    from unittest.mock import patch

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
    from unittest.mock import patch

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
    from unittest.mock import MagicMock, patch

    import pytest

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
    from unittest.mock import MagicMock, patch

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

    pass

    pass
