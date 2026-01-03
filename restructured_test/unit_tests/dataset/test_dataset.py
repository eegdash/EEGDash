import os
import time
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

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
            ds = EEGDashDataset(
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
