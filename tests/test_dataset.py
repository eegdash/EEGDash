import time
from pathlib import Path

import pytest

from eegdash.api import EEGDash, EEGDashDataset
from eegdash.dataset import EEGChallengeDataset

RELEASES = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"]
FILES_PER_RELEASE = [1342, 1405, 1812, 3342, 3326, 1227, 3100, 2320, 2885, 2516, 3397]

RELEASE_FILES = list(zip(RELEASES, FILES_PER_RELEASE))

CACHE_DIR = (Path.home() / "mne_data" / "eeg_challenge_cache").resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_release(release):
    ds = EEGChallengeDataset(release=release, mini=False, cache_dir=CACHE_DIR)
    getattr(ds, "description", None)
    return ds


@pytest.fixture(scope="session")
def warmed_mongo():
    try:
        EEGDash()
    except Exception:
        pytest.skip("Mongo not reachable")


def test_eeg_challenge_dataset_initialization():
    """Test the initialization of EEGChallengeDataset."""
    dataset = EEGChallengeDataset(release="R5", mini=False, cache_dir=CACHE_DIR)

    release = "R5"
    expected_bucket_prefix = f"s3://nmdatasets/NeurIPS25/{release}_L100_bdf"
    assert dataset.s3_bucket == expected_bucket_prefix, (
        f"Unexpected s3_bucket: {dataset.s3_bucket} (expected {expected_bucket_prefix})"
    )

    # Expected components (kept explicit for readability & easier future edits)
    expected_dataset = "ds005509"
    expected_subject = "sub-NDARAC350XUM"
    expected_task = "DespicableMe"
    expected_suffix = (
        f"{expected_dataset}/{expected_subject}/eeg/"
        f"{expected_subject}_task-{expected_task}_eeg.set"
    )

    expected_full_path = f"{dataset.s3_bucket}/{expected_suffix}"
    first_file = dataset.datasets[0].s3file

    assert first_file == expected_full_path, (
        "Mismatch in first dataset s3 file path.\n"
        f"Got     : {first_file}\n"
        f"Expected: {expected_full_path}"
    )


@pytest.mark.parametrize("release, number_files", RELEASE_FILES)
def test_eeg_challenge_dataset_amount_files(release, number_files):
    dataset = EEGChallengeDataset(release=release, mini=False, cache_dir=CACHE_DIR)
    assert len(dataset.datasets) == number_files


@pytest.mark.parametrize("release", RELEASES)
def test_mongodb_load_benchmark(benchmark, warmed_mongo, release):
    # Group makes the report nicer when comparing releases
    benchmark.group = "EEGChallengeDataset.load"

    result = benchmark.pedantic(
        _load_release,
        args=(release,),
        iterations=1,  # I/O-bound → 1 iteration per round
        rounds=5,  # take min/median across several cold-ish runs
        warmup_rounds=1,  # do one warmup round
    )

    assert result is not None


@pytest.mark.parametrize("release", RELEASES)
def test_mongodb_load_under_sometime(release):
    start_time = time.perf_counter()
    _ = EEGChallengeDataset(release=release, cache_dir=CACHE_DIR)
    duration = time.perf_counter() - start_time
    assert duration < 30, f"{release} took {duration:.2f}s"


@pytest.mark.parametrize("mini", [True, False])
@pytest.mark.parametrize("release", RELEASES)
def test_consuming_one_raw(release, mini):
    if mini:
        cache_dir = CACHE_DIR / "mini"
        cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        cache_dir = CACHE_DIR
    dataset_obj = EEGChallengeDataset(
        release=release,
        task="RestingState",
        cache_dir=cache_dir,
        mini=mini,
    )
    raw = dataset_obj.datasets[0].raw
    assert raw is not None


@pytest.mark.parametrize("eeg_dash_instance", [None, EEGDash()])
def test_eeg_dash_integration(eeg_dash_instance, release="R5", mini=True):
    dataset_obj = EEGChallengeDataset(
        release=release,
        task="RestingState",
        cache_dir=CACHE_DIR,
        mini=mini,
        eeg_dash_instance=eeg_dash_instance,
    )
    raw = dataset_obj.datasets[0].raw
    assert raw is not None


def test_eeg_dash_integration_warning():
    """Test that EEGChallengeDataset emits the expected UserWarning on init."""
    release = "R5"
    mini = True
    with pytest.warns(UserWarning) as record:
        _ = EEGChallengeDataset(
            release=release,
            task="RestingState",
            cache_dir=CACHE_DIR,
            mini=mini,
        )
    # There may be multiple warnings, check that at least one matches expected text
    found = any("EEG 2025 Competition Data Notice" in str(w.message) for w in record)
    assert found, (
        "Expected competition warning not found in warnings emitted by EEGChallengeDataset"
    )


def test_eeg_dashdataset():
    """Test that EEGDashDataset emits the expected UserWarning on init."""
    with pytest.warns(UserWarning) as record:
        _ = EEGDashDataset(
            dataset="ds005505",
            task="RestingState",
            cache_dir=CACHE_DIR,
        )
    # There may be multiple warnings, check that at least one matches expected text
    found = any("EEG 2025 Competition Data Notice" in str(w.message) for w in record)
    assert found, (
        "Expected competition warning not found in warnings emitted by EEGChallengeDataset"
    )
