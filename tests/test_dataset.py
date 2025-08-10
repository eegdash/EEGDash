import pytest
from eegdash.api import EEGDash
from eegdash.dataset import EEGChallengeDataset

RELEASES = ["R1","R2","R3","R4","R5","R6","R7","R8","R9","R10","R11"]
FILES_PER_RELEASE = [1342, 1405, 1812, 3342, 3326, 1227, 3100, 2320, 2885, 2516, 3397]

RELEASE_FILES = list(zip(RELEASES, FILES_PER_RELEASE))

def _load_release(release):
    ds = EEGChallengeDataset(release=release)
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
    dataset = EEGChallengeDataset(release="R5")
    assert dataset.s3_bucket == "s3://nmdatasets/NeurIPS25//R5_L100"
    assert (
        dataset.datasets[0].s3file
        == "s3://nmdatasets/NeurIPS25//R5_L100/ds005509/sub-NDARAC350XUM/eeg/sub-NDARAC350XUM_task-DespicableMe_eeg.set"
    )

@pytest.mark.parametrize("release, number_files", RELEASE_FILES)
def test_eeg_challenge_dataset_amount_files(release, number_files):
    dataset = EEGChallengeDataset(release=release)
    assert len(dataset.datasets) == number_files


@pytest.mark.parametrize("release", RELEASES)
def test_mongodb_load_benchmark(benchmark, warmed_mongo, release):
    # Group makes the report nicer when comparing releases
    benchmark.group = "EEGChallengeDataset.load"
    result = benchmark.pedantic(
        _load_release,
        args=(release,),
        iterations=1,      # I/O-bound → 1 iteration per round
        rounds=5,          # take min/median across several cold-ish runs
        warmup_rounds=1,   # do one warmup round
    )
    assert result is not None