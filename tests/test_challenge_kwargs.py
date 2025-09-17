from pathlib import Path

import pytest

from eegdash.api import EEGDashDataset
from eegdash.dataset.dataset import EEGChallengeDataset


@pytest.fixture(scope="module")
def cache_dir():
    """Provide a shared cache directory for tests that need to cache datasets."""
    from pathlib import Path

    from eegdash.paths import get_default_cache_dir

    cache_dir = Path(get_default_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def test_challenge_dataset_passes_task_and_dataset_filters(cache_dir: Path):
    ds = EEGChallengeDataset(
        release="R5",
        cache_dir=cache_dir,
        mini=False,
        task="RestingState",
    )

    assert len(ds.datasets) == 329
    assert ds.datasets[0].record["dataset"] == "ds005509"


def test_eegdashdataset_ignores_empty_query_when_kwargs_present(cache_dir: Path):
    _ = EEGDashDataset(
        query={},
        cache_dir=cache_dir,
        dataset="ds005509",
        task="RestingState",
    )


def test_challenge_dataset_task_list_propagation(cache_dir: Path):
    tasks = ["RestingState", "DespicableMe"]

    _ = EEGChallengeDataset(
        release="R5",
        cache_dir=cache_dir,
        mini=False,
        task=tasks,
    )


def test_eegdashdataset_allows_raw_query_and_kwargs(cache_dir: Path):
    _ = EEGDashDataset(
        query={"subject": {"$in": ["NDARAU708TL8", "NDARAP785CTE"]}},
        cache_dir=cache_dir,
        dataset="ds005509",
        task="RestingState",
    )


def test_challenge_dataset_mini_populates_subjects(cache_dir: Path):
    _ = EEGChallengeDataset(
        release="R5",
        cache_dir=cache_dir,
        mini=True,
        task="RestingState",
    )
