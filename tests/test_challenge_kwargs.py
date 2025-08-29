from eegdash.api import EEGDashDataset
from eegdash.dataset import EEGChallengeDataset


def test_challenge_dataset_passes_task_and_dataset_filters(tmp_path):
    ds = EEGChallengeDataset(
        release="R5",
        cache_dir=str(tmp_path),
        mini=False,
        task="RestingState",
    )

    assert len(ds.datasets) == 329
    assert ds.datasets[0].record["dataset"] == "ds005509"


def test_eegdashdataset_ignores_empty_query_when_kwargs_present(tmp_path):
    _ = EEGDashDataset(
        query={},
        cache_dir=str(tmp_path),
        dataset="ds005509",
        task="RestingState",
    )


def test_challenge_dataset_task_list_propagation(tmp_path):
    tasks = ["RestingState", "DespicableMe"]

    _ = EEGChallengeDataset(
        release="R5",
        cache_dir=str(tmp_path),
        mini=False,
        task=tasks,
    )


def test_eegdashdataset_allows_raw_query_and_kwargs(tmp_path):
    _ = EEGDashDataset(
        query={"subject": {"$in": ["NDARAU708TL8", "NDARAP785CTE"]}},
        cache_dir=str(tmp_path),
        dataset="ds005509",
        task="RestingState",
    )


def test_challenge_dataset_mini_populates_subjects(tmp_path):
    _ = EEGChallengeDataset(
        release="R5",
        cache_dir=str(tmp_path),
        mini=True,
        task="RestingState",
    )
