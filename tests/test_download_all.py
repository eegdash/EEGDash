from unittest.mock import patch

import eegdash.dataset.dataset as dataset_mod
from eegdash.dataset.dataset import EEGDashDataset
from eegdash.schemas import create_record


def _make_records(dataset_id: str, count: int):
    records = []
    for idx in range(count):
        subject = f"{idx:02d}"
        bids_relpath = f"sub-{subject}/eeg/sub-{subject}_task-test_eeg.bdf"
        records.append(
            create_record(
                dataset=dataset_id,
                storage_base="s3://example-bucket",
                bids_relpath=bids_relpath,
                subject=subject,
                task="test",
            )
        )
    return records


def test_download_all_skips_when_disabled(tmp_path):
    records = _make_records("ds000001", 2)
    dataset = EEGDashDataset(cache_dir=tmp_path, records=records, download=False)

    with patch.object(
        dataset_mod.EEGDashRaw, "_download_required_files", autospec=True
    ) as download_mock:
        dataset.download_all(n_jobs=2)

    download_mock.assert_not_called()


def test_download_all_only_missing_files(tmp_path):
    records = _make_records("ds000002", 2)
    dataset = EEGDashDataset(cache_dir=tmp_path, records=records)

    cached_path = dataset.datasets[0].filecache
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_text("cached")

    with patch.object(
        dataset_mod.EEGDashRaw, "_download_required_files", autospec=True
    ) as download_mock:
        dataset.download_all(n_jobs=1)

    assert download_mock.call_count == 1
    called_datasets = {call.args[0] for call in download_mock.call_args_list}
    assert dataset.datasets[1] in called_datasets
    assert dataset.datasets[0] not in called_datasets


def test_download_all_uses_parallel(tmp_path, monkeypatch):
    records = _make_records("ds000003", 3)
    dataset = EEGDashDataset(cache_dir=tmp_path, records=records)

    parallel_calls = []

    class ParallelSpy:
        def __init__(self, n_jobs=None, prefer=None):
            self.n_jobs = n_jobs
            self.prefer = prefer
            self.tasks = []
            parallel_calls.append(self)

        def __call__(self, tasks):
            self.tasks = list(tasks)
            return [task() for task in self.tasks]

    monkeypatch.setattr(dataset_mod, "Parallel", ParallelSpy)

    with patch.object(
        dataset_mod.EEGDashRaw, "_download_required_files", autospec=True
    ) as download_mock:
        dataset.download_all(n_jobs=2)

    assert len(parallel_calls) == 1
    assert parallel_calls[0].n_jobs == 2
    assert parallel_calls[0].prefer == "threads"
    assert download_mock.call_count == len(dataset.datasets)
