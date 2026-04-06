from unittest.mock import patch

import pytest

import eegdash.dataset.dataset as dataset_mod
from eegdash import EEGDash
from eegdash.dataset.dataset import EEGDashDataset

OPENNEURO_DATASET = "ds000248"


@pytest.fixture(scope="session")
def ds000248_records():
    eegdash = EEGDash()
    try:
        records = eegdash.find({"dataset": OPENNEURO_DATASET}, limit=2)
    except Exception as exc:
        pytest.skip(f"EEGDash API unavailable for {OPENNEURO_DATASET}: {exc}")
    if len(records) < 2:
        pytest.skip(f"Not enough records returned for {OPENNEURO_DATASET}")
    return records


def _build_dataset(tmp_path, records, *, download=True):
    return EEGDashDataset(cache_dir=tmp_path, records=records, download=download)


def test_download_all_skips_when_disabled(tmp_path, ds000248_records):
    dataset = _build_dataset(tmp_path, [ds000248_records[0]], download=False)

    with patch.object(
        dataset_mod.EEGDashRaw, "_download_required_files", autospec=True
    ) as download_mock:
        dataset.download_all(n_jobs=2)

    download_mock.assert_not_called()


def test_download_all_only_missing_files(tmp_path, ds000248_records):
    records = ds000248_records[:2]
    dataset = _build_dataset(tmp_path, records)

    ds0 = dataset.datasets[0]
    ds1 = dataset.datasets[1]

    # Ensure dataset[1] files are missing FIRST, before caching ds0.
    # Both datasets share the same bids_root, so dep_paths can overlap
    # (e.g. shared sidecar files). Removing ds1 after caching ds0 could
    # accidentally delete a file ds0 also depends on.
    all_paths_1 = [ds1.filecache] + list(ds1._dep_paths)
    for p in all_paths_1:
        p.unlink(missing_ok=True)

    # Cache ALL files for dataset[0]: main file + every dependency
    all_paths_0 = [ds0.filecache] + list(ds0._dep_paths)
    for p in all_paths_0:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("cached")

    with patch.object(
        dataset_mod.EEGDashRaw, "_download_required_files", autospec=True
    ) as download_mock:
        dataset.download_all(n_jobs=1)

    # dataset[0] is fully cached -> should NOT be in download calls
    called_datasets = {call.args[0] for call in download_mock.call_args_list}
    assert ds0 not in called_datasets, (
        f"ds0 should be skipped (cached). "
        f"filecache={ds0.filecache.exists()}, "
        f"deps={[p.exists() for p in ds0._dep_paths]}"
    )
    assert ds1 in called_datasets, "ds1 should have been downloaded"


def test_download_all_uses_parallel(tmp_path, monkeypatch, ds000248_records):
    records = ds000248_records[:2]
    dataset = _build_dataset(tmp_path, records)

    for ds in dataset.datasets:
        ds.filecache.unlink(missing_ok=True)
        for dep_path in getattr(ds, "_dep_paths", []):
            dep_path.unlink(missing_ok=True)

    parallel_calls = []

    class ParallelSpy:
        def __init__(self, n_jobs=None, prefer=None):
            self.n_jobs = n_jobs
            self.prefer = prefer
            self.tasks = []
            parallel_calls.append(self)

        def __call__(self, tasks):
            self.tasks = list(tasks)
            results = []
            for task in self.tasks:
                if callable(task):
                    results.append(task())
                else:
                    func, args, kwargs = task
                    results.append(func(*args, **kwargs))
            return results

    monkeypatch.setattr(dataset_mod, "Parallel", ParallelSpy)

    with patch.object(
        dataset_mod.EEGDashRaw, "_download_required_files", autospec=True
    ) as download_mock:
        dataset.download_all(n_jobs=2)

    assert len(parallel_calls) == 1
    assert parallel_calls[0].n_jobs == 2
    assert parallel_calls[0].prefer == "threads"
    assert download_mock.call_count == len(dataset.datasets)
