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


def _dummy_record(dataset: str, ext: str = ".set") -> dict:
    # Minimal record used by EEGDashBaseDataset without triggering IO
    # bidspath must start with dataset root
    bidspath = f"{dataset}/sub-01/ses-01/eeg/sub-01_ses-01_task-test_run-01_eeg{ext}"
    return {
        "data_name": f"{dataset}_sub-01_ses-01_task-test_run-01_eeg{ext}",
        "dataset": dataset,
        "bidspath": bidspath,
        "bidsdependencies": [],
        # BIDS entities used to construct BIDSPath
        "subject": "01",
        "session": "01",
        "task": "test",
        "run": "01",
        # Not used in this test path, but present in real records
        "modality": "eeg",
        "sampling_frequency": 100.0,
        "nchans": 3,
        "ntimes": 100,
    }


@pytest.mark.parametrize(
    "release,dataset_id",
    [("R5", "ds005509")],
)
def test_dataset_folder_suffixes(cache_dir: Path, release: str, dataset_id: str):
    # Baseline EEGDashDataset should use plain dataset folder
    rec = _dummy_record(dataset_id)
    ds_plain = EEGDashDataset(cache_dir=cache_dir, records=[rec])
    base = ds_plain.datasets[0]
    assert base.bids_root == cache_dir / dataset_id
    assert str(base.filecache).startswith(str((cache_dir / dataset_id).resolve()))

    # EEGChallengeDataset mini=True should suffix with -bdf-mini
    ds_min = EEGChallengeDataset(release=release, cache_dir=cache_dir, records=[rec])
    base_min = ds_min.datasets[0]
    assert base_min.bids_root == cache_dir / f"{dataset_id}-bdf-mini"
    assert str(base_min.filecache).startswith(
        str((cache_dir / f"{dataset_id}-bdf-mini").resolve())
    )

    # EEGChallengeDataset mini=False should suffix with -bdf
    ds_full = EEGChallengeDataset(
        release=release, cache_dir=cache_dir, mini=False, records=[rec]
    )
    base_full = ds_full.datasets[0]
    assert base_full.bids_root == cache_dir / f"{dataset_id}-bdf"
    assert str(base_full.filecache).startswith(
        str((cache_dir / f"{dataset_id}-bdf").resolve())
    )
