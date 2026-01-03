from pathlib import Path

import pytest

from eegdash.dataset import EEGChallengeDataset, EEGDashDataset


def _dummy_record(dataset: str, ext: str = ".set") -> dict:
    # Minimal record used by Raw without triggering IO
    # bids_relpath must be the path within the dataset
    bids_relpath = f"sub-01/ses-01/eeg/sub-01_ses-01_task-test_run-01_eeg{ext}"
    return {
        "data_name": f"{dataset}_sub-01_ses-01_task-test_run-01_eeg{ext}",
        "dataset": dataset,
        "bidspath": f"{dataset}/{bids_relpath}",
        "bids_relpath": bids_relpath,
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
        "storage": {
            "backend": "s3",
            "base": f"s3://openneuro.org/{dataset}",
            "raw_key": bids_relpath,
            "dep_keys": [],
        },
    }


@pytest.mark.parametrize(
    "release,dataset_id",
    [("R5", "EEG2025r5")],
)
def test_dataset_folder_suffixes(tmp_path: Path, release: str, dataset_id: str):
    # Baseline EEGDashDataset should use plain dataset folder
    rec = _dummy_record(dataset_id)
    ds_plain = EEGDashDataset(cache_dir=str(tmp_path), records=[rec])
    base = ds_plain.datasets[0]
    assert base.bids_root == tmp_path / dataset_id
    assert str(base.filecache).startswith(str((tmp_path / dataset_id).resolve()))

    # EEGChallengeDataset mini=True should now plain dataset folder (suffixes removed)
    rec_mini = _dummy_record(f"{dataset_id}mini")
    ds_min = EEGChallengeDataset(
        release=release, cache_dir=str(tmp_path), records=[rec_mini]
    )
    base_min = ds_min.datasets[0]
    # User removed suffix logic, so it should be just the dataset ID (which IS the mini ID)
    assert base_min.bids_root == tmp_path / f"{dataset_id}mini"
    assert str(base_min.filecache).startswith(
        str((tmp_path / f"{dataset_id}mini").resolve())
    )

    # EEGChallengeDataset mini=False should use EEG2025r{X} (no suffix)
    ds_full = EEGChallengeDataset(
        release=release, cache_dir=str(tmp_path), mini=False, records=[rec]
    )
    base_full = ds_full.datasets[0]
    assert base_full.bids_root == tmp_path / dataset_id
    assert str(base_full.filecache).startswith(str((tmp_path / dataset_id).resolve()))
