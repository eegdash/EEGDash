from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import mne
import pandas as pd
import pytest

import eegdash.dataset.dataset as dataset_mod
from eegdash.dataset import EEGDashDataset
from eegdash.features.datasets import FeaturesConcatDataset, FeaturesDataset
from eegdash.features.serialization import load_features_concat_dataset


def _record(dataset: str, run: str, backend: str = "s3") -> dict:
    bids_relpath = f"sub-01/ses-01/eeg/sub-01_ses-01_task-test_run-{run}_eeg.set"
    storage_base = f"s3://openneuro.org/{dataset}"
    storage_raw = bids_relpath
    if backend == "local":
        storage_base = f"/local/{dataset}"
        storage_raw = bids_relpath

    return {
        "data_name": f"{dataset}_sub-01_ses-01_task-test_run-{run}_eeg.set",
        "dataset": dataset,
        "bidspath": f"{dataset}/{bids_relpath}",
        "bids_relpath": bids_relpath,
        "subject": "01",
        "session": "01",
        "task": "test",
        "run": run,
        "modality": "eeg",
        "suffix": "eeg",
        "datatype": "eeg",
        "extension": ".set",
        "sampling_frequency": 128.0,
        "nchans": 2,
        "ntimes": 16,
        "entities": {
            "subject": "01",
            "session": "01",
            "task": "test",
            "run": run,
        },
        "entities_mne": {
            "subject": "01",
            "session": "01",
            "task": "test",
            "run": run,
        },
        "storage": {
            "backend": backend,
            "base": storage_base,
            "raw_key": storage_raw,
            "dep_keys": [
                f"sub-01/ses-01/eeg/sub-01_ses-01_task-test_run-{run}_events.tsv"
            ],
        },
    }


@pytest.mark.parametrize(
    "cached_main,cached_dep,n_jobs,expected_downloaded",
    [
        ({0, 1}, {0, 1}, 1, set()),
        ({0}, {0}, 1, {1}),
        ({1}, set(), 2, {0, 1}),
    ],
    ids=["fully-cached", "one-missing-main", "missing-dependency"],
)
def test_download_all_respects_cache_state(
    tmp_path: Path,
    cached_main: set[int],
    cached_dep: set[int],
    n_jobs: int,
    expected_downloaded: set[int],
):
    records = [_record("ds-cache", "01"), _record("ds-cache", "02")]
    dataset = EEGDashDataset(cache_dir=tmp_path, dataset="ds-cache", records=records)

    for idx, ds in enumerate(dataset.datasets):
        all_paths = [ds.filecache, *ds._dep_paths]
        for p in all_paths:
            p.unlink(missing_ok=True)

        if idx in cached_main:
            ds.filecache.parent.mkdir(parents=True, exist_ok=True)
            ds.filecache.write_text("cached-main")

        if idx in cached_dep:
            for dep_path in ds._dep_paths:
                dep_path.parent.mkdir(parents=True, exist_ok=True)
                dep_path.write_text("cached-dep")

    with (
        patch.object(
            dataset_mod.EEGDashRaw, "_download_required_files", autospec=True
        ) as download_mock,
        patch.object(dataset, "_download_dataset_files") as globals_mock,
    ):
        dataset.download_all(n_jobs=n_jobs)

    called_indices = {
        i
        for i, ds in enumerate(dataset.datasets)
        if any(call.args[0] is ds for call in download_mock.call_args_list)
    }
    assert called_indices == expected_downloaded
    globals_mock.assert_called_once()


@pytest.mark.parametrize(
    "ids_to_load,with_raw_info,expected_subjects",
    [
        (None, False, {"sub-01", "sub-02"}),
        ([1], True, {"sub-02"}),
    ],
    ids=["load-all-no-raw-info", "subset-with-raw-info"],
)
def test_features_roundtrip_metadata_and_kwargs(
    tmp_path: Path,
    ids_to_load: list[int] | None,
    with_raw_info: bool,
    expected_subjects: set[str],
):
    metadata = pd.DataFrame(
        {
            "i_window_in_trial": [0, 1],
            "i_start_in_trial": [0, 8],
            "i_stop_in_trial": [8, 16],
            "target": [0, 1],
        }
    )

    info = (
        mne.create_info(["Cz", "Pz"], sfreq=128.0, ch_types="eeg")
        if with_raw_info
        else None
    )

    ds1 = FeaturesDataset(
        features=pd.DataFrame({"feat_a": [1.0, 2.0], "feat_b": [3.0, 4.0]}),
        metadata=metadata,
        description={"subject": "sub-01", "session": "ses-A"},
        raw_info=info,
        raw_preproc_kwargs={"resample": 128},
        features_kwargs={"feature_set": "basic"},
    )
    ds2 = FeaturesDataset(
        features=pd.DataFrame({"feat_a": [5.0, 6.0], "feat_b": [7.0, 8.0]}),
        metadata=metadata,
        description={"subject": "sub-02", "session": "ses-B"},
        raw_info=info,
        window_kwargs={"window_size": 8},
        window_preproc_kwargs={"clip": True},
    )

    concat = FeaturesConcatDataset([ds1, ds2])
    save_dir = tmp_path / "features"
    save_dir.mkdir(parents=True, exist_ok=True)
    concat.save(str(save_dir))

    loaded = load_features_concat_dataset(save_dir, ids_to_load=ids_to_load)
    loaded_metadata = loaded.get_metadata()

    assert set(loaded_metadata["subject"].unique()) == expected_subjects
    assert set(loaded_metadata["session"].unique()) == {
        "ses-A" if s == "sub-01" else "ses-B" for s in expected_subjects
    }

    loaded_subjects = {ds.description["subject"] for ds in loaded.datasets}
    assert loaded_subjects == expected_subjects

    if with_raw_info:
        assert all(ds.raw_info is not None for ds in loaded.datasets)
    else:
        assert all(ds.raw_info is None for ds in loaded.datasets)

    first = loaded.datasets[0]
    assert isinstance(first.metadata, pd.DataFrame)
    assert "target" in first.metadata.columns
    assert first.features.shape[1] == 2
