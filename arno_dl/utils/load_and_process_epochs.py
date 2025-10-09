# %%
import os
from pathlib import Path

from braindecode.datasets.base import BaseConcatDataset
from braindecode.datautil import load_concat_dataset
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)
from eegdash.api import EEGDashDataset
from eegdash.dataset.dataset import EEGChallengeDataset


def process_data(
    releases,
    tasks,
    target_name=None,
    cache_dir=Path("/mnt/v1/arno/eeg2025"),
    output_dir="data",
):
    for release in releases:
        missing = []
        for task in tasks:
            cached_data_folder_name = output_dir + "/" + release + "_" + task
            if not os.path.exists(cached_data_folder_name):
                missing.append(task)

        if missing == []:
            continue  # skip this one

        params = dict(
            {
                "cache_dir": cache_dir,
                "task": tasks,
            }
        )
        if target_name is not None:
            params["target_name"] = target_name

        if release[0] == "R" and release != "R12":
            hbn_flag = True
            ds_data = EEGChallengeDataset(
                **params, release=release, mini=False, download=False
            )
        elif release == "R12":
            hbn_flag = True
            ds_data = EEGDashDataset(**params, dataset="HBN-R12_L100", download=False)
        elif os.path.exists(os.path.join(cache_dir, release)):
            hbn_flag = False
            ds_data = EEGDashDataset(**params, dataset=release, download=False)
        else:
            hbn_flag = False
            ds_data = EEGDashDataset(**params, dataset=release, download=True)

        # Preprocess all datasets together to avoid threading conflicts
        if hbn_flag:
            sub_rm = [
                "NDARWV769JM7",
                "NDARME789TD2",
                "NDARUA442ZVF",
                "NDARJP304NK1",
                "NDARTY128YLU",
                "NDARDW550GU6",
                "NDARLD243KRE",
                "NDARUJ292JXV",
                "NDARBA381JGH",
            ]
            all_datasets = BaseConcatDataset(
                [
                    ds
                    for ds in ds_data.datasets
                    if not ds.description.subject in sub_rm
                    and ds.raw.n_times >= 4 * SFREQ
                    and len(ds.raw.ch_names) == 129
                ]
            )
            ch_names = [
                "E22",
                "E9",
                "E33",
                "E24",
                "E11",
                "E124",
                "E122",
                "E29",
                "E6",
                "E111",
                "E45",
                "E36",
                "E104",
                "E108",
                "E42",
                "E55",
                "E93",
                "E58",
                "E52",
                "E62",
                "E92",
                "E96",
                "E70",
                "Cz",
            ]
            preprocessors = [
                Preprocessor(
                    "pick_channels",
                    ch_names=ch_names,
                ),
                Preprocessor("resample", sfreq=128),
                Preprocessor("filter", l_freq=1, h_freq=55, picks=ch_names),
            ]
        else:
            all_datasets = BaseConcatDataset([ds for ds in ds_data.datasets])
            preprocessors = [
                Preprocessor("resample", sfreq=128),
                Preprocessor("filter", l_freq=1, h_freq=55),
            ]

        preprocess(all_datasets, preprocessors, n_jobs=16)  # Reduced from -1 to 2
        print("Preprocessing completed successfully!")

        for task in tasks:
            if len(all_datasets) > 0:
                print(f"Preprocessing {len(all_datasets.datasets)} datasets...")

                # extract windows and save to disk
                windows_ds = create_fixed_length_windows(
                    all_datasets,
                    start_offset_samples=0,
                    stop_offset_samples=None,
                    window_size_samples=256,
                    window_stride_samples=256,
                    drop_last_window=True,
                    preload=False,  # Keep preload=False to save memory
                )

                # save to disk
                cached_data_folder_name = output_dir + "/" + release + "_" + task
                os.makedirs(cached_data_folder_name, exist_ok=True)
                windows_ds.save(cached_data_folder_name, overwrite=True)

                # reload to create metadata_df.pkl
                windows_ds = load_concat_dataset(cached_data_folder_name, preload=False)
                print(
                    f"Number of datasets in {cached_data_folder_name}: {len(windows_ds.datasets)}"
                )
                print(
                    f"number of samples in {cached_data_folder_name} : {len(windows_ds)}"
                )


if __name__ == "__main__":
    process_data(
        ["ds003061"], ["P300"], cache_dir="/System/Volumes/Data/data/matlab/pca_averef/"
    )
