from pathlib import Path

from eegdash.const import RELEASE_TO_OPENNEURO_DATASET_MAP
from eegdash.dataset.dataset import EEGChallengeDataset
from eegdash.paths import get_default_cache_dir


def test_offline_real_data_end_to_end():
    """Use real data like in the tutorial: prefetch (online) then go offline.

    - Prefetch via EEGChallengeDataset (mini release) to the user cache
    - Instantiate an offline EEGChallengeDataset pointing at the cache
    - Compare raw shapes for one subject and basic description columns
    """
    release = "R2"
    _ = RELEASE_TO_OPENNEURO_DATASET_MAP[release]
    cache_dir = Path(get_default_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Online: construct challenge dataset (mini) and prefetch first subject
    # Limit to a single subject to keep the test lean
    subject_id = "NDARAB793GL3"  # part of R2 mini set
    ds_online = EEGChallengeDataset(
        release=release,
        cache_dir=cache_dir,
        task="RestingState",
        mini=True,
        subject=subject_id,
    )
    assert len(ds_online.datasets) > 0
    first_online = ds_online.datasets[0]
    # Trigger download of this subject's files (raw + sidecars)
    _ = first_online.raw

    # Offline: enumerate locally cached data
    ds_offline = EEGChallengeDataset(
        release=release,
        cache_dir=cache_dir,
        task="RestingState",
        download=False,
        subject=subject_id,
    )
    assert len(ds_offline.datasets) == 1
    first_offline = ds_offline.datasets[0]

    # Compare raw shapes for the same subject online vs offline
    shape_online = first_online.raw.get_data().shape
    shape_offline = first_offline.raw.get_data().shape
    assert shape_online == shape_offline

    # Basic description columns present
    for col in ("subject", "task"):
        assert col in ds_offline.description.columns


def test_offline_real_bidspath_and_cache_suffix():
    """Verify bidspath root and local cache folder for real data (tutorial style)."""
    release = "R2"
    dataset_id = RELEASE_TO_OPENNEURO_DATASET_MAP[release]
    cache_dir = Path(get_default_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)

    subject_id = "NDARAB793GL3"
    ds_offline = EEGChallengeDataset(
        release=release,
        cache_dir=cache_dir,
        task="RestingState",
        download=False,
        subject=subject_id,
    )
    assert len(ds_offline.datasets) == 1
    base = ds_offline.datasets[0]
    # bidspath must start with dataset id (not suffixed cache folder)
    assert base.record["bidspath"].split("/")[0] == dataset_id
    # local BIDS root points to suffixed folder used by the challenge
    assert (cache_dir / f"{dataset_id}-bdf-mini").exists()
    assert base.bids_root == cache_dir / f"{dataset_id}-bdf-mini"


def test_offline_real_records_description_shape():
    """Reconstruct from records and compare description row counts (tutorial-like)."""
    release = "R2"
    cache_dir = Path(get_default_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)

    subject_id = "NDARAB793GL3"
    ds_offline = EEGChallengeDataset(
        release=release,
        cache_dir=cache_dir,
        task="RestingState",
        download=False,
        subject=subject_id,
    )
    assert len(ds_offline.datasets) == 1

    # Recreate a dataset from the exact offline records
    records = [bd.record for bd in ds_offline.datasets]
    ds_from_records = EEGChallengeDataset(
        release=release, cache_dir=cache_dir, task="RestingState", records=records
    )

    assert ds_offline.description.shape[0] == ds_from_records.description.shape[0]


def test_online_vs_records_vs_offline_single_subject():
    """Compare online vs records-injection vs offline for a single subject.

    Ensures consistent row counts and identical raw data shapes across modes.
    """
    release = "R2"
    subject_id = "NDARAB793GL3"
    cache_dir = Path(get_default_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Online for a single subject, and prefetch raw
    ds_online = EEGChallengeDataset(
        release=release,
        cache_dir=cache_dir,
        task="RestingState",
        mini=True,
        subject=subject_id,
    )
    assert len(ds_online.datasets) == 1
    online_base = ds_online.datasets[0]
    _ = online_base.raw

    # From records (inject the online records directly)
    records = [d.record for d in ds_online.datasets]
    ds_records = EEGChallengeDataset(
        release=release,
        cache_dir=cache_dir,
        task="RestingState",
        records=records,
    )
    assert len(ds_records.datasets) == 1

    # Offline: enumerate from cache for same subject
    ds_offline = EEGChallengeDataset(
        release=release,
        cache_dir=cache_dir,
        task="RestingState",
        download=False,
        subject=subject_id,
    )
    assert len(ds_offline.datasets) == 1

    # Compare row counts in description
    assert ds_online.description.shape[0] == 1
    assert ds_records.description.shape[0] == 1
    assert ds_offline.description.shape[0] == 1

    # Compare raw shapes across modes
    shape_online = ds_online.datasets[0].raw.get_data().shape
    shape_records = ds_records.datasets[0].raw.get_data().shape
    shape_offline = ds_offline.datasets[0].raw.get_data().shape
    assert shape_online == shape_records == shape_offline
