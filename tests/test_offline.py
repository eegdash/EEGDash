from pathlib import Path


from eegdash.const import RELEASE_TO_OPENNEURO_DATASET_MAP
from eegdash.dataset.dataset import EEGChallengeDataset


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()


def _make_tutorial_like_cache(tmp_path: Path, dataset_id: str) -> Path:
    """Create a minimal local BIDS cache structure similar to the tutorial.

    We simulate a pre-fetched mini release folder "<dataset_id>-bdf-mini" with two
    subjects, each having two RestingState runs (01, 02). File presence is
    sufficient; tests do not read EEG contents.
    """
    root = tmp_path / f"{dataset_id}-bdf-mini"
    # Minimal dataset_description.json to satisfy BIDS tools
    (root).mkdir(parents=True, exist_ok=True)
    desc = root / "dataset_description.json"
    if not desc.exists():
        desc.write_text('{"Name": "test", "BIDSVersion": "1.6.0"}')
    # Subject A: two runs
    _touch(
        root
        / "sub-NDARAB793GL3"
        / "ses-01"
        / "eeg"
        / "sub-NDARAB793GL3_ses-01_task-RestingState_run-01_eeg.edf"
    )
    _touch(
        root
        / "sub-NDARAB793GL3"
        / "ses-01"
        / "eeg"
        / "sub-NDARAB793GL3_ses-01_task-RestingState_run-02_eeg.edf"
    )
    # Subject B: two runs
    _touch(
        root
        / "sub-NDARAM675UR8"
        / "ses-01"
        / "eeg"
        / "sub-NDARAM675UR8_ses-01_task-RestingState_run-01_eeg.edf"
    )
    _touch(
        root
        / "sub-NDARAM675UR8"
        / "ses-01"
        / "eeg"
        / "sub-NDARAM675UR8_ses-01_task-RestingState_run-02_eeg.edf"
    )
    return root


def test_offline_basic_and_dedup(tmp_path: Path):
    """Offline EEGChallengeDataset finds one RestingState recording per subject.

    Mirrors the tutorial's Step 2 (offline) with the run deduplication implied by
    the curated DB (keep run-01 when run is not specified).
    """
    dataset_id = RELEASE_TO_OPENNEURO_DATASET_MAP["R2"]
    _make_tutorial_like_cache(tmp_path, dataset_id)

    ds_offline = EEGChallengeDataset(
        release="R2", cache_dir=tmp_path, task="RestingState", download=False
    )

    # We created 2 subjects; without specifying run, only run-01 should be included
    assert len(ds_offline.datasets) == 2
    runs = sorted([d.record.get("run") for d in ds_offline.datasets])
    assert runs == ["01", "01"]


def test_offline_filter_by_subject(tmp_path: Path):
    dataset_id = RELEASE_TO_OPENNEURO_DATASET_MAP["R2"]
    _make_tutorial_like_cache(tmp_path, dataset_id)

    ds_sub = EEGChallengeDataset(
        release="R2",
        cache_dir=tmp_path,
        download=False,
        subject="NDARAB793GL3",
    )
    assert len(ds_sub.datasets) == 1
    rec = ds_sub.datasets[0].record
    assert rec["subject"] == "NDARAB793GL3"
    assert rec["task"] == "RestingState"
    assert rec["run"] == "01"


def test_offline_bidspath_and_cache_suffix(tmp_path: Path):
    """Bidspath starts with dataset id; cache path uses suffixed folder.

    Equivalent to the tutorial's notion of "offline_root = cache_dir/<dataset>-bdf-mini".
    """
    dataset_id = RELEASE_TO_OPENNEURO_DATASET_MAP["R2"]
    cache_folder = f"{dataset_id}-bdf-mini"
    _make_tutorial_like_cache(tmp_path, dataset_id)

    ds_offline = EEGChallengeDataset(
        release="R2", cache_dir=tmp_path, task="RestingState", download=False
    )
    assert len(ds_offline.datasets) == 2
    base = ds_offline.datasets[0]
    # bidspath must start with dataset id (not suffixed cache folder)
    assert base.record["bidspath"].split("/")[0] == dataset_id
    # local BIDS root points to suffixed folder
    assert base.bids_root == tmp_path / cache_folder


def test_offline_vs_records_description_shape(tmp_path: Path):
    """Descriptions built offline match shape when reconstructed from records.

    Mirrors the tutorial's Step 4.1, using record injection to simulate "online".
    """
    dataset_id = RELEASE_TO_OPENNEURO_DATASET_MAP["R2"]
    _make_tutorial_like_cache(tmp_path, dataset_id)

    ds_offline = EEGChallengeDataset(
        release="R2", cache_dir=tmp_path, task="RestingState", download=False
    )

    # Recreate a dataset from the exact records (record-injection path)
    records = [bd.record for bd in ds_offline.datasets]
    ds_from_records = EEGChallengeDataset(
        release="R2", cache_dir=tmp_path, task="RestingState", records=records
    )

    # At minimum, number of recordings (rows) must match
    assert ds_offline.description.shape[0] == ds_from_records.description.shape[0]
    # And offline descriptions should include core BIDS entities
    for col in ("subject", "task", "run"):
        assert col in ds_offline.description.columns
