from pathlib import Path

import pytest

from eegdash.api import EEGDashDataset


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()


def make_minimal_bids(tmp_path: Path, dataset_id: str, folder_name: str | None = None):
    """Create minimal BIDS-like structure under tmp_path/folder_name or dataset_id.

    The filenames will always embed the dataset_id in bidspath semantics; the folder
    name can include suffixes to simulate cache suffixing (e.g., ds-xxx-bdf-mini).
    """
    root = tmp_path / (folder_name or dataset_id)
    # Create a few EEG files with different entities
    _touch(
        root / "sub-01" / "ses-01" / "eeg" / "sub-01_ses-01_task-rest_run-01_eeg.edf"
    )
    _touch(
        root / "sub-02" / "ses-01" / "eeg" / "sub-02_ses-01_task-rest_run-01_eeg.edf"
    )
    _touch(root / "sub-02" / "ses-02" / "eeg" / "sub-02_ses-02_task-eo_run-01_eeg.bdf")
    return root


def test_offline_match_all(tmp_path: Path):
    dataset_id = "ds-local"
    make_minimal_bids(tmp_path, dataset_id)
    ds = EEGDashDataset(cache_dir=tmp_path, dataset=dataset_id, download=False)
    assert len(ds.datasets) == 3


def test_offline_filter_subject(tmp_path: Path):
    dataset_id = "ds-local"
    make_minimal_bids(tmp_path, dataset_id)
    ds = EEGDashDataset(
        cache_dir=tmp_path, dataset=dataset_id, subject="01", download=False
    )
    assert len(ds.datasets) == 1
    rec = ds.datasets[0].record
    assert rec["subject"] == "01"
    assert rec["task"] == "rest"


def test_offline_filter_lists(tmp_path: Path):
    dataset_id = "ds-local"
    make_minimal_bids(tmp_path, dataset_id)
    ds = EEGDashDataset(
        cache_dir=tmp_path,
        dataset=dataset_id,
        subject=["01", "02"],
        task=["rest"],
        download=False,
    )
    # two rest recordings across subjects
    assert len(ds.datasets) == 2
    tasks = sorted([d.record["task"] for d in ds.datasets])
    assert tasks == ["rest", "rest"]


def test_offline_filter_session(tmp_path: Path):
    dataset_id = "ds-local"
    make_minimal_bids(tmp_path, dataset_id)
    ds = EEGDashDataset(
        cache_dir=tmp_path, dataset=dataset_id, session="02", download=False
    )
    assert len(ds.datasets) == 1
    rec = ds.datasets[0].record
    assert rec["session"] == "02"
    assert rec["task"] == "eo"


def test_offline_bidspath_and_suffix_rewrite(tmp_path: Path, monkeypatch):
    """Bidspath should start with dataset id (no suffix) while files are stored
    under suffixed cache root when s3_bucket indicates preprocessing.
    Also ensure no S3 is touched in offline path.
    """
    dataset_id = "ds-local"
    folder_name = f"{dataset_id}-bdf-mini"
    make_minimal_bids(tmp_path, dataset_id, folder_name=folder_name)

    # Make S3 usage explode if called; offline should not call it
    import eegdash.api as api_mod

    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError(
                "S3FileSystem should not be instantiated in offline mode"
            )

    monkeypatch.setattr(api_mod, "S3FileSystem", Boom)

    ds = EEGDashDataset(
        cache_dir=tmp_path,
        dataset=dataset_id,
        download=False,
        s3_bucket="s3://example/some_bdf_mini_bucket",
        eeg_dash_instance=object(),  # prevent constructing real EEGDash (which touches S3FileSystem)
    )
    assert len(ds.datasets) == 3
    base = ds.datasets[0]

    # Records should keep bidspath starting with dataset id (no suffix)
    assert base.record["bidspath"].split("/")[0] == dataset_id

    # Local writes should target suffixed folder
    assert base.bids_root == tmp_path / folder_name
    assert str(base.filecache).startswith(str((tmp_path / folder_name).resolve()))


def test_offline_missing_dir_raises(tmp_path: Path):
    dataset_id = "ds-does-not-exist"
    with pytest.raises(ValueError, match="Offline mode is enabled, but local data_dir"):
        EEGDashDataset(cache_dir=tmp_path, dataset=dataset_id, download=False)
