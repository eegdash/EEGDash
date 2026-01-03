from unittest.mock import patch

import pytest

from eegdash.dataset.bids_dataset import EEGBIDSDataset
from eegdash.dataset.dataset import EEGDashDataset, EEGDashRaw


def test_dataset_download_all_njobs(tmp_path):
    # Coverage for download_all branches
    record = {
        "dataset": "ds_dl",
        "data_name": "ds_dl_f.set",
        "bidspath": "ds_dl/f.set",
        "bids_relpath": "f.set",
        "storage": {"base": "s3://b", "backend": "s3", "raw_key": "f.set"},
    }

    with patch("eegdash.api.EEGDash") as MockDash:
        MockDash.return_value.find.return_value = [record]
        with patch("eegdash.dataset.dataset.downloader.get_s3_filesystem"):
            ds = EEGDashDataset(cache_dir=str(tmp_path), dataset="ds_dl", download=True)

            # Patch the method on the CLASS so Parallel/Delayed sees it
            with patch.object(EEGDashRaw, "_download_required_files") as mock_dl:
                # Test n_jobs=1
                ds.download_all(n_jobs=1)
                assert mock_dl.call_count == 1

                # Test n_jobs=2 (Parallel branch)
                mock_dl.reset_mock()
                ds.download_all(n_jobs=2)
                assert mock_dl.call_count == 1


def test_dataset_offline_enrichment(tmp_path):
    # Coverage for lines 284-298 in dataset.py (Offline mode enrichment)
    d = tmp_path / "ds_offline"
    d.mkdir()
    subj_dir = d / "sub-01" / "eeg"
    subj_dir.mkdir(parents=True)
    f = subj_dir / "sub-01_task-rest_eeg.set"
    f.touch()

    # participants.tsv
    (d / "participants.tsv").write_text("participant_id\tage\nsub-01\t25")
    (d / "dataset_description.json").write_text('{"Name": "DS"}')

    # Mock discover_local_bids_records with valid v2 record
    record = {
        "dataset": "ds_offline",
        "data_name": "ds_offline_sub-01_task-rest_eeg.set",
        "bidspath": "ds_offline/sub-01/eeg/sub-01_task-rest_eeg.set",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
        "subject": "01",
        "storage": {"base": str(d), "backend": "local"},
    }

    with patch(
        "eegdash.dataset.dataset.discover_local_bids_records", return_value=[record]
    ):
        ds = EEGDashDataset(
            cache_dir=str(tmp_path), dataset="ds_offline", download=False
        )
        # Verify enrichment
        assert ds.datasets[0].description["age"] == "25"


def test_bids_dataset_inheritance_full(tmp_path):
    # Coverage for _get_bids_file_inheritance recursion (Line 688+)
    root = tmp_path / "bids_root"
    root.mkdir()
    (root / "dataset_description.json").touch()
    (root / "participants.tsv").write_text("participant_id\tage\nsub-01\t30")

    sub = root / "sub-01"
    sub.mkdir()
    (sub / "sub-01_eeg.json").write_text('{"Task": "inherited"}')

    eeg_dir = sub / "eeg"
    eeg_dir.mkdir()
    f = eeg_dir / "sub-01_task-test_eeg.set"
    f.touch()

    ds = EEGBIDSDataset(data_dir=str(root), dataset="bids_root")
    # This should trigger recursion up to root
    inh_files = ds._get_bids_file_inheritance(eeg_dir, "sub-01_task-test", "json")
    # It finds sub-01_eeg.json which is at sub-01 level, the parent of eeg_dir.
    # Recursion: eeg_dir -> sub -> root (finds dataset_description.json and stops)
    assert any("sub-01_eeg.json" in str(p) for p in inh_files)


def test_dataset_various_errors(tmp_path):
    # Coverage for rare error branches
    # 1. No dataset in records fallback
    with pytest.raises(ValueError, match="provide a 'dataset' argument"):
        EEGDashDataset(
            cache_dir=str(tmp_path), records=[{"bidspath": "p"}], download=True
        )

    # 2. Offline mode but directory missing
    with pytest.raises(ValueError, match="Offline mode is enabled, but local data_dir"):
        EEGDashDataset(cache_dir=str(tmp_path), dataset="missing_ds", download=False)


def test_bids_dataset_search_modalities(tmp_path):
    # Coverage for multiple modalities in _init_bids_paths
    d = tmp_path / "ds_multi"
    d.mkdir()
    # EEG uses .set, MEG uses .fif
    for mod, ext in [("eeg", "set"), ("meg", "fif")]:
        mod_dir = d / "sub-01" / mod
        mod_dir.mkdir(parents=True)
        (mod_dir / f"sub-01_{mod}.{ext}").touch()

    (d / "dataset_description.json").touch()

    # must match ONE of MODALITIES
    ds = EEGBIDSDataset(data_dir=str(d), dataset="ds_multi", modalities=["eeg", "meg"])
    assert len(ds.files) >= 2
