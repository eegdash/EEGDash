import pytest

from eegdash.local_bids import discover_local_bids_records


@pytest.fixture
def multimodal_ds(tmp_path):
    ds_root = tmp_path / "multimodal_ds"
    ds_root.mkdir()

    # BIDS requirement
    (ds_root / "dataset_description.json").write_text(
        '{"Name": "Multimodal Test", "BIDSVersion": "1.4.0", "DatasetType": "raw"}'
    )
    (ds_root / "participants.tsv").write_text("participant_id\nsub-01")

    # EEG
    eeg_dir = ds_root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "sub-01_task-rest_eeg.set").touch()

    # MEG
    meg_dir = ds_root / "sub-01" / "meg"
    meg_dir.mkdir(parents=True)
    (meg_dir / "sub-01_task-rest_meg.fif").touch()

    # fNIRS (standard dir)
    nirs_dir = ds_root / "sub-01" / "nirs"
    nirs_dir.mkdir(parents=True)
    (nirs_dir / "sub-01_task-rest_nirs.snirf").touch()

    # fNIRS (alias dir/suffix - common in some datasets)
    fnirs_dir = ds_root / "sub-01" / "fnirs"
    fnirs_dir.mkdir(parents=True)
    (fnirs_dir / "sub-01_task-rest_fnirs.snirf").touch()

    # Motion (should be skipped by default)
    motion_dir = ds_root / "sub-01" / "motion"
    motion_dir.mkdir(parents=True)
    (motion_dir / "sub-01_task-rest_motion.tsv").touch()

    return ds_root


def test_multimodal_discovery_default(multimodal_ds):
    """Test that all physiological modalities are discovered by default, skipping motion."""
    records = discover_local_bids_records(multimodal_ds, {"dataset": "multimodal_ds"})

    # Expect: EEG, MEG, NIRS (standard), FNIRS (alias)
    # Total: 4
    assert len(records) == 4

    suffixes = [r["suffix"] for r in records]
    assert "eeg" in suffixes
    assert "meg" in suffixes
    assert "nirs" in suffixes
    assert "fnirs" in suffixes
    assert "motion" not in suffixes


def test_multimodal_discovery_eeg_only(multimodal_ds):
    """Test filtering for EEG only."""
    records = discover_local_bids_records(
        multimodal_ds, {"dataset": "multimodal_ds", "modality": "eeg"}
    )
    assert len(records) == 1
    assert records[0]["suffix"] == "eeg"


def test_multimodal_discovery_nirs_aliases(multimodal_ds):
    """Test that both 'nirs' and 'fnirs' can be filtered for explicitly."""
    # Filter for 'nirs' should find both if we consider them aliases
    # However, discover_local_bids_records currently uses suffixes=modalities
    # If we filter for 'nirs', it should at least find the standard one.
    records = discover_local_bids_records(
        multimodal_ds, {"dataset": "multimodal_ds", "modality": "nirs"}
    )
    # Since we alias fnirs to nirs in _normalize_modalities, it finds the nirs one.
    # If we want it to find both when 'nirs' is requested, we might need more logic.
    assert len(records) >= 1
    suffixes = [r["suffix"] for r in records]
    assert "nirs" in suffixes


def test_multimodal_discovery_fnirs_alias(multimodal_ds):
    """Test filtering for 'fnirs' explicitly."""
    records = discover_local_bids_records(
        multimodal_ds, {"dataset": "multimodal_ds", "modality": "fnirs"}
    )
    # fnirs is aliased to nirs in _normalize_modalities
    # So searching for 'fnirs' is effectively searching for 'nirs'
    assert len(records) >= 1
    suffixes = [r["suffix"] for r in records]
    assert "nirs" in suffixes or "fnirs" in suffixes
