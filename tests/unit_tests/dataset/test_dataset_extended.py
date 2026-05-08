from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, write_raw_bids

from eegdash.dataset.dataset import EEGChallengeDataset, EEGDashDataset


@pytest.fixture
def mock_dash_client():
    mock = MagicMock()
    mock.find.return_value = []
    return mock


@pytest.fixture
def toy_bids_dataset(tmp_path):
    """Create a valid BIDS dataset using MNE-BIDS."""
    bids_root = tmp_path / "ds1"
    bids_root.mkdir()

    # Create dummy raw data
    sfreq = 100
    info = mne.create_info(ch_names=["O1", "O2", "Cz"], sfreq=sfreq, ch_types="eeg")
    data = np.zeros((3, sfreq * 10))  # 10 seconds
    raw = mne.io.RawArray(data, info)

    # Write to BIDS
    bids_path = BIDSPath(
        subject="01", task="rest", datatype="eeg", root=bids_root, extension=".set"
    )
    # verbose=False suppresses prints
    write_raw_bids(
        raw,
        bids_path,
        verbose=False,
        overwrite=True,
        allow_preload=True,
        format="EEGLAB",
    )

    return bids_root


def test_eegdashdataset_records_init(toy_bids_dataset):
    # Determine the actual filename created by MNE-BIDS
    # Check what exists (using EEGLAB usually .set)
    files = list((toy_bids_dataset / "sub-01/eeg").glob("*.set"))

    assert files, "No BIDS file created?"
    fpath = files[0]
    filename = fpath.name
    relpath = f"sub-01/eeg/{filename}"
    bidspath = f"ds1/{relpath}"

    # Init directly with records (no DB call)
    records = [
        {
            "dataset": "ds1",
            "bids_relpath": relpath,
            "bidspath": bidspath,
            "storage": {
                "backend": "local",
                "base": str(toy_bids_dataset),
            },  # ds1 is the dataset folder
            "entities_mne": {"subject": "01", "task": "rest"},
        }
    ]

    # We pass cache_dir as the parent of ds1 directory
    ds = EEGDashDataset(
        cache_dir=toy_bids_dataset.parent,
        dataset="ds1",
        records=records,
        download=False,
    )
    assert len(ds.datasets) == 1
    assert ds.datasets[0].bidspath.subject == "01"

    # Ensure it loads correctly (no Bad EDF error)
    raw = ds.datasets[0].raw
    assert len(raw) == 1000  # 10s * 100Hz


def test_eegdashdataset_infer_dataset_id(tmp_path):
    # Infer dataset ID from records
    records = [
        {
            "dataset": "ds_inferred",
            "bids_relpath": "file.edf",
            "bidspath": "ds_inferred/file.edf",
            "storage": {"backend": "local", "base": str(tmp_path)},
        }
    ]
    # Create directory to satisfy offline check
    (tmp_path / "ds_inferred").mkdir(parents=True, exist_ok=True)

    ds = EEGDashDataset(cache_dir=tmp_path, records=records, download=False)
    assert ds.query["dataset"] == "ds_inferred"


def test_eegdashdataset_deduplication(toy_bids_dataset):
    files = list((toy_bids_dataset / "sub-01/eeg").glob("*.set"))
    fpath = files[0]
    filename = fpath.name
    relpath = f"sub-01/eeg/{filename}"
    bidspath = f"ds1/{relpath}"

    # Duplicate records
    rec = {
        "dataset": "ds1",
        "bids_relpath": relpath,
        "bidspath": bidspath,
        "storage": {"backend": "local", "base": str(toy_bids_dataset)},
        "entities_mne": {"subject": "01", "task": "rest"},
    }
    records = [rec, rec]

    # Without dedupe
    ds = EEGDashDataset(
        cache_dir=toy_bids_dataset.parent,
        dataset="ds1",
        records=records,
        download=False,
    )
    assert len(ds.datasets) == 2

    # With dedupe
    ds_dedupe = EEGDashDataset(
        cache_dir=toy_bids_dataset.parent,
        dataset="ds1",
        records=records,
        download=False,
        _dedupe_records=True,
    )
    assert len(ds_dedupe.datasets) == 1


def test_eegdashdataset_offline_missing_dir(tmp_path):
    # Download=False but dir missing
    with pytest.raises(ValueError, match="Offline mode is enabled, but local data_dir"):
        EEGDashDataset(cache_dir=tmp_path, dataset="ds_missing", download=False)


def test_eegchallenge_validation(tmp_path):
    # Unknown release
    with pytest.raises(ValueError, match="Unknown release: RX"):
        EEGChallengeDataset(release="RX", cache_dir=tmp_path)

    # Query with dataset key
    with pytest.raises(ValueError, match="Query using the parameters `dataset`"):
        EEGChallengeDataset(release="R1", cache_dir=tmp_path, query={"dataset": "foo"})


def test_eegchallenge_mini_subject_validation(toy_bids_dataset):
    # Mock map
    with patch(
        "eegdash.dataset.dataset.SUBJECT_MINI_RELEASE_MAP", {"R-Test": ["001", "002"]}
    ):
        with patch(
            "eegdash.dataset.dataset.RELEASE_TO_OPENNEURO_DATASET_MAP",
            {"R-Test": "ds1"},
        ):  # Match toy dataset ID
            # Use toy data for valid record
            files = list((toy_bids_dataset / "sub-01/eeg").glob("*.set"))
            fpath = files[0]
            filename = fpath.name
            relpath = f"sub-01/eeg/{filename}"
            bidspath = f"ds1/{relpath}"

            dummy_records = [
                {
                    "dataset": "ds1",
                    "bids_relpath": relpath,
                    "bidspath": bidspath,
                    "storage": {"backend": "local", "base": str(toy_bids_dataset)},
                    "entities_mne": {
                        "subject": "001"
                    },  # Map 01 to 001 logic? Or just use "01" if logic permits.
                    # Actually challenge dataset enforces subject list.
                    # Let's mock map to include "01" to match our toy.
                }
            ]

    # Retry with "01" support in mock
    with patch(
        "eegdash.dataset.dataset.SUBJECT_MINI_RELEASE_MAP", {"R-Test": ["01", "02"]}
    ):
        with patch(
            "eegdash.dataset.dataset.RELEASE_TO_OPENNEURO_DATASET_MAP",
            {"R-Test": "ds1"},
        ):
            files = list((toy_bids_dataset / "sub-01/eeg").glob("*.set"))
            fpath = files[0]
            filename = fpath.name
            relpath = f"sub-01/eeg/{filename}"
            bidspath = f"ds1/{relpath}"

            dummy_records = [
                {
                    "dataset": "ds1",
                    "bids_relpath": relpath,
                    "bidspath": bidspath,
                    "storage": {"backend": "local", "base": str(toy_bids_dataset)},
                    "entities_mne": {"subject": "01", "task": "rest"},
                }
            ]

            ds = EEGChallengeDataset(
                release="R-Test",
                cache_dir=toy_bids_dataset.parent,
                subject="01",
                download=False,
                records=dummy_records,
            )
            assert ds.query["subject"] == "01"  # Single subject

            # Invalid subject for mini
            with pytest.raises(ValueError, match="Some requested subject"):
                EEGChallengeDataset(
                    release="R-Test",
                    cache_dir=toy_bids_dataset.parent,
                    subject="99",
                    mini=True,
                )


@pytest.mark.parametrize(
    ("query", "records", "kwargs", "expected_dataset"),
    [
        (
            {"dataset": "ds_query"},
            [
                {
                    "dataset": "ds_query",
                    "bidspath": "ds_query/a.set",
                    "bids_relpath": "a.set",
                }
            ],
            {"task": "RestingState"},
            "ds_query",
        ),
        (
            None,
            [
                {
                    "dataset": "ds_kwargs",
                    "bidspath": "ds_kwargs/a.set",
                    "bids_relpath": "a.set",
                }
            ],
            {"dataset": "ds_kwargs", "subject": "01"},
            "ds_kwargs",
        ),
        (
            {},
            [
                {
                    "dataset": "ds_inferred",
                    "bidspath": "ds_inferred/a.set",
                    "bids_relpath": "a.set",
                }
            ],
            {},
            "ds_inferred",
        ),
    ],
)
def test_eegdashdataset_constructor_dataset_resolution(
    tmp_path, query, records, kwargs, expected_dataset
):
    with patch("eegdash.dataset.dataset.EEGDashRaw"):
        ds = EEGDashDataset(
            cache_dir=tmp_path,
            query=query,
            records=records,
            download=True,
            **kwargs,
        )
    assert ds.query["dataset"] == expected_dataset


@pytest.mark.parametrize(
    "query,records,kwargs",
    [
        (None, [], {}),
        ({}, [{"bidspath": "no-dataset/a.set", "bids_relpath": "a.set"}], {}),
    ],
)
def test_eegdashdataset_constructor_missing_dataset_raises(
    tmp_path, query, records, kwargs
):
    with pytest.raises(ValueError, match="You must provide a 'dataset' argument"):
        EEGDashDataset(
            cache_dir=tmp_path,
            query=query,
            records=records,
            download=True,
            **kwargs,
        )


@pytest.mark.parametrize("cache_input", ["", None])
def test_eegdashdataset_constructor_uses_default_cache_dir(tmp_path, cache_input):
    records = [
        {"dataset": "ds_cache", "bidspath": "ds_cache/a.set", "bids_relpath": "a.set"}
    ]
    with (
        patch(
            "eegdash.dataset.dataset.get_default_cache_dir", return_value=str(tmp_path)
        ),
        patch("eegdash.dataset.dataset.EEGDashRaw"),
    ):
        ds = EEGDashDataset(
            cache_dir=cache_input,
            records=records,
            download=True,
        )
    assert str(ds.cache_dir) == str(tmp_path)


def test_eegdashdataset_constructor_creates_missing_cache_dir(tmp_path):
    missing_cache = tmp_path / "new-cache-dir"
    records = [
        {"dataset": "ds_cache", "bidspath": "ds_cache/a.set", "bids_relpath": "a.set"}
    ]
    with patch("eegdash.dataset.dataset.EEGDashRaw"):
        _ = EEGDashDataset(cache_dir=missing_cache, records=records, download=True)
    assert missing_cache.exists()


@pytest.mark.parametrize("on_error", ["raise", "warn", "skip"])
def test_eegdashdataset_constructor_passes_on_error_to_raw(tmp_path, on_error):
    records = [
        {
            "dataset": "ds_on_error",
            "bidspath": "ds_on_error/a.set",
            "bids_relpath": "a.set",
        }
    ]
    with patch("eegdash.dataset.dataset.EEGDashRaw") as mock_raw:
        _ = EEGDashDataset(
            cache_dir=tmp_path,
            records=records,
            download=True,
            on_error=on_error,
        )
    assert mock_raw.call_args.kwargs["on_error"] == on_error


@pytest.mark.parametrize(
    ("database", "auth_token", "expected_kwargs"),
    [
        (None, None, {}),
        ("eegdash_staging", None, {"database": "eegdash_staging"}),
        (None, "token-123", {"auth_token": "token-123"}),
        (
            "eegdash_staging",
            "token-123",
            {"database": "eegdash_staging", "auth_token": "token-123"},
        ),
    ],
)
def test_eegdashdataset_constructor_builds_eegdash_client_kwargs(
    tmp_path, database, auth_token, expected_kwargs
):
    with (
        patch("eegdash.api.EEGDash") as mock_eegdash,
        patch(
            "eegdash.dataset.dataset.EEGDashDataset._find_datasets",
            return_value=[MagicMock()],
        ),
        patch(
            "eegdash.dataset.dataset.downloader.get_s3_filesystem",
            return_value=MagicMock(),
        ),
    ):
        _ = EEGDashDataset(
            cache_dir=tmp_path,
            dataset="ds_remote",
            database=database,
            auth_token=auth_token,
            download=True,
        )

    mock_eegdash.assert_called_once_with(**expected_kwargs)


@pytest.mark.parametrize(
    ("s3_bucket", "expected_backend"),
    [(None, "https"), ("s3://mirror-bucket", "s3")],
)
def test_eegdashdataset_constructor_source_override_with_s3_bucket(
    tmp_path, s3_bucket, expected_backend
):
    records = [
        {
            "dataset": "ds_source",
            "bidspath": "ds_source/a.set",
            "bids_relpath": "a.set",
            "storage": {"backend": "https", "base": "https://original.example"},
        }
    ]
    with patch("eegdash.dataset.dataset.EEGDashRaw"):
        ds = EEGDashDataset(
            cache_dir=tmp_path,
            records=records,
            download=True,
            s3_bucket=s3_bucket,
        )

    if s3_bucket:
        assert ds.records[0]["storage"]["base"] == s3_bucket
    else:
        assert ds.records[0]["storage"]["base"] == "https://original.example"
    assert ds.records[0]["storage"]["backend"] == expected_backend
