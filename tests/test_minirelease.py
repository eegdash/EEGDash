import os
from pathlib import Path

import numpy as np
import pytest

from eegdash.dataset.dataset import EEGChallengeDataset


@pytest.fixture(scope="session")
def warmed_mongo(cache_dir: Path):
    """Skip tests gracefully if Mongo is not reachable."""
    try:
        # Lazy import to avoid circulars; constructing EEGChallengeDataset will touch DB
        _ = EEGChallengeDataset(release="R5", mini=True, cache_dir=cache_dir)
    except Exception:
        pytest.skip("Mongo not reachable")


def test_minirelease_vs_full_counts_and_subjects(warmed_mongo, cache_dir: Path):
    """Mini release should have fewer files and (typically) fewer subjects than full release."""
    release = "R5"

    ds_mini = EEGChallengeDataset(release=release, mini=True, cache_dir=cache_dir)
    ds_full = EEGChallengeDataset(release=release, mini=False, cache_dir=cache_dir)

    # File count: mini must be strictly smaller than full
    assert len(ds_mini.datasets) < len(ds_full.datasets)

    # Subject cardinality: mini should be strictly less than full, and > 0
    subj_mini = ds_mini.description["subject"].nunique()
    subj_full = ds_full.description["subject"].nunique()
    assert subj_mini > 0
    assert subj_mini < subj_full


def test_minirelease_subject_raw_equivalence(warmed_mongo, cache_dir: Path):
    """For a subject present in the mini set, loading that subject in mini vs full yields identical raw data."""
    release = "R5"

    # Pick a concrete subject from the mini set to avoid guessing
    ds_mini_all = EEGChallengeDataset(release=release, mini=True, cache_dir=cache_dir)
    assert len(ds_mini_all.datasets) > 0
    subject = ds_mini_all.description["subject"].iloc[0]

    ds_mini = EEGChallengeDataset(
        release=release,
        mini=True,
        cache_dir=cache_dir,
    )
    ds_full = EEGChallengeDataset(
        release=release, mini=False, cache_dir=cache_dir, subject=subject
    )

    assert len(ds_mini.datasets) > 0
    assert len(ds_full.datasets) > 0

    # Identify a common BIDS file (bidspath) present in both (bucket prefixes differ between mini/full)
    mini_paths = {d.record["bidspath"] for d in ds_mini.datasets}
    full_paths = {d.record["bidspath"] for d in ds_full.datasets}
    intersection = mini_paths & full_paths
    assert intersection, "No common recordings found for the chosen subject"

    common_path = next(iter(intersection))
    mini_idx = next(
        i for i, d in enumerate(ds_mini.datasets) if d.record["bidspath"] == common_path
    )
    full_idx = next(
        i for i, d in enumerate(ds_full.datasets) if d.record["bidspath"] == common_path
    )

    raw_mini = ds_mini.datasets[mini_idx].raw
    raw_full = ds_full.datasets[full_idx].raw

    # Basic metadata equivalence
    assert raw_mini.info["sfreq"] == raw_full.info["sfreq"]
    assert raw_mini.info["nchan"] == raw_full.info["nchan"]
    assert raw_mini.ch_names == raw_full.ch_names

    # Compare a small data slice to ensure content equality (avoid loading entire arrays into memory)
    n_samples = min(1000, raw_mini.n_times, raw_full.n_times)
    assert n_samples > 0
    data_mini = raw_mini.get_data(picks=[0], start=0, stop=n_samples)
    data_full = raw_full.get_data(picks=[0], start=0, stop=n_samples)
    assert np.allclose(data_mini, data_full, rtol=1e-6, atol=0), (
        "Raw data mismatch between mini and full"
    )


@pytest.mark.slow
def test_minirelease_consume_everything(warmed_mongo, cache_dir: Path):
    """Smoke-test that a few mini-release recordings can be read.

    Loading every recording in the mini release is network- and I/O-heavy and
    makes the default test suite slow and flaky. Set ``EEGDASH_TEST_LOAD_ALL=1``
    to restore the exhaustive behavior when desired.
    """
    release = "R5"
    ds_mini = EEGChallengeDataset(release=release, mini=True, cache_dir=cache_dir)

    load_all = os.getenv("EEGDASH_TEST_LOAD_ALL", "").strip() == "1"
    n_smoke = int(os.getenv("EEGDASH_TEST_MAX_RAW_LOADS", "1"))
    to_load = ds_mini.datasets if load_all else ds_mini.datasets[: max(n_smoke, 0)]

    for dataset in to_load:
        raw = dataset.raw  # noqa: F841
        description = dataset.description  # noqa: F841
        assert raw is not None
        assert description is not None
