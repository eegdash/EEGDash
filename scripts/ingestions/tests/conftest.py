"""Pytest configuration and shared fixtures for ingestion tests.

Fixtures defined here are auto-discovered by pytest in every test file
under ``tests/``. Keep this file small — module-specific fixtures live
next to the tests that use them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from eegdash.testing import data_path

# Make the ingestion package importable without `pip install -e .` —
# this is the smallest layer that lets `from _set_parser import …` work
# from the test files. Once the package is properly installed via the
# pyproject.toml, this becomes redundant; for now it keeps `pytest`
# discoverable from any CWD.
_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

# Also make tests/ itself importable so subfolder tests can write
# `from _helpers.builders import …` without having to spell out
# `tests._helpers.builders` everywhere.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))


# ─── Fixture paths (lazily fetched from eegdash-testing-data) ──────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures"  # only inline: records/


def _testing_data_root() -> Path:
    """Resolve the testing-data corpus root, skipping the test on failure."""
    try:
        return data_path()
    except Exception as exc:  # fetch failure → skip cleanly
        pytest.skip(f"eegdash-testing-data unavailable: {exc}")


@pytest.fixture(scope="session")
def eeg_fixtures_dir() -> Path:
    """Path to EEG-modality fixtures (EDF/BDF/BV/SET) inside the testing-data corpus."""
    return _testing_data_root() / "eeg"


@pytest.fixture(scope="session")
def ieeg_fixtures_dir() -> Path:
    """Path to iEEG-modality fixtures (BrainVision triple, MEF3 header)."""
    return _testing_data_root() / "ieeg"


@pytest.fixture(scope="session")
def meg_fixtures_dir() -> Path:
    """Path to MEG-modality fixtures (FIFF samples from MNE-Python BSD-3)."""
    return _testing_data_root() / "meg"


@pytest.fixture(scope="session")
def digest_snapshots_dir() -> Path:
    """Directory containing ``digest_snapshots/{inputs,outputs}/ds_snapshot_*``."""
    return _testing_data_root() / "digest_snapshots"
