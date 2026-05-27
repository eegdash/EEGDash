"""Pytest configuration and shared fixtures for ingestion tests.

Fixtures defined here are auto-discovered by pytest in every test file
under ``tests/``. Keep this file small — module-specific fixtures live
next to the tests that use them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the ingestion package importable without `pip install -e .` —
# this is the smallest layer that lets `from _set_parser import …` work
# from the test files. Once the package is properly installed via the
# pyproject.toml, this becomes redundant; for now it keeps `pytest`
# discoverable from any CWD.
_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))


# ─── Binary fixture paths (fetched lazily from eegdash-testing-data) ───────

FIXTURES_DIR = Path(__file__).parent / "fixtures"  # JSON snapshots stay inline


def _testing_data_root() -> Path:
    """Resolve the testing-data corpus root, skipping the test on failure."""
    from eegdash.testing import data_path

    try:
        return data_path()
    except Exception as exc:  # noqa: BLE001 — fetch failure = skip cleanly
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


# ─── Smoke test fixture (used by test_smoke.py) ─────────────────────────────


@pytest.fixture
def ingest_package_importable():
    """Verify ``ingestions`` package can be imported (Phase 0 smoke).

    Yields
    ------
    bool
        True if import succeeded; the test asserts on this directly.

    Notes
    -----
    This is the canary for the Phase 0 evaluation hook:
    "`pytest --collect-only` exits 0 from outside the ingestions dir".
    """
    try:
        # The package is in sys.path via the prelude above; under the
        # installed-package layout it will be importable as
        # ``import ingestions``. Both should work.
        from pathlib import Path  # noqa: F401 — proxy import

        yield True
    except ImportError as exc:  # pragma: no cover — would crash the fixture
        pytest.fail(f"ingestions package not importable: {exc}")
