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


# ─── Paths ──────────────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures"
EEG_FIXTURES = FIXTURES_DIR / "eeg"
IEEG_FIXTURES = FIXTURES_DIR / "ieeg"
MEG_FIXTURES = FIXTURES_DIR / "meg"


@pytest.fixture(scope="session")
def eeg_fixtures_dir() -> Path:
    """Path to EEG-modality fixtures (EDF/BDF/BV/SET).

    Returns
    -------
    Path
        Directory containing the small CC0 EEG samples used by parser tests.
    """
    return EEG_FIXTURES


@pytest.fixture(scope="session")
def ieeg_fixtures_dir() -> Path:
    """Path to iEEG-modality fixtures (BrainVision triple)."""
    return IEEG_FIXTURES


@pytest.fixture(scope="session")
def meg_fixtures_dir() -> Path:
    """Path to MEG-modality fixtures (FIFF samples from MNE-Python BSD-3)."""
    return MEG_FIXTURES


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
