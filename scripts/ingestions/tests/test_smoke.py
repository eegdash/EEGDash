"""Smoke tests — Phase 0 evaluation canary.

These tests verify that the package layout works at all. They are
deliberately minimal: if any of them fails, Phase 0 has not landed,
and no further phase can proceed.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_package_layout_collects():
    """`pytest --collect-only` must succeed from the ingestion root."""
    # If pytest is running this test, collection has already succeeded.
    # The assertion is the test's existence; we use a trivial check
    # so the function body is not empty.
    assert True


def test_fixtures_directory_exists(eeg_fixtures_dir: Path) -> None:
    """The CC0 fixture corpus is present.

    Without these fixtures, Phase 1 parser tests cannot run.
    """
    assert eeg_fixtures_dir.exists(), (
        f"fixtures dir missing at {eeg_fixtures_dir} — "
        "run `cp -r ../eegdash-viewer/tests/fixtures/eeg tests/fixtures/`"
    )
    edf_files = list(eeg_fixtures_dir.glob("*.edf"))
    assert len(edf_files) >= 1, "expected at least one .edf fixture"


def test_attribution_file_present(eeg_fixtures_dir: Path) -> None:
    """Fixture provenance must be documented.

    Every fixture binary in this directory is derived from a CC0 or
    BSD-licensed source. The attribution file is the contract that
    keeps this redistributable.
    """
    attribution = eeg_fixtures_dir / "LICENSE-ATTRIBUTION.md"
    assert attribution.exists(), (
        "tests/fixtures/eeg/LICENSE-ATTRIBUTION.md must document the "
        "source of each binary fixture (BIDS dataset accession + license)."
    )
    text = attribution.read_text()
    assert "CC0" in text, "attribution must mention the CC0 license"


@pytest.mark.parametrize(
    ("modality", "suffix"),
    [
        ("eeg", ".edf"),
        ("eeg", ".bdf"),
        ("eeg", ".vhdr"),
        ("eeg", ".set"),
        ("ieeg", ".vhdr"),
        ("meg", ".fif"),
    ],
)
def test_modality_format_coverage(
    modality: str,
    suffix: str,
    eeg_fixtures_dir: Path,
    ieeg_fixtures_dir: Path,
    meg_fixtures_dir: Path,
) -> None:
    """At least one fixture exists for each (modality, format) pair.

    This is the matrix that proves the test corpus covers every reader
    we are expected to test. If a future contributor adds a new reader
    (CTF, SNIRF, etc.), the matrix grows.
    """
    fixtures_root = {
        "eeg": eeg_fixtures_dir,
        "ieeg": ieeg_fixtures_dir,
        "meg": meg_fixtures_dir,
    }[modality]
    matches = list(fixtures_root.glob(f"*{suffix}"))
    assert matches, (
        f"no {modality}{suffix} fixture in {fixtures_root}. "
        "Extend tests/fixtures/ per ROBUSTNESS/07-DETAILS.md § Phase 1."
    )
