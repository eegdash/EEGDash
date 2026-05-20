"""Characterisation tests for the pure helpers inside ``3_digest.py``.

Phase 8 of the robustness programme. These tests pin the CURRENT
behaviour of the small, pure-function helpers inside ``3_digest.py``
so we can refactor the file's mega-functions
(``digest_from_manifest`` = 631 LOC, ``extract_record`` = 521 LOC, ...)
under a safety net.

The bias here is intentional: we DON'T test the mega-functions yet.
We test the leaves first — entity parsing, neuro-file detection — and
in later commits the mega-functions are decomposed under the same
safety pattern.

The module is loaded via ``importlib`` because its filename starts
with a digit (``3_digest.py`` is not a legal Python identifier).
This is a known pattern for CLI-style script files.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

# ─── Module loader ──────────────────────────────────────────────────────────

_DIGEST_PATH = Path(__file__).parent.parent / "3_digest.py"


@pytest.fixture(scope="module")
def digest() -> ModuleType:
    """Load ``3_digest.py`` as an importable module despite its numeric name."""
    spec = importlib.util.spec_from_file_location("digest_under_test", _DIGEST_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── parse_bids_entities_from_path — characterisation ──────────────────────


def test_parse_entities_full_path_with_ses_and_run(digest: ModuleType) -> None:
    """The fully-entity-typed BIDS path returns every field populated."""
    path = "ds002893/sub-001/ses-01/eeg/sub-001_ses-01_task-rest_run-01_eeg.edf"
    result = digest.parse_bids_entities_from_path(path)
    assert result["subject"] == "001"
    assert result["session"] == "01"
    assert result["task"] == "rest"
    assert result["run"] == "01"
    assert result["modality"] == "eeg"
    assert result["datatype"] == "eeg"
    assert result["suffix"] == "eeg"


def test_parse_entities_minimal_path_no_session_no_run(digest: ModuleType) -> None:
    """The minimal BIDS path (sub + task only) omits session/run gracefully."""
    path = "ds002336/sub-xp101/eeg/sub-xp101_task-motorloc_eeg.vhdr"
    result = digest.parse_bids_entities_from_path(path)
    assert result["subject"] == "xp101"
    assert result["task"] == "motorloc"
    assert result["modality"] == "eeg"
    # Missing keys must be absent (not None / not "")
    assert "session" not in result
    assert "run" not in result


@pytest.mark.parametrize(
    ("path", "expected_subject"),
    [
        ("ds00/sub-01/eeg/sub-01_eeg.edf", "01"),
        ("ds00/sub-001/eeg/sub-001_eeg.edf", "001"),
        ("ds00/sub-A1B2/eeg/sub-A1B2_eeg.edf", "A1B2"),
        ("ds00/sub-xp101/eeg/sub-xp101_eeg.edf", "xp101"),
    ],
)
def test_parse_entities_subject_parses_alphanumeric(
    digest: ModuleType, path: str, expected_subject: str
) -> None:
    """Subject IDs can be alphanumeric (``sub-xp101`` is legal BIDS)."""
    result = digest.parse_bids_entities_from_path(path)
    assert result["subject"] == expected_subject


@pytest.mark.parametrize(
    ("ext", "modality"),
    [
        (".edf", "eeg"),
        (".bdf", "eeg"),
        (".vhdr", "eeg"),
        (".set", "eeg"),
    ],
)
def test_parse_entities_recovers_modality_from_path(
    digest: ModuleType, ext: str, modality: str
) -> None:
    """Modality is derived from the BIDS datatype directory, not the extension."""
    path = f"ds00/sub-01/{modality}/sub-01_task-rest_{modality}{ext}"
    result = digest.parse_bids_entities_from_path(path)
    assert result["modality"] == modality
    assert result["datatype"] == modality


# ─── is_neuro_data_file — characterisation ────────────────────────────────


@pytest.mark.parametrize(
    ("filename", "is_neuro"),
    [
        # Binary recording formats ARE neuro data
        ("sub-001_eeg.edf", True),
        ("sub-001_eeg.bdf", True),
        ("sub-001_eeg.vhdr", True),
        ("sub-001_eeg.set", True),
        # Sidecars are NOT
        ("sub-001_eeg.json", False),
        ("sub-001_channels.tsv", False),
        ("participants.tsv", False),
        ("dataset_description.json", False),
        # BIDS folder-structure files are NOT
        ("README.md", False),
        ("CHANGES", False),
    ],
)
def test_is_neuro_data_file_classifies_correctly(
    digest: ModuleType, filename: str, is_neuro: bool
) -> None:
    """Recording binaries → True; sidecars/metadata → False."""
    assert digest.is_neuro_data_file(filename) is is_neuro


def test_is_neuro_data_file_is_case_sensitive_on_extension(
    digest: ModuleType,
) -> None:
    """An ``.EDF`` uppercase extension follows the same classification."""
    # The function's documented behaviour: it normalises to lowercase
    # internally, so both pass. Pinning that here.
    assert digest.is_neuro_data_file("sub-001_eeg.EDF") is True
    assert digest.is_neuro_data_file("sub-001_eeg.Edf") is True


# ─── A deliberate canary: the mega-functions exist with documented sizes ───
# ─── (informational; not a regression — refactor lands in later commits) ──


def test_megafunction_line_counts_are_known_baseline(digest: ModuleType) -> None:
    """Document the current LOC of the mega-functions as the Phase 8 baseline.

    These are the targets for decomposition. The test asserts each is
    BIG, which is *deliberately* the opposite of normal — when the
    decomposition lands, this test gets REMOVED in that PR.
    """
    import inspect

    big_functions = {
        "digest_from_manifest": 631,
        "extract_record": 521,
        "extract_dataset_metadata": 360,
        "digest_dataset": 302,
    }
    for name, baseline_loc in big_functions.items():
        fn = getattr(digest, name, None)
        if fn is None:
            pytest.skip(f"{name} not found — function may have been split already")
        source_lines = inspect.getsourcelines(fn)[0]
        actual_loc = len(source_lines)
        # Phase 8 LOC ceiling is 80 (per ROBUSTNESS/02-STYLE_GUIDE § 6).
        # Once a function drops below 100, the baseline assertion below
        # FAILS — that's the signal that decomposition has landed.
        assert actual_loc > 100, (
            f"{name} dropped below 100 LOC ({actual_loc}). Phase 8 "
            "decomposition appears complete — REMOVE this test in "
            "the decomposition PR, replace with per-helper unit tests."
        )
        # Track drift downward — if a refactor accidentally GROWS a
        # mega-function, that's a problem worth noticing.
        assert actual_loc <= baseline_loc + 20, (
            f"{name} grew from {baseline_loc} → {actual_loc} LOC. "
            "Mega-functions should only shrink."
        )
