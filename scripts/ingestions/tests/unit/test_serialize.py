"""Unit tests for ``_serialize.py`` — deterministic JSON / dataset-id helpers.

``_serialize.py`` powers the dataset-listing utilities shared by every
``1_fetch_sources/*`` script. Critical reproducibility surface (stable
bytes, deterministic dataset_id, deduplication) — the regressions caught
here are silent bit-drift in the source-listing artefacts.
"""

from __future__ import annotations

import json
import re as _re
import sys
from pathlib import Path

import pytest

from _serialize import (
    SUBJECT_COUNT_PATTERNS,
    deduplicate_dataset_ids,
    extract_subjects_count,
    extract_surname,
    extract_year,
    generate_dataset_id,
    normalize_dataset,
    save_datasets_deterministically,
    setup_paths,
)

# ─── setup_paths ──────────────────────────────────────────────────────────


def test_setup_paths_is_idempotent():
    """Second call must not grow sys.path further (idempotency invariant)."""
    before = len(sys.path)
    setup_paths()
    after_first = len(sys.path)
    setup_paths()
    after_second = len(sys.path)
    assert after_first == after_second
    assert after_first - before <= 2


# ─── extract_subjects_count ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(None, 0, id="empty_none"),
        pytest.param("", 0, id="empty_string"),
        pytest.param("This study had 42 subjects", 42, id="had_N_subjects"),
        pytest.param("Data from 17 participants", 17, id="from_N_participants"),
        pytest.param("N = 25 healthy controls", 25, id="N_equals_with_qualifier"),
        pytest.param("n = 30", 30, id="n_lowercase_equals_space"),
        pytest.param("N=8 patients", 8, id="N_uppercase_no_space"),
        pytest.param("5 individuals", 5, id="individuals"),
        pytest.param("12 volunteers", 12, id="volunteers"),
        pytest.param("8 children", 8, id="children"),
        pytest.param("15 adults", 15, id="adults"),
        pytest.param("recorded from 20 subjects", 20, id="recorded_from"),
        pytest.param("data from 7 patients", 7, id="data_from_patients"),
        pytest.param("ID 99999 subjects", 0, id="value_above_10000_rejected"),
        pytest.param("This is just a description", 0, id="no_keyword"),
        pytest.param("Music perception study", 0, id="unrelated_text"),
    ],
)
def test_subjects_count_extracts(text, expected: int):
    """Empty / pattern variants / sanity-rejection matrix."""
    assert extract_subjects_count(text) == expected


# ─── extract_surname ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # Firstname Lastname
        pytest.param("John Smith", "Smith", id="firstname-lastname"),
        pytest.param("Alice Marie Johnson", "Johnson", id="multi-middle-name"),
        # Lastname, Firstname (BibTeX-style)
        pytest.param("Smith, John", "Smith", id="bibtex-comma"),
        pytest.param("Johnson, Alice Marie", "Johnson", id="bibtex-multi"),
        # Initials shouldn't end up as the surname
        pytest.param("J. Smith", "Smith", id="single-initial-firstname"),
        pytest.param("F. M. Lastname", "Lastname", id="two-initials"),
        # Suffixes (Jr/Sr/III/Ph.D.) stripped
        pytest.param("John Smith Jr.", "Smith", id="suffix-jr"),
        pytest.param("Mary Johnson Ph.D.", "Johnson", id="suffix-phd"),
        pytest.param("Robert Brown III", "Brown", id="suffix-iii"),
        # Accented characters normalised to ASCII for ID stability
        pytest.param("José García", "Garcia", id="unicode-spanish"),
        pytest.param("Müller", "Muller", id="unicode-umlaut"),
        # None / empty / length-1 → None
        pytest.param("", None, id="empty"),
        pytest.param(None, None, id="none"),
        pytest.param("   ", None, id="whitespace-only"),
        pytest.param("X", None, id="length-1"),
    ],
)
def test_surname_extracts(name, expected):
    """Name formats / suffixes / unicode / invalid input matrix."""
    assert extract_surname(name) == expected


# ─── extract_year ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # ISO date / datetime
        pytest.param("2024-01-15T10:30:00Z", "2024", id="iso-datetime"),
        pytest.param("2024-01-15", "2024", id="iso-date"),
        pytest.param("2024", "2024", id="year-only"),
        # Found in arbitrary text
        pytest.param("published in 2023", "2023", id="in-text"),
        pytest.param("doi:10.5281/zenodo.2019", "2019", id="in-doi"),
        # None / empty / no match → None
        pytest.param(None, None, id="none"),
        pytest.param("", None, id="empty"),
        pytest.param("some text with no year", None, id="no-year"),
        # Only 19xx / 20xx considered sensible
        pytest.param("1850 study", None, id="century-too-old"),
        pytest.param("2100 prediction", None, id="century-too-new"),
        pytest.param("1999 published", "1999", id="1999-edge"),
        pytest.param("2099 forecast", "2099", id="2099-edge"),
    ],
)
def test_extract_year(text, expected):
    """ISO / free-text / range-limit / no-match matrix."""
    assert extract_year(text) == expected


# ─── generate_dataset_id ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("args", "kwargs", "expected"),
    [
        pytest.param(
            ("zenodo", ["John Smith"], "2024"),
            {},
            "Smith2024",
            id="surname-year-format",
        ),
        pytest.param(
            ("zenodo", ["", "X", "Alice Johnson"], "2023"),
            {},
            "Johnson2023",
            id="walks-list-for-extractable-surname",
        ),
        pytest.param(
            ("figshare",),
            {"authors": [], "fallback_id": "12345"},
            "figshare_12345",
            id="fallback-to-source-id",
        ),
        pytest.param(
            ("scidb",),
            {},
            "scidb_unknown",
            id="unknown-when-nothing-works",
        ),
        # index=2 → suffix _3 (1-indexed in users' eyes)
        pytest.param(
            ("zenodo", ["Smith, J."], "2024"),
            {"index": 2},
            "Smith2024_3",
            id="appends-disambiguating-index",
        ),
    ],
)
def test_generate_dataset_id(args, kwargs, expected):
    """Surname-year / fallback / index matrix."""
    assert generate_dataset_id(*args, **kwargs) == expected


# ─── deduplicate_dataset_ids ──────────────────────────────────────────────
# Kept standalone: each case has a structurally different assertion shape
# (ordered-list equality vs distinctness via len(set()) vs other-fields
# preservation) — parametrising would force a branch-on-id inside the body
# which defeats the readability win.


def test_dedup_handles_no_duplicates():
    datasets = [{"dataset_id": "Smith2023"}, {"dataset_id": "Johnson2024"}]
    out = deduplicate_dataset_ids(datasets)
    assert [d["dataset_id"] for d in out] == ["Smith2023", "Johnson2024"]


def test_dedup_renames_duplicates_with_suffix():
    datasets = [
        {"dataset_id": "Smith2024"},
        {"dataset_id": "Smith2024"},
        {"dataset_id": "Smith2024"},
    ]
    out = deduplicate_dataset_ids(datasets)
    ids = [d["dataset_id"] for d in out]
    assert ids[0] == "Smith2024"
    assert len(set(ids)) == 3


def test_dedup_preserves_other_fields():
    """Only ``dataset_id`` is rewritten; other fields untouched."""
    datasets = [
        {"dataset_id": "X", "name": "alpha", "year": 2023},
        {"dataset_id": "X", "name": "beta", "year": 2024},
    ]
    out = deduplicate_dataset_ids(datasets)
    assert out[0]["name"] == "alpha"
    assert out[1]["name"] == "beta"
    assert out[0]["year"] == 2023
    assert out[1]["year"] == 2024


# ─── save_datasets_deterministically ──────────────────────────────────────


def test_save_deterministic_produces_stable_bytes(tmp_path: Path):
    """Saving twice with the same input → byte-identical files."""
    datasets = [
        {"dataset_id": "Z", "tasks": ["rest"]},
        {"dataset_id": "A", "tasks": ["motor", "language"]},
    ]
    out1 = tmp_path / "first.json"
    out2 = tmp_path / "second.json"
    save_datasets_deterministically(datasets, out1)
    save_datasets_deterministically(datasets, out2)
    assert out1.read_bytes() == out2.read_bytes()


def test_save_sorts_by_dataset_id(tmp_path: Path):
    """Output is sorted by ``dataset_id`` for stable diffs across runs."""
    datasets = [{"dataset_id": "Zzz"}, {"dataset_id": "Aaa"}, {"dataset_id": "Mmm"}]
    out = tmp_path / "sorted.json"
    save_datasets_deterministically(datasets, out)
    payload = json.loads(out.read_text())
    assert isinstance(payload, list)
    ids = [d["dataset_id"] for d in payload]
    assert ids == sorted(ids)


@pytest.mark.parametrize(
    ("kwargs", "expected_unique"),
    [
        pytest.param({}, 2, id="default-dedups"),
        pytest.param({"deduplicate_ids": False}, 1, id="skip-dedup-preserves-dupes"),
    ],
)
def test_save_deduplication_toggle(tmp_path: Path, kwargs, expected_unique):
    """``deduplicate_ids`` flag: default dedups, False preserves duplicate IDs."""
    datasets = [{"dataset_id": "Smith2024"}, {"dataset_id": "Smith2024"}]
    out = tmp_path / "x.json"
    save_datasets_deterministically(datasets, out, **kwargs)
    payload = json.loads(out.read_text())
    ids = [d["dataset_id"] for d in payload]
    assert len(set(ids)) == expected_unique


# ─── normalize_dataset ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(
            {"dataset_id": "X", "name": "Test", "year": None, "modalities": ["eeg"]},
            {"dataset_id": "X", "name": "Test", "modalities": ["eeg"]},
            id="strips-none-values",
        ),
        pytest.param(
            {"dataset_id": "X", "ages": [42, 21, 35]},
            {"dataset_id": "X", "ages": [42, 21, 35]},
            id="preserves-list-order",
        ),
        pytest.param(
            {"dataset_id": "X", "metadata": {"keep": "yes", "drop": None}},
            {"dataset_id": "X", "metadata": {"keep": "yes"}},
            id="recurses-into-nested-dicts",
        ),
    ],
)
def test_normalize_dataset(raw, expected):
    """None-stripping + list-order preservation + nested-dict recursion."""
    assert normalize_dataset(raw) == expected


# ─── SUBJECT_COUNT_PATTERNS sanity check ──────────────────────────────────


def test_subject_count_patterns_compile():
    """Every pattern is a valid regex (catches typos at import time)."""
    for pattern in SUBJECT_COUNT_PATTERNS:
        _re.compile(pattern)
