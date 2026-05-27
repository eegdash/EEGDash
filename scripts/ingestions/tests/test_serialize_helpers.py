"""Unit tests for ``_serialize.py`` — was at 0% coverage (C1.2).

``_serialize.py`` provides the dataset-listing utilities shared by
the 7 source-fetch scripts in ``1_fetch_sources/``. Despite being a
critical reproducibility path (deterministic JSON, dataset_id
generation, deduplication), it had NO tests before this commit.

close the dead zone by exercising every
public helper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_INGEST_DIR = Path(__file__).resolve().parent.parent
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
    """Calling setup_paths twice doesn't accumulate duplicate sys.path entries."""
    before = len(sys.path)
    setup_paths()
    after_first = len(sys.path)
    setup_paths()
    after_second = len(sys.path)
    # Either no-op or only-once: either way, the second call must not
    # grow sys.path further.
    assert after_first == after_second
    # And first call shouldn't blow up sys.path either (we're in test
    # context where the paths are already present).
    assert after_first - before <= 2


# ─── extract_subjects_count ────────────────────────────────────────────────


def test_subjects_count_empty_input_returns_zero():
    assert extract_subjects_count(None) == 0
    assert extract_subjects_count("") == 0


def test_subjects_count_finds_simple_phrasing():
    assert extract_subjects_count("This study had 42 subjects") == 42
    assert extract_subjects_count("Data from 17 participants") == 17
    assert extract_subjects_count("N = 25 healthy controls") == 25


def test_subjects_count_handles_n_equals_pattern():
    assert extract_subjects_count("n = 30") == 30
    assert extract_subjects_count("N=8 patients") == 8


def test_subjects_count_rejects_unreasonable_counts():
    """Values outside (0, 10000) are sanity-rejected (likely false matches)."""
    assert extract_subjects_count("ID 99999 subjects") == 0
    # Note: the regex matches digits before keywords, so the value is
    # tested against the < 10000 sanity bound.


def test_subjects_count_handles_multiple_phrasings():
    """Several common phrasings — all should match."""
    cases = {
        "5 individuals": 5,
        "12 volunteers": 12,
        "8 children": 8,
        "15 adults": 15,
        "recorded from 20 subjects": 20,
        "data from 7 patients": 7,
    }
    for text, expected in cases.items():
        assert extract_subjects_count(text) == expected, f"{text=}"


def test_subjects_count_returns_zero_when_no_pattern_matches():
    assert extract_subjects_count("This is just a description") == 0
    assert extract_subjects_count("Music perception study") == 0


# ─── extract_surname ──────────────────────────────────────────────────────


def test_surname_extracts_from_firstname_lastname():
    assert extract_surname("John Smith") == "Smith"
    assert extract_surname("Alice Marie Johnson") == "Johnson"


def test_surname_extracts_from_comma_separated():
    """'Lastname, Firstname' is the BibTeX-style format."""
    assert extract_surname("Smith, John") == "Smith"
    assert extract_surname("Johnson, Alice Marie") == "Johnson"


def test_surname_handles_initials_in_middle():
    """'F. M. Lastname' or 'F. Lastname' — initials shouldn't be the surname."""
    result = extract_surname("J. Smith")
    assert result == "Smith"
    result = extract_surname("F. M. Lastname")
    assert result == "Lastname"


def test_surname_strips_common_suffixes():
    """Jr / Sr / III / Ph.D. shouldn't end up as the surname."""
    assert extract_surname("John Smith Jr.") == "Smith"
    assert extract_surname("Mary Johnson Ph.D.") == "Johnson"
    assert extract_surname("Robert Brown III") == "Brown"


def test_surname_normalizes_unicode():
    """Accented characters get stripped to ASCII for ID stability."""
    assert extract_surname("José García") == "Garcia"
    assert extract_surname("Müller") == "Muller"


def test_surname_returns_none_for_empty_or_invalid():
    assert extract_surname("") is None
    assert extract_surname(None) is None
    assert extract_surname("   ") is None
    # Length-1 result after normalization should be None
    assert extract_surname("X") is None


# ─── extract_year ──────────────────────────────────────────────────────────


def test_extract_year_iso_date():
    assert extract_year("2024-01-15T10:30:00Z") == "2024"
    assert extract_year("2024-01-15") == "2024"


def test_extract_year_year_only():
    assert extract_year("2024") == "2024"


def test_extract_year_finds_in_arbitrary_text():
    assert extract_year("published in 2023") == "2023"
    assert extract_year("doi:10.5281/zenodo.2019") == "2019"


def test_extract_year_returns_none_for_no_year():
    assert extract_year(None) is None
    assert extract_year("") is None
    assert extract_year("some text with no year") is None


def test_extract_year_only_19xx_or_20xx():
    """Only sensible centuries — '1850' or '2100' shouldn't be valid."""
    assert extract_year("1850 study") is None
    assert extract_year("2100 prediction") is None
    assert extract_year("1999 published") == "1999"
    assert extract_year("2099 forecast") == "2099"


# ─── generate_dataset_id ──────────────────────────────────────────────────


def test_dataset_id_surname_year_format():
    assert generate_dataset_id("zenodo", ["John Smith"], "2024") == "Smith2024"


def test_dataset_id_uses_first_with_extractable_surname():
    """Walks the authors list until it finds an extractable surname."""
    assert (
        generate_dataset_id("zenodo", ["", "X", "Alice Johnson"], "2023")
        == "Johnson2023"
    )


def test_dataset_id_falls_back_to_source_id_when_no_surname():
    assert (
        generate_dataset_id("figshare", authors=[], fallback_id="12345")
        == "figshare_12345"
    )


def test_dataset_id_unknown_when_nothing_works():
    assert generate_dataset_id("scidb") == "scidb_unknown"


def test_dataset_id_appends_disambiguating_index():
    """index > 0 adds a suffix for collision resolution."""
    result = generate_dataset_id("zenodo", ["Smith, J."], "2024", index=2)
    assert result == "Smith2024_3"  # index=2 → suffix _3 (1-indexed in users' eyes)


# ─── deduplicate_dataset_ids ──────────────────────────────────────────────


def test_dedup_handles_no_duplicates():
    datasets = [
        {"dataset_id": "Smith2023"},
        {"dataset_id": "Johnson2024"},
    ]
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
    # First is unchanged; subsequent ones get _N suffixes
    assert ids[0] == "Smith2024"
    assert ids[1] != ids[0]
    assert ids[2] != ids[0]
    assert ids[2] != ids[1]
    assert len(set(ids)) == 3  # all distinct after dedup


def test_dedup_preserves_other_fields():
    """The function only rewrites dataset_id; other fields untouched."""
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
    """Calling save twice with the same input produces byte-identical files."""
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
    """The output list is sorted by dataset_id for stable diff across runs."""
    datasets = [
        {"dataset_id": "Zzz"},
        {"dataset_id": "Aaa"},
        {"dataset_id": "Mmm"},
    ]
    out = tmp_path / "sorted.json"
    save_datasets_deterministically(datasets, out)
    payload = json.loads(out.read_text())
    # Output is a list of dataset dicts sorted by dataset_id.
    assert isinstance(payload, list)
    ids = [d["dataset_id"] for d in payload]
    assert ids == sorted(ids)


def test_save_deduplicates_by_default(tmp_path: Path):
    """Duplicate dataset_ids are renamed by default."""
    datasets = [
        {"dataset_id": "Smith2024"},
        {"dataset_id": "Smith2024"},
    ]
    out = tmp_path / "dedup.json"
    save_datasets_deterministically(datasets, out)
    payload = json.loads(out.read_text())
    ids = [d["dataset_id"] for d in payload]
    assert len(set(ids)) == 2  # both unique after dedup


def test_save_can_skip_deduplication(tmp_path: Path):
    """``deduplicate_ids=False`` preserves duplicate IDs as-is."""
    datasets = [
        {"dataset_id": "Smith2024"},
        {"dataset_id": "Smith2024"},
    ]
    out = tmp_path / "dup.json"
    save_datasets_deterministically(datasets, out, deduplicate_ids=False)
    payload = json.loads(out.read_text())
    ids = [d["dataset_id"] for d in payload]
    assert ids == ["Smith2024", "Smith2024"]


# ─── normalize_dataset ────────────────────────────────────────────────────


def test_normalize_dataset_strips_none_values():
    """``None`` values are removed from the output dict."""
    raw = {
        "dataset_id": "X",
        "name": "Test",
        "year": None,
        "modalities": ["eeg"],
    }
    out = normalize_dataset(raw)
    assert "year" not in out
    assert out["name"] == "Test"
    assert out["modalities"] == ["eeg"]


def test_normalize_dataset_preserves_list_order():
    """The docstring's promise about ages — list order preserved (not sorted)."""
    raw = {"dataset_id": "X", "ages": [42, 21, 35]}
    out = normalize_dataset(raw)
    assert out["ages"] == [42, 21, 35]


def test_normalize_dataset_recurses_into_nested_dicts():
    """None values in nested dicts also get removed."""
    raw = {
        "dataset_id": "X",
        "metadata": {"keep": "yes", "drop": None},
    }
    out = normalize_dataset(raw)
    assert out["metadata"] == {"keep": "yes"}
    assert "drop" not in out["metadata"]


# ─── SUBJECT_COUNT_PATTERNS sanity check ──────────────────────────────────


def test_subject_count_patterns_compile():
    """Every pattern is a valid regex (catches typos at import time)."""
    import re as _re

    for pattern in SUBJECT_COUNT_PATTERNS:
        _re.compile(pattern)  # ValueError on bad regex
