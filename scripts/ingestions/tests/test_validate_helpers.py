"""Unit tests for ``_validate.py`` — was at 0% coverage (C1.2).

``_validate.py`` is the schema gate used by Stages 4 (validate output)
and 5 (inject). Its ``VALID_STORAGE_PATTERNS`` dict is the only thing
between MongoDB and a misrouted record's storage URL. Despite this it
had ZERO tests before this commit.

"""

from __future__ import annotations

from pathlib import Path

import pytest

_INGEST_DIR = Path(__file__).resolve().parent.parent
from _validate import (
    DATA_QUALITY_FIELDS,
    NEURO_EXTENSIONS,
    RECOMMENDED_DATASET_FIELDS,
    RECOMMENDED_RECORD_FIELDS,
    VALID_SOURCES,
    VALID_STORAGE_PATTERNS,
    ValidationResult,
    validate_storage_url,
)

# ─── VALID_STORAGE_PATTERNS — happy paths per source ──────────────────────


@pytest.mark.parametrize(
    ("source", "url"),
    [
        ("openneuro", "s3://openneuro.org/ds002893"),
        ("openneuro", "s3://openneuro.org/ds002893/sub-01/eeg/file.edf"),
        ("nemar", "s3://nemar/nm000176"),
        ("nemar", "s3://nmdatasets/nm000176"),
        ("osf", "https://files.osf.io/v1/resources/abc/providers/osfstorage/"),
        ("figshare", "https://figshare.com/ndownloader/files/12345"),
        ("figshare", "https://ndownloader.figshare.com/files/12345"),
        ("figshare", "https://mydomain.figshare.com/articles/12345"),
        ("zenodo", "https://zenodo.org/record/12345"),
        ("zenodo", "https://zenodo.org/records/12345"),
        ("scidb", "https://scidb.cn/dataset/abc"),
        ("scidb", "https://www.scidb.cn/dataset/abc"),
        ("datarn", "https://webdav.data.ru.nl/test/path"),
        ("gin", "https://gin.g-node.org/EEGManyLabs/study"),
    ],
)
def test_storage_url_accepts_canonical_per_source(source: str, url: str):
    """Each source's known-good URL shape is accepted."""
    ok, msg = validate_storage_url(source, url)
    assert ok is True, f"{source=} {url=} → {msg=}"
    assert msg == ""


# ─── VALID_STORAGE_PATTERNS — cross-source mismatches rejected ────────────


@pytest.mark.parametrize(
    ("source", "wrong_url"),
    [
        # OpenNeuro URL on NEMAR dataset (the pre-PR-#327 misrouting bug)
        ("nemar", "s3://openneuro.org/nm000176"),
        # NEMAR URL on OpenNeuro
        ("openneuro", "s3://nemar/ds002893"),
        # Zenodo URL on Figshare
        ("figshare", "https://zenodo.org/record/12345"),
        # OSF URL on Zenodo
        ("zenodo", "https://files.osf.io/v1/abc"),
        # Random URL on OpenNeuro
        ("openneuro", "https://example.com/ds002893"),
        # NEMAR URL on Zenodo
        ("zenodo", "s3://nemar/12345"),
    ],
)
def test_storage_url_rejects_cross_source_mismatch(source: str, wrong_url: str):
    """A URL with the wrong source's pattern is rejected with a message."""
    ok, msg = validate_storage_url(source, wrong_url)
    assert ok is False
    assert source in msg
    assert wrong_url in msg


def test_storage_url_unknown_source_passes_through():
    """Unknown sources can't be validated — return True with empty message."""
    ok, msg = validate_storage_url("brand_new_source", "https://example.com/")
    assert ok is True
    assert msg == ""


def test_storage_url_must_match_at_start():
    """Patterns are anchored — URL with the right shape mid-string is rejected."""
    ok, _msg = validate_storage_url("openneuro", "garbage_s3://openneuro.org/ds123")
    assert ok is False


# ─── Constant-set invariants ──────────────────────────────────────────────


def test_valid_sources_includes_canonical():
    """Every Source mentioned in ADR 0001 / cycle 1 must be in VALID_SOURCES."""
    expected = {
        "openneuro",
        "nemar",
        "gin",
        "figshare",
        "zenodo",
        "osf",
        "scidb",
        "datarn",
    }
    assert expected.issubset(VALID_SOURCES)


def test_valid_storage_patterns_cover_all_valid_sources_except_hbn():
    """``hbn`` is in VALID_SOURCES but doesn't have a storage pattern —
    that's intentional (HBN is a logical grouping, not a Source per se).

    Every OTHER valid source should have a storage pattern.
    """
    sources_with_patterns = set(VALID_STORAGE_PATTERNS.keys())
    sources_without_patterns = VALID_SOURCES - sources_with_patterns
    # Only ``hbn`` is allowed to lack a pattern.
    assert sources_without_patterns <= {"hbn"}


def test_neuro_extensions_includes_canonical_formats():
    """The 7 BIDS-supported neuro file formats must be present."""
    expected = {".edf", ".bdf", ".vhdr", ".set", ".fif", ".snirf"}
    assert expected.issubset(NEURO_EXTENSIONS)


def test_data_quality_fields_match_record_schema():
    """The data-quality fields enforced for ingest match the Record schema."""
    assert "nchans" in DATA_QUALITY_FIELDS
    assert "sampling_frequency" in DATA_QUALITY_FIELDS


def test_recommended_fields_are_string_lists():
    """Recommended field lists are non-empty lists of strings."""
    assert all(
        isinstance(f, str) and f
        for f in RECOMMENDED_RECORD_FIELDS + RECOMMENDED_DATASET_FIELDS
    )


def test_storage_patterns_compile():
    """Every regex pattern compiles (catches typos at import time)."""
    import re

    for source, pattern in VALID_STORAGE_PATTERNS.items():
        try:
            re.compile(pattern)
        except re.error as e:
            pytest.fail(f"VALID_STORAGE_PATTERNS[{source!r}] = {pattern!r} fails: {e}")


# ─── ValidationResult ──────────────────────────────────────────────────────


def test_validation_result_initialises_empty():
    """A fresh ValidationResult has empty errors + warnings + zero counts."""
    r = ValidationResult()
    assert r.errors == []
    assert r.warnings == []
    assert r.stats["datasets_checked"] == 0
    assert r.stats["records_checked"] == 0
    assert r.stats["storage_errors"] == 0


def test_validation_result_stats_keys_are_stable():
    """Pins the stats key names so downstream consumers don't break silently."""
    r = ValidationResult()
    expected_keys = {
        "datasets_checked",
        "records_checked",
        "storage_errors",
        "missing_mandatory",
        "missing_recommended",
        "empty_datasets",
        "invalid_source",
        "zip_placeholders",
        "missing_nchans",
        "missing_sampling_frequency",
    }
    assert set(r.stats.keys()) == expected_keys
