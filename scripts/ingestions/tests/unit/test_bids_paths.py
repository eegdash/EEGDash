"""Tests for ``_bids.py`` — BIDS structure validation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from _bids import (
    BIDS_DATASET_ZIP_PATTERN,
    BIDS_OPTIONAL_FILES,
    BIDS_REQUIRED_FILES,
    BIDS_SUBJECT_PATTERN,
    collect_bids_matches,
    count_bad_channels,
    find_channels_tsv,
    validate_bids_structure_from_files,
    validate_bids_structure_from_names,
)

# ─── collect_bids_matches ─────────────────────────────────────────────────


def test_collect_matches_required_files_case_insensitive():
    """Required-file lookup is case-insensitive."""
    out = collect_bids_matches(
        ["Dataset_Description.JSON", "other.txt"],
        required_files=["dataset_description.json"],
        optional_files=[],
    )
    assert "dataset_description.json" in out["required_found"]


def test_collect_returns_empty_for_empty_input():
    out = collect_bids_matches(
        [],
        required_files=["dataset_description.json"],
        optional_files=[],
    )
    assert out["required_found"] == []
    assert out["subject_files"] == []
    assert out["bids_zip_files"] == []


def test_collect_finds_subject_folders():
    out = collect_bids_matches(
        ["sub-01", "sub-02", "README.md"],
        required_files=[],
        optional_files=[],
    )
    assert sorted(out["subject_files"]) == ["sub-01", "sub-02"]
    assert out["subject_zips"] == []


def test_collect_finds_subject_zips():
    """Subject zips are flagged separately for downstream extraction."""
    out = collect_bids_matches(
        ["sub-01.zip", "sub-02.zip", "data.csv"],
        required_files=[],
        optional_files=[],
    )
    assert sorted(out["subject_zips"]) == ["sub-01.zip", "sub-02.zip"]
    assert sorted(out["subject_files"]) == ["sub-01.zip", "sub-02.zip"]


def test_collect_with_dataset_zip_pattern_finds_bids_zips():
    """BIDS_DATASET_ZIP_PATTERN matches 'bids'/'dataset' prefixes and '_bids.zip' suffix; plain 'data.zip' does not match."""
    out = collect_bids_matches(
        ["dataset_bids_v1.zip", "study_bids.zip", "data.zip", "results.csv"],
        required_files=[],
        optional_files=[],
        dataset_zip_pattern=BIDS_DATASET_ZIP_PATTERN,
    )
    assert "dataset_bids_v1.zip" in out["bids_zip_files"]
    assert "study_bids.zip" in out["bids_zip_files"]
    assert "data.zip" not in out["bids_zip_files"]
    assert "results.csv" not in out["bids_zip_files"]


def test_collect_dataset_zip_matcher_search_finds_pattern_anywhere():
    """``matcher='search'`` finds patterns anywhere in the filename."""
    files = ["prefix_data_bids.zip"]
    search_result = collect_bids_matches(
        files,
        required_files=[],
        optional_files=[],
        dataset_zip_pattern=BIDS_DATASET_ZIP_PATTERN,
        dataset_zip_matcher="search",
    )
    assert "prefix_data_bids.zip" in search_result["bids_zip_files"]


# ─── validate_bids_structure_from_names ────────────────────────────────────


@pytest.mark.parametrize(
    ("names", "kwargs", "checks"),
    [
        pytest.param(
            ["dataset_description.json", "sub-01", "sub-02", "README"],
            {},
            {
                "is_bids": True,
                "subject_count": 2,
                "bids_files_found_includes": "dataset_description.json",
            },
            id="test_validate_structure_recognises_bids_root",
        ),
        pytest.param(
            ["data1.csv", "data2.csv", "README.txt"],
            {},
            {"is_bids": False, "subject_count": 0},
            id="test_validate_structure_rejects_non_bids_layout",
        ),
        pytest.param(
            ["sub-01.zip", "sub-02.zip", "sub-03.zip"],
            {},
            {"is_bids": True, "subject_count": 3, "has_subject_zips": True},
            id="test_validate_structure_accepts_subject_zips_alone",
        ),
        pytest.param(
            ["sub-01"],
            {},
            {"is_bids": False},
            id="test_validate_structure_requires_min_subjects",
        ),
        pytest.param(
            ["data_bids.zip"],
            {"dataset_zip_pattern": BIDS_DATASET_ZIP_PATTERN},
            {"is_bids": True, "has_bids_zip": True},
            id="test_validate_structure_accepts_dataset_zip",
        ),
        pytest.param(
            ["sub-01", "sub-02"],
            {"include_subject_files": True},
            {"subject_files_sorted": ["sub-01", "sub-02"]},
            id="test_validate_structure_include_subject_files_returns_list",
        ),
        pytest.param(
            [f"sub-{i:03d}" for i in range(50)],
            {"include_subject_files": True, "subject_files_limit": 5},
            {"subject_files_len": 5},
            id="test_validate_structure_subject_files_limit_respected",
        ),
    ],
)
def test_validate_structure_from_names(names, kwargs, checks):
    """validate_bids_structure_from_names behaves correctly across layouts."""
    out = validate_bids_structure_from_names(names, **kwargs)
    if "is_bids" in checks:
        assert out["is_bids"] is checks["is_bids"]
    if "subject_count" in checks:
        assert out["subject_count"] == checks["subject_count"]
    if "bids_files_found_includes" in checks:
        assert checks["bids_files_found_includes"] in out["bids_files_found"]
    if "has_subject_zips" in checks:
        assert out["has_subject_zips"] is checks["has_subject_zips"]
    if "has_bids_zip" in checks:
        assert out["has_bids_zip"] is checks["has_bids_zip"]
    if "subject_files_sorted" in checks:
        assert sorted(out["subject_files"]) == checks["subject_files_sorted"]
    if "subject_files_len" in checks:
        assert len(out["subject_files"]) == checks["subject_files_len"]


# ─── validate_bids_structure_from_files ────────────────────────────────────


@pytest.mark.parametrize(
    ("files", "name_key", "checks"),
    [
        pytest.param(
            [
                {"name": "dataset_description.json"},
                {"name": "sub-01"},
                {"name": "sub-02"},
            ],
            "name",
            {"is_bids": True, "subject_count": 2},
            id="test_validate_from_files_extracts_name_key",
        ),
        pytest.param(
            [],
            "name",
            {"is_bids": False, "subject_count": 0},
            id="test_validate_from_files_handles_empty_list",
        ),
        pytest.param(
            [{"size": 100}, {"size": 200}],
            "name",
            {"is_bids": False},
            id="test_validate_from_files_tolerates_missing_name_key",
        ),
    ],
)
def test_validate_from_files(files, name_key, checks):
    """validate_bids_structure_from_files handles dict-list inputs; missing name keys are treated as empty strings."""
    out = validate_bids_structure_from_files(files, name_key=name_key)
    if "is_bids" in checks:
        assert out["is_bids"] is checks["is_bids"]
    if "subject_count" in checks:
        assert out["subject_count"] == checks["subject_count"]


# ─── find_channels_tsv ─────────────────────────────────────────────────────


def test_find_channels_tsv_returns_parent_default_when_no_match(tmp_path: Path):
    """When no channels.tsv exists, returns ``parent/channels.tsv`` (path may not exist)."""
    eeg_file = tmp_path / "sub-01_eeg.edf"
    eeg_file.touch()
    out = find_channels_tsv(eeg_file)
    assert out.name == "channels.tsv"
    assert out.parent == tmp_path


def test_find_channels_tsv_prefers_canonical_filename(tmp_path: Path):
    """``channels.tsv`` (no prefix) is returned if it exists."""
    eeg_file = tmp_path / "sub-01_eeg.edf"
    eeg_file.touch()
    (tmp_path / "channels.tsv").touch()
    out = find_channels_tsv(eeg_file)
    assert out.name == "channels.tsv"


def test_find_channels_tsv_falls_back_to_prefix_match(tmp_path: Path):
    """Falls back to ``<stem>_channels.tsv`` when no bare ``channels.tsv`` exists."""
    eeg_file = tmp_path / "sub-01_task-rest_eeg.edf"
    eeg_file.touch()
    prefix_tsv = tmp_path / "sub-01_task-rest_channels.tsv"
    prefix_tsv.touch()
    out = find_channels_tsv(eeg_file)
    assert out.name == "sub-01_task-rest_channels.tsv"


# ─── count_bad_channels ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param(
            "name\ttype\tstatus\nCz\teeg\tgood\nFz\teeg\tgood\n",
            0,
            id="test_count_bad_channels_returns_zero_when_no_bad",
        ),
        pytest.param(
            "name\ttype\tstatus\nCz\teeg\tgood\nFz\teeg\tbad\nOz\teeg\tbad\n",
            2,
            id="test_count_bad_channels_counts_bad_status",
        ),
        pytest.param(
            "name\tstatus\nCz\tBAD\nFz\tBad\nOz\tbad\nPz\tgood\n",
            3,
            id="test_count_bad_channels_case_insensitive",
        ),
        pytest.param(
            "name\ttype\nCz\teeg\nFz\teeg\n",
            None,
            id="test_count_bad_channels_returns_none_for_no_status_column",
        ),
    ],
)
def test_count_bad_channels_tsv_content(tmp_path: Path, content, expected):
    """count_bad_channels counts bad-status rows; no status column returns None."""
    tsv = tmp_path / "channels.tsv"
    tsv.write_text(content, encoding="utf-8")
    assert count_bad_channels(tsv) == expected


def test_count_bad_channels_returns_none_for_missing_file(tmp_path: Path):
    """Missing file → None (distinct from zero bad channels)."""
    assert count_bad_channels(tmp_path / "missing.tsv") is None


def test_count_bad_channels_tolerates_malformed_tsv(tmp_path: Path):
    """A garbage file returns None or int, never raises."""
    tsv = tmp_path / "channels.tsv"
    tsv.write_bytes(b"\x00\xff\x01 garbage")
    result = count_bad_channels(tsv)
    assert result is None or isinstance(result, int)


# ─── Constants ─────────────────────────────────────────────────────────────


def test_bids_required_files_contains_dataset_description():
    assert "dataset_description.json" in BIDS_REQUIRED_FILES


def test_bids_optional_files_contains_canonical_set():
    for f in ("participants.tsv", "readme", "changes"):
        assert f in BIDS_OPTIONAL_FILES


def test_bids_subject_pattern_matches_canonical_names():
    """``sub-XX``, ``sub-001``, ``sub-A1B2`` all match; ``Sub-`` is case-insensitive."""
    for name in ("sub-01", "sub-001", "sub-A1B2", "Sub-99"):
        assert BIDS_SUBJECT_PATTERN.match(name), f"{name=} should match"


def test_bids_subject_pattern_rejects_non_subject_names():
    for name in ("ses-01", "data1", "subject-01", "sub_01"):
        assert not BIDS_SUBJECT_PATTERN.match(name), f"{name=} should not match"
