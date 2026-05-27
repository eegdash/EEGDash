"""Tests for ``_bids.py`` — BIDS structure validation .

Was at 0% coverage. These helpers are used by the source-listing
adapters (figshare, zenodo, osf, etc.) to detect "is this dataset
actually BIDS" before deciding to clone it. A regression here would
either mis-ingest non-BIDS datasets or skip valid BIDS ones.
"""

from __future__ import annotations

from pathlib import Path

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
    """BIDS_DATASET_ZIP_PATTERN matches files starting with 'bids' or
    'dataset', or ending in '_bids.zip'. Plain 'data.zip' does NOT match."""
    out = collect_bids_matches(
        ["dataset_bids_v1.zip", "study_bids.zip", "data.zip", "results.csv"],
        required_files=[],
        optional_files=[],
        dataset_zip_pattern=BIDS_DATASET_ZIP_PATTERN,
    )
    assert "dataset_bids_v1.zip" in out["bids_zip_files"]
    assert "study_bids.zip" in out["bids_zip_files"]  # _bids.zip suffix
    # Plain 'data.zip' / 'results.csv' do NOT match the pattern
    assert "data.zip" not in out["bids_zip_files"]
    assert "results.csv" not in out["bids_zip_files"]


def test_collect_dataset_zip_matcher_search_finds_pattern_anywhere():
    """``matcher='search'`` finds patterns anywhere in the filename."""
    files = ["prefix_data_bids.zip"]
    pattern = BIDS_DATASET_ZIP_PATTERN
    search_result = collect_bids_matches(
        files,
        required_files=[],
        optional_files=[],
        dataset_zip_pattern=pattern,
        dataset_zip_matcher="search",
    )
    # `search` finds it; the pattern is `(bids|dataset).*\.zip` and
    # the file ends in `_bids.zip` which the alternate clause matches.
    assert "prefix_data_bids.zip" in search_result["bids_zip_files"]


# ─── validate_bids_structure_from_names ────────────────────────────────────


def test_validate_structure_recognises_bids_root():
    """A directory with dataset_description.json + sub-XX entries is BIDS."""
    out = validate_bids_structure_from_names(
        ["dataset_description.json", "sub-01", "sub-02", "README"]
    )
    assert out["is_bids"] is True
    assert out["subject_count"] == 2
    assert "dataset_description.json" in out["bids_files_found"]


def test_validate_structure_rejects_non_bids_layout():
    """A flat directory of CSVs with no BIDS indicators → not BIDS."""
    out = validate_bids_structure_from_names(["data1.csv", "data2.csv", "README.txt"])
    assert out["is_bids"] is False
    assert out["subject_count"] == 0


def test_validate_structure_accepts_subject_zips_alone():
    """Multiple ``sub-XX.zip`` files at root → BIDS (no dataset_description
    required if subjects are clearly present)."""
    out = validate_bids_structure_from_names(["sub-01.zip", "sub-02.zip", "sub-03.zip"])
    assert out["is_bids"] is True
    assert out["subject_count"] == 3
    assert out["has_subject_zips"] is True


def test_validate_structure_requires_min_subjects():
    """A single subject doesn't meet ``subject_min_count=2`` default."""
    out = validate_bids_structure_from_names(["sub-01"])
    # No required files, only 1 subject → not BIDS
    assert out["is_bids"] is False


def test_validate_structure_accepts_dataset_zip():
    """A ``data_bids.zip`` file alone passes the BIDS check."""
    out = validate_bids_structure_from_names(
        ["data_bids.zip"],
        dataset_zip_pattern=BIDS_DATASET_ZIP_PATTERN,
    )
    assert out["is_bids"] is True
    assert out["has_bids_zip"] is True


def test_validate_structure_include_subject_files_returns_list():
    """When ``include_subject_files=True``, the response includes the list."""
    out = validate_bids_structure_from_names(
        ["sub-01", "sub-02"],
        include_subject_files=True,
    )
    assert sorted(out["subject_files"]) == ["sub-01", "sub-02"]


def test_validate_structure_subject_files_limit_respected():
    """Subject list is capped at ``subject_files_limit``."""
    many = [f"sub-{i:03d}" for i in range(50)]
    out = validate_bids_structure_from_names(
        many, include_subject_files=True, subject_files_limit=5
    )
    assert len(out["subject_files"]) == 5


# ─── validate_bids_structure_from_files ────────────────────────────────────


def test_validate_from_files_extracts_name_key():
    """``files`` is a list of dicts; pull each file's name by ``name_key``."""
    files = [
        {"name": "dataset_description.json"},
        {"name": "sub-01"},
        {"name": "sub-02"},
    ]
    out = validate_bids_structure_from_files(files, name_key="name")
    assert out["is_bids"] is True
    assert out["subject_count"] == 2


def test_validate_from_files_handles_empty_list():
    out = validate_bids_structure_from_files([], name_key="name")
    assert out["is_bids"] is False
    assert out["subject_count"] == 0


def test_validate_from_files_tolerates_missing_name_key():
    """Files dicts without the name key get treated as empty names."""
    files = [{"size": 100}, {"size": 200}]
    out = validate_bids_structure_from_files(files, name_key="name")
    assert out["is_bids"] is False


# ─── find_channels_tsv ─────────────────────────────────────────────────────


def test_find_channels_tsv_returns_parent_default_when_no_match(tmp_path: Path):
    """When no channels.tsv exists, returns the default (``parent/channels.tsv``)
    even though the path doesn't exist — caller decides what to do."""
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
    """If no ``channels.tsv`` exists, look for ``<prefix>_channels.tsv``
    where prefix matches the data-file stem."""
    eeg_file = tmp_path / "sub-01_task-rest_eeg.edf"
    eeg_file.touch()
    prefix_tsv = tmp_path / "sub-01_task-rest_channels.tsv"
    prefix_tsv.touch()
    out = find_channels_tsv(eeg_file)
    assert out.name == "sub-01_task-rest_channels.tsv"


# ─── count_bad_channels ────────────────────────────────────────────────────


def test_count_bad_channels_returns_none_for_missing_file(tmp_path: Path):
    """Missing file → None (distinguish from 'zero bad channels' which is 0)."""
    assert count_bad_channels(tmp_path / "missing.tsv") is None


def test_count_bad_channels_returns_zero_when_no_bad(tmp_path: Path):
    """A channels.tsv with status='good' entries → 0 bad."""
    tsv = tmp_path / "channels.tsv"
    tsv.write_text("name\ttype\tstatus\nCz\teeg\tgood\nFz\teeg\tgood\n")
    assert count_bad_channels(tsv) == 0


def test_count_bad_channels_counts_bad_status(tmp_path: Path):
    tsv = tmp_path / "channels.tsv"
    tsv.write_text("name\ttype\tstatus\nCz\teeg\tgood\nFz\teeg\tbad\nOz\teeg\tbad\n")
    assert count_bad_channels(tsv) == 2


def test_count_bad_channels_case_insensitive():
    """status values 'BAD' / 'Bad' / 'bad' all count."""
    # Use a tmp file
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tsv", delete=False, encoding="utf-8"
    ) as f:
        f.write("name\tstatus\nCz\tBAD\nFz\tBad\nOz\tbad\nPz\tgood\n")
        tsv_path = Path(f.name)

    try:
        assert count_bad_channels(tsv_path) == 3
    finally:
        tsv_path.unlink()


def test_count_bad_channels_returns_none_for_no_status_column(tmp_path: Path):
    """A TSV without a 'status' column → None (no annotation)."""
    tsv = tmp_path / "channels.tsv"
    tsv.write_text("name\ttype\nCz\teeg\nFz\teeg\n")
    assert count_bad_channels(tsv) is None


def test_count_bad_channels_tolerates_malformed_tsv(tmp_path: Path):
    """A garbage file → None, never raises."""
    tsv = tmp_path / "channels.tsv"
    tsv.write_bytes(b"\x00\xff\x01 garbage")
    # Should not raise; returns None or 0 depending on parse outcome
    result = count_bad_channels(tsv)
    assert result is None or isinstance(result, int)


# ─── Constants ─────────────────────────────────────────────────────────────


def test_bids_required_files_contains_dataset_description():
    """The minimal BIDS marker file is dataset_description.json."""
    assert "dataset_description.json" in BIDS_REQUIRED_FILES


def test_bids_optional_files_contains_canonical_set():
    """The optional-files list matches what BIDS docs recommend."""
    # All should be present (case-sensitive match expected)
    for f in ("participants.tsv", "readme", "changes"):
        assert f in BIDS_OPTIONAL_FILES


def test_bids_subject_pattern_matches_canonical_names():
    """``sub-XX``, ``sub-001``, ``sub-A1B2`` all match. ``Sub-`` is case-insensitive."""
    for name in ("sub-01", "sub-001", "sub-A1B2", "Sub-99"):
        assert BIDS_SUBJECT_PATTERN.match(name), f"{name=} should match"


def test_bids_subject_pattern_rejects_non_subject_names():
    """``ses-01``, ``data1``, etc. don't match."""
    for name in ("ses-01", "data1", "subject-01", "sub_01"):
        assert not BIDS_SUBJECT_PATTERN.match(name), f"{name=} should not match"
