"""Tests for _file_utils.py: BIDS classification, git-annex pointer resolution, inline-sidecar gating."""

from __future__ import annotations

from pathlib import Path

import pytest

from _file_utils import (
    get_annex_file_key,
    get_annex_file_size,
    is_bids_file,
    is_bids_root_file,
    parse_annex_size,
    read_inline_sidecar,
)

# ─── is_bids_file / is_bids_root_file ────────────────────────────────────


def test_is_bids_file_recognises_subject_prefix():
    assert is_bids_file("sub-01_eeg.edf") is True
    assert is_bids_file("data/sub-01/eeg/sub-01_eeg.vhdr") is True


def test_is_bids_file_recognises_canonical_extensions():
    # FIF and SNIRF are NOT in BIDS_DATA_EXTENSIONS; pinned so future expansion is visible.
    for ext in (".edf", ".bdf", ".vhdr", ".set", ".cnt", ".nwb"):
        assert is_bids_file(f"random_name{ext}") is True, (
            f"BIDS data extension {ext} should match"
        )


def test_is_bids_file_recognises_root_files():
    assert is_bids_file("dataset_description.json") is True
    assert is_bids_file("README") is True
    assert is_bids_file("participants.tsv") is True


def test_is_bids_file_rejects_random_csv():
    assert is_bids_file("results.csv") is False
    assert is_bids_file("analysis.py") is False


# ─── is_bids_root_file ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        pytest.param("dataset_description.json", True, id="canonical_json"),
        pytest.param("README", True, id="readme"),
        pytest.param("DATASET_DESCRIPTION.JSON", True, id="case_insensitive"),
        pytest.param("sub-01_eeg.edf", False, id="subject_file_not_root"),
        pytest.param("results.csv", False, id="non_bids_csv"),
    ],
)
def test_is_bids_root_file(path: str, expected: bool):
    assert is_bids_root_file(path) is expected


# ─── parse_annex_size ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        pytest.param("MD5E-s12345--abcdef.fif", 12345, id="md5_key"),
        pytest.param("SHA256E-s9999999--deadbeef.edf", 9999999, id="sha256_key"),
        pytest.param("/some/path/MD5E-s512000--abc123def.fif", 512000, id="full_path"),
        pytest.param("sub-01_eeg.edf", None, id="no_annex_key"),
        pytest.param("random text", None, id="plain_text"),
    ],
)
def test_parse_annex_size(key: str, expected: int | None):
    # regex uses `search` so full paths match too (see `full_path` case)
    assert parse_annex_size(key) == expected


# ─── get_annex_file_key (symlink + smudged-pointer paths) ─────────────────


def test_get_annex_file_key_returns_none_for_regular_file(tmp_path: Path):
    f = tmp_path / "regular.txt"
    f.write_text("just text, not annex")
    assert get_annex_file_key(f) is None


def test_get_annex_file_key_from_symlink(tmp_path: Path):
    annex_key = "MD5E-s12345--abc123def456.fif"
    link = tmp_path / "data.fif"
    target = Path(f".git/annex/objects/XX/YY/{annex_key}/{annex_key}")
    link.symlink_to(target)
    out = get_annex_file_key(link)
    assert out == annex_key


def test_get_annex_file_key_from_smudged_pointer(tmp_path: Path):
    # Smudged-pointer mode: file exists but holds only the annex path, not data.
    annex_key = "SHA256E-s99999--deadbeef.edf"
    pointer = tmp_path / "data.edf"
    pointer.write_text(f"/annex/objects/{annex_key}\n")
    out = get_annex_file_key(pointer)
    assert out == annex_key


def test_get_annex_file_key_returns_none_for_oversized_file(tmp_path: Path):
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * 1000)  # > 256
    assert get_annex_file_key(big) is None


def test_get_annex_file_key_returns_none_for_pointer_without_annex_path(tmp_path: Path):
    sidecar = tmp_path / "small.json"
    sidecar.write_text('{"key": "value"}')
    assert get_annex_file_key(sidecar) is None


# ─── get_annex_file_size ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        pytest.param("symlink_with_key", 4096, id="symlink_key"),
        pytest.param("real_file", 1234, id="real_file_size"),
        pytest.param("missing", 0, id="missing_returns_zero"),
    ],
)
def test_get_annex_file_size(tmp_path: Path, fixture: str, expected: int):
    if fixture == "symlink_with_key":
        annex_key = "MD5E-s4096--abc123.fif"
        path = tmp_path / "data.fif"
        target = Path(f".git/annex/objects/XX/YY/{annex_key}/{annex_key}")
        path.symlink_to(target)
    elif fixture == "real_file":
        path = tmp_path / "real.bin"
        path.write_bytes(b"x" * 1234)
    else:  # missing
        path = tmp_path / "missing.txt"

    assert get_annex_file_size(path) == expected


# ─── read_inline_sidecar ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("fixture", "expected_check"),
    [
        pytest.param("utf8_tsv", "sub-01", id="reads_utf8_tsv"),
        pytest.param("oversized", None, id="rejects_oversized"),
        pytest.param(
            "non_allowlisted_ext", None, id="rejects_non_allowlisted_extension"
        ),
        pytest.param("symlink", None, id="rejects_symlinks"),
        pytest.param("missing", None, id="missing_returns_none"),
        pytest.param("empty_tsv", "", id="empty_file_returns_empty_string"),
    ],
)
def test_read_inline_sidecar(tmp_path: Path, fixture: str, expected_check: str | None):
    # Empty allowlisted file yields "" (not None) — empty content is valid.
    if fixture == "utf8_tsv":
        f = tmp_path / "participants.tsv"
        f.write_text("participant_id\tage\nsub-01\t30\n")
        out = read_inline_sidecar(f)
        assert out is not None
        assert expected_check in out
        return
    elif fixture == "oversized":
        f = tmp_path / "huge.tsv"
        f.write_bytes(b"x" * (6 * 1024 * 1024))  # 6 MB
    elif fixture == "non_allowlisted_ext":
        f = tmp_path / "data.edf"
        f.write_bytes(b"some binary")
    elif fixture == "symlink":
        target = tmp_path / "real.tsv"
        target.write_text("data")
        f = tmp_path / "link.tsv"
        f.symlink_to(target)
    elif fixture == "missing":
        f = tmp_path / "missing.tsv"
    elif fixture == "empty_tsv":
        f = tmp_path / "empty.tsv"
        f.write_text("")

    assert read_inline_sidecar(f) == expected_check


def test_read_inline_sidecar_accepts_known_basenames(tmp_path: Path):
    for name in ("README", "CHANGES", "LICENSE"):
        f = tmp_path / name
        f.write_text("content for " + name)
        assert read_inline_sidecar(f) is not None
