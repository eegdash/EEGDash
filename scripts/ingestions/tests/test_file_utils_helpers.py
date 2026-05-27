"""Tests for _file_utils.py helpers — BIDS classification + annex (C4.3).

Network-listing adapters were covered in C1.5 + C3.1. This file
adds the pure helpers + git-annex pointer resolution + inline-
sidecar gating that the digest pipeline relies on.
"""

from __future__ import annotations

from pathlib import Path

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
    """``sub-01_*`` files anywhere in path → True."""
    assert is_bids_file("sub-01_eeg.edf") is True
    assert is_bids_file("data/sub-01/eeg/sub-01_eeg.vhdr") is True


def test_is_bids_file_recognises_canonical_extensions():
    """BIDS data extensions configured in BIDS_DATA_EXTENSIONS → True.

    Note: FIF and SNIRF are NOT in the current set — those formats arrived
    in BIDS later and aren't enumerated in this list. Pinned so a future
    extension expansion is visible.
    """
    for ext in (".edf", ".bdf", ".vhdr", ".set", ".cnt", ".nwb"):
        assert is_bids_file(f"random_name{ext}") is True, (
            f"BIDS data extension {ext} should match"
        )


def test_is_bids_file_recognises_root_files():
    """README, dataset_description.json etc. are BIDS root markers."""
    assert is_bids_file("dataset_description.json") is True
    assert is_bids_file("README") is True
    assert is_bids_file("participants.tsv") is True


def test_is_bids_file_rejects_random_csv():
    assert is_bids_file("results.csv") is False
    assert is_bids_file("analysis.py") is False


def test_is_bids_root_file_canonical_only():
    """Only canonical root-level markers; sub-XX files don't qualify."""
    assert is_bids_root_file("dataset_description.json") is True
    assert is_bids_root_file("README") is True
    # sub-01_eeg.edf is BIDS but NOT a root file
    assert is_bids_root_file("sub-01_eeg.edf") is False
    assert is_bids_root_file("results.csv") is False


def test_is_bids_root_file_case_insensitive():
    """``DATASET_DESCRIPTION.JSON`` still matches."""
    assert is_bids_root_file("DATASET_DESCRIPTION.JSON") is True


# ─── parse_annex_size ─────────────────────────────────────────────────────


def test_parse_annex_size_from_md5_key():
    """``MD5E-s12345--abc`` → 12345."""
    out = parse_annex_size("MD5E-s12345--abcdef.fif")
    assert out == 12345


def test_parse_annex_size_from_sha256_key():
    out = parse_annex_size("SHA256E-s9999999--deadbeef.edf")
    assert out == 9999999


def test_parse_annex_size_full_path_works():
    """The regex is `search`, so a full path is fine."""
    out = parse_annex_size("/some/path/MD5E-s512000--abc123def.fif")
    assert out == 512000


def test_parse_annex_size_returns_none_for_non_annex():
    """A path without the annex key shape → None."""
    assert parse_annex_size("sub-01_eeg.edf") is None
    assert parse_annex_size("random text") is None


# ─── get_annex_file_key (symlink + smudged-pointer paths) ─────────────────


def test_get_annex_file_key_returns_none_for_regular_file(tmp_path: Path):
    """A real (non-annex) file returns None."""
    f = tmp_path / "regular.txt"
    f.write_text("just text, not annex")
    assert get_annex_file_key(f) is None


def test_get_annex_file_key_from_symlink(tmp_path: Path):
    """A symlink pointing into ``.git/annex/objects/`` → the SHA key."""
    annex_key = "MD5E-s12345--abc123def456.fif"
    link = tmp_path / "data.fif"
    target = Path(f".git/annex/objects/XX/YY/{annex_key}/{annex_key}")
    link.symlink_to(target)
    out = get_annex_file_key(link)
    assert out == annex_key


def test_get_annex_file_key_from_smudged_pointer(tmp_path: Path):
    """A small regular file whose content is ``/annex/objects/...`` →
    the SHA key. (Smudged-pointer mode.)"""
    annex_key = "SHA256E-s99999--deadbeef.edf"
    pointer = tmp_path / "data.edf"
    pointer.write_text(f"/annex/objects/{annex_key}\n")
    out = get_annex_file_key(pointer)
    assert out == annex_key


def test_get_annex_file_key_returns_none_for_oversized_file(tmp_path: Path):
    """A regular file > 256B isn't a smudged pointer → None."""
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * 1000)  # > 256
    assert get_annex_file_key(big) is None


def test_get_annex_file_key_returns_none_for_pointer_without_annex_path(tmp_path: Path):
    """A small file without /annex/objects/ in content → None."""
    sidecar = tmp_path / "small.json"
    sidecar.write_text('{"key": "value"}')
    assert get_annex_file_key(sidecar) is None


# ─── get_annex_file_size ──────────────────────────────────────────────────


def test_get_annex_file_size_from_symlink_with_key(tmp_path: Path):
    """For an annex symlink, size is parsed from the key."""
    annex_key = "MD5E-s4096--abc123.fif"
    link = tmp_path / "data.fif"
    target = Path(f".git/annex/objects/XX/YY/{annex_key}/{annex_key}")
    link.symlink_to(target)
    size = get_annex_file_size(link)
    # Size is parsed from the key, not from the empty symlink target
    assert size == 4096


def test_get_annex_file_size_real_file_returns_real_size(tmp_path: Path):
    """A real file returns its actual byte size."""
    f = tmp_path / "real.bin"
    f.write_bytes(b"x" * 1234)
    size = get_annex_file_size(f)
    assert size == 1234


def test_get_annex_file_size_missing_returns_zero(tmp_path: Path):
    """A non-existent path → 0."""
    size = get_annex_file_size(tmp_path / "missing.txt")
    assert size == 0


# ─── read_inline_sidecar ──────────────────────────────────────────────────


def test_read_inline_sidecar_reads_utf8_tsv(tmp_path: Path):
    """A small .tsv with UTF-8 content is returned."""
    f = tmp_path / "participants.tsv"
    f.write_text("participant_id\tage\nsub-01\t30\n")
    out = read_inline_sidecar(f)
    assert out is not None
    assert "sub-01" in out


def test_read_inline_sidecar_rejects_oversized(tmp_path: Path):
    """A file over ~5 MB → None (too big to inline)."""
    f = tmp_path / "huge.tsv"
    f.write_bytes(b"x" * (6 * 1024 * 1024))  # 6 MB
    assert read_inline_sidecar(f) is None


def test_read_inline_sidecar_rejects_non_allowlisted_extension(tmp_path: Path):
    """An .edf binary file is NOT an inline sidecar candidate."""
    f = tmp_path / "data.edf"
    f.write_bytes(b"some binary")
    assert read_inline_sidecar(f) is None


def test_read_inline_sidecar_accepts_known_basenames(tmp_path: Path):
    """README / CHANGES / LICENSE without extension pass the allowlist."""
    for name in ("README", "CHANGES", "LICENSE"):
        f = tmp_path / name
        f.write_text("content for " + name)
        assert read_inline_sidecar(f) is not None


def test_read_inline_sidecar_rejects_symlinks(tmp_path: Path):
    """Symlinks (likely annex pointers) → None (caller filters separately)."""
    target = tmp_path / "real.tsv"
    target.write_text("data")
    link = tmp_path / "link.tsv"
    link.symlink_to(target)
    assert read_inline_sidecar(link) is None


def test_read_inline_sidecar_returns_empty_for_empty_file(tmp_path: Path):
    """An empty allowlisted file returns ``""`` (not None) — empty is a
    valid sidecar."""
    f = tmp_path / "empty.tsv"
    f.write_text("")
    out = read_inline_sidecar(f)
    assert out == ""


def test_read_inline_sidecar_missing_returns_none(tmp_path: Path):
    """Non-existent path → None."""
    assert read_inline_sidecar(tmp_path / "missing.tsv") is None
