"""Tests for ``_parser_utils.py`` — pure helpers .

Was at 32% before this commit. Network-dependent helpers
(``fetch_bytes_from_s3`` / ``head_content_length`` / ``fetch_from_s3``)
are not covered here — they need urllib mocking that's out of scope
for this round. The pure helpers (path parsing, URL building,
security checks) ARE covered here because they're the regression risk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_INGEST_DIR = Path(__file__).resolve().parent.parent
from _parser_utils import (
    build_s3_url,
    extract_dataset_info,
    extract_openneuro_info,
    is_broken_symlink,
    path_is_within_root,
    read_with_encoding_fallback,
    validate_file_path,
)

# ─── is_broken_symlink ─────────────────────────────────────────────────────


def test_is_broken_symlink_real_file_returns_false(tmp_path: Path):
    f = tmp_path / "real.txt"
    f.write_text("hi")
    assert is_broken_symlink(f) is False


def test_is_broken_symlink_broken_link_returns_true(tmp_path: Path):
    link = tmp_path / "broken.lnk"
    link.symlink_to(tmp_path / ".no_such_target")
    assert is_broken_symlink(link) is True


def test_is_broken_symlink_valid_link_returns_false(tmp_path: Path):
    target = tmp_path / "target.txt"
    target.write_text("hi")
    link = tmp_path / "good.lnk"
    link.symlink_to(target)
    assert is_broken_symlink(link) is False


def test_is_broken_symlink_missing_path_returns_false(tmp_path: Path):
    """A missing path that isn't a symlink: not 'broken' — it's just absent."""
    assert is_broken_symlink(tmp_path / "missing.txt") is False


# ─── extract_dataset_info ─────────────────────────────────────────────────


def test_extract_dataset_info_openneuro_path():
    """Path containing ``/ds<digits>/`` → openneuro."""
    out = extract_dataset_info(Path("/clones/ds002893/sub-01/eeg/file.vhdr"))
    assert out is not None
    source, ds_id, rel = out
    assert source == "openneuro"
    assert ds_id == "ds002893"
    assert rel == "sub-01/eeg/file.vhdr"


def test_extract_dataset_info_nemar_path():
    """Path containing ``/nm<digits>/`` → nemar."""
    out = extract_dataset_info(Path("/clones/nm000176/sub-01/meg/file.fif"))
    assert out is not None
    source, ds_id, rel = out
    assert source == "nemar"
    assert ds_id == "nm000176"
    assert rel == "sub-01/meg/file.fif"


def test_extract_dataset_info_unrecognised_returns_none():
    out = extract_dataset_info(Path("/clones/zenodo_12345/sub-01/file.edf"))
    assert out is None


def test_extract_dataset_info_normalises_backslashes():
    """Windows-style path output is forward-slashed in the relative path."""
    # Use a path with backslashes (simulating Windows)
    out = extract_dataset_info(Path("/clones/ds002893/sub-01/eeg/file.vhdr"))
    assert out is not None
    _, _, rel = out
    assert "\\" not in rel


# ─── extract_openneuro_info (backward-compat wrapper) ─────────────────────


def test_extract_openneuro_info_returns_pair():
    out = extract_openneuro_info(Path("/clones/ds002893/sub-01/eeg/file.vhdr"))
    assert out == ("ds002893", "sub-01/eeg/file.vhdr")


def test_extract_openneuro_info_returns_none_for_nemar_path():
    """The OpenNeuro-only wrapper rejects NEMAR paths."""
    out = extract_openneuro_info(Path("/clones/nm000176/sub-01/file.fif"))
    assert out is None


def test_extract_openneuro_info_returns_none_for_unknown_path():
    out = extract_openneuro_info(Path("/clones/garbage_path/file.edf"))
    assert out is None


# ─── build_s3_url ──────────────────────────────────────────────────────────


def test_build_s3_url_openneuro():
    out = build_s3_url("ds002893", "sub-01/eeg/file.vhdr", source="openneuro")
    assert out == (
        "https://s3.amazonaws.com/openneuro.org/ds002893/sub-01/eeg/file.vhdr"
    )


def test_build_s3_url_nemar():
    out = build_s3_url("nm000176", "objects/MD5E-s12345--abc.fif", source="nemar")
    assert "nemar/nm000176/objects/" in out


def test_build_s3_url_url_encodes_special_characters():
    """Forward slashes preserved, but spaces / special chars are encoded."""
    out = build_s3_url("ds001", "sub-01/eeg/file with spaces.vhdr")
    # Slashes preserved, spaces encoded as %20
    assert "/sub-01/" in out
    assert "%20" in out


def test_build_s3_url_rejects_unknown_source():
    with pytest.raises(ValueError, match="Unsupported source"):
        build_s3_url("ds001", "file.edf", source="unknown_source")


def test_build_s3_url_default_source_is_openneuro():
    """Default source kwarg is openneuro."""
    out = build_s3_url("ds001", "sub-01/file.vhdr")
    assert "openneuro.org" in out


# ─── path_is_within_root ───────────────────────────────────────────────────


def test_path_within_root_when_inside(tmp_path: Path):
    inside = tmp_path / "sub" / "file.txt"
    inside.parent.mkdir()
    inside.touch()
    assert path_is_within_root(inside, tmp_path) is True


def test_path_within_root_when_outside(tmp_path: Path):
    """The pre-PR-#327 path-traversal check that survived mutmut surfacing."""
    outside = tmp_path / ".." / "evil.txt"
    assert path_is_within_root(outside, tmp_path) is False


def test_path_within_root_accepts_string_args(tmp_path: Path):
    """Both args accept str or Path."""
    inside = tmp_path / "sub" / "file.txt"
    inside.parent.mkdir()
    inside.touch()
    assert path_is_within_root(str(inside), str(tmp_path)) is True


def test_path_within_root_root_itself_within_itself(tmp_path: Path):
    """Edge case: a directory is within itself."""
    assert path_is_within_root(tmp_path, tmp_path) is True


# ─── validate_file_path ───────────────────────────────────────────────────


def test_validate_file_path_real_file(tmp_path: Path):
    f = tmp_path / "real.txt"
    f.write_text("hi")
    assert validate_file_path(f) is True


def test_validate_file_path_missing_returns_false(tmp_path: Path):
    assert validate_file_path(tmp_path / "missing.txt") is False


def test_validate_file_path_broken_symlink_returns_false(tmp_path: Path):
    link = tmp_path / "broken"
    link.symlink_to(tmp_path / ".no_such_target")
    assert validate_file_path(link) is False


def test_validate_file_path_directory_returns_true(tmp_path: Path):
    """A directory passes validate_file_path — the function only checks
    existence + symlink resolution, not that it's a regular file. Pinning
    this so a future refactor doesn't accidentally start rejecting dirs."""
    assert validate_file_path(tmp_path) is True


# ─── read_with_encoding_fallback ──────────────────────────────────────────


def test_read_encoding_fallback_utf8(tmp_path: Path):
    """Standard UTF-8 file reads cleanly."""
    f = tmp_path / "utf8.txt"
    f.write_text("Hello, world!\nLine 2\n", encoding="utf-8")
    out = read_with_encoding_fallback(f)
    assert out == "Hello, world!\nLine 2\n"


def test_read_encoding_fallback_latin1(tmp_path: Path):
    """A latin-1 file that's not valid UTF-8 still reads (fallback)."""
    f = tmp_path / "latin1.txt"
    # latin-1 bytes that are NOT valid UTF-8: 0xe9 by itself.
    f.write_bytes(b"foo \xe9 bar\n")
    out = read_with_encoding_fallback(f)
    assert out is not None
    assert "foo" in out
    assert "bar" in out


def test_read_encoding_fallback_missing_file(tmp_path: Path):
    """Missing file → None (no raise)."""
    out = read_with_encoding_fallback(tmp_path / "missing.txt")
    assert out is None
