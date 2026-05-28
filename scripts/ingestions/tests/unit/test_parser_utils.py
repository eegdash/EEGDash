"""Tests for ``_parser_utils.py`` — pure helpers .

Network-dependent helpers (``fetch_bytes_from_s3`` / ``head_content_length`` /
``fetch_from_s3``) are not covered here — they need urllib mocking that's out of
scope for this round. The pure helpers (path parsing, URL building, security checks)
ARE covered here because they're the regression risk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        pytest.param("real_file", False, id="real-file"),
        pytest.param("broken_link", True, id="broken"),
        pytest.param("valid_link", False, id="valid-link"),
        pytest.param("missing_path", False, id="missing"),
    ],
)
def test_is_broken_symlink(tmp_path: Path, setup: str, expected: bool):
    """is_broken_symlink returns True only for dangling symlinks."""
    if setup == "real_file":
        path = tmp_path / "real.txt"
        path.write_text("hi")
    elif setup == "broken_link":
        path = tmp_path / "broken.lnk"
        path.symlink_to(tmp_path / ".no_such_target")
    elif setup == "valid_link":
        target = tmp_path / "target.txt"
        target.write_text("hi")
        path = tmp_path / "good.lnk"
        path.symlink_to(target)
    else:  # missing_path — not a symlink, just absent
        path = tmp_path / "missing.txt"
    assert is_broken_symlink(path) is expected


# ─── extract_dataset_info ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "expected_source", "expected_id", "expected_rel"),
    [
        pytest.param(
            "/clones/ds002893/sub-01/eeg/file.vhdr",
            "openneuro",
            "ds002893",
            "sub-01/eeg/file.vhdr",
            id="openneuro_ds_prefix",
        ),
        pytest.param(
            "/clones/nm000176/sub-01/meg/file.fif",
            "nemar",
            "nm000176",
            "sub-01/meg/file.fif",
            id="nemar_nm_prefix",
        ),
    ],
)
def test_extract_dataset_info_recognised_source(
    path: str, expected_source: str, expected_id: str, expected_rel: str
):
    """Recognised source-id prefixes resolve to (source, dataset_id, relpath)."""
    out = extract_dataset_info(Path(path))
    assert out is not None
    source, ds_id, rel = out
    assert source == expected_source
    assert ds_id == expected_id
    assert rel == expected_rel


def test_extract_dataset_info_unrecognised_returns_none():
    out = extract_dataset_info(Path("/clones/zenodo_12345/sub-01/file.edf"))
    assert out is None


def test_extract_dataset_info_normalises_backslashes():
    """Windows-style path output is forward-slashed in the relative path."""
    out = extract_dataset_info(Path("/clones/ds002893/sub-01/eeg/file.vhdr"))
    assert out is not None
    _, _, rel = out
    assert "\\" not in rel


# ─── extract_openneuro_info (backward-compat wrapper) ─────────────────────


@pytest.mark.parametrize(
    ("path_str", "expected"),
    [
        pytest.param(
            "/clones/ds002893/sub-01/eeg/file.vhdr",
            ("ds002893", "sub-01/eeg/file.vhdr"),
            id="pair",
        ),
        pytest.param(
            "/clones/nm000176/sub-01/file.fif",
            None,
            id="nemar",
        ),
        pytest.param(
            "/clones/garbage_path/file.edf",
            None,
            id="unknown",
        ),
    ],
)
def test_extract_openneuro_info(path_str: str, expected):
    """OpenNeuro-only wrapper returns a (ds_id, relpath) pair or None."""
    assert extract_openneuro_info(Path(path_str)) == expected


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


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        pytest.param("inside", True, id="inside"),
        pytest.param("outside", False, id="outside"),
        pytest.param("string_args", True, id="string-arg"),
        pytest.param("self", True, id="self"),
    ],
)
def test_path_within_root(tmp_path: Path, setup: str, expected: bool):
    """path_is_within_root guards against path-traversal (pre-PR-#327 regression)."""
    if setup == "inside":
        path = tmp_path / "sub" / "file.txt"
        path.parent.mkdir()
        path.touch()
        assert path_is_within_root(path, tmp_path) is expected
    elif setup == "outside":
        outside = tmp_path / ".." / "evil.txt"
        assert path_is_within_root(outside, tmp_path) is expected
    elif setup == "string_args":
        path = tmp_path / "sub" / "file.txt"
        path.parent.mkdir()
        path.touch()
        assert path_is_within_root(str(path), str(tmp_path)) is expected
    else:  # self — a directory is within itself
        assert path_is_within_root(tmp_path, tmp_path) is expected


# ─── validate_file_path ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        pytest.param("real_file", True, id="real"),
        pytest.param("missing", False, id="missing"),
        pytest.param("broken_symlink", False, id="broken-symlink"),
        pytest.param("directory", True, id="dir"),
    ],
)
def test_validate_file_path(tmp_path: Path, setup: str, expected: bool):
    """validate_file_path checks existence + symlink resolution, not regular-file type.

    Pinning the directory=True case so a future refactor doesn't accidentally
    start rejecting dirs.
    """
    if setup == "real_file":
        path = tmp_path / "real.txt"
        path.write_text("hi")
    elif setup == "missing":
        path = tmp_path / "missing.txt"
    elif setup == "broken_symlink":
        path = tmp_path / "broken"
        path.symlink_to(tmp_path / ".no_such_target")
    else:  # directory
        path = tmp_path
    assert validate_file_path(path) is expected


# ─── read_with_encoding_fallback ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("setup", "check"),
    [
        pytest.param("utf8", "exact", id="utf8"),
        pytest.param("latin1", "contains", id="latin1"),
        pytest.param("missing", "none", id="missing"),
    ],
)
def test_read_encoding_fallback(tmp_path: Path, setup: str, check: str):
    """read_with_encoding_fallback handles UTF-8, latin-1 fallback, and missing files."""
    if setup == "utf8":
        f = tmp_path / "utf8.txt"
        f.write_text("Hello, world!\nLine 2\n", encoding="utf-8")
        out = read_with_encoding_fallback(f)
        assert out == "Hello, world!\nLine 2\n"
    elif setup == "latin1":
        # latin-1 bytes that are NOT valid UTF-8: 0xe9 by itself.
        f = tmp_path / "latin1.txt"
        f.write_bytes(b"foo \xe9 bar\n")
        out = read_with_encoding_fallback(f)
        assert out is not None
        assert "foo" in out
        assert "bar" in out
    else:  # missing
        out = read_with_encoding_fallback(tmp_path / "missing.txt")
        assert out is None
