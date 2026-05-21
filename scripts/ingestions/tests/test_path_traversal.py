"""Regression tests for the Phase 9 audit-3 path-traversal hardening.

Three findings are addressed in the same commit:

- **F1** — ``_set_parser`` now uses ``validate_file_path`` for the
  existence check (so broken git-annex symlinks behave consistently
  with the other parsers).
- **F2** — ``_vhdr_parser.extract_vhdr_references`` rejects DataFile/
  MarkerFile references that resolve outside the .vhdr's parent
  directory (defense against a malicious .vhdr that names
  ``DataFile=../../../etc/passwd``).
- **F3** — A new helper ``_parser_utils.path_is_within_root`` formalises
  the containment check and is what F2 calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow tests to import the parser modules without installing the package.
_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

from _parser_utils import path_is_within_root
from _vhdr_parser import extract_vhdr_references

# ─── path_is_within_root ──────────────────────────────────────────────────


def test_within_root_basic(tmp_path: Path) -> None:
    """Normal sub-path is contained."""
    sub = tmp_path / "sub-01" / "eeg" / "file.set"
    sub.parent.mkdir(parents=True)
    sub.touch()
    assert path_is_within_root(sub, tmp_path) is True


def test_within_root_dot_dot_escape_is_rejected(tmp_path: Path) -> None:
    """Path traversal via ``..`` is rejected after resolution."""
    sub = tmp_path / "child"
    sub.mkdir()
    outside = sub / ".." / ".." / "tmp"
    # `outside` resolves to tmp_path.parent / tmp / which is outside `sub`.
    assert path_is_within_root(outside, sub) is False


def test_within_root_self_returns_true(tmp_path: Path) -> None:
    """A path equal to root is itself in-root."""
    assert path_is_within_root(tmp_path, tmp_path) is True


def test_within_root_nonexistent_path_still_evaluated(tmp_path: Path) -> None:
    """The function tests STRUCTURE; existence is irrelevant.

    A path that doesn't exist but lexically belongs in the tree is
    accepted. Callers that need existence-AND-containment should call
    ``.exists()`` separately.
    """
    candidate = tmp_path / "future" / "file.set"
    # The parent doesn't exist yet, but the resolved structural answer
    # is still "inside tmp_path".
    assert path_is_within_root(candidate, tmp_path) is True


def test_within_root_symlink_out_of_tree_is_rejected(tmp_path: Path) -> None:
    """A symlink that points OUT of the tree fails the check.

    Phase 9 audit-3 F2: the threat model includes symlinks that look
    inside-tree by name but resolve outside (e.g., a malicious dataset
    archive that ships ``sub-01/data.set -> /etc/passwd``).
    """
    outside = tmp_path.parent / "outside_target.set"
    outside.touch()
    try:
        inside = tmp_path / "sub-01" / "data.set"
        inside.parent.mkdir(parents=True)
        inside.symlink_to(outside)
        # The symlink LOOKS inside tmp_path by name; resolve() reveals
        # the truth.
        assert path_is_within_root(inside, tmp_path) is False
    finally:
        if outside.exists():
            outside.unlink()


# ─── F2: extract_vhdr_references rejects out-of-tree DataFile= ───────────


def _write_vhdr(parent: Path, datafile: str, markerfile: str = "") -> Path:
    """Write a minimal .vhdr file with a chosen DataFile=/MarkerFile= line."""
    vhdr = parent / "sub-01_eeg.vhdr"
    content = "Brain Vision Data Exchange Header File Version 1.0\n"
    content += "[Common Infos]\n"
    content += f"DataFile={datafile}\n"
    if markerfile:
        content += f"MarkerFile={markerfile}\n"
    content += "Codepage=UTF-8\n"
    vhdr.write_text(content)
    return vhdr


def test_extract_vhdr_references_accepts_in_tree_sibling(tmp_path: Path) -> None:
    """Normal sibling .eeg file resolves and is marked exists=True."""
    vhdr = _write_vhdr(tmp_path, "sub-01_eeg.eeg")
    (tmp_path / "sub-01_eeg.eeg").write_bytes(b"\x00" * 16)
    refs = extract_vhdr_references(vhdr)
    assert refs["datafile"] == "sub-01_eeg.eeg"
    assert refs["datafile_exists"] is True


def test_extract_vhdr_references_rejects_dotdot_path(tmp_path: Path) -> None:
    """A DataFile= that escapes the parent dir is recorded but NOT
    marked exists, even if the target happens to exist on disk."""
    # Create a file outside the .vhdr's parent.
    outside_file = tmp_path.parent / "evil_target.eeg"
    outside_file.write_bytes(b"\x00")
    try:
        inner = tmp_path / "dataset_root"
        inner.mkdir()
        vhdr = _write_vhdr(inner, "../evil_target.eeg")
        refs = extract_vhdr_references(vhdr)
        # Field is still surfaced (we don't lie about what's in the file)
        assert refs["datafile"] == "../evil_target.eeg"
        # ... but it MUST NOT be treated as a valid sibling.
        assert refs["datafile_exists"] is False
    finally:
        if outside_file.exists():
            outside_file.unlink()


def test_extract_vhdr_references_rejects_absolute_path(tmp_path: Path) -> None:
    """An absolute path in DataFile= is rejected too."""
    vhdr = _write_vhdr(tmp_path, "/etc/passwd")
    refs = extract_vhdr_references(vhdr)
    assert refs["datafile"] == "/etc/passwd"
    assert refs["datafile_exists"] is False


def test_extract_vhdr_references_rejects_dotdot_markerfile(tmp_path: Path) -> None:
    """MarkerFile= gets the same containment treatment as DataFile=."""
    outside_marker = tmp_path.parent / "evil.vmrk"
    outside_marker.write_bytes(b"\x00")
    try:
        inner = tmp_path / "dataset_root"
        inner.mkdir()
        vhdr = _write_vhdr(inner, "sub-01_eeg.eeg", markerfile="../evil.vmrk")
        refs = extract_vhdr_references(vhdr)
        assert refs["markerfile"] == "../evil.vmrk"
        assert refs["markerfile_exists"] is False
    finally:
        if outside_marker.exists():
            outside_marker.unlink()


def test_extract_vhdr_references_subdirectory_sibling_is_accepted(
    tmp_path: Path,
) -> None:
    """A subdirectory sibling is still within the .vhdr's parent → OK."""
    sub = tmp_path / "data"
    sub.mkdir()
    (sub / "channels.eeg").write_bytes(b"\x00")
    vhdr = _write_vhdr(tmp_path, "data/channels.eeg")
    refs = extract_vhdr_references(vhdr)
    assert refs["datafile_exists"] is True


# ─── F3: _set_parser uses validate_file_path now ──────────────────────────


def test_set_parser_handles_broken_symlink(tmp_path: Path) -> None:
    """A broken git-annex-style symlink returns None, not a crash.

    Before F3, the parser used a raw .exists() check that returns True
    for broken symlinks (the symlink itself exists; its target doesn't).
    scipy.io.loadmat would then crash on the dangling target.
    """
    from _set_parser import parse_set_metadata

    broken_link = tmp_path / "broken.set"
    broken_link.symlink_to(tmp_path / ".does_not_exist")
    assert parse_set_metadata(broken_link) is None
