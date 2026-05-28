"""Regression tests for path-traversal hardening.

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

from pathlib import Path

# Allow tests to import the parser modules without installing the package.
from _parser_utils import path_is_within_root
from _set_parser import parse_set_metadata
from _vhdr_parser import extract_vhdr_references
from _vhdr_parser import extract_vhdr_references as _refs

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

    The threat model includes symlinks that look
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
    marked exists, even when the target file ACTUALLY EXISTS where the
    traversal resolves.

    This is the critical test for the security check at the heart of
    ``path_is_within_root(...) AND data_path.exists()``.
    The naive bug (``or`` instead of ``and``, surfaced as mutmut
    mutant 80) only manifests when BOTH:
        - the path traversal puts the resolved target outside parent
        - the target file actually exists on disk

    The earlier version of this test had the outside file at the wrong
    path, so ``data_path.exists()`` was also False — both ``and`` and
    ``or`` returned False, and the test couldn't distinguish them.
    """
    inner = tmp_path / "dataset_root"
    inner.mkdir()
    # CRITICAL: place the target where ``../evil_target.eeg`` resolves
    # from inside ``inner``, i.e. at ``tmp_path/evil_target.eeg``.
    # If this file isn't created at that exact path, ``data_path.exists()``
    # returns False and the test passes by accident even with a broken
    # security check.
    outside_target = tmp_path / "evil_target.eeg"
    outside_target.write_bytes(b"\x00")
    try:
        vhdr = _write_vhdr(inner, "../evil_target.eeg")
        # Sanity: the target really does exist where the traversal points

        assert (vhdr.parent / "../evil_target.eeg").resolve().exists(), (
            "fixture bug: target file not at the resolved-traversal path"
        )
        refs = _refs(vhdr)
        # Field is still surfaced (we don't lie about what's in the file)
        assert refs["datafile"] == "../evil_target.eeg"
        # ... but it MUST NOT be treated as a valid sibling.
        # If a code change weakens this (e.g., ``or`` instead of ``and``
        # — mutmut mutant 80), the existence check overrides the
        # containment check and this assertion fails.
        assert refs["datafile_exists"] is False
    finally:
        if outside_target.exists():
            outside_target.unlink()


def test_extract_vhdr_references_rejects_absolute_path(tmp_path: Path) -> None:
    """An absolute path in DataFile= is rejected too."""
    vhdr = _write_vhdr(tmp_path, "/etc/passwd")
    refs = extract_vhdr_references(vhdr)
    assert refs["datafile"] == "/etc/passwd"
    assert refs["datafile_exists"] is False


def test_extract_vhdr_references_rejects_dotdot_markerfile(tmp_path: Path) -> None:
    """MarkerFile= gets the same containment treatment as DataFile=.

    Same fixture shape as ``test_extract_vhdr_references_rejects_dotdot_path``:
    the marker file is placed AT the path where the ``../`` traversal
    resolves, so ``data_path.exists()`` is True and the test
    distinguishes ``AND`` (correct) from ``OR`` (mutant 80 analog).
    """
    inner = tmp_path / "dataset_root"
    inner.mkdir()
    outside_marker = tmp_path / "evil.vmrk"  # where ../evil.vmrk resolves
    outside_marker.write_bytes(b"\x00")
    try:
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

    broken_link = tmp_path / "broken.set"
    broken_link.symlink_to(tmp_path / ".does_not_exist")
    assert parse_set_metadata(broken_link) is None
