"""Tests for the git-annex SHA-key extractor used during NEMAR digestion."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_INGESTIONS_DIR = str(Path(__file__).resolve().parents[2] / "scripts" / "ingestions")


@pytest.fixture()
def get_annex_file_key():
    """Import ``get_annex_file_key`` from ``scripts/ingestions/_file_utils.py``."""
    if _INGESTIONS_DIR not in sys.path:
        sys.path.insert(0, _INGESTIONS_DIR)
    from _file_utils import get_annex_file_key as fn  # type: ignore

    return fn


_REAL_KEY = (
    "SHA256E-s121363992--47368a490fb602d7ed40698bcf062fa9ff938b8dd03d5ce25d30a64d11c5dace.set"
)


def test_extracts_key_from_smudged_pointer(tmp_path, get_annex_file_key):
    p = tmp_path / "sub-01_eeg.set"
    p.write_text(f"/annex/objects/{_REAL_KEY}\n")
    assert get_annex_file_key(p) == _REAL_KEY


def test_extracts_key_from_symlink_target(tmp_path, get_annex_file_key):
    target_dir = tmp_path / ".git" / "annex" / "objects" / "Vf" / "4X" / _REAL_KEY
    target_dir.mkdir(parents=True)
    target_file = target_dir / _REAL_KEY
    target_file.write_bytes(b"")  # not used, just so the path exists

    pointer = tmp_path / "sub-01_eeg.set"
    rel_target = os.path.relpath(target_file, pointer.parent)
    os.symlink(rel_target, pointer)

    assert get_annex_file_key(pointer) == _REAL_KEY


def test_returns_none_for_directly_tracked_sidecar(tmp_path, get_annex_file_key):
    sidecar = tmp_path / "dataset_description.json"
    sidecar.write_text('{"Name": "Example"}\n')
    assert get_annex_file_key(sidecar) is None


def test_returns_none_for_real_binary_file(tmp_path, get_annex_file_key):
    binary = tmp_path / "something.bdf"
    # Larger than the 256B sniff window — must be skipped.
    binary.write_bytes(b"\x00" * 4096)
    assert get_annex_file_key(binary) is None


def test_returns_none_for_missing_path(tmp_path, get_annex_file_key):
    assert get_annex_file_key(tmp_path / "does-not-exist") is None


def test_rejects_basename_that_is_not_a_full_key(tmp_path, get_annex_file_key):
    # A pointer that mentions /annex/objects/ but whose basename does not
    # match the strict SHA-key shape must be rejected — we never want to
    # forward garbage to S3.
    p = tmp_path / "broken.set"
    p.write_text("/annex/objects/garbage-not-a-key.set\n")
    assert get_annex_file_key(p) is None
