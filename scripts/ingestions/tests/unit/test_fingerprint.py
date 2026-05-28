"""Unit tests for the manifest-fingerprint helpers in ``_fingerprint``.

Fingerprints are content-addressed identifiers used by the ingestion
pipeline to detect when a dataset has actually changed (so the digest
step can skip unchanged inputs). They must be:

- **Deterministic**: same input → same fingerprint.
- **Stable across orderings**: file-order in the manifest must NOT
  affect the output (we sort).
- **Sensitive to the dataset id**: same files but different dataset
  → different fingerprint (so seeded by dataset_id + source).
- **Robust to absent fields**: missing ``size``, missing inner files,
  empty manifests — all must produce a fingerprint, not crash.
"""

from __future__ import annotations

import string

import pytest

from _fingerprint import _hash_entries, fingerprint_from_manifest

# ─── Determinism ───────────────────────────────────────────────────────────


def test_fingerprint_is_deterministic():
    """Calling twice with the same input returns the same fingerprint."""
    manifest = {"files": [{"path": "a.edf", "size": 100}]}
    f1 = fingerprint_from_manifest("ds001", "openneuro", manifest)
    f2 = fingerprint_from_manifest("ds001", "openneuro", manifest)
    assert f1 == f2


def test_fingerprint_is_a_hex_sha256():
    """The fingerprint format must be a stable 64-char lowercase hex string."""
    f = fingerprint_from_manifest("ds001", "openneuro", {})
    assert len(f) == 64
    assert all(c in string.hexdigits.lower() for c in f)


# ─── Stability across orderings ────────────────────────────────────────────


def test_fingerprint_ignores_file_order():
    """Re-ordering ``files`` in the manifest must NOT change the fingerprint."""
    f_a_b = fingerprint_from_manifest(
        "ds001",
        "openneuro",
        {"files": [{"path": "a.edf", "size": 100}, {"path": "b.edf", "size": 200}]},
    )
    f_b_a = fingerprint_from_manifest(
        "ds001",
        "openneuro",
        {"files": [{"path": "b.edf", "size": 200}, {"path": "a.edf", "size": 100}]},
    )
    assert f_a_b == f_b_a


# ─── Sensitivity to changes ────────────────────────────────────────────────


def test_fingerprint_changes_with_dataset_id():
    """Same files, different dataset → different fingerprint (seed is per-dataset)."""
    manifest = {"files": [{"path": "a.edf", "size": 100}]}
    f1 = fingerprint_from_manifest("ds001", "openneuro", manifest)
    f2 = fingerprint_from_manifest("ds002", "openneuro", manifest)
    assert f1 != f2


def test_fingerprint_changes_with_source():
    """Same files, different source → different fingerprint."""
    manifest = {"files": [{"path": "a.edf", "size": 100}]}
    f1 = fingerprint_from_manifest("ds001", "openneuro", manifest)
    f2 = fingerprint_from_manifest("ds001", "nemar", manifest)
    assert f1 != f2


def test_fingerprint_changes_when_a_file_size_changes():
    """One byte different in a file size → entirely different fingerprint
    (sha256's avalanche property)."""
    f1 = fingerprint_from_manifest(
        "ds001", "openneuro", {"files": [{"path": "a.edf", "size": 100}]}
    )
    f2 = fingerprint_from_manifest(
        "ds001", "openneuro", {"files": [{"path": "a.edf", "size": 101}]}
    )
    assert f1 != f2


def test_fingerprint_changes_when_a_file_path_changes():
    """Different path → different fingerprint."""
    f1 = fingerprint_from_manifest(
        "ds001", "openneuro", {"files": [{"path": "a.edf", "size": 100}]}
    )
    f2 = fingerprint_from_manifest(
        "ds001", "openneuro", {"files": [{"path": "b.edf", "size": 100}]}
    )
    assert f1 != f2


# ─── Robustness to absent / weird inputs ──────────────────────────────────


def test_fingerprint_empty_manifest():
    """An empty manifest still produces a (constant) fingerprint."""
    f = fingerprint_from_manifest("ds001", "openneuro", {})
    assert len(f) == 64


def test_fingerprint_manifest_without_files_key():
    """A manifest lacking the ``files`` key produces same output as empty."""
    f_empty = fingerprint_from_manifest("ds001", "openneuro", {})
    f_no_files = fingerprint_from_manifest("ds001", "openneuro", {"other_field": 42})
    assert f_empty == f_no_files


def test_fingerprint_file_missing_size_treats_as_zero():
    """A file entry with no ``size`` defaults to size=0."""
    f_no_size = fingerprint_from_manifest(
        "ds001", "openneuro", {"files": [{"path": "a.edf"}]}
    )
    f_size_zero = fingerprint_from_manifest(
        "ds001", "openneuro", {"files": [{"path": "a.edf", "size": 0}]}
    )
    assert f_no_size == f_size_zero


def test_fingerprint_uses_name_when_path_missing():
    """The fingerprint accepts ``name`` as a fallback for ``path``."""
    f_path = fingerprint_from_manifest(
        "ds001", "openneuro", {"files": [{"path": "a.edf", "size": 100}]}
    )
    f_name = fingerprint_from_manifest(
        "ds001", "openneuro", {"files": [{"name": "a.edf", "size": 100}]}
    )
    assert f_path == f_name


def test_fingerprint_files_without_path_or_name_are_skipped():
    """A file entry with neither ``path`` nor ``name`` is dropped silently."""
    f_with = fingerprint_from_manifest(
        "ds001", "openneuro", {"files": [{"path": "a.edf", "size": 100}]}
    )
    f_plus_bogus = fingerprint_from_manifest(
        "ds001",
        "openneuro",
        {"files": [{"path": "a.edf", "size": 100}, {"size": 999}]},
    )
    assert f_with == f_plus_bogus


# ─── _hash_entries primitive ───────────────────────────────────────────────


def test_hash_entries_empty_input_is_constant():
    """``_hash_entries([])`` must be a stable constant — used as the
    fingerprint for an empty manifest."""
    h1 = _hash_entries([])
    h2 = _hash_entries([])
    assert h1 == h2
    assert len(h1) == 64


def test_hash_entries_seeded_differs_from_unseeded():
    """A non-empty seed produces a different hash than the unseeded form."""
    unseeded = _hash_entries([("a", 1)])
    seeded = _hash_entries([("a", 1)], seed="x")
    assert unseeded != seeded


@pytest.mark.parametrize(
    ("entries_a", "entries_b"),
    [
        # Order doesn't matter because _hash_entries sorts.
        ([("a", 1), ("b", 2)], [("b", 2), ("a", 1)]),
        ([("z", 9), ("a", 1)], [("a", 1), ("z", 9)]),
    ],
)
def test_hash_entries_sort_invariance(
    entries_a: list[tuple[str, int]], entries_b: list[tuple[str, int]]
) -> None:
    """Re-ordering input entries does not change the hash."""
    assert _hash_entries(entries_a) == _hash_entries(entries_b)
