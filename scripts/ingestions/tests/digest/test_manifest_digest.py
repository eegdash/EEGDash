"""Tests for the extracted manifest-digest Seam (``_manifest_digest.py``).

The headline assertion is structural: the manifest path must no longer pull in the
``3_digest`` CLI module (the import cycle that the extraction removed). The rest
exercise ``_manifest_digest`` directly from its new home, including the two builders
that the via-``3_digest`` helper tests do not reach.
"""

from __future__ import annotations

import subprocess
import sys

import _manifest_digest
from tests._helpers import INGEST_DIR


def test_manifest_digest_does_not_import_3_digest():
    """De-circularization proof: importing _manifest_digest must NOT load 3_digest.py."""
    code = (
        "import sys, _manifest_digest; "
        "bad = [m for m, mod in sys.modules.items() "
        "if getattr(mod, '__file__', None) and mod.__file__.endswith('3_digest.py')]; "
        "print('LOADED_3DIGEST:' + ','.join(bad) if bad else 'CLEAN')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(INGEST_DIR),
        timeout=120,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "CLEAN" in proc.stdout, f"{proc.stdout}\n{proc.stderr}"


def test_enumerate_via_manifest_is_owned_by_the_seam():
    assert _manifest_digest._enumerate_via_manifest.__module__ == "_manifest_digest"


def test_enumerate_via_manifest_empty_files_returns_manifest_only():
    result, total = _manifest_digest._enumerate_via_manifest(
        "ds005", {"source": "openneuro", "files": []}, "2026-01-01T00:00:00Z"
    )
    assert total == 0
    assert result.digest_method == "manifest_only"
    assert result.dataset_meta["dataset_id"] == "ds005"
    assert result.records == []


def test_enumerate_via_manifest_reconciles_source_from_id_pattern():
    # Manifest claims 'zenodo' but the ds##### id pattern wins (S3-misroute guard).
    result, _ = _manifest_digest._enumerate_via_manifest(
        "ds002893", {"source": "zenodo", "files": []}, "2026-01-01T00:00:00Z"
    )
    assert result.dataset_meta["source"] == "openneuro"


def test_fetch_subject_count_falls_back_without_urls():
    # No dataset_description.json / participants.tsv entries -> no HTTP, returns fallback.
    assert _manifest_digest._fetch_subject_count_via_http([], 7) == 7
    assert (
        _manifest_digest._fetch_subject_count_via_http(
            [{"path": "sub-01/eeg/sub-01_task-rest_eeg.edf"}], 3
        )
        == 3
    )


def test_importing_record_enumerator_does_not_eagerly_load_manifest_digest():
    """Pin the lazy-import invariant that keeps the manifest path acyclic.

    _manifest_digest imports EnumerationResult from record_enumerator at module top,
    so record_enumerator must reference _manifest_digest only at runtime (inside
    ManifestEnumerator.enumerate). If a future edit hoists that import to module
    top-level, this fails in CI instead of with a partial-init ImportError at runtime.
    """
    code = (
        "import sys, record_enumerator; "
        "print('EAGER' if '_manifest_digest' in sys.modules else 'LAZY')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(INGEST_DIR),
        timeout=120,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "LAZY" in proc.stdout, (
        f"record_enumerator eagerly imported _manifest_digest (cycle risk): {proc.stdout}"
    )


class _FakeResp:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        pass


class _FakeClient:
    def __init__(self, content: bytes):
        self._content = content

    def get(self, url, timeout=None):
        return _FakeResp(self._content)


def test_fetch_subject_count_reads_subjects_from_dataset_description(monkeypatch):
    monkeypatch.setattr(
        _manifest_digest, "_http_client", lambda: _FakeClient(b'{"Subjects": 42}')
    )
    files = [{"path": "dataset_description.json", "download_url": "http://x/dd.json"}]
    assert _manifest_digest._fetch_subject_count_via_http(files, 0) == 42


def test_fetch_subject_count_counts_participants_tsv_rows(monkeypatch):
    tsv = b"participant_id\tage\nsub-01\t30\nsub-02\t25\nsub-03\t40\n"
    monkeypatch.setattr(_manifest_digest, "_http_client", lambda: _FakeClient(tsv))
    files = [{"path": "participants.tsv", "download_url": "http://x/p.tsv"}]
    # 4 lines minus the header row -> 3 subjects.
    assert _manifest_digest._fetch_subject_count_via_http(files, 0) == 3


def test_is_bids_data_zip_matches_known_patterns():
    assert _manifest_digest._is_bids_data_zip("sub-01_eeg.zip") is True
    assert _manifest_digest._is_bids_data_zip("data.zip") is True
    assert _manifest_digest._is_bids_data_zip("README.txt") is False
    assert _manifest_digest._is_bids_data_zip("derivatives/code.zip") is False
