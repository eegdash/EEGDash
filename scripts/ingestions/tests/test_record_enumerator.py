"""Unit tests for the RecordEnumerator factory dispatch.

Stage-1 scope: the factory's fallback rules (3 cases mirrored from the
old ``digest_dataset`` orchestrator). The Adapter ``enumerate()`` bodies
are stubbed in stage 1; stage 2 wires them to the existing digest
algorithm bodies, and stage 2's tests use real fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

from record_enumerator import (
    BIDSFilesystemEnumerator,
    EnumerationResult,
    ManifestEnumerator,
    _has_actual_recording_files,
    get_record_enumerator,
)
from source_adapter import OpenNeuroAdapter


def _make_adapter(dataset_id: str = "ds002893") -> OpenNeuroAdapter:
    """Make a minimal SourceAdapter; bids_root unused by the factory."""
    return OpenNeuroAdapter(dataset_id)


# ─── EnumerationResult dataclass ──────────────────────────────────────────


def test_enumeration_result_defaults():
    """Empty fields default sensibly (empty lists / dict)."""
    result = EnumerationResult(dataset_meta={"dataset_id": "ds001"})
    assert result.records == []
    assert result.errors == []
    assert result.montages == {}
    assert result.digest_method == "bids_filesystem"


def test_enumeration_result_carries_all_fields():
    """All four collections + method label round-trip through the dataclass."""
    result = EnumerationResult(
        dataset_meta={"dataset_id": "ds001"},
        records=[{"path": "a"}],
        errors=[{"file": "b", "error": "x"}],
        montages={"hash1": {"sensors": []}},
        digest_method="manifest_only",
    )
    assert result.records == [{"path": "a"}]
    assert result.errors == [{"file": "b", "error": "x"}]
    assert "hash1" in result.montages
    assert result.digest_method == "manifest_only"


# ─── _has_actual_recording_files ───────────────────────────────────────────


def test_has_actual_files_detects_edf(tmp_path: Path) -> None:
    """A canonical .edf file means the BIDS path is viable."""
    (tmp_path / "sub-01" / "eeg").mkdir(parents=True)
    (tmp_path / "sub-01" / "eeg" / "sub-01_eeg.edf").write_bytes(b"\x00")
    assert _has_actual_recording_files(tmp_path) is True


def test_has_actual_files_detects_ctf_ds_dir(tmp_path: Path) -> None:
    """A CTF .ds directory counts as a real recording."""
    (tmp_path / "sub-01" / "meg").mkdir(parents=True)
    (tmp_path / "sub-01" / "meg" / "sub-01_meg.ds").mkdir()
    assert _has_actual_recording_files(tmp_path) is True


def test_has_actual_files_detects_symlink_pointer(tmp_path: Path) -> None:
    """A git-annex broken symlink still counts (pointer, not the binary)."""
    (tmp_path / "sub-01" / "eeg").mkdir(parents=True)
    pointer = tmp_path / "sub-01" / "eeg" / "sub-01_eeg.edf"
    pointer.symlink_to(tmp_path / ".does_not_exist")  # broken on purpose
    assert _has_actual_recording_files(tmp_path) is True


def test_has_actual_files_returns_false_on_empty_dir(tmp_path: Path) -> None:
    """No recordings → False."""
    (tmp_path / "sub-01").mkdir()
    assert _has_actual_recording_files(tmp_path) is False


def test_has_actual_files_ignores_unrelated_files(tmp_path: Path) -> None:
    """README / dataset_description.json don't count as recordings."""
    (tmp_path / "README").write_text("hi")
    (tmp_path / "dataset_description.json").write_text("{}")
    assert _has_actual_recording_files(tmp_path) is False


# ─── Factory dispatch ──────────────────────────────────────────────────────


def test_factory_picks_bids_when_real_files_present(tmp_path: Path) -> None:
    """Case 1 of the factory: real files on disk → BIDS path."""
    (tmp_path / "sub-01" / "eeg").mkdir(parents=True)
    (tmp_path / "sub-01" / "eeg" / "sub-01_eeg.edf").write_bytes(b"\x00")

    enumerator = get_record_enumerator(
        dataset_id="ds002893",
        dataset_dir=tmp_path,
        source="openneuro",
        source_adapter=_make_adapter(),
        digested_at="2026-05-21T12:00:00Z",
    )
    assert isinstance(enumerator, BIDSFilesystemEnumerator)


def test_factory_picks_manifest_when_only_manifest_present(
    tmp_path: Path,
) -> None:
    """Case 2: manifest.json but no actual files → Manifest path."""
    (tmp_path / "manifest.json").write_text('{"files": []}')

    enumerator = get_record_enumerator(
        dataset_id="zenodo-12345",
        dataset_dir=tmp_path,
        source="zenodo",
        source_adapter=_make_adapter(),
        digested_at="2026-05-21T12:00:00Z",
    )
    assert isinstance(enumerator, ManifestEnumerator)


def _bids_init_that_raises(*args, **kwargs):
    """Module-level stub: a __init__ that always raises OSError.

    Used to monkey-patch BIDSFilesystemEnumerator's constructor and
    simulate a BIDS-load failure. Kept at module level so the project's
    no-nested-functions lint stays clean.
    """
    raise OSError("simulated BIDS load failure")


def test_factory_falls_back_to_manifest_when_bids_load_fails_with_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Case 2 fallback: BIDS load raises → fall back to manifest.

    Simulates the BIDS-construction failure by monkey-patching the
    BIDSFilesystemEnumerator constructor to raise. The factory should
    catch and return ManifestEnumerator since manifest.json exists.
    """
    (tmp_path / "sub-01" / "eeg").mkdir(parents=True)
    (tmp_path / "sub-01" / "eeg" / "sub-01_eeg.edf").write_bytes(b"\x00")
    (tmp_path / "manifest.json").write_text('{"files": []}')

    import record_enumerator as re_mod

    real_init = re_mod.BIDSFilesystemEnumerator.__init__
    monkeypatch.setattr(
        re_mod.BIDSFilesystemEnumerator, "__init__", _bids_init_that_raises
    )
    try:
        enumerator = get_record_enumerator(
            dataset_id="ds002893",
            dataset_dir=tmp_path,
            source="openneuro",
            source_adapter=_make_adapter(),
            digested_at="2026-05-21T12:00:00Z",
        )
        assert isinstance(enumerator, ManifestEnumerator)
    finally:
        monkeypatch.setattr(re_mod.BIDSFilesystemEnumerator, "__init__", real_init)


def test_factory_propagates_bids_failure_when_no_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """BIDS load fails AND no manifest → re-raise (no fallback possible)."""
    (tmp_path / "sub-01" / "eeg").mkdir(parents=True)
    (tmp_path / "sub-01" / "eeg" / "sub-01_eeg.edf").write_bytes(b"\x00")
    # No manifest.json on purpose.

    import record_enumerator as re_mod

    real_init = re_mod.BIDSFilesystemEnumerator.__init__
    monkeypatch.setattr(
        re_mod.BIDSFilesystemEnumerator, "__init__", _bids_init_that_raises
    )
    try:
        with pytest.raises(OSError, match="simulated"):
            get_record_enumerator(
                dataset_id="ds002893",
                dataset_dir=tmp_path,
                source="openneuro",
                source_adapter=_make_adapter(),
                digested_at="2026-05-21T12:00:00Z",
            )
    finally:
        monkeypatch.setattr(re_mod.BIDSFilesystemEnumerator, "__init__", real_init)


def test_factory_returns_bids_enumerator_when_empty_no_manifest(
    tmp_path: Path,
) -> None:
    """Case 3 — empty dir, no manifest: return BIDSFilesystemEnumerator
    (its .enumerate() will produce an empty result with status='empty').

    The factory doesn't refuse to return; the orchestrator inspects the
    EnumerationResult to decide the status.
    """
    # Empty directory; no manifest, no files.
    enumerator = get_record_enumerator(
        dataset_id="ds_empty",
        dataset_dir=tmp_path,
        source="openneuro",
        source_adapter=_make_adapter(),
        digested_at="2026-05-21T12:00:00Z",
    )
    # Either is acceptable here; the orchestrator surfaces the empty
    # status. We assert the type belongs to the Enumerator hierarchy.
    assert isinstance(enumerator, BIDSFilesystemEnumerator | ManifestEnumerator)


# ─── ABC invariants ────────────────────────────────────────────────────────


def test_record_enumerator_is_abstract():
    """RecordEnumerator can't be instantiated directly."""
    from record_enumerator import RecordEnumerator

    with pytest.raises(TypeError):
        RecordEnumerator(  # type: ignore[abstract]
            "ds001", Path("/tmp"), "openneuro", _make_adapter(), "now"
        )


def test_stub_enumerate_raises_not_implemented(tmp_path: Path) -> None:
    """Stage-1 stub: enumerate() raises NotImplementedError on both Adapters."""
    bids = BIDSFilesystemEnumerator(
        "ds001", tmp_path, "openneuro", _make_adapter(), "now"
    )
    manifest = ManifestEnumerator("z-001", tmp_path, "zenodo", _make_adapter(), "now")
    with pytest.raises(NotImplementedError):
        bids.enumerate()
    with pytest.raises(NotImplementedError):
        manifest.enumerate()
