"""Unit tests for cascade-with-provenance .

The technical-metadata cascade in :func:`_extract_technical_metadata`
has 4 ordered steps:

1. ``EEGBIDSDataset`` attribute getters (``mne_bids``)
2. Modality JSON sidecar (``modality_sidecar``)
3. ``channels.tsv`` (``channels_tsv``)
4. Binary parser + MNE fallback (``binary_parser`` / ``mne_fallback``)

These tests pin the provenance dict's behaviour: first writer wins,
clamped fields lose their provenance, all-None inputs yield all-None
provenance.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))


def _load_digest():
    spec = importlib.util.spec_from_file_location(
        "_provenance_target", _INGEST_DIR / "3_digest.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── _empty_provenance / _stamp_provenance ────────────────────────────────


def test_empty_provenance_has_all_four_fields():
    digest = _load_digest()
    p = digest._empty_provenance()
    assert set(p.keys()) == {
        "sampling_frequency",
        "nchans",
        "ntimes",
        "ch_names",
    }
    assert all(v is None for v in p.values())


def test_stamp_provenance_records_source_when_value_filled():
    """None → not-None at a step records the step's source name."""
    digest = _load_digest()
    p = digest._empty_provenance()
    digest._stamp_provenance(
        p,
        "modality_sidecar",
        field="sampling_frequency",
        old_value=None,
        new_value=250.0,
    )
    assert p["sampling_frequency"] == "modality_sidecar"


def test_stamp_provenance_first_writer_wins():
    """When two steps would fill the same field, the first one stamped wins.

    Mirrors the cascade's ``X = X or new_X`` pattern: the second step
    sees ``X`` as already set and never overwrites.
    """
    digest = _load_digest()
    p = digest._empty_provenance()
    digest._stamp_provenance(
        p,
        "mne_bids",
        field="nchans",
        old_value=None,
        new_value=64,
    )
    # Simulate a later step trying to stamp; it shouldn't overwrite.
    digest._stamp_provenance(
        p,
        "binary_parser",
        field="nchans",
        old_value=64,
        new_value=64,
    )
    assert p["nchans"] == "mne_bids"


def test_stamp_provenance_skips_when_value_unchanged():
    """A step that found the field already set leaves provenance untouched."""
    digest = _load_digest()
    p = digest._empty_provenance()
    p["ntimes"] = "channels_tsv"
    digest._stamp_provenance(
        p,
        "binary_parser",
        field="ntimes",
        old_value=5000,
        new_value=5000,
    )
    assert p["ntimes"] == "channels_tsv"


def test_stamp_provenance_skips_when_new_value_still_none():
    """A step that returned None for the field doesn't record provenance."""
    digest = _load_digest()
    p = digest._empty_provenance()
    digest._stamp_provenance(
        p,
        "binary_parser",
        field="ch_names",
        old_value=None,
        new_value=None,
    )
    assert p["ch_names"] is None


# ─── _clamp_metadata_extremes: clears provenance on reject ───────────────


def test_clamp_clears_provenance_when_sfreq_rejected_zero():
    """sfreq <= 0 is rejected; its provenance entry is cleared."""
    digest = _load_digest()
    provenance = {
        "sampling_frequency": "mne_bids",
        "nchans": "mne_bids",
        "ntimes": None,
        "ch_names": None,
    }
    sf, n = digest._clamp_metadata_extremes(
        sampling_frequency=0.0,
        nchans=64,
        ch_names=None,
        bids_relpath="sub-01/eeg/sub-01_eeg.edf",
        provenance=provenance,
    )
    assert sf is None
    assert provenance["sampling_frequency"] is None  # cleared
    # nchans untouched
    assert n == 64
    assert provenance["nchans"] == "mne_bids"


def test_clamp_clears_provenance_when_nchans_rejected_zero():
    """nchans <= 0 is rejected; its provenance entry is cleared."""
    digest = _load_digest()
    provenance = {
        "sampling_frequency": "mne_bids",
        "nchans": "modality_sidecar",
        "ntimes": None,
        "ch_names": None,
    }
    _sf, n = digest._clamp_metadata_extremes(
        sampling_frequency=250.0,
        nchans=-1,
        ch_names=None,
        bids_relpath="sub-01/eeg/sub-01_eeg.edf",
        provenance=provenance,
    )
    assert n is None
    assert provenance["nchans"] is None  # cleared


def test_clamp_does_not_clear_when_just_suspicious_not_rejected():
    """sfreq > 1MHz is warned about but kept; provenance stays."""
    digest = _load_digest()
    provenance = {
        "sampling_frequency": "binary_parser",
        "nchans": None,
        "ntimes": None,
        "ch_names": None,
    }
    sf, _ = digest._clamp_metadata_extremes(
        sampling_frequency=2_000_000.0,
        nchans=None,
        ch_names=None,
        bids_relpath="sub-01/eeg/sub-01_eeg.edf",
        provenance=provenance,
    )
    assert sf == 2_000_000.0  # not rejected, just warned
    assert provenance["sampling_frequency"] == "binary_parser"  # preserved


def test_clamp_provenance_kwarg_optional():
    """``provenance=None`` keeps backward-compatible behaviour."""
    digest = _load_digest()
    sf, n = digest._clamp_metadata_extremes(
        sampling_frequency=-1.0,
        nchans=64,
        ch_names=None,
        bids_relpath="sub-01/eeg/sub-01_eeg.edf",
        # no provenance kwarg
    )
    assert sf is None
    assert n == 64


# ─── End-to-end: BIDS-fs snapshot Record has provenance ───────────────────


def test_bids_snapshot_record_has_metadata_provenance():
    """The ds_snapshot_vhdr Record has a _metadata_provenance field.

    Pins the integration-level contract: extract_record stamps the
    provenance from _extract_technical_metadata onto the Record (when
    at least one field was resolved). The VHDR fixture has no sidecar
    JSONs, so the cascade falls through to the binary parser.
    """
    import json as json_mod

    snapshot_path = (
        _INGEST_DIR
        / "tests"
        / "fixtures"
        / "digest_snapshots"
        / "outputs"
        / "ds_snapshot_vhdr"
        / "ds_snapshot_vhdr_records.json"
    )
    payload = json_mod.loads(snapshot_path.read_text())
    record = payload["records"][0]
    assert "_metadata_provenance" in record, (
        "Records produced via _extract_technical_metadata must carry "
        "_metadata_provenance when at least one field was resolved."
    )
    provenance = record["_metadata_provenance"]
    # All 4 keys present.
    assert set(provenance.keys()) == {
        "sampling_frequency",
        "nchans",
        "ntimes",
        "ch_names",
    }
    # Values are one of the 5 known sources or None.
    valid_sources = {
        "mne_bids",
        "modality_sidecar",
        "channels_tsv",
        "binary_parser",
        "mne_fallback",
        None,
    }
    for field, source in provenance.items():
        assert source in valid_sources, (
            f"unknown provenance source '{source}' for {field}"
        )


def test_bids_snapshot_provenance_matches_expected_cascade():
    """The VHDR fixture has no sidecars; cascade should fall through.

    Specifically:
    - sampling_frequency, nchans, ch_names → binary_parser (.vhdr reader)
    - ntimes → mne_fallback (VHDR doesn't include n_times in the header;
      MNE computes it from the binary companion)
    """
    import json as json_mod

    snapshot_path = (
        _INGEST_DIR
        / "tests"
        / "fixtures"
        / "digest_snapshots"
        / "outputs"
        / "ds_snapshot_vhdr"
        / "ds_snapshot_vhdr_records.json"
    )
    payload = json_mod.loads(snapshot_path.read_text())
    provenance = payload["records"][0]["_metadata_provenance"]
    assert provenance["sampling_frequency"] == "binary_parser"
    assert provenance["nchans"] == "binary_parser"
    assert provenance["ch_names"] == "binary_parser"
    assert provenance["ntimes"] == "mne_fallback"


def test_manifest_snapshot_records_have_no_provenance():
    """Manifest path bypasses _extract_technical_metadata.

    Records created by ``_enumerate_via_manifest`` don't go through
    the cascade, so they have no ``_metadata_provenance`` field.
    Pins the contract so a future refactor that accidentally adds
    provenance to the manifest path triggers this test.
    """
    import json as json_mod

    snapshot_path = (
        _INGEST_DIR
        / "tests"
        / "fixtures"
        / "digest_snapshots"
        / "outputs"
        / "ds_snapshot_manifest"
        / "ds_snapshot_manifest_records.json"
    )
    payload = json_mod.loads(snapshot_path.read_text())
    for record in payload["records"]:
        assert "_metadata_provenance" not in record, (
            "Manifest-path Records shouldn't carry provenance — the "
            "cascade only runs in the BIDS-fs path."
        )
