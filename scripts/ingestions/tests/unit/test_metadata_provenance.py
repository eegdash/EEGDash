"""Unit tests for the technical-metadata cascade provenance in ``_extract_technical_metadata``.

The cascade has 4 ordered steps: ``mne_bids``, ``modality_sidecar``,
``channels_tsv``, ``binary_parser`` / ``mne_fallback``. First writer wins;
clamped fields lose their provenance; all-None inputs yield all-None provenance.
"""

from __future__ import annotations

import json as json_mod

from _helpers import load_digest

from eegdash.testing import data_file

# ─── _empty_provenance / _stamp_provenance ────────────────────────────────


def test_empty_provenance_has_all_four_fields():
    digest = load_digest()
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
    digest = load_digest()
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
    """When two steps would fill the same field, the first one stamped wins."""
    digest = load_digest()
    p = digest._empty_provenance()
    digest._stamp_provenance(
        p,
        "mne_bids",
        field="nchans",
        old_value=None,
        new_value=64,
    )
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
    digest = load_digest()
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
    digest = load_digest()
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
    digest = load_digest()
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
    assert provenance["sampling_frequency"] is None
    assert n == 64
    assert provenance["nchans"] == "mne_bids"


def test_clamp_clears_provenance_when_nchans_rejected_zero():
    """nchans <= 0 is rejected; its provenance entry is cleared."""
    digest = load_digest()
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
    assert provenance["nchans"] is None


def test_clamp_does_not_clear_when_just_suspicious_not_rejected():
    """sfreq > 1MHz is warned about but kept; provenance stays."""
    digest = load_digest()
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
    assert sf == 2_000_000.0
    assert provenance["sampling_frequency"] == "binary_parser"


def test_clamp_provenance_kwarg_optional():
    """``provenance=None`` keeps backward-compatible behaviour."""
    digest = load_digest()
    sf, n = digest._clamp_metadata_extremes(
        sampling_frequency=-1.0,
        nchans=64,
        ch_names=None,
        bids_relpath="sub-01/eeg/sub-01_eeg.edf",
    )
    assert sf is None
    assert n == 64


# ─── End-to-end: BIDS-fs snapshot Record has provenance ───────────────────


def test_bids_snapshot_record_has_metadata_provenance():
    """The ds_snapshot_vhdr Record carries a ``_metadata_provenance`` field.

    Pins the integration contract: ``extract_record`` stamps provenance from
    ``_extract_technical_metadata`` when at least one field was resolved.
    The VHDR fixture has no sidecar JSONs, so the cascade reaches the binary parser.
    """
    snapshot_path = data_file(
        "digest_snapshots/outputs/ds_snapshot_vhdr/ds_snapshot_vhdr_records.json"
    )
    payload = json_mod.loads(snapshot_path.read_text())
    record = payload["records"][0]
    assert "_metadata_provenance" in record, (
        "Records produced via _extract_technical_metadata must carry "
        "_metadata_provenance when at least one field was resolved."
    )
    provenance = record["_metadata_provenance"]
    assert set(provenance.keys()) == {
        "sampling_frequency",
        "nchans",
        "ntimes",
        "ch_names",
        "duration_seconds",
    }
    valid_sources = {
        "mne_bids",
        "modality_sidecar",
        "channels_tsv",
        "binary_parser",
        "mne_fallback",
        "sidecar_arithmetic",
        "derived",
        None,
    }
    for field, source in provenance.items():
        assert source in valid_sources, (
            f"unknown provenance source '{source}' for {field}"
        )


def test_bids_snapshot_provenance_matches_expected_cascade():
    """The VHDR fixture has no sidecars; cascade should fall through to binary_parser.

    - sampling_frequency, nchans, ch_names → binary_parser (.vhdr reader)
    - ntimes → mne_fallback (VHDR header omits n_times; MNE computes it from the binary)
    """
    snapshot_path = data_file(
        "digest_snapshots/outputs/ds_snapshot_vhdr/ds_snapshot_vhdr_records.json"
    )
    payload = json_mod.loads(snapshot_path.read_text())
    provenance = payload["records"][0]["_metadata_provenance"]
    assert provenance["sampling_frequency"] == "binary_parser"
    assert provenance["nchans"] == "binary_parser"
    assert provenance["ch_names"] == "binary_parser"
    assert provenance["ntimes"] == "mne_fallback"


def test_manifest_snapshot_records_have_no_provenance():
    """Manifest path bypasses ``_extract_technical_metadata``; no provenance field expected.

    Pins the contract so a future refactor that accidentally adds provenance to
    the manifest path triggers this test.
    """
    snapshot_path = data_file(
        "digest_snapshots/outputs/ds_snapshot_manifest/ds_snapshot_manifest_records.json"
    )
    payload = json_mod.loads(snapshot_path.read_text())
    for record in payload["records"]:
        assert "_metadata_provenance" not in record, (
            "Manifest-path Records shouldn't carry provenance — the "
            "cascade only runs in the BIDS-fs path."
        )
