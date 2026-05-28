"""Tests for the MetadataCascade module .

The cascade extracts technical metadata (sampling_frequency / nchans /
ntimes / ch_names) from up to 5 sources, in order, with first-writer-wins
provenance stamping. This module exercises each step in isolation plus
the integration of the full chain.

The snapshot gate (``tests/test_digest_snapshot.py``) provides the
byte-identical guarantee against the legacy implementation; the tests
here pin down the public contract of the module itself.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest  # noqa: F401 — kept for `monkeypatch` discovery

import _metadata_cascade as mc
from _metadata_cascade import (
    CascadeContext,
    CascadeResult,
    MetadataCascade,
    MneBidsStep,
)

# ─── CascadeContext + CascadeResult ───────────────────────────────────────


def test_cascade_context_derives_ext_and_root():
    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    ctx = CascadeContext(
        bids_dataset=bids_dataset,
        bids_file="sub-01/eeg/sub-01_task-rest_eeg.vhdr",
    )
    assert ctx.ext == ".vhdr"
    assert ctx.bids_file_path == Path("sub-01/eeg/sub-01_task-rest_eeg.vhdr")
    assert ctx.bids_root == Path("/tmp/bids")


def test_cascade_result_defaults_are_none():
    result = CascadeResult()
    assert result.sampling_frequency is None
    assert result.nchans is None
    assert result.ntimes is None
    assert result.ch_names is None
    assert result.fif_is_split is False
    assert result.fif_continuations_ok is True
    assert set(result.provenance.keys()) == {
        "sampling_frequency",
        "nchans",
        "ntimes",
        "ch_names",
    }
    assert all(v is None for v in result.provenance.values())


def test_cascade_result_stamp_only_fires_on_none_to_non_none_transition():
    """``stamp`` must match the legacy ``_stamp_provenance`` semantics:
    stamps iff (old is None AND new is not None AND prov was None).

    Specifically, ``(old=0, new=500)`` MUST NOT stamp — the old value
    was already a non-None falsy assignment and provenance should
    remain whatever first source claimed it (or None).
    """
    r = CascadeResult()

    # Case 1: legitimate first-writer transition (None -> 500).
    r.stamp("first_source", "sampling_frequency", old=None, new=500.0)
    assert r.provenance["sampling_frequency"] == "first_source"

    # Case 2: second writer must NOT overwrite (first-writer-wins).
    r.stamp("second_source", "sampling_frequency", old=500.0, new=750.0)
    assert r.provenance["sampling_frequency"] == "first_source"

    # Case 3: (old=0, new=500) — legacy SKIPS this case; new must too.
    r2 = CascadeResult()
    r2.stamp("any_source", "nchans", old=0, new=500)
    assert r2.provenance["nchans"] is None, (
        "stamp must not fire when old is a non-None falsy value "
        "(legacy _stamp_provenance used `old is None` check)"
    )

    # Case 4: (old=None, new=None) — no stamp.
    r3 = CascadeResult()
    r3.stamp("any_source", "ntimes", old=None, new=None)
    assert r3.provenance["ntimes"] is None


# ─── MneBidsStep ──────────────────────────────────────────────────────────


def test_mne_bids_step_fills_from_attribute_getters():
    """Step 1: pulls sfreq/nchans/ntimes from EEGBIDSDataset attribute getters."""
    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    bids_dataset.get_bids_file_attribute.side_effect = lambda key, _file: {
        "sfreq": "500",
        "nchans": "64",
        "ntimes": "1000",
    }[key]
    bids_dataset.channel_labels.return_value = ["F1", "F2", "Cz"]

    ctx = CascadeContext(bids_dataset, "sub-01_eeg.vhdr")
    result = CascadeResult()

    MneBidsStep().fill(ctx, result)

    assert result.sampling_frequency == 500.0
    assert result.nchans == 3  # channel_labels count overrides sidecar nchans
    assert result.ntimes == 1000
    assert result.ch_names == ["F1", "F2", "Cz"]
    assert result.provenance == {
        "sampling_frequency": "mne_bids",
        "nchans": "mne_bids",
        "ntimes": "mne_bids",
        "ch_names": "mne_bids",
    }


def test_mne_bids_step_handles_missing_sidecar_gracefully():
    """OSError from attribute getter -> step leaves fields None."""
    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    bids_dataset.get_bids_file_attribute.side_effect = FileNotFoundError(
        "sidecar missing"
    )
    bids_dataset.channel_labels.side_effect = OSError("annex broken")

    ctx = CascadeContext(bids_dataset, "sub-01_eeg.vhdr")
    result = CascadeResult()
    MneBidsStep().fill(ctx, result)

    assert result.sampling_frequency is None
    assert result.nchans is None
    assert all(v is None for v in result.provenance.values())


# ─── ModalitySidecarStep ──────────────────────────────────────────────────


def test_modality_sidecar_step_only_fills_unset_fields(monkeypatch):
    """Step 2: fills sfreq/nchans from modality sidecar IFF still None."""

    # Behaves like the real helper: returns (sf, nchans), defaulting if None.
    monkeypatch.setattr(
        mc,
        "extract_sfreq_nchans_from_modality_sidecar",
        lambda _path, _root, sf_in, nc_in: (sf_in or 250.0, nc_in or 32),
    )

    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"

    ctx = CascadeContext(bids_dataset, "sub-01_eeg.vhdr")
    result = CascadeResult(sampling_frequency=None, nchans=None)

    mc.ModalitySidecarStep().fill(ctx, result)

    assert result.sampling_frequency == 250.0
    assert result.nchans == 32
    assert result.provenance["sampling_frequency"] == "modality_sidecar"
    assert result.provenance["nchans"] == "modality_sidecar"


def test_modality_sidecar_step_does_not_overwrite_filled_fields(monkeypatch):
    """If Step 1 filled sampling_frequency, Step 2 must not overwrite."""

    monkeypatch.setattr(
        mc,
        "extract_sfreq_nchans_from_modality_sidecar",
        lambda _p, _r, sf, nc: (sf or 250.0, nc or 32),
    )

    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    ctx = CascadeContext(bids_dataset, "sub-01_eeg.vhdr")

    # Step 1 already filled sampling_frequency.
    result = CascadeResult(sampling_frequency=500.0, nchans=None)
    result.provenance["sampling_frequency"] = "mne_bids"

    mc.ModalitySidecarStep().fill(ctx, result)

    assert result.sampling_frequency == 500.0  # unchanged
    assert result.provenance["sampling_frequency"] == "mne_bids"  # first-writer wins
    assert result.nchans == 32  # nchans WAS unset, so step fills
    assert result.provenance["nchans"] == "modality_sidecar"


# ─── ChannelsTsvStep ──────────────────────────────────────────────────────


def test_channels_tsv_step_fills_from_helper(monkeypatch):
    monkeypatch.setattr(
        mc,
        "extract_sfreq_nchans_from_channels_tsv",
        lambda _p, _r, sf, nc: (sf or 1000.0, nc or 19),
    )
    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    ctx = CascadeContext(bids_dataset, "sub-01_eeg.vhdr")
    result = CascadeResult()

    mc.ChannelsTsvStep().fill(ctx, result)

    assert result.sampling_frequency == 1000.0
    assert result.nchans == 19
    assert result.provenance["sampling_frequency"] == "channels_tsv"
    assert result.provenance["nchans"] == "channels_tsv"


# ─── BinaryParserStep ─────────────────────────────────────────────────────


def test_binary_parser_step_uses_registry(monkeypatch):
    """Step 4: dispatches to _format_parser_registry per file extension."""

    monkeypatch.setattr(
        mc,
        "get_parser_for_extension",
        lambda ext: (
            (
                lambda _p: {
                    "sampling_frequency": 256.0,
                    "nchans": 16,
                    "n_times": 100000,
                }
            )
            if ext == ".edf"
            else None
        ),
    )

    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    ctx = CascadeContext(bids_dataset, "sub-01_eeg.edf")
    result = CascadeResult()  # all fields None

    mc.BinaryParserStep().fill(ctx, result)

    assert result.sampling_frequency == 256.0
    assert result.nchans == 16
    assert result.ntimes == 100000
    assert result.provenance["sampling_frequency"] == "binary_parser"


def test_binary_parser_step_skipped_when_all_fields_filled(monkeypatch):
    """If Steps 1-3 already filled everything, Step 4 must be a no-op."""

    # MagicMock tracks call counts without needing a nested function.
    fake_parser = MagicMock(return_value={"sampling_frequency": 999.0, "nchans": 1})
    monkeypatch.setattr(
        mc, "get_parser_for_extension", MagicMock(return_value=fake_parser)
    )

    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    ctx = CascadeContext(bids_dataset, "sub-01_eeg.edf")
    result = CascadeResult(
        sampling_frequency=500.0, nchans=64, ntimes=1000, ch_names=["F1"]
    )

    mc.BinaryParserStep().fill(ctx, result)

    fake_parser.assert_not_called()  # parser never invoked
    assert result.sampling_frequency == 500.0  # unchanged


# ─── MneFallbackStep ──────────────────────────────────────────────────────


def test_mne_fallback_step_fills_vhdr_ntimes(monkeypatch):
    """Step 5: VHDR ntimes via MNE when binary parser couldn't get it."""

    fake_raw = MagicMock()
    fake_raw.n_times = 200000
    fake_raw.close = MagicMock()

    fake_mne = MagicMock()
    fake_mne.io.read_raw_brainvision.return_value = fake_raw

    monkeypatch.setattr(mc, "mne", fake_mne)

    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    ctx = CascadeContext(bids_dataset, "sub-01_eeg.vhdr")
    result = CascadeResult(sampling_frequency=500.0, nchans=32, ntimes=None)

    mc.MneFallbackStep().fill(ctx, result)

    assert result.ntimes == 200000
    assert result.provenance["ntimes"] == "mne_fallback"
    fake_raw.close.assert_called_once()


def test_mne_fallback_step_fif_split_metadata(monkeypatch):
    """Step 5: FIF split detection populates fif_is_split."""

    monkeypatch.setattr(
        mc,
        "_parse_fif_with_mne",
        lambda _path: (
            {
                "sampling_frequency": 1000.0,
                "nchans": 306,
                "n_times": 60000,
                "ch_names": ["MEG001"],
            },
            True,
        ),
    )

    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    ctx = CascadeContext(bids_dataset, "sub-01_meg.fif")
    result = CascadeResult()

    mc.MneFallbackStep().fill(ctx, result)

    assert result.fif_is_split is True
    assert result.sampling_frequency == 1000.0
    assert result.provenance["sampling_frequency"] == "mne_fallback"


# ─── Integration: full cascade ────────────────────────────────────────────


def test_metadata_cascade_runs_all_steps_in_order(monkeypatch):
    """End-to-end: simulate each step contributing one field."""
    bids_dataset = MagicMock()
    bids_dataset.bidsdir = "/tmp/bids"
    bids_dataset.get_bids_file_attribute.side_effect = lambda key, _f: (
        "500" if key == "sfreq" else None
    )
    bids_dataset.channel_labels.return_value = None

    # Step 2 contributes nchans
    monkeypatch.setattr(
        mc,
        "extract_sfreq_nchans_from_modality_sidecar",
        lambda _p, _r, sf, nc: (sf, nc or 32),
    )
    # Step 3 contributes nothing extra
    monkeypatch.setattr(
        mc,
        "extract_sfreq_nchans_from_channels_tsv",
        lambda _p, _r, sf, nc: (sf, nc),
    )
    # Step 4 contributes ntimes + ch_names
    monkeypatch.setattr(
        mc,
        "get_parser_for_extension",
        lambda _ext: lambda _p: {"n_times": 10000, "ch_names": ["A1", "A2"]},
    )

    ctx = CascadeContext(bids_dataset, "sub-01_eeg.edf")
    result = MetadataCascade().run(ctx)

    assert result.sampling_frequency == 500.0
    assert result.nchans == 32
    assert result.ntimes == 10000
    assert result.ch_names == ["A1", "A2"]
    assert result.provenance == {
        "sampling_frequency": "mne_bids",
        "nchans": "modality_sidecar",
        "ntimes": "binary_parser",
        "ch_names": "binary_parser",
    }


def test_metadata_cascade_default_steps_in_correct_order():
    """The default step ordering is what produces snapshot-identical bytes."""
    cascade = MetadataCascade()
    assert [type(s).__name__ for s in cascade.steps] == [
        "MneBidsStep",
        "ModalitySidecarStep",
        "ChannelsTsvStep",
        "BinaryParserStep",
        "MneFallbackStep",
    ]
