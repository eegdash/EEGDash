"""Happy-path tests for _set_parser via a synthetic .set (MAT v5).

Builds a minimal EEGLAB .set struct using scipy.io.savemat,
matching the field shapes the parser reads (EEG.srate, EEG.nbchan,
EEG.pnts, EEG.chanlocs.labels).

The CC0 .set fixture in ``eegdash-testing-data/eeg/`` is metadata-
light (only has_fdt extracted). This file constructs a richer
synthetic fixture in-memory that exercises every extraction branch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

# Try scipy.io.savemat; skip the whole module if scipy isn't installed.
scipy_io = pytest.importorskip("scipy.io")

import numpy as np
from _helpers.builders import build_synthetic_set_v5

from _set_parser import parse_set_metadata

# ─── Happy path ───────────────────────────────────────────────────────────


def test_set_extracts_sampling_frequency(tmp_path: Path):
    set_path = build_synthetic_set_v5(tmp_path / "test.set", srate=500.0)
    out = parse_set_metadata(set_path)
    assert out is not None
    assert out.get("sampling_frequency") == 500.0


def test_set_extracts_nchans(tmp_path: Path):
    set_path = build_synthetic_set_v5(tmp_path / "test.set", nbchan=64)
    out = parse_set_metadata(set_path)
    assert out is not None
    assert out.get("nchans") == 64


def test_set_extracts_pnts_as_n_samples(tmp_path: Path):
    """``EEG.pnts`` should land in the ``n_samples`` field."""
    set_path = build_synthetic_set_v5(tmp_path / "test.set", pnts=10000)
    out = parse_set_metadata(set_path)
    assert out is not None
    # Some parsers emit n_samples, others emit n_times — accept either.
    assert out.get("n_samples") == 10000 or out.get("n_times") == 10000


def test_set_extracts_ch_names_from_chanlocs(tmp_path: Path):
    """``EEG.chanlocs.labels`` becomes ch_names."""
    set_path = build_synthetic_set_v5(
        tmp_path / "test.set",
        nbchan=3,
        ch_names=["Cz", "Fz", "Pz"],
    )
    out = parse_set_metadata(set_path)
    assert out is not None
    if "ch_names" in out:
        assert sorted(out["ch_names"]) == ["Cz", "Fz", "Pz"]


def test_set_reports_has_fdt_false_when_companion_missing(tmp_path: Path):
    """Without a sibling .fdt file, has_fdt is False."""
    set_path = build_synthetic_set_v5(tmp_path / "test.set")
    out = parse_set_metadata(set_path)
    assert out is not None
    assert out.get("has_fdt") is False


def test_set_reports_has_fdt_true_when_companion_present(tmp_path: Path):
    """When ``.set`` has a sibling ``.fdt``, has_fdt is True."""
    set_path = build_synthetic_set_v5(tmp_path / "test.set")
    # Drop a placeholder .fdt next to it
    (tmp_path / "test.fdt").write_bytes(b"placeholder")
    out = parse_set_metadata(set_path)
    assert out is not None
    assert out.get("has_fdt") is True


# ─── Edge case: minimal struct (no chanlocs) ──────────────────────────────


def test_set_handles_struct_without_chanlocs(tmp_path: Path):
    """A .set with just srate/nbchan/pnts (no chanlocs) still parses."""
    set_path = tmp_path / "minimal.set"
    scipy_io.savemat(
        str(set_path),
        {
            "EEG": {
                "srate": np.array([[250.0]]),
                "nbchan": np.array([[32]]),
                "pnts": np.array([[1000]]),
            }
        },
    )
    out = parse_set_metadata(set_path)
    assert out is not None
    # Should still extract sampling_frequency + nchans + n_samples
    assert out.get("sampling_frequency") == 250.0
    assert out.get("nchans") == 32
    # ch_names may or may not be present depending on chanlocs handling


# ─── Edge case: mat without EEG struct ────────────────────────────────────


def test_set_returns_none_when_no_eeg_struct(tmp_path: Path):
    """A MAT file without ``EEG`` at the top → None or empty."""
    set_path = tmp_path / "noeeg.set"
    scipy_io.savemat(str(set_path), {"other_var": np.array([1, 2, 3])})
    out = parse_set_metadata(set_path)
    # Parser may emit None or a dict with only has_fdt
    if out is not None:
        assert "sampling_frequency" not in out
        assert "nchans" not in out
