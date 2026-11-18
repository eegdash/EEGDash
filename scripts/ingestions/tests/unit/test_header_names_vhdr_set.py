"""BrainVision and EEGLAB records take the header's channel names when channels.tsv lists another number.

The catalogue-wide search of 2026-11-03 (1,919 records in 18 datasets with
``len(ch_names) != nchans``) found the #517 rule in place for EDF/BDF only. In
the BrainVision datasets the header count is right and the sidecar names are
not those of the file:

- on003702: the vhdr has 64 channels; channels.tsv lists 59 (no Fp1, Fpz, M1,
  M2, EOG).
- on004917 (and on005571): the vhdr has 64 channels; channels.tsv adds ``Ref``
  and ``Gnd``.

MNE reads the vhdr's names and, with the counts unequal, MNE-BIDS keeps them.
Fixtures in ``tests/fixtures/header_names`` are the deposit files, byte for
byte (vhdr from the NEMAR annex object, matching its key; channels.tsv from
github.com/nemarDatasets/<id>).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import _metadata_cascade as mc
from _metadata_cascade import (
    PROV_BINARY_PARSER,
    BinaryParserStep,
    CascadeContext,
    CascadeResult,
)

DATA = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "header_names"


def _sidecar_names(name: str) -> list[str]:
    lines = (DATA / name).read_text(encoding="utf-8").splitlines()[1:]
    return [ln.split("\t", 1)[0] for ln in lines if ln]


def _vhdr_names(name: str) -> list[str]:
    text = (DATA / name).read_text(encoding="latin-1")
    section = text.split("[Channel Infos]", 1)[1].split("\n[", 1)[0]
    return [
        ln.split("=", 1)[1].split(",", 1)[0]
        for ln in section.splitlines()
        if ln.startswith("Ch") and "=" in ln
    ]


CASES = {
    "on003702": (
        "on003702_sub-17_task-SocialMemoryCuing_eeg.vhdr",
        "on003702_sub-17_task-SocialMemoryCuing_channels.tsv",
        "sub-17/eeg/sub-17_task-SocialMemoryCuing_eeg.vhdr",
    ),
    "on004917": (
        "on004917_sub-17_task-pdm_eeg.vhdr",
        "on004917_sub-17_task-pdm_channels.tsv",
        "sub-17/eeg/sub-17_task-pdm_eeg.vhdr",
    ),
}


def _result(names: list[str]) -> CascadeResult:
    """What MneBidsStep left: the channels.tsv names and their count."""
    result = CascadeResult(nchans=len(names), ch_names=list(names))
    for field in ("nchans", "ch_names"):
        result.provenance[field] = "mne_bids"
    return result


def _ctx(tmp_path: Path, dataset: str) -> CascadeContext:
    vhdr, _, rel = CASES[dataset]
    root = tmp_path / dataset
    (root / rel).parent.mkdir(parents=True)
    shutil.copy(DATA / vhdr, root / rel)
    return CascadeContext(SimpleNamespace(bidsdir=str(root)), str(root / rel))


def test_fixtures_are_the_deposit_files():
    """The counts the search reported: 64 header channels, 59 and 66 sidecar names."""
    assert len(_vhdr_names(CASES["on003702"][0])) == 64
    assert len(_sidecar_names(CASES["on003702"][1])) == 59
    assert len(_vhdr_names(CASES["on004917"][0])) == 64
    assert _sidecar_names(CASES["on004917"][1])[-2:] == ["Ref", "Gnd"]


@pytest.mark.parametrize("dataset", sorted(CASES))
def test_vhdr_takes_header_names_when_counts_differ(tmp_path, dataset):
    _vhdr, tsv, _ = CASES[dataset]
    result = _result(_sidecar_names(tsv))

    BinaryParserStep().fill(_ctx(tmp_path, dataset), result)

    assert result.ch_names == _vhdr_names(CASES[dataset][0])
    assert result.nchans == len(result.ch_names) == 64
    assert result.provenance["ch_names"] == PROV_BINARY_PARSER


def test_vhdr_keeps_sidecar_names_when_counts_agree(tmp_path):
    """Equal counts: the sidecar's names stay (they may carry the curated labels)."""
    header = _vhdr_names(CASES["on004917"][0])
    curated = [f"X{i}" for i in range(len(header))]
    result = _result(curated)

    BinaryParserStep().fill(_ctx(tmp_path, "on004917"), result)

    assert result.ch_names == curated
    assert result.provenance["ch_names"] == "mne_bids"


@pytest.mark.parametrize(
    ("md", "expected"),
    [
        ({"ch_names": ["Fp1", "Fp2", "Cz"], "nchans": 3}, ["Fp1", "Fp2", "Cz"]),
        ({"ch_names": ["Fp1", "Fp2", "Cz"]}, ["Fp1", "Fp2", "Cz"]),
        ({"ch_names": ["Fp1", "Fp1", "Cz"], "nchans": 3}, None),  # MNE renames
        ({"ch_names": ["Fp1", "Fp2"], "nchans": 3}, None),  # not one per channel
        ({"ch_names": [], "nchans": 3}, None),
        ({"nchans": 3}, None),
    ],
)
def test_parsed_header_names_only_when_they_are_the_files(md, expected):
    assert mc._parsed_header_names(md) == expected


def test_set_takes_parser_names_when_counts_differ(monkeypatch, tmp_path):
    """EEGLAB: chanlocs labels, one per channel, win over a shorter channels.tsv."""
    header = [f"E{i}" for i in range(35)]
    monkeypatch.setattr(
        mc,
        "get_parser_for_extension",
        lambda ext: lambda path: {"ch_names": header, "nchans": 35},
    )
    rel = "sub-002/ses-001/eeg/sub-002_ses-001_task-sleep_eeg.set"
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_bytes(b"")
    result = _result(header[:34])

    BinaryParserStep().fill(
        CascadeContext(SimpleNamespace(bidsdir=str(tmp_path)), str(tmp_path / rel)),
        result,
    )

    assert result.ch_names == header
    assert result.nchans == 35
