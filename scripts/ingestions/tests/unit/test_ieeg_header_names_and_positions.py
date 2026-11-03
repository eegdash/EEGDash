"""iEEG records: the data file's channel names, their bad channels and their positions.

Survey of 2026-10-26 (live catalogue vs deposits; fixtures in
``tests/fixtures/ieeg_positions`` at the repository root, read from the NEMAR mirrors):

- on006910 (and on006914, on005545, on004551): every record has ``nchans`` from
  the EDF header (49) but the 128 ``ch_names`` of channels.tsv (``A1``..``B64``).
  The EDF holds bipolar ``POL B10-B18``.. signals. The cascade took the header's
  count but kept the sidecar's names (``ch_before or md["ch_names"]``). Raw is
  the truth: when the counts differ, the header's names are the record's.
- on006910 channels.tsv marks 20 of its contact names bad (``A31``, ``A49``..);
  none is a channel of the EDF, so none is a bad channel of the record.
- Channels that name contacts without being a row of electrodes.tsv (bipolar
  ``RAFA1-RAFA2`` in on004789/on004809, ``POL B10-B18``, ``EEG A1-Ref``) had no
  sensor in the stored montage. They get derived sensors (midpoint of the two
  contacts, or the one contact), marked with ``position_source``; a layout whose
  channels are all exact rows (nm000358) keeps its live hash.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import _metadata_cascade as mc
from _edf_header import edf_channel_names
from _metadata_cascade import (
    PROV_BINARY_PARSER,
    PROV_REMOTE_HEADER,
    BinaryParserStep,
    CascadeContext,
    CascadeResult,
    MetadataCascade,
    RemoteHeaderStep,
    SidecarArithmeticStep,
)
from _montage import extract_layout
from _record_extractor import _bad_channels_in_data, _extract_channel_status_counts

DATA = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "ieeg_positions"


def _sidecar_names(channels_tsv: str) -> list[str]:
    lines = (DATA / channels_tsv).read_text(encoding="utf-8").splitlines()[1:]
    return [ln.split("\t", 1)[0] for ln in lines if ln]


def _labels(dataset: str) -> list[str]:
    return json.loads((DATA / "edf_header_labels.json").read_text())[dataset]["labels"]


ON006910_SIDECAR = _sidecar_names("on006910_sub-098_ses-3_task-auditory_channels.tsv")
ON006910_LABELS = _labels("on006910")  # 49 channels + "EDF Annotations"
ON006910_HEADER = [n for n in ON006910_LABELS if n != "EDF Annotations"]


def _column(values, width: int) -> bytes:
    return b"".join(f"{v:<{width}}".encode("latin-1") for v in values)


def _edf_bytes(labels: list[str], spr: int = 10, n_records: int = 1) -> bytes:
    """A complete EDF file (header + zero samples) that MNE reads."""
    ns = len(labels)
    main = (
        f"{'0':<8}{'X':<80}{'X':<80}01.01.2001.00.00{256 * (ns + 1):<8}{'':<44}"
        f"{n_records:<8}{'1':<8}{ns:<4}"
    ).encode("latin-1")
    column = _column
    signals = b"".join(
        (
            b"".join(lab.encode("latin-1").ljust(16) for lab in labels),
            column([""] * ns, 80),
            column(["uV"] * ns, 8),
            column(["-3200"] * ns, 8),
            column(["3200"] * ns, 8),
            column(["-32768"] * ns, 8),
            column(["32767"] * ns, 8),
            column([""] * ns, 80),
            column([spr] * ns, 8),
            column([""] * ns, 32),
        )
    )
    assert len(main) == 256
    assert len(signals) == 256 * ns
    return main + signals + b"\0\0" * spr * ns * n_records


def _ctx(root: Path, rel: str) -> CascadeContext:
    return CascadeContext(SimpleNamespace(bidsdir=str(root)), str(root / rel))


_REL = "sub-098/ses-3/ieeg/sub-098_ses-3_task-auditory_ieeg.edf"


def _annexed(tmp_path: Path, dataset: str) -> CascadeContext:
    key = "SHA256E-s74245656--" + "d" * 64 + ".edf"
    root = tmp_path / dataset
    (root / _REL).parent.mkdir(parents=True)
    os.symlink(f"../../../.git/annex/objects/Aa/Bb/{key}/{key}", root / _REL)
    return _ctx(root, _REL)


class _BlobReader:
    """RangeReader stand-in serving an in-memory file."""

    blob = b""

    def __init__(self, url, **_k):
        self.url = url

    def read(self, offset, length):
        return self.blob[offset : offset + length]

    read_once = read


def _late_steps():
    return [
        s
        for s in MetadataCascade().steps
        if isinstance(s, (RemoteHeaderStep, SidecarArithmeticStep))
    ]


def _sidecar_result(names: list[str]) -> CascadeResult:
    """What MneBidsStep left: channels.tsv names, their count, the sidecar rate."""
    result = CascadeResult(sampling_frequency=10.0, nchans=len(names), ch_names=names)
    for field in ("sampling_frequency", "nchans", "ch_names"):
        result.provenance[field] = "mne_bids"
    return result


# ── raw is truth: header names when the sidecar lists another number ──


def test_edf_channel_names_as_mne_reads_them():
    """Stripped, latin-1, annotation signal dropped; collisions are not names."""
    labels = ["POL B10-B18", "EEG Ä1-Ref", "EDF Annotations"]
    block = b"".join(lab.encode("latin-1").ljust(16) for lab in labels)
    assert edf_channel_names(block, 3) == ["POL B10-B18", "EEG Ä1-Ref"]
    twice = b"POL A1".ljust(16) * 2
    assert edf_channel_names(twice, 2) is None
    assert edf_channel_names(block[:20], 3) is None


def test_shallow_clone_takes_header_names_when_counts_differ(monkeypatch, tmp_path):
    """on006910: 128 sidecar names, 49 EDF channels -> the 49 header names."""
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    monkeypatch.setattr(_BlobReader, "blob", _edf_bytes(ON006910_LABELS))
    monkeypatch.setattr(mc._remote_header, "RangeReader", _BlobReader)
    ctx = _annexed(tmp_path, "on006910")
    assert len(ON006910_SIDECAR) == 128
    result = _sidecar_result(ON006910_SIDECAR)

    for step in _late_steps():
        step.fill(ctx, result)

    assert result.ch_names == ON006910_HEADER
    assert result.nchans == 49 == len(result.ch_names)
    assert result.provenance["ch_names"] == PROV_REMOTE_HEADER


@pytest.mark.parametrize(
    "header",
    [
        pytest.param([f"EEG {n}" for n in ON006910_SIDECAR], id="same-count"),
        pytest.param(["POL A1"] * 128 + ["POL B1"], id="colliding-labels"),
    ],
)
def test_sidecar_names_kept_otherwise(monkeypatch, tmp_path, header):
    """Equal counts, or header labels MNE would rename (``-0``/``-1``): no change."""
    monkeypatch.setenv("EEGDASH_REMOTE_HEADERS", "missing")
    monkeypatch.setattr(_BlobReader, "blob", _edf_bytes(header))
    monkeypatch.setattr(mc._remote_header, "RangeReader", _BlobReader)
    ctx = _annexed(tmp_path, "on006910")
    result = _sidecar_result(list(ON006910_SIDECAR))

    for step in _late_steps():
        step.fill(ctx, result)

    assert result.ch_names == ON006910_SIDECAR
    assert result.provenance["ch_names"] == "mne_bids"


def test_local_edf_takes_header_names_when_counts_differ(tmp_path):
    """Full clone: the MNE parser reads the file, the names are the header's."""
    pytest.importorskip("mne")
    root = tmp_path / "on006910"
    (root / _REL).parent.mkdir(parents=True)
    (root / _REL).write_bytes(_edf_bytes(ON006910_HEADER))
    result = _sidecar_result(list(ON006910_SIDECAR))

    BinaryParserStep().fill(_ctx(root, _REL), result)

    assert result.ch_names == ON006910_HEADER
    assert result.nchans == 49
    assert result.provenance["ch_names"] == PROV_BINARY_PARSER


# ── bad channels are channels of the data file ──


def test_bad_channels_absent_from_the_data_are_dropped(tmp_path):
    """on006910: 20 contact names bad in channels.tsv, none of them in the EDF."""
    ieeg = tmp_path / "on006910" / "sub-098" / "ses-3" / "ieeg"
    ieeg.mkdir(parents=True)
    tsv = "on006910_sub-098_ses-3_task-auditory_channels.tsv"
    (ieeg / "sub-098_ses-3_task-auditory_channels.tsv").write_bytes(
        (DATA / tsv).read_bytes()
    )
    status = _extract_channel_status_counts(
        SimpleNamespace(bidsdir=str(tmp_path / "on006910")),
        str(ieeg / "sub-098_ses-3_task-auditory_ieeg.edf"),
    )
    assert status["bad_channels_count"] == 20
    assert {"A31", "A49", "A64"} <= set(status["bad_channels"])

    assert _bad_channels_in_data(status, ON006910_HEADER) == {
        "bad_channels": [],
        "bad_channels_count": 0,
    }
    # with the sidecar's own names (counts agree) every bad channel stays
    assert _bad_channels_in_data(status, ON006910_SIDECAR) == status
    # a subset stays, in channels.tsv order; no ch_names drops nothing
    kept = _bad_channels_in_data(status, ["A64", "B1", "A31"])
    assert kept == {"bad_channels": ["A31", "A64"], "bad_channels_count": 2}
    assert _bad_channels_in_data(status, None) == status


# ── stored montages: derived sensors next to the measured ones ──


def _tree(tmp_path: Path, dataset: str, rel_dir: str, files: dict[str, str]) -> Path:
    folder = tmp_path / dataset / rel_dir
    folder.mkdir(parents=True)
    for target, fixture in files.items():
        (folder / target).write_bytes((DATA / fixture).read_bytes())
    return folder


def _measured(doc):
    return [s for s in doc["sensors"] if "position_source" not in s]


def _derived(doc):
    return {s["name"]: s for s in doc["sensors"] if "position_source" in s}


def test_bipolar_edf_montage_gets_pair_midpoints(tmp_path):
    """on006910 sub-098 ses-3: live hash f96934d546d71406 for the measured layout."""
    folder = _tree(
        tmp_path,
        "on006910",
        "sub-098/ses-3/ieeg",
        {
            "sub-098_ses-3_electrodes.tsv": "on006910_sub-098_ses-3_electrodes.tsv",
            "sub-098_ses-3_coordsystem.json": "on006910_sub-098_ses-3_coordsystem.json",
        },
    )
    data_file = folder / "sub-098_ses-3_task-auditory_ieeg.edf"
    root = tmp_path / "on006910"

    h0, measured_only = extract_layout(data_file, root, "ieeg")
    h, doc = extract_layout(data_file, root, "ieeg", ch_names=ON006910_HEADER)

    assert h0 == "f96934d546d71406"  # the live montage of this record
    assert h != h0
    assert _measured(doc) == measured_only["sensors"]
    derived = _derived(doc)
    assert len(derived) == 47
    assert doc["n_sensors"] == len(measured_only["sensors"]) + 47
    xyz = {s["name"]: (s["x"], s["y"], s["z"]) for s in measured_only["sensors"]}
    pair = derived["POL B10-B18"]
    assert pair["position_source"] == "bipolar_midpoint"
    assert pair["contacts"] == ["B10", "B18"]
    assert (pair["x"], pair["y"], pair["z"]) == pytest.approx(
        tuple((a + b) / 2 for a, b in zip(xyz["B10"], xyz["B18"], strict=True))
    )
    assert pair["contact_distance"] == pytest.approx(math.dist(xyz["B10"], xyz["B18"]))
    assert (doc["space_declared"], doc["units_declared"]) == ("Other", "mm")


@pytest.mark.parametrize(
    ("dataset", "folder", "stem", "live_hash", "n_pairs"),
    [
        pytest.param(
            "on004789",
            "sub-R1148P/ses-0/ieeg",
            "sub-R1148P_ses-0_task-FR1",
            "bbec83a1e12dc97a",
            37,
            id="on004789",
        ),
        pytest.param(
            "on004809",
            "sub-R1106M/ses-0/ieeg",
            "sub-R1106M_ses-0_task-catFR1",
            "dd5236990ccf8f11",
            116,
            id="on004809",
        ),
    ],
)
def test_acq_bipolar_montage_gets_pair_midpoints(
    tmp_path, dataset, folder, stem, live_hash, n_pairs
):
    space = "space-MNI152NLin6ASym"
    ieeg = _tree(
        tmp_path,
        dataset,
        folder,
        {
            f"{stem}_{space}_electrodes.tsv": f"{dataset}_{stem}_{space}_electrodes.tsv",
            f"{stem}_{space}_coordsystem.json": f"{dataset}_{stem}_{space}_coordsystem.json",
        },
    )
    names = _sidecar_names(f"{dataset}_{stem}_acq-bipolar_channels.tsv")
    data_file = ieeg / f"{stem}_acq-bipolar_ieeg.edf"

    h0, measured_only = extract_layout(data_file, tmp_path / dataset, "ieeg")
    _h, doc = extract_layout(data_file, tmp_path / dataset, "ieeg", ch_names=names)

    assert h0 == live_hash
    derived = _derived(doc)
    assert sorted(derived) == sorted(names)
    assert len(derived) == n_pairs
    assert {s["position_source"] for s in derived.values()} == {"bipolar_midpoint"}
    assert _measured(doc) == measured_only["sensors"]


def test_exact_named_channels_keep_the_live_montage(tmp_path):
    """nm000358 sub-03 ses-5: nothing derived, hash 1d870db57a42d77e unchanged."""
    folder = _tree(
        tmp_path,
        "nm000358",
        "sub-03/ses-5/ieeg",
        {
            "sub-03_ses-5_electrodes.tsv": "nm000358_sub-03_ses-5_electrodes.tsv",
            "sub-03_ses-5_coordsystem.json": "nm000358_sub-03_ses-5_coordsystem.json",
        },
    )
    names = _sidecar_names("nm000358_sub-03_ses-5_task-naturalistic_channels.tsv")
    data_file = folder / "sub-03_ses-5_task-naturalistic_ieeg.vhdr"
    root = tmp_path / "nm000358"

    with_names = extract_layout(data_file, root, "ieeg", ch_names=names)
    assert with_names == extract_layout(data_file, root, "ieeg")
    assert with_names[0] == "1d870db57a42d77e"
    assert _derived(with_names[1]) == {}


def test_eeg_layouts_get_no_derived_sensors(tmp_path):
    """Only iEEG: a scalp ``Fp1-F7`` channel stays without a derived sensor."""
    folder = tmp_path / "ds" / "sub-01" / "eeg"
    folder.mkdir(parents=True)
    (folder / "sub-01_electrodes.tsv").write_text(
        "name\tx\ty\tz\nFp1\t-0.03\t0.08\t0.0\nF7\t-0.07\t0.04\t0.0\n"
        "Cz\t0\t0\t0.1\nPz\t0\t-0.07\t0.07\n"
    )
    data_file = folder / "sub-01_task-x_eeg.edf"
    root = tmp_path / "ds"
    assert extract_layout(data_file, root, "eeg", ch_names=["Fp1-F7"]) == (
        extract_layout(data_file, root, "eeg")
    )
