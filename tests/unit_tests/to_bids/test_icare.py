"""eegdash.to_bids.icare: the BDSP layout, merged groups, channel types, lossless EDF, resume.

The source is a fake I-CARE 2.0 release of hand-written WFDB records (``_wfdb.py``): two
patients in two splits, a few hour segments with EEG, ECG, REF and OTHER records, no
network. The tests that write data need ``wfdb``, which the adapter reads records with.
"""

from __future__ import annotations

import csv
import json
import os
import socket
import sys
from fractions import Fraction
from pathlib import Path

import mne
import numpy as np
import pytest
from _wfdb import write_patient, write_record

import eegdash.api
import eegdash.corpus as corpus_mod
from eegdash.to_bids import ADAPTERS, get_adapter, icare, to_bids
from eegdash.to_bids.adapter import Header
from eegdash.to_bids.icare import ICARE

DSID = "icare-2.0"
COMMENTS = ["Utility frequency: 60", "Start time: 4:07:23", "End time: 4:59:59"]
# 0284, hour 4, segment 1: EEG channels and their (gain, baseline)
EEG_0284 = {
    "Fp1": (2.0, 0),  # exact, and the full int16 range is used (saturated codes)
    "Fp2": (17.980017665, 7),  # no exact 8-character range: rounded
    "T3": (3.0, 0),  # exact on a range that encloses the codes
    "Cz": (10.0, -5),
    "Pz": (2.0, 0),  # three missing samples
    "Fpz": (2.0, 0),
}
EEG_0300 = ["Fp1", "F3", "C3", "O1"]
IN_TREE = [
    "sub-0284/eeg/sub-0284_task-icu_acq-h004_run-001_eeg.edf",
    "sub-0284/eeg/sub-0284_task-icu_acq-h005_run-002_eeg.edf",
    "sub-0284/eeg/sub-0284_task-icu_acq-h005_run-003_eeg.edf",
    "sub-0300/eeg/sub-0300_task-icu_acq-h010_run-001_eeg.edf",
    "sub-0300/eeg/sub-0300_task-icu_acq-h011_run-002_eeg.edf",
]
CHANNELS_0284_H004 = [
    ("Fp1", "EEG"),
    ("Fp2", "EEG"),
    ("T3", "EEG"),
    ("Cz", "EEG"),
    ("Pz", "EEG"),
    ("Fpz", "EEG"),
    ("ECG", "ECG"),
    ("SpO2", "MISC"),
    ("LOC", "EOG"),
]


def _refuse(*args, **kwargs):
    raise AssertionError("the network was used")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(eegdash.api, "EEGDash", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)


def _codes(n_times, n_signals, seed):
    return np.random.default_rng(seed).integers(-2000, 2000, (n_times, n_signals))


def _segment_0284_h004(folder):
    n = 12 * 256
    eeg = _codes(n, len(EEG_0284), 1)
    eeg[:2, 0] = [32767, -32767]  # saturated codes in Fp1
    eeg[[5, 6, 100], 4] = icare.MISSING  # Pz
    gains, baselines = zip(*EEG_0284.values())
    kw = dict(sfreq=256, comments=COMMENTS)
    write_record(folder, "0284_001_004_EEG", list(EEG_0284), eeg, gains=gains, baselines=baselines, **kw)  # fmt: skip
    write_record(folder, "0284_001_004_ECG", ["ECG"], _codes(n, 1, 2), gains=[10.0], baselines=[0], **kw)  # fmt: skip
    other = np.column_stack([np.full(n, 97), _codes(n, 1, 3)[:, 0]])
    write_record(folder, "0284_001_004_OTHER", ["SpO2", "LOC"], other, gains=[0.001, 2.0], baselines=[0, 0], **kw)  # fmt: skip


@pytest.fixture
def src(tmp_path):
    root = tmp_path / "bdsp-icare" / "2.0"
    root.mkdir(parents=True)
    (root / "bdsp_subject_crosswalk.csv").write_text("icare_patient_id,site\n")
    p = root / "training" / "0284"
    write_patient(
        p,
        "0284",
        {
            "Hospital": "A",
            "Age": "53",
            "Sex": "Male",
            "ROSC": "nan",
            "OHCA": "True",
            "Shockable Rhythm": "True",
            "TTM": "33",
            "Outcome": "Good",
            "CPC": "1",
        },
    )
    _segment_0284_h004(p)
    n, eeg = 12 * 256, list(EEG_0284)
    two = dict(gains=[2.0] * len(eeg), baselines=[0] * len(eeg), sfreq=256)
    write_record(p, "0284_002_005_EEG", eeg, _codes(n, len(eeg), 4), **two)
    write_record(p, "0284_003_005_EEG", eeg, _codes(n, len(eeg), 5), **two)
    short = dict(gains=[2.0], baselines=[0], sfreq=256)
    write_record(p, "0284_003_005_ECG", ["ECG"], _codes(n - 256, 1, 6), **short)
    write_record(p, "0284_004_006_ECG", ["ECG"], _codes(n, 1, 7), **short)  # no EEG
    (p / "0284_005_007_EKG.hea").write_text("0284_005_007_EKG 0 256 0\n")
    q = root / "test" / "0300"
    write_patient(
        q,
        "0300",
        {
            "Age": "61",
            "Sex": "Female",
            "ROSC": "12",
            "OHCA": "False",
            "VFib": "True",
            "TTM": "36",
            "Outcome": "Poor",
            "CPC": "5",
            "Foo Bar": "x",
        },
    )
    four = dict(gains=[4.0] * 4, baselines=[0] * 4, sfreq=250)
    n = 3125  # 12.5 s: EDF data records of 2.5 s
    write_record(q, "0300_001_010_EEG", EEG_0300, _codes(n, 4, 8), units=None, **four)
    write_record(q, "0300_001_010_REF", ["A1", "M1"], _codes(n, 2, 9), gains=[4.0] * 2, baselines=[0] * 2, sfreq=250)  # fmt: skip
    write_record(q, "0300_002_011_EEG", EEG_0300, _codes(3000, 4, 10), units="uV", **four)  # fmt: skip
    return root


def _run(src, cache, **kwargs):
    return to_bids(DSID, src, cache_dir=cache, **kwargs)


def _tsv(path, delimiter="\t"):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def _files(root):
    return sorted(p.relative_to(root).as_posix() for p in root.glob("sub-*/**/*.edf"))


def _snapshot(root):
    """Name -> (size, mtime) of every file under *root* but ``.eegdash/``."""
    out = {}
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            p = Path(dirpath, f)
            rel = p.relative_to(root).as_posix()
            if not rel.startswith(".eegdash/"):
                out[rel] = (p.stat().st_size, p.stat().st_mtime_ns)
    return out


def _edf_codes(path):
    """The int16 samples of a plain EDF file, samples x signals, read from its bytes."""
    raw = Path(path).read_bytes()
    n_records, ns = int(raw[236:244]), int(raw[252:256])
    spr_at = 256 + ns * (
        16 + 80 + 8 * 5 + 80
    )  # label, transducer, unit, ranges, filter
    spr = {int(raw[spr_at + 8 * k : spr_at + 8 * k + 8]) for k in range(ns)}
    assert len(spr) == 1
    spr = spr.pop()
    data = np.frombuffer(raw[256 * (ns + 1) :], dtype="<i2")
    assert data.size == n_records * ns * spr
    return data.reshape(n_records, ns, spr).transpose(0, 2, 1).reshape(-1, ns)


def _wfdb_segment(folder, key, groups):
    """WFDB's codes (missing ones as the baseline), physical values and gains of a segment."""
    wfdb = pytest.importorskip("wfdb")
    codes, phys, gains = [], [], []
    for group in groups:
        d = wfdb.rdrecord(str(folder / f"{key}_{group}"), physical=False)
        p = wfdb.rdrecord(str(folder / f"{key}_{group}"))
        c = d.d_signal.astype(np.int64)
        for k, b in enumerate(d.baseline):
            c[c[:, k] == icare.MISSING, k] = b
        codes.append(c)
        phys.append(p.p_signal)
        gains += d.adc_gain
    return np.column_stack(codes), np.column_stack(phys), np.array(gains)


# ---------------------------------------------------------------- the layout


def test_the_adapter_is_registered():
    assert ADAPTERS["icare"] == "eegdash.to_bids.icare:ICARE"
    assert get_adapter("icare") is ICARE
    assert ICARE.versions == ("2.0",) and not ICARE.linkable
    assert ICARE.extension == ".edf"


def test_parse_name():
    assert icare.parse_name("0284_001_004_EEG.hea", "0284") == ("001", "004", "EEG")
    assert icare.parse_name("0284_012_072_OTHER.hea", "0284") == ("012", "072", "OTHER")
    for wrong in ("0285_001_004_EEG.hea", "0284_001_EEG.hea", "0284_001_004_EEG.mat"):
        assert "not 0284_<segment>_<hour>_<group>.hea" in icare.parse_name(
            wrong, "0284"
        )


def test_segments_give_the_entities_and_the_groups(src):
    adapter = ICARE(src, "2.0")
    recs = list(adapter.recordings())
    assert [(r.subject, r.task, r.acquisition, r.run, r.key) for r in recs] == [
        ("0284", "icu", "h004", "001", "0284_001_004"),
        ("0284", "icu", "h005", "002", "0284_002_005"),
        ("0284", "icu", "h005", "003", "0284_003_005"),
        ("0300", "icu", "h010", "001", "0300_001_010"),
        ("0300", "icu", "h011", "002", "0300_002_011"),
    ]
    assert recs[0].source == Path("training/0284/0284_001_004_EEG.hea")
    assert [r.extra["groups"] for r in recs] == [
        "EEG,ECG,OTHER",
        "EEG",
        "EEG,ECG",
        "EEG,REF",
        "EEG",
    ]
    assert adapter.splits == {"0284": "training", "0300": "test"}
    assert "training/: 1 segment(s) without an EEG record; left out" in adapter.warnings
    assert any("0284_005_007_EKG.hea: group 'EKG'" in w for w in adapter.warnings)


def test_a_folder_that_is_not_the_release_is_refused(src, tmp_path):
    other = tmp_path / "other"
    (other / "0284").mkdir(parents=True)
    with pytest.raises(ValueError, match="has no training/, validation/, test/"):
        _run(other, tmp_path / "cache")
    with pytest.raises(ValueError, match="unknown to the icare adapter"):
        to_bids("icare-2.1", src, cache_dir=tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


# ---------------------------------------------------------------- channels


def test_the_channel_table():
    table = icare._channel_table()
    eeg = "Fp1 Fp2 F7 F8 F3 F4 T3 T4 C3 C4 T5 T6 P3 P4 O1 O2 Fz Cz Pz Fpz Oz F9"
    for name in eeg.split():
        assert table.lookup(f"EEG {name}") == (name, "EEG", True), name
    for name in "ECG ECG1 ECG2 ECGL ECGR".split():
        assert table.lookup(f"ECG {name}") == (name, "ECG", True), name
    ref = "RAT1 RAT2 REF C2 A1 A2 BIP1 BIP2 BIP3 BIP4 Cb2 M1 M2 In1-Ref2 In1-Ref3"
    for name in ref.split():  # positions some are, but reference channels: not EEG
        assert table.lookup(f"REF {name}") == (name, "MISC", True), name
    other = {"SpO2": "MISC", "EMG1": "EMG", "EMG3": "EMG", "LAT1": "EMG"}
    other |= {"LEG2": "EMG", "LOC": "EOG", "ROC": "EOG"}
    for name, kind in other.items():
        assert table.lookup(f"OTHER {name}") == (name, kind, True), name
    assert table.lookup("EEG Fp1-F7") == ("Fp1-F7", "MISC", True)  # a bipolar pair
    assert table.lookup("OTHER XYZ") == ("OTHER XYZ", "MISC", False)  # counted


def test_the_adapter_types_a_header_it_is_given(tmp_path):
    labels = ("EEG Fp1", "EEG T3", "ECG ECG", "REF A1", "OTHER LOC")
    header = Header(labels, ("uV",) * 5, 256.0, 2560, None)
    channels = ICARE(tmp_path, "2.0").channels(None, header)
    assert [(c.name, c.type, c.description) for c in channels] == [
        ("Fp1", "EEG", "EEG Fp1"),
        ("T3", "EEG", "EEG T3"),
        ("ECG", "ECG", "ECG ECG"),
        ("A1", "MISC", "REF A1"),
        ("LOC", "EOG", "OTHER LOC"),
    ]


# ----------------------------------------------------- the EDF number fields


def test_the_physical_range_is_exact_when_8_characters_hold_it():
    full = icare.physical_range(2.0, 0, -1000, 1000)
    assert full == dict(
        unit="uV", dmin=-32768, dmax=32767, pmin="-16384", pmax="16383.5", exact=True
    )
    # 1/3 has no decimal: a range on every third code that encloses the codes
    third = icare.physical_range(3.0, 0, -1000, 1001)
    assert third == dict(
        unit="uV", dmin=-1002, dmax=1002, pmin="-334", pmax="334", exact=True
    )
    shifted = icare.physical_range(10.0, -5, -100, 100)
    assert (shifted["pmin"], shifted["pmax"], shifted["exact"]) == (
        "-3276.3",
        "3277.2",
        True,
    )
    for f in (full, third, shifted):  # the line through the two ends is WFDB's
        assert Fraction(f["pmax"]) - Fraction(f["pmin"]) > 0
    # no exact range: the full int16 range, each end rounded to its last decimal place
    gain, base = 17.980017665, 7
    r = icare.physical_range(gain, base, -3000, 3000)
    assert r["exact"] is False and (r["dmin"], r["dmax"]) == (-32768, 32767)
    assert (r["pmin"], r["pmax"]) == ("-1822.86", "1822.023")
    g = Fraction(repr(gain))
    for d, p in ((r["dmin"], r["pmin"]), (r["dmax"], r["pmax"])):
        half = Fraction(1, 2 * 10 ** len(p.split(".")[1]))
        assert len(p) <= 8 and abs(Fraction(p) - (d - base) / g) <= half


def test_a_range_too_wide_for_microvolts_is_written_in_millivolts():
    r = icare.physical_range(0.001, 0, -32768, 32767)  # 32,768,000 uV: 9 characters
    assert r == dict(
        unit="mV", dmin=-32768, dmax=32767, pmin="-32768", pmax="32767", exact=True
    )
    assert icare.physical_range(2.0, 0, 0, 1, "mV")["unit"] == "mV"  # declared
    assert icare.physical_range(2.0, 0, 0, 1, "nu")["unit"] == "uV"
    assert icare.physical_range(1.0, 0, 0, 1, "%")["unit"] == "%"  # not a voltage
    with pytest.raises(ValueError, match="not a calibrated channel"):
        icare.physical_range(0.0, 0, 0, 1)


def test_the_record_layout_keeps_every_sample_when_it_can():
    assert icare.record_layout(3072, 256) == (256, 12, "1", 0)
    assert icare.record_layout(3125, 250) == (625, 5, "2.5", 0)
    # 3001 is prime and 1/256 s has no 8-character decimal: the part second is dropped
    assert icare.record_layout(3001, 256) == (256, 11, "1", 185)
    assert icare.record_layout(798, 199.5) == (798, 1, "4", 0)
    with pytest.raises(ValueError, match="no EDF data record fits"):
        icare.record_layout(100, 199.5)


# ------------------------------------------------------------- the tree


def test_the_tree_paths_channels_and_sidecars(src, tmp_path):
    pytest.importorskip("wfdb")
    report = _run(src, tmp_path / "cache")
    assert report.ok, report.problems
    root = tmp_path / "cache" / DSID
    assert _files(root) == IN_TREE
    assert all(not (root / p).is_symlink() for p in IN_TREE)
    eeg = root / "sub-0284" / "eeg"
    rows = _tsv(eeg / "sub-0284_task-icu_acq-h004_run-001_channels.tsv")
    assert [(r["name"], r["type"]) for r in rows] == CHANNELS_0284_H004
    assert rows[0]["description"] == "EEG Fp1" and rows[6]["description"] == "ECG ECG"
    side = json.loads((eeg / "sub-0284_task-icu_acq-h004_run-001_eeg.json").read_text())
    assert side["TaskName"] == "icu" and side["PowerLineFrequency"] == 60
    assert side["SamplingFrequency"] == 256 and side["RecordingDuration"] == 12
    assert side["HourAfterArrest"] == 4 and side["SegmentIndex"] == 1
    assert side["StartTimeAfterArrest"] == "4:07:23"
    assert side["EndTimeAfterArrest"] == "4:59:59"
    assert side["SourceRecords"] == [
        "0284_001_004_EEG",
        "0284_001_004_ECG",
        "0284_001_004_OTHER",
    ]
    assert side["MissingSamples"] == {"Pz": 3}
    assert side["PhysicalRangeRounded"] == ["Fp2"]
    assert "'nu'" in side["UnitsNote"]
    counts = {k: side[k] for k in side if k.endswith("ChannelCount")}
    assert counts == {
        "EEGChannelCount": 6,
        "EOGChannelCount": 1,
        "ECGChannelCount": 1,
        "EMGChannelCount": 0,
        "MiscChannelCount": 1,
        "TriggerChannelCount": 0,
    }
    # the ECG record of another length is left out of its run, with a warning
    rows = _tsv(eeg / "sub-0284_task-icu_acq-h005_run-003_channels.tsv")
    assert [r["type"] for r in rows] == ["EEG"] * 6
    assert any("0284_003_005_ECG: 256 Hz, 2816 samples" in w for w in report.warnings)
    # the REF record's A1 and M1 are MISC; no unit declared: microvolts (as MNE reads uV)
    rows = _tsv(root / "sub-0300/eeg/sub-0300_task-icu_acq-h010_run-001_channels.tsv")
    assert [(r["name"], r["type"], r["units"]) for r in rows][-2:] == [
        ("A1", "MISC", "µV"),
        ("M1", "MISC", "µV"),
    ]
    side = json.loads(
        (root / "sub-0300/eeg/sub-0300_task-icu_acq-h011_run-002_eeg.json").read_text()
    )
    assert "UnitsNote" not in side and "MissingSamples" not in side
    assert side["PowerLineFrequency"] == "n/a" and side["RecordingDuration"] == 12
    assert not list(root.glob("sub-*/**/*_events.tsv"))


def test_participants_from_the_clinical_files(src, tmp_path):
    pytest.importorskip("wfdb")
    _run(src, tmp_path / "cache")
    root = tmp_path / "cache" / DSID
    rows = _tsv(root / "participants.tsv")
    assert list(rows[0]) == [
        "participant_id",
        "split",
        "hospital",
        "age",
        "sex",
        "rosc",
        "ohca",
        "shockable_rhythm",
        "ttm",
        "outcome",
        "cpc",
        "foo_bar",
    ]
    assert [list(r.values()) for r in rows] == [
        [
            "sub-0284",
            "training",
            "A",
            "53",
            "M",
            "n/a",
            "True",
            "True",
            "33",
            "Good",
            "1",
            "n/a",
        ],  # fmt: skip
        [
            "sub-0300",
            "test",
            "n/a",
            "61",
            "F",
            "12",
            "False",
            "True",
            "36",
            "Poor",
            "5",
            "x",
        ],  # fmt: skip
    ]
    described = json.loads((root / "participants.json").read_text())
    assert list(described) == list(rows[0])[1:]
    assert described["age"]["Units"] == "year"
    assert "VFib" in described["shockable_rhythm"]["Description"]
    assert described["foo_bar"] == {
        "Description": "The release's patient file field `Foo Bar`."
    }


def test_the_dataset_files(src, tmp_path):
    pytest.importorskip("wfdb")
    _run(src, tmp_path / "cache")
    root = tmp_path / "cache" / DSID
    desc = json.loads((root / "dataset_description.json").read_text())
    assert desc["Name"].startswith("I-CARE: International Cardiac Arrest")
    assert desc["SourceDatasets"][0]["URL"] == "https://bdsp.io/content/bdsp-icare/2.0/"
    assert "Not redistributable" in desc["License"]
    assert desc["GeneratedBy"] == [{"Name": "eegdash.to_bids", "Version": "icare/1"}]
    readme = (root / "README.md").read_text()
    assert readme.startswith("# I-CARE 2.0 (BDSP)")
    assert "5 recordings of 2 patients (test 1, training 1)" in readme
    assert "250 Hz (2), 256 Hz (3)" in readme
    assert "without an EEG record" in readme and "none in this tree" in readme


# ------------------------------------------------------------ lossless data


def test_the_edf_holds_wfdbs_codes_and_physical_values(src, tmp_path):
    pytest.importorskip("wfdb")
    _run(src, tmp_path / "cache")
    folder = src / "training" / "0284"
    codes, phys, gains = _wfdb_segment(folder, "0284_001_004", ["EEG", "ECG", "OTHER"])
    edf = tmp_path / "cache" / DSID / IN_TREE[0]
    assert np.array_equal(_edf_codes(edf), codes)  # every 16-bit code kept
    raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
    got = raw.get_data().T  # volts
    names = [n for n, _ in CHANNELS_0284_H004]
    missing = np.isnan(phys)
    assert missing.sum() == 3 and (got[missing] == 0).all()  # as the baseline
    exact = [k for k, n in enumerate(names) if n != "Fp2"]
    # Fp1 (gain 2): MNE's arithmetic gives WFDB's values to the last bit
    assert np.array_equal(got[:, 0], phys[:, 0] * 1e-6)
    ok = ~missing[:, exact]
    assert np.allclose(got[:, exact][ok] * 1e6, phys[:, exact][ok], rtol=1e-12, atol=0)
    # Fp2 (gain 17.98...): rounded ends (0.005 uV): within a tenth of a step of WFDB's
    k = names.index("Fp2")
    assert np.abs(got[:, k] * 1e6 - phys[:, k]).max() * gains[k] < 0.1
    # 0300: 12.5 s in 2.5-s data records, no sample dropped; no unit declared: uV
    folder = src / "test" / "0300"
    codes, phys, _ = _wfdb_segment(folder, "0300_001_010", ["EEG", "REF"])
    edf = tmp_path / "cache" / DSID / IN_TREE[3]
    assert np.array_equal(_edf_codes(edf), codes)
    got = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR").get_data().T
    assert got.shape == (3125, 6) and np.array_equal(got, phys * 1e-6)


# ------------------------------------------------- resume, dry run, wfdb


def test_a_second_run_writes_nothing_and_a_new_segment_is_added(src, tmp_path):
    pytest.importorskip("wfdb")
    first = _run(src, tmp_path / "cache")
    root = tmp_path / "cache" / DSID
    before = _snapshot(root)
    second = _run(src, tmp_path / "cache")
    assert second.ok and second.counts["kept"] == first.counts["written"] == 5
    assert second.counts["written"] == 0 and _snapshot(root) == before
    folder = src / "test" / "0300"
    write_record(folder, "0300_003_012_EEG", EEG_0300, _codes(3000, 4, 11), gains=[4.0] * 4, baselines=[0] * 4, sfreq=250)  # fmt: skip
    third = _run(src, tmp_path / "cache")
    assert third.ok and third.counts["kept"] == 5 and third.counts["written"] == 1
    assert len(_files(root)) == 6


def test_a_cut_run_resumes_without_reading_the_finished_subjects_again(
    src, tmp_path, monkeypatch
):
    pytest.importorskip("wfdb")
    import eegdash.to_bids.core as core  # noqa: PLC0415

    real_convert, real_read = core.convert_recording, core.read_header

    def cut_at_0300(ctx, rec, prior=None):
        if rec.subject == "0300":
            raise KeyboardInterrupt("job cut")
        return real_convert(ctx, rec, prior)

    monkeypatch.setattr(core, "convert_recording", cut_at_0300)
    with pytest.raises(KeyboardInterrupt):
        _run(src, tmp_path / "cache")
    monkeypatch.setattr(core, "convert_recording", real_convert)
    read = []
    monkeypatch.setattr(core, "read_header", lambda p: read.append(p) or real_read(p))
    report = _run(src, tmp_path / "cache")
    assert report.ok and report.counts["kept"] == 3 and report.counts["written"] == 2
    assert all("0300" in p.name for p in read) and len(read) == 2


def test_a_dry_run_writes_nothing_but_its_report(src, tmp_path, capsys):
    pytest.importorskip("wfdb")
    cache = tmp_path / "cache"
    out = tmp_path / "reports" / "dry.json"
    args = ["bids", DSID, "--src", str(src), "--cache-dir", str(cache), "--dry-run"]
    assert corpus_mod.main([*args, "--json", str(out)]) == 0
    assert "(dry run, nothing written)" in capsys.readouterr().out
    assert not cache.exists()
    assert json.loads(out.read_text())["counts"]["planned"] == 5


def test_without_wfdb_every_recording_fails_with_the_reason(src, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wfdb", None)
    report = _run(src, tmp_path / "cache")
    assert not report.ok and report.counts["failed"] == 5
    assert all("pip install wfdb" in r["reason"] for r in report.rows)


# ------------------------------------------------------------------ chain


def test_bids_then_files_verify_shards_convert(src, tmp_path, capsys):
    pytest.importorskip("wfdb")
    zarr = pytest.importorskip("zarr")
    cache = tmp_path / "cache"
    pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({k: list(map(float, v)) for k, v in pos.items()}))
    base = [DSID, "--cache-dir", str(cache)]
    assert corpus_mod.main(["bids", *base, "--src", str(src)]) == 0
    listing = cache / f"{DSID}.files.csv"
    assert corpus_mod.main(["files", *base, "--out", str(listing)]) == 0
    assert corpus_mod.main(["verify", *base, "--level", "hash"]) == 0
    assert corpus_mod.main(["shards", *base]) == 0
    out = tmp_path / "out"
    args = ["convert", *base, "--out", str(out), "--positions", str(bank)]
    assert corpus_mod.main(args) == 0
    part = out / "parts" / f"{DSID}_000"
    summary = json.loads((part / "summary.json").read_text())
    assert summary["skipped_by_reason"] == {} and summary["n_sessions"] == 5
    store = zarr.open_group(part / "samples.zarr", mode="r")
    channels = sorted(tuple(store[k].attrs["channels"]) for k in store.group_keys())
    # EEG-typed channels only: no ECG, A1 or M1 of the REF record, LOC or SpO2
    assert channels == [
        ("C3", "F3", "Fp1", "O1"),
        ("Cz", "Fp1", "Fp2", "Fpz", "Pz", "T3"),
    ]
    for k in store.group_keys():
        assert np.isfinite(store[k]["data"][:]).all()
