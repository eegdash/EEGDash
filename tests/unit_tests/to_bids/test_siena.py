"""eegdash.to_bids.siena: both seizure-list layouts (with midnight), run numbering, participants,
channel types, seizure placement, and one conversion of a tiny synthetic release read back by
MNE-BIDS. The EDF files are hand-written (``_edf.py``); no network, no MNE-BIDS writer.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from _edf import write_edf

from eegdash.to_bids import get_adapter, siena, to_bids

LAYOUT_A = """PN00

Data Sampling Rate: 512 Hz
*************************

Channels in EDF files:
**********************
Channel 1: Fp1
Channel 5: 1
Channel 30: EKG 1

Seizure n 1
File name: PN00-1.edf
Registration start time: 19.39.33
Registration end time:  20.22.58
Seizure start time: 19.58.36
Seizure end time: 19.59.46

Seizure n 2
File name: PN00-2.edf
Registration start time: 23.50.00
Registration end time: 00.40.00
Seizure start time: 00.05.10
Seizure end time: 00:06:00
"""

LAYOUT_B = """File name: PN01.edf
Registration start time: 19.00.44
Registration end time: 08.29.41

Seizure n 1
Start time: 21:51:02
End time: 21.51.56

Seizure n 2
Start time: 07.53.17
End time: 07.54.31
"""

SUBJECT_INFO = """patient_id, age_years, gender, seizure, localization, lateralization, eeg_channel, number_seizures, rec_time_minutes
PN00, 55, Male, IAS, T, R, 29, 5, 198
PN10, 25, Female, FBTC, F, Bilateral, 20, 10, 1002
"""


# ---------------------------------------------------------------- parsers


def test_layout_a_onsets_durations_and_midnight():
    p = siena.parse_seizure_list(LAYOUT_A)
    assert p["sfreq"] == 512 and p["channels"] == {1: "Fp1", 5: "1", 30: "EKG 1"}
    assert p["warnings"] == []
    s1, s2 = p["seizures"]
    # 19:58:36 - 19:39:33; the second crosses midnight
    assert (s1["file"], s1["onset"], s1["duration"]) == ("PN00-1.edf", 1143.0, 70.0)
    assert (s2["file"], s2["onset"], s2["duration"]) == ("PN00-2.edf", 910.0, 50.0)
    assert s2["registration_length"] == 3000.0


def test_layout_b_one_file_several_seizures_across_midnight():
    p = siena.parse_seizure_list(LAYOUT_B)
    assert [s["file"] for s in p["seizures"]] == ["PN01.edf", "PN01.edf"]
    assert p["warnings"] == []
    # 21:51:02 and 07:53:17 after 19:00:44
    assert [s["onset"] for s in p["seizures"]] == [10218.0, 46353.0]
    assert [s["duration"] for s in p["seizures"]] == [54.0, 74.0]


def test_a_seizure_without_end_is_reported_not_emitted():
    p = siena.parse_seizure_list(
        "File name: X.edf\nRegistration start time: 10.00.00\nSeizure n 1\n"
        "Start time: 10.01.00\n"
    )
    assert p["seizures"] == [] and any("no end time" in w for w in p["warnings"])


def test_clock_parsing():
    assert siena.parse_clock(" 07.53.17") == 7 * 3600 + 53 * 60 + 17
    assert siena.parse_clock("21:51:02") == 21 * 3600 + 51 * 60 + 2
    with pytest.raises(ValueError):
        siena.parse_clock("25.00.00")


def test_run_numbering():
    assert siena.assign_runs(["PN01.edf"]) == [("01", "PN01.edf")]
    assert siena.assign_runs(["PN12-4.edf", "PN12-3.edf", "PN12-1.2.edf"]) == [
        ("01", "PN12-1.2.edf"),
        ("02", "PN12-3.edf"),
        ("03", "PN12-4.edf"),
    ]
    names = [f"PN00-{i}.edf" for i in (10, 2, 1)]
    assert [f for _, f in siena.assign_runs(names)] == [
        "PN00-1.edf",
        "PN00-2.edf",
        "PN00-10.edf",
    ]


def test_records_and_file_matching():
    assert siena.parse_records("PN00/PN00-1.edf\nPN00/PN00-2.edf\nPN01/PN01.edf\n") == {
        "PN00": ["PN00-1.edf", "PN00-2.edf"],
        "PN01": ["PN01.edf"],
    }
    files = ["PN12-1.2.edf", "PN12-3.edf"]
    assert siena.match_file("pn12-1.2", files) == ("PN12-1.2.edf", None)
    assert siena.match_file("PN12-1.edf", files) == (None, None)
    # the lists' misspellings seen in the release (dry run of 2026-10-12)
    f, note = siena.match_file("PNO6-4.edf", ["PN06-1.edf", "PN06-4.edf"])
    assert f == "PN06-4.edf" and "read as PN06-4.edf" in note
    assert siena.match_file("PN01.edf", ["PN01-1.edf"])[0] == "PN01-1.edf"
    assert siena.match_file("PN11-.edf", ["PN11-1.edf"])[0] == "PN11-1.edf"
    assert siena.match_file("PN11-.edf", ["PN11-1.edf", "PN11-2.edf"]) == (None, None)


def test_participants_rows_and_sidecar_keys():
    rows = siena.parse_subject_info(SUBJECT_INFO)
    assert rows[0] == {
        "participant_id": "sub-PN00",
        "age": 55,
        "sex": "M",
        "seizure_type": "IAS",
        "localization": "T",
        "lateralization": "R",
        "n_seizures": 5,
        "recording_minutes": 198,
        "eeg_channels": 29,
    }
    assert rows[1]["sex"] == "F" and rows[1]["lateralization"] == "Bilateral"
    assert list(rows[0]) == siena.PARTICIPANT_COLUMNS


def test_channel_types_as_in_the_siena_headers():
    """`EEG Fp1`-style labels, one `EKG EKG`, and SPO2, HR, PLET, MK and spare channels
    named with numbers or letters (dry run of 2026-10-12).
    """
    ct = siena.channel_type
    assert ct("EKG EKG") == "ecg" and ct("EKG 1") == "ecg"
    eeg = ["EEG Fp1", "EEG FP2", "EEG T3", "EEG Fc1", "EEG F10", "EEG CZ", "Fp1", "Cz"]
    assert all(ct(n) == "eeg" for n in [*eeg, "P9"])
    assert all(
        ct(n) == "misc" for n in ["SPO2", "HR", "PLET", "MK", "1", "2", "61", "A", "B"]
    )


def test_parsers_import_without_mne():
    code = (
        "import sys; sys.modules['mne_bids'] = None; sys.modules['mne'] = None; "
        "from eegdash.to_bids import siena; "
        "print(len(siena.parse_seizure_list(sys.stdin.read())['seizures']))"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], input=LAYOUT_B, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "2"


def test_seizures_placed_by_the_header_clock_clipped_or_dropped():
    """PN10-7.8.9: the list's start is mistyped by ten hours, the header is right;
    PN00-3: the end time is an hour past the file; a seizure after the end is dropped.
    """
    meas = datetime(2017, 1, 1, 16, 49, 25, tzinfo=timezone.utc)
    reg = siena.parse_clock("06.49.25")

    def row(n, st, en):
        cur = {"file": "x", "reg_start": reg, "reg_end": None}
        return siena.seizure_row(n, cur, siena.parse_clock(st), siena.parse_clock(en))

    warn: list[str] = []
    kept = siena.place_seizures(
        [row(7, "17.35.13", "17.36.01"), row(8, "18.20.24", "18.20.42")],
        meas,
        14955.0,
        "PN10/PN10-7.8.9.edf",
        warn,
    )
    assert [(s["onset"], s["duration"]) for s in kept] == [
        (2748.0, 48.0),
        (5459.0, 18.0),
    ]
    assert len(warn) == 1 and "onsets from the header clock" in warn[0]
    warn = []
    meas = datetime(2016, 1, 1, 18, 15, 44, tzinfo=timezone.utc)
    reg = siena.parse_clock("18.15.44")
    kept = siena.place_seizures(
        [row(3, "18.28.29", "19.29.29"), row(4, "19.00.00", "19.00.30")],
        meas,
        2509.0,
        "PN00/PN00-3.edf",
        warn,
    )
    assert len(kept) == 1 and kept[0]["onset"] == 765.0
    assert kept[0]["duration"] == pytest.approx(2509.0 - 765.0)
    assert "clipped from 3660 s to 1744 s" in warn[0] and "dropped" in warn[1]


def test_extra_text_on_a_time_line_is_reported():
    p = siena.parse_seizure_list(
        "File name: PN10-2.edf\nRegistration start time: 09.30.15\n"
        "Registration end time: 11.52.29\nSeizure start time: 11.40.13\n"
        "Seizure end time: 11.41.04 oppure 11.40.43\n"
    )
    assert len(p["seizures"]) == 1 and p["seizures"][0]["duration"] == 51.0
    assert p["warnings"] == [
        "line 5: first time kept, the rest of the line ignored: 'oppure 11.40.43'"
    ]


# ------------------------------------------------------ a synthetic release

SIENA_LABELS = ["Fp1", "T3", "1", "Fc1", "EKG 1"]  # old spellings, a spare, one EKG


def _edf(path: Path, start: datetime, seconds: int = 60) -> None:
    write_edf(path, SIENA_LABELS, sfreq=512, seconds=seconds, start=start)


def _release(root: Path) -> Path:
    src = root / "siena"
    _edf(src / "PN00/PN00-1.edf", datetime(2020, 1, 1, 19, 39, 33))
    _edf(src / "PN00/PN00-2.edf", datetime(2020, 1, 2, 23, 59, 50))
    (src / "PN00/Seizures-list-PN00.txt").write_text(
        "PN00\n\nData Sampling Rate: 512 Hz\n\n"
        "Seizure n 1\nFile name: PN00-1.edf\nRegistration start time: 19.39.33\n"
        "Registration end time: 19.40.33\n"
        "Seizure start time: 19.39.53\nSeizure end time: 19.40.03\n\n"
        "Seizure n 2\nFile name: PN00-2.edf\nRegistration start time: 23.59.50\n"
        "Registration end time: 00.00.50\n"
        "Seizure start time: 00.00.10\nSeizure end time: 00:00:25\n"
    )
    (src / "subject_info.csv").write_text(
        "patient_id, age_years, gender, seizure, localization, lateralization, "
        "eeg_channel, number_seizures, rec_time_minutes\n"
        "PN00, 55, Male, IAS, T, R, 3, 2, 2\n"
    )
    (src / "RECORDS").write_text("PN00/PN00-1.edf\nPN00/PN00-2.edf\n")
    (src / "LICENSE.txt").write_text("Creative Commons Attribution 4.0 International\n")
    return src


def _tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _words(text: str) -> int:
    return len(" ".join(x for x in text.splitlines() if not x.startswith("- ")).split())


def test_a_synthetic_release(tmp_path):
    src, cache = _release(tmp_path), tmp_path / "cache"
    dry = to_bids("siena-1.0.0", src, cache_dir=cache, dry_run=True)
    assert dry.ok and dry.counts["planned"] == 2 and not cache.exists()

    report = to_bids("siena-1.0.0", src, cache_dir=cache)
    assert report.ok, report.problems
    assert (report.counts["written"], report.counts["subjects"]) == (2, 1)
    assert not any("EDF header starts at" in w for w in report.warnings)
    out = cache / "siena-1.0.0"
    eeg = out / "sub-PN00/eeg"
    edf1 = eeg / "sub-PN00_task-monitoring_run-01_eeg.edf"
    assert (
        edf1.is_symlink()
        and edf1.read_bytes() == (src / "PN00/PN00-1.edf").read_bytes()
    )
    assert (eeg / "sub-PN00_task-monitoring_run-02_eeg.edf").exists()
    ev1 = _tsv(eeg / "sub-PN00_task-monitoring_run-01_events.tsv")
    ev2 = _tsv(eeg / "sub-PN00_task-monitoring_run-02_events.tsv")
    assert [
        (float(e["onset"]), float(e["duration"]), e["trial_type"]) for e in ev1
    ] == [(20.0, 10.0, "seizure")]
    assert [(float(e["onset"]), float(e["duration"])) for e in ev2] == [(20.0, 15.0)]
    chans = _tsv(eeg / "sub-PN00_task-monitoring_run-01_channels.tsv")
    assert {c["name"]: c["type"] for c in chans} == {
        "Fp1": "EEG",
        "T3": "EEG",
        "1": "MISC",
        "Fc1": "EEG",
        "EKG 1": "ECG",
    }
    assert [c["name"] for c in chans] == SIENA_LABELS  # names unchanged, header order
    side = json.loads((eeg / "sub-PN00_task-monitoring_run-01_eeg.json").read_text())
    assert side["PowerLineFrequency"] == 50 and side["MISCChannelCount"] == 1
    assert (
        side["Manufacturer"] == "EB Neuro; Natus" and side["TaskName"] == "monitoring"
    )
    assert {
        "ManufacturersModelName",
        "SoftwareVersions",
        "CapManufacturer",
        "Instructions",
        "InstitutionAddress",
        "InstitutionalDepartmentName",
        "DeviceSerialNumber",
        "HardwareFilters",
    } <= set(side)  # the validator's list
    assert not list(out.rglob("*_electrodes.tsv")) and not list(out.rglob("*.lock"))

    parts = _tsv(out / "participants.tsv")
    assert [
        (p["participant_id"], p["age"], p["sex"], p["seizure_type"]) for p in parts
    ] == [("sub-PN00", "55", "M", "IAS")]
    assert list(parts[0]) == siena.PARTICIPANT_COLUMNS
    assert set(json.loads((out / "participants.json").read_text())) == set(
        siena.PARTICIPANT_COLUMNS
    )
    desc = json.loads((out / "dataset_description.json").read_text())
    assert desc["Name"] == "Siena Scalp EEG Database (BIDS)"
    assert desc["License"] == "CC-BY-4.0" and len(desc["Authors"]) == 3
    assert "Ethical Committee of the University of Siena" in desc["EthicsApprovals"][0]
    assert desc["Funding"][0].startswith("PANACEE")
    assert desc["GeneratedBy"] == [{"Name": "eegdash.to_bids", "Version": "siena/1"}]
    readme = (out / "README.md").read_text()
    assert (
        readme.startswith("# Siena Scalp EEG Database") and "PN00/PN00-1.edf" in readme
    )
    data, conversion = readme.split("## This BIDS version")
    conversion = conversion.split("### Original files")[0]
    assert 280 <= _words(data) <= 380 and 180 <= _words(conversion) <= 300
    assert "channel '1' typed MISC (not EEG or ECG) (all 2 files)" in readme
    assert "### Source inconsistencies" in readme
    # two 1-min files
    assert (
        "0 hours of EEG (the recording times of `subject_info.csv` sum to 0 hours)"
        in readme
    )
    assert (out / "CHANGES").read_text().startswith("1.0.0 2020-08-11\n")
    assert (out / "LICENSE").read_text().startswith("Creative Commons")
    assert (out / "sourcedata/Seizures-list-PN00.txt").exists()
    assert (out / "sourcedata/subject_info.csv").exists()

    from mne_bids import BIDSPath, read_raw_bids  # noqa: PLC0415

    bp = BIDSPath(subject="PN00", task="monitoring", run="01", datatype="eeg", root=out)
    raw = read_raw_bids(bp, verbose="ERROR")
    assert raw.ch_names == SIENA_LABELS and raw.info["sfreq"] == 512
    assert list(raw.annotations.description) == ["seizure"]
    assert raw.annotations.onset[0] == pytest.approx(20.0)


def test_sourcedata_holds_only_the_kept_subjects(tmp_path):
    src = _release(tmp_path)
    _edf(src / "PN01/PN01.edf", datetime(2020, 1, 3, 10, 0, 0))
    (src / "PN01/Seizures-list-PN01.txt").write_text("PN01\n")
    with open(src / "subject_info.csv", "a") as f:
        f.write("PN01, 40, Female, WIAS, F, L, 3, 0, 1\n")
    with open(src / "RECORDS", "a") as f:
        f.write("PN01/PN01.edf\n")
    exclude = tmp_path / "exclude.csv"
    exclude.write_text("key,match,reason\nPN01,subject,asked to leave\n")
    report = to_bids("siena-1.0.0", src, cache_dir=tmp_path / "cache", exclude=exclude)
    assert report.ok and report.counts["excluded"] == 1
    data = tmp_path / "cache" / "siena-1.0.0" / "sourcedata"
    assert sorted(p.name for p in data.iterdir()) == [
        "RECORDS",
        "Seizures-list-PN00.txt",
        "subject_info.csv",
    ]
    assert "PN01" not in (data / "subject_info.csv").read_text()
    assert (data / "RECORDS").read_text() == "PN00/PN00-1.edf\nPN00/PN00-2.edf\n"
    kept = to_bids("siena-1.0.0", src, cache_dir=tmp_path / "all")
    data = tmp_path / "all" / "siena-1.0.0" / "sourcedata"
    assert (data / "subject_info.csv").read_bytes() == (
        src / "subject_info.csv"
    ).read_bytes()
    assert kept.ok and (data / "Seizures-list-PN01.txt").exists()


def test_a_mismatched_header_clock_is_warned(tmp_path):
    src = _release(tmp_path)
    _edf(src / "PN00/PN00-1.edf", datetime(2020, 1, 1, 18, 0, 0))
    report = to_bids("siena-1.0.0", src, cache_dir=tmp_path / "cache", dry_run=True)
    (warning,) = [w for w in report.warnings if "EDF header starts at" in w]
    # the header clock puts the seizure outside the file: the list is kept
    assert "18:00:00" in warning and "onsets from the list" in warning


def test_not_the_siena_release(tmp_path):
    assert get_adapter("siena") is siena.Siena
    (tmp_path / "src").mkdir()
    with pytest.raises(ValueError, match="not the Siena release"):
        to_bids("siena-1.0.0", tmp_path / "src", cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="unknown to the siena adapter"):
        to_bids("siena-2.0.0", tmp_path / "src", cache_dir=tmp_path / "cache")
