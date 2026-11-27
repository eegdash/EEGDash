"""eegdash.to_bids.tueg: the TUH EEG layout, sessions with a year, montages, channels, held out.

The source is a fake TUH EEG 2.0.1 release of hand-written EDF files (``_edf.py``): no
network, no real subject code. The packaged held-out table holds hashes of real codes, so
the tests that leave recordings out use a table of their own, with hashed keys.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import socket
from pathlib import Path

import pytest
from _edf import write_edf

import eegdash.api
import eegdash.corpus as corpus_mod
from eegdash.to_bids import ADAPTERS, get_adapter, read_exclusions, to_bids, tueg
from eegdash.to_bids.adapter import TABLES, Header
from eegdash.to_bids.tueg import TUEG

DSID = "tueg-2.0.1"
AR = [
    "EEG FP1-REF",
    "EEG Fp2-REF",  # case kept
    "EEG CZ-REF",
    "EEG T1-REF",
    "EEG LOC-REF",
    "EEG ROC-REF",
    "EKG1-REF",
    "EMG-REF",
    "PHOTIC-REF",
    "IBI",
    "BURSTS",
    "SUPPR",
    "EEG 26-REF",
    "EEG XYZ-REF",  # no rule: MISC, unmapped
]
AR_CHANNELS = [
    ("FP1", "EEG"),
    ("Fp2", "EEG"),
    ("CZ", "EEG"),
    ("T1", "EEG"),
    ("LOC", "EOG"),
    ("ROC", "EOG"),
    ("EKG1", "ECG"),
    ("EMG", "EMG"),
    ("PHOTIC", "TRIG"),
    ("IBI", "MISC"),
    ("BURSTS", "MISC"),
    ("SUPPR", "MISC"),
    ("26", "MISC"),
    ("EEG XYZ-REF", "MISC"),
]
LE = ["EEG FP1-LE", "EEG A1-LE", "EEG A2-LE", "EEG CZ-LE", "EKG1-LE"]
LE_CHANNELS = [
    ("FP1", "EEG"),
    ("A1", "EEG"),
    ("A2", "EEG"),
    ("CZ", "EEG"),
    ("EKG1", "ECG"),
]
SOURCES = {  # relative to edf/ -> labels
    # one subject, session 012 in two years, the same file name in both
    "000/aaaaaaaa/s012_2014/01_tcp_ar/aaaaaaaa_s012_t000.edf": AR,
    "000/aaaaaaaa/s012_2016/01_tcp_ar/aaaaaaaa_s012_t000.edf": AR,
    "000/aaaaaaaa/s013_2016/03_tcp_ar_a/aaaaaaaa_s013_t001.edf": AR,
    "000/aaaaaaab/s001_2010/02_tcp_le/aaaaaaab_s001_t000.edf": LE,
    "000/aaaaaaab/s001_2010/02_tcp_le/aaaaaaab_s001_t001.edf": LE,  # held out
    "000/aaaaaaab/s002_2011/04_tcp_le_a/aaaaaaab_s002_t000.edf": LE,
    "001/aaaaaaac/s001_2012/01_tcp_ar/aaaaaaac_s001_t000.edf": AR,  # held out
}
IN_TREE = [
    "sub-aaaaaaaa/ses-012y2014/eeg/sub-aaaaaaaa_ses-012y2014_task-clinical_acq-ar_run-000_eeg.edf",
    "sub-aaaaaaaa/ses-012y2016/eeg/sub-aaaaaaaa_ses-012y2016_task-clinical_acq-ar_run-000_eeg.edf",
    "sub-aaaaaaaa/ses-013y2016/eeg/sub-aaaaaaaa_ses-013y2016_task-clinical_acq-ara_run-001_eeg.edf",
    "sub-aaaaaaab/ses-001y2010/eeg/sub-aaaaaaab_ses-001y2010_task-clinical_acq-le_run-000_eeg.edf",
    "sub-aaaaaaab/ses-002y2011/eeg/sub-aaaaaaab_ses-002y2011_task-clinical_acq-lea_run-000_eeg.edf",
]
HELD_RECORDING = "TUAB v3.0.1 train recording (benchmark)"
HELD_SUBJECT = "TUEV v2.0.1 train subject (benchmark)"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class HeldOutHere(TUEG):
    """TUEG with a held-out table of the test's own (hashed keys), in place of the packaged one."""

    held_out_csv: Path | None = None

    def held_out(self):
        return read_exclusions(self.held_out_csv)


def _refuse(*args, **kwargs):
    raise AssertionError("the network was used")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(eegdash.api, "EEGDash", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)


@pytest.fixture
def src(tmp_path):
    root = tmp_path / "tuh_eeg" / "v2.0.1"
    (root / "DOCS").mkdir(parents=True)
    (root / "AAREADME.txt").write_text("TUH EEG Corpus v2.0.1\n")
    for k, (rel, labels) in enumerate(sorted(SOURCES.items())):
        write_edf(root / "edf" / rel, labels, seconds=4, seed=k)
    return root


@pytest.fixture
def held(tmp_path, monkeypatch):
    """A held-out table that leaves out one recording (by stem) and one subject, hashed."""
    path = tmp_path / "heldout.csv"
    path.write_text(
        "key,match,reason,hashed\n"
        f"{_sha('aaaaaaab_s001_t001')},recording,{HELD_RECORDING},1\n"
        f"{_sha('aaaaaaac')},subject,{HELD_SUBJECT},1\n"
    )
    monkeypatch.setattr(HeldOutHere, "held_out_csv", path)
    return path


def _run(src, cache, adapter=HeldOutHere, **kwargs):
    return to_bids(DSID, src, cache_dir=cache, adapter=adapter, **kwargs)


def _tsv(path, delimiter="\t"):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def _files(root):
    return sorted(p.relative_to(root).as_posix() for p in root.glob("sub-*/**/*.edf"))


def _snapshot(root):
    """Name -> (size, mtime, link target) of every file under *root* but ``.eegdash/``."""
    out = {}
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            p = Path(dirpath, f)
            rel = p.relative_to(root).as_posix()
            if not rel.startswith(".eegdash/"):
                st = p.lstat()
                target = os.readlink(p) if p.is_symlink() else None
                out[rel] = (st.st_size, st.st_mtime_ns, target)
    return out


# ---------------------------------------------------------------- the layout


def test_the_adapter_is_registered():
    assert ADAPTERS == {"tueg": "eegdash.to_bids.tueg:TUEG"}
    assert get_adapter("tueg") is TUEG
    assert TUEG.versions == ("2.0.1",) and TUEG.linkable and TUEG.extension == ".edf"


def test_parse_path():
    montages = tueg._meta()["montages"]
    rel = Path("000/aaaaaaaa/s012_2014/04_tcp_le_a/aaaaaaaa_s012_t003.edf")
    assert tueg.parse_path(rel, montages) == {
        "subject": "aaaaaaaa",
        "session": "012y2014",
        "acquisition": "lea",
        "run": "003",
    }
    wrong = {
        "000/aaaaaaaa/s012_2014/aaaaaaaa_s012_t000.edf": "not at edf/",
        "000/aaaaaaaa/s012/01_tcp_ar/aaaaaaaa_s012_t000.edf": "not s<NNN>_<YYYY>",
        "000/aaaaaaaa/s012_2014/05_tcp_xx/aaaaaaaa_s012_t000.edf": "montage folder",
        "000/aaaaaaaa/s012_2014/01_tcp_ar/aaaaaaab_s012_t000.edf": "file name",
        "000/aaaaaaaa/s012_2014/01_tcp_ar/aaaaaaaa_s013_t000.edf": "file name",
        "abc/aaaaaaaa/s012_2014/01_tcp_ar/aaaaaaaa_s012_t000.edf": "not a number",
    }
    for path, reason in wrong.items():
        assert reason in tueg.parse_path(Path(path), montages), path


def test_sessions_montages_and_runs_give_the_paths(src, held, tmp_path):
    report = _run(src, tmp_path / "cache")
    assert report.ok, report.problems
    root = tmp_path / "cache" / DSID
    assert _files(root) == IN_TREE
    # the two files of one name, in two years, are two recordings
    two = [r for r in report.rows if r["key"] == "aaaaaaaa_s012_t000"]
    assert len({r["bids_path"] for r in two}) == 2
    assert {r["status"] for r in two} == {"written"}
    assert {re.search(r"_acq-(\w+?)_", p)[1] for p in IN_TREE} == {
        "ar",
        "le",
        "ara",
        "lea",
    }
    link = root / IN_TREE[0]
    assert link.is_symlink() and os.readlink(link) == str(
        src.resolve() / "edf" / sorted(SOURCES)[0]
    )
    assert not list(root.glob("sub-*/**/*_events.tsv"))
    assert not (root / "task-clinical_events.json").exists()
    assert _tsv(root / "participants.tsv") == [
        {"participant_id": "sub-aaaaaaaa"},
        {"participant_id": "sub-aaaaaaab"},
    ]
    assert not (root / "participants.json").exists()


def test_the_sidecar_and_the_dataset_files(src, held, tmp_path):
    _run(src, tmp_path / "cache")
    root = tmp_path / "cache" / DSID
    side = json.loads(
        next(root.glob("sub-aaaaaaab/**/*acq-le_run-000_eeg.json")).read_text()
    )
    assert side["TaskName"] == "clinical"
    assert side["PowerLineFrequency"] == 60 and side["EEGReference"] == "n/a"
    assert side["SamplingFrequency"] == 256 and side["RecordingDuration"] == 4
    assert side["EEGChannelCount"] == 4 and side["ECGChannelCount"] == 1
    desc = json.loads((root / "dataset_description.json").read_text())
    assert desc["Name"] == "TUH EEG Corpus (TUEG)"
    assert desc["SourceDatasets"] == [
        {
            "URL": "https://isip.piconepress.com/projects/nedc/html/tuh_eeg/",
            "Version": "2.0.1",
        }
    ]
    assert "forbid redistribution" in desc["License"]
    assert "isip.piconepress.com/projects/nedc/html/tuh_eeg/" in desc["License"]
    assert desc["GeneratedBy"] == [{"Name": "eegdash.to_bids", "Version": "tueg/2"}]
    readme = (root / "README.md").read_text()
    assert readme.startswith("# TUH EEG Corpus (TUEG) 2.0.1")
    assert "5 recordings of 2 subjects" in readme
    assert "`acq-ar` 2, `acq-ara` 1, `acq-le` 1, `acq-lea` 1" in readme
    assert "`EEG XYZ-REF` (3 recordings)" in readme
    assert "does not define them" in readme  # why EEGReference is n/a
    flat = readme.replace("\n", " ")
    assert (
        "TUAB" in flat
        and "training subjects (cross-validation within the training set) is clean"
        in flat
    )
    assert "a score on TUEV's evaluation set should not be reported" in flat
    assert "cannot be matched" in readme
    assert f"| {HELD_RECORDING} | 1 |" in readme
    assert f"| {HELD_SUBJECT} | 1 |" in readme
    assert not (root / "LICENSE").exists() and not (root / "sourcedata").exists()


# ---------------------------------------------------------------- channels


def test_the_channel_table():
    table = tueg._channel_table()
    for label, (name, kind) in zip([*AR, *LE], [*AR_CHANNELS, *LE_CHANNELS]):
        assert table.lookup(label) == (name, kind, label != "EEG XYZ-REF"), label
    assert table.lookup("EEG EKG-REF") == ("EKG", "ECG", True)
    assert table.lookup("PHOTIC PH") == ("PHOTIC PH", "TRIG", True)
    assert table.lookup("EEG SP1-REF") == ("SP1", "EEG", True)
    assert table.lookup("EEG T10-REF") == ("T10", "EEG", True)
    assert table.lookup("EEG 30-LE") == ("30", "MISC", True)
    assert table.lookup("EEG C3P-REF") == ("C3P", "EEG", True)
    # the labels EXP 37b found unmapped over the whole archive (2026-11-22)
    seen = {
        "ECG EKG-REF": ("EKG", "ECG"),
        "EEG LUC-LE": ("LUC", "EOG"),
        "EEG RLC-REF": ("RLC", "EOG"),
        "RESP ABDOMEN-REF": ("RESP ABDOMEN", "RESP"),
        "EEG RESP1-REF": ("RESP1", "RESP"),
        "DC3-DC": ("DC3", "MISC"),
        "PULSE RATE": ("PULSE RATE", "MISC"),
        "EEG PG1-LE": ("PG1", "EEG"),
        "EEG X1-REF": ("X1", "MISC"),
        "EEG 1X10_LAT_01-": ("1X10_LAT_01", "MISC"),
        "EEG C3-P3": ("C3-P3", "MISC"),
        "EEG FP1-F7": ("FP1-F7", "MISC"),
    }
    for label, (name, kind) in seen.items():
        assert table.lookup(label) == (name, kind, True), label
    assert table.lookup("EEG C3-REF") == (
        "C3",
        "EEG",
        True,
    )  # a referential label is not a bipolar pair
    assert table.lookup("EEG FOO-REF") == (
        "EEG FOO-REF",
        "MISC",
        False,
    )  # still no rule: unmapped, counted


def test_channels_tsv_rows(src, held, tmp_path):
    report = _run(src, tmp_path / "cache")
    root = tmp_path / "cache" / DSID
    ar = _tsv(next(root.glob("sub-aaaaaaaa/ses-012y2014/eeg/*_channels.tsv")))
    assert [(r["name"], r["type"]) for r in ar] == AR_CHANNELS
    assert [r["description"] for r in ar] == AR
    le = _tsv(next(root.glob("sub-aaaaaaab/ses-001y2010/eeg/*_channels.tsv")))
    assert [(r["name"], r["type"]) for r in le] == LE_CHANNELS
    assert report.to_dict()["unmapped"] == {"EEG XYZ-REF": 3}


def test_the_adapter_types_a_header_it_is_given(tmp_path):
    adapter = TUEG(tmp_path, "2.0.1")
    header = Header(tuple(LE), ("uV",) * len(LE), 250.0, 2500, None)
    channels = adapter.channels(None, header)
    assert [(c.name, c.type, c.mapped) for c in channels] == [
        (n, t, True) for n, t in LE_CHANNELS
    ]


# ------------------------------------------------------------- held out


def test_a_held_out_recording_and_subject_are_left_out(src, held, tmp_path):
    report = _run(src, tmp_path / "cache")
    root = tmp_path / "cache" / DSID
    excluded = {e["source"]: e["reason"] for e in report.to_dict()["excluded"]}
    assert excluded == {
        "edf/000/aaaaaaab/s001_2010/02_tcp_le/aaaaaaab_s001_t001.edf": HELD_RECORDING,
        "edf/001/aaaaaaac/s001_2012/01_tcp_ar/aaaaaaac_s001_t000.edf": HELD_SUBJECT,
    }
    assert not (root / "sub-aaaaaaac").exists()
    assert not list(root.glob("sub-aaaaaaab/**/*run-001*"))
    assert report.counts["written"] == 5 and report.counts["excluded"] == 2
    kept = _run(src, tmp_path / "all", keep_held_out=True)
    assert kept.held_out_kept == 2 and kept.counts["written"] == 7
    readme = (tmp_path / "all" / DSID / "README.md").read_text()
    assert "None in this run" in readme


def test_the_packaged_held_out_table(src):
    path = TABLES / "tueg-2.0.1.heldout.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["key", "match", "reason", "hashed"]
    assert len(rows) == 3283
    held = TUEG(src, "2.0.1").held_out()
    assert len(held) == 3283
    assert sum(e.match == "recording" for e in held) == 2993
    assert sum(e.match == "subject" for e in held) == 290
    assert all(e.hashed for e in held)
    assert all(re.fullmatch(r"[0-9a-f]{64}", e.key) for e in held)  # no clear code
    assert len({e.key for e in held}) == 3283
    assert {e.reason for e in held if e.match == "subject"} == {HELD_SUBJECT}
    assert {
        e.reason.split(" recording")[0] for e in held if e.match == "recording"
    } == {
        "TUAB v3.0.1 train",
        "TUAB v3.0.1 eval",
    }


def test_the_packaged_table_is_applied_by_default(src, tmp_path):
    report = _run(src, tmp_path / "cache", adapter=None)
    assert report.ok and report.counts["written"] == 7  # no fake code is held out
    readme = (tmp_path / "cache" / DSID / "README.md").read_text()
    assert "None in this run" in readme


# ------------------------------------------------------- source, dry run, resume


def test_a_folder_that_is_not_the_tuh_release_is_refused(src, tmp_path):
    other = tmp_path / "other"
    write_edf(other / "s1" / "s1_1.edf", AR)
    with pytest.raises(ValueError, match="no edf/ or AAREADME.txt: not the TUH EEG"):
        _run(other, tmp_path / "cache")
    (other / "edf").mkdir()
    with pytest.raises(ValueError, match="no AAREADME.txt: not the TUH EEG"):
        _run(other, tmp_path / "cache")
    with pytest.raises(ValueError, match="unknown to the tueg adapter"):
        to_bids("tueg-2.0.2", src, cache_dir=tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


def test_a_file_outside_the_layout_is_left_out_with_a_warning(src, held, tmp_path):
    write_edf(src / "edf/000/aaaaaaaa/s014_2017/09_tcp_xx/aaaaaaaa_s014_t000.edf", AR)
    report = _run(src, tmp_path / "cache")
    assert report.ok and report.counts["written"] == 5
    assert any(
        "09_tcp_xx/aaaaaaaa_s014_t000.edf: montage folder" in w for w in report.warnings
    )
    readme = (tmp_path / "cache" / DSID / "README.md").read_text()
    assert "09_tcp_xx" in readme


def test_a_dry_run_writes_nothing_but_its_report(src, tmp_path, capsys):
    cache = tmp_path / "cache"
    out = tmp_path / "reports" / "dry.json"
    before = _snapshot(src)
    args = ["bids", DSID, "--src", str(src), "--cache-dir", str(cache)]
    assert corpus_mod.main([*args, "--dry-run", "--json", str(out)]) == 0
    assert "(dry run, nothing written)" in capsys.readouterr().out
    assert not cache.exists() and _snapshot(src) == before
    assert sorted(p.name for p in out.parent.iterdir()) == ["dry.csv", "dry.json"]
    report = json.loads(out.read_text())
    assert report["counts"]["planned"] == 7 and report["unmapped"] == {"EEG XYZ-REF": 4}


def test_a_second_run_writes_nothing_and_a_new_file_is_added(src, held, tmp_path):
    first = _run(src, tmp_path / "cache")
    root = tmp_path / "cache" / DSID
    before = _snapshot(root)
    second = _run(src, tmp_path / "cache")
    assert second.ok and second.counts["kept"] == first.counts["written"] == 5
    assert second.counts["written"] == 0 and _snapshot(root) == before
    write_edf(src / "edf/000/aaaaaaab/s003_2012/01_tcp_ar/aaaaaaab_s003_t000.edf", AR)
    third = _run(src, tmp_path / "cache")
    assert third.ok and third.counts["kept"] == 5 and third.counts["written"] == 1
    assert len(_files(root)) == 6
