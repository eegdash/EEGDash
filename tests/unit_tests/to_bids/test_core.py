"""eegdash.to_bids: plan, guards, links, sidecars, exclusions, resume, shards, report, chain.

Every source tree is a handful of hand-written EDF files (``_edf.py``) read by a test
adapter defined here; no network, no MNE-BIDS writer. The chain test (``bids`` then
``files``, ``verify``, ``shards``, ``convert``) needs ``zarr``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import socket
from pathlib import Path

import mne
import numpy as np
import pytest
from _edf import write_edf

import eegdash.api
import eegdash.corpus as corpus_mod
from eegdash.to_bids import (
    Adapter,
    ChannelTable,
    Event,
    Exclusion,
    Recording,
    get_adapter,
    to_bids,
)

LABELS = [
    "EEG FP1-REF",
    "EEG FP2-REF",
    "EEG C3-REF",
    "EEG C4-REF",
    "EEG CZ-REF",
    "EKG1",
    "ODD 7",  # no rule: MISC, unmapped
]
NAMES = ["FP1", "FP2", "C3", "C4", "CZ", "EKG1", "ODD 7"]
DSID = "fake-1.0"


class Fake(Adapter):
    """A source ``<subject>/<subject>_<run>.edf`` beside a ``FAKE`` marker."""

    name = "fake"
    versions = ("1.0",)
    TABLE = ChannelTable(
        [
            (r"EEG (\w+)-(REF|LE)", r"\1", "EEG"),
            (r"EKG\d*", r"\g<0>", "ECG"),
            (r"PHOTIC", r"\g<0>", "TRIG"),
        ]
    )

    def check_source(self):
        if not (self.src / "FAKE").is_file():
            raise ValueError(f"{self.src}: not the fake release")

    def recordings(self):
        for path in sorted(self.src.glob("*/*.edf")):
            run = path.stem.rsplit("_", 1)[-1]
            yield Recording(
                path.relative_to(self.src),
                subject=path.parent.name,
                task="rest",
                run=run,
            )

    def channels(self, rec, header):
        return self.TABLE.channels(header)

    def sidecar(self, rec, header):
        return {"PowerLineFrequency": 60, "EEGReference": "average", "TaskName": "x"}

    def events(self, rec, header):
        return [Event(1.0, 0.5, "blink")] if rec.run == "1" else None

    def held_out(self):
        key = hashlib.sha256(b"s1_2").hexdigest()
        return [
            Exclusion("s9", "subject", "benchmark subject"),
            Exclusion(key, "recording", "benchmark recording", hashed=True),
        ]

    def participants(self, subjects):
        rows = [{"participant_id": f"sub-{s}", "age": 30} for s in subjects]
        return rows, {"age": {"Description": "Age", "Units": "year"}}

    def dataset_description(self):
        return {"Name": "Fake", "License": "CC0", "Authors": ["n/a"]}

    def readme(self, summary):
        return (
            f"# Fake\n\n{len(summary['recordings'])} recordings, {summary['hours']} h\n"
        )

    def sourcedata(self, subjects):
        kept = ",".join(subjects) + "\n"
        return {"FAKE": (self.src / "FAKE").read_bytes(), "kept.txt": kept.encode()}


class Collide(Fake):
    """Two source files on one BIDS path."""

    def recordings(self):
        for rec in super().recordings():
            yield Recording(rec.source, subject=rec.subject, task="rest", run="1")


class Underscore(Fake):
    """A subject label BIDS cannot hold."""

    def recordings(self):
        for rec in super().recordings():
            yield Recording(rec.source, subject=f"{rec.subject}_x", task="rest")


class Swapped(Fake):
    """s1's and s2's first runs swap places: another source on each path."""

    def recordings(self):
        swap = {"s1": "s2", "s2": "s1"}
        for rec in super().recordings():
            subject = (
                swap.get(rec.subject, rec.subject) if rec.run == "1" else rec.subject
            )
            yield Recording(rec.source, subject=subject, task="rest", run=rec.run)


class WithoutS2(Fake):
    """A plan that s2 has left."""

    def recordings(self):
        yield from (r for r in super().recordings() if r.subject != "s2")


class Reordered(Fake):
    """channels.tsv rows in another order than the reader's channels."""

    def channels(self, rec, header):
        return list(reversed(self.TABLE.channels(header)))


class AsBdf(Fake):
    """EDF sources linked as BDF data files."""

    extension = ".bdf"


class AsGdf(Fake):
    """A data format EEGDash has no reader for."""

    extension = ".gdf"


class Written(Fake):
    """Not linkable: a ``.dat`` source (its labels, one per line) written as an EDF."""

    linkable = False
    EXTRA: list[str] = []

    def recordings(self):
        for path in sorted(self.src.glob("*/*.dat")):
            run = path.stem.rsplit("_", 1)[-1]
            yield Recording(
                path.relative_to(self.src),
                subject=path.parent.name,
                task="rest",
                run=run,
            )

    def write_data(self, rec, dest):
        labels = (self.src / rec.source).read_text().split("\n")
        write_edf(dest, [*labels, *self.EXTRA], annotations=True)


class WrittenAgain(Written):
    """Revision 2 writes one more channel."""

    revision = 2
    EXTRA = ["PHOTIC"]


def _refuse(*args, **kwargs):
    raise AssertionError("the network was used")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(eegdash.api, "EEGDash", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)


@pytest.fixture
def src(tmp_path):
    """s1: two runs (the second held out by hash); s2: one run; s9: a held-out subject."""
    root = tmp_path / "release"
    root.mkdir()
    (root / "FAKE").write_text("fake release\n")
    for k, rel in enumerate(
        ["s1/s1_1.edf", "s1/s1_2.edf", "s2/s2_1.edf", "s9/s9_1.edf"]
    ):
        write_edf(root / rel, LABELS, annotations=True, seed=k)
    return root


def _run(src, cache, **kwargs):
    return to_bids(DSID, src, cache_dir=cache, adapter=Fake, **kwargs)


def _tsv(path, delimiter="\t"):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def _snapshot(root):
    """Name -> (size, mtime, inode, link target) of every file and link under *root*."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for f in filenames:
            p = Path(dirpath, f)
            st = p.lstat()
            target = os.readlink(p) if p.is_symlink() else None
            out[p.relative_to(root).as_posix()] = (
                st.st_size,
                st.st_mtime_ns,
                st.st_ino,
                target,
            )
    return out


def _tree(root):
    """The snapshot without the converter's own folder."""
    return {k: v for k, v in _snapshot(root).items() if not k.startswith(".eegdash/")}


# ----------------------------------------------------------------- the plan


def test_entities_give_the_paths(src, tmp_path):
    report = _run(src, tmp_path / "cache")
    assert report.ok, report.problems
    root = tmp_path / "cache" / DSID
    assert sorted(p.relative_to(root).as_posix() for p in root.glob("sub-*/eeg/*")) == [
        "sub-s1/eeg/sub-s1_task-rest_run-1_channels.tsv",
        "sub-s1/eeg/sub-s1_task-rest_run-1_eeg.edf",
        "sub-s1/eeg/sub-s1_task-rest_run-1_eeg.json",
        "sub-s1/eeg/sub-s1_task-rest_run-1_events.tsv",
        "sub-s2/eeg/sub-s2_task-rest_run-1_channels.tsv",
        "sub-s2/eeg/sub-s2_task-rest_run-1_eeg.edf",
        "sub-s2/eeg/sub-s2_task-rest_run-1_eeg.json",
        "sub-s2/eeg/sub-s2_task-rest_run-1_events.tsv",
    ]
    rec = Recording(
        Path("a/b.edf"), "01", "rest", session="2", acquisition="ar", run="3"
    )
    assert rec.key == "b" and rec.stem == "sub-01_ses-2_task-rest_acq-ar_run-3"
    assert rec.folder == Path("sub-01", "ses-2", "eeg")


@pytest.mark.parametrize(
    ("adapter", "match"),
    [(Collide, "named by two source files"), (Underscore, "not alphanumeric")],
)
def test_a_collision_or_a_bad_label_is_refused_before_any_write(
    src, tmp_path, adapter, match
):
    with pytest.raises(ValueError, match=match):
        to_bids(DSID, src, cache_dir=tmp_path / "cache", adapter=adapter)
    assert not (tmp_path / "cache").exists()


def test_the_adapter_and_the_version_are_checked(src, tmp_path):
    with pytest.raises(ValueError, match="version '2.0' unknown"):
        to_bids("fake-2.0", src, cache_dir=tmp_path, adapter=Fake)
    with pytest.raises(ValueError, match="does not convert 'other'"):
        to_bids("other-1.0", src, cache_dir=tmp_path, adapter=Fake)
    with pytest.raises(ValueError, match="no adapter"):
        to_bids("nothing-1.0", src, cache_dir=tmp_path)
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="not the fake release"):
        to_bids(DSID, tmp_path / "empty", cache_dir=tmp_path / "cache", adapter=Fake)
    assert get_adapter(f"{__name__}:Fake") is Fake


# ----------------------------------------------------------------- the guards


def test_the_output_may_not_be_in_the_source_nor_the_source_in_the_output(
    src, tmp_path
):
    with pytest.raises(ValueError, match="inside the source"):
        _run(src, src / "cache")
    inner = tmp_path / "c" / DSID / "release"
    inner.mkdir(parents=True)
    (inner / "FAKE").write_text("fake release\n")
    with pytest.raises(ValueError, match="inside the output"):
        _run(inner, tmp_path / "c")
    before = _snapshot(src)
    for link in ("symlink", "hard", "copy"):
        assert _run(src, tmp_path / link, link=link).ok
    assert _snapshot(src) == before  # names, sizes, mtimes, inodes


# ------------------------------------------------------------- the data file


def test_the_three_link_modes_and_a_link_to_another_source(src, tmp_path):
    rel = "sub-s1/eeg/sub-s1_task-rest_run-1_eeg.edf"
    source = (src / "s1" / "s1_1.edf").resolve()
    for link in ("symlink", "hard", "copy"):
        _run(src, tmp_path / link, link=link)
    sym, hard, copy = (tmp_path / m / DSID / rel for m in ("symlink", "hard", "copy"))
    assert sym.is_symlink() and os.readlink(sym) == str(source)
    assert not hard.is_symlink() and hard.stat().st_ino == source.stat().st_ino
    assert not copy.is_symlink() and copy.stat().st_ino != source.stat().st_ino
    assert copy.read_bytes() == source.read_bytes()
    sym.unlink()
    sym.symlink_to(src / "s2" / "s2_1.edf")  # now another source's
    report = _run(src, tmp_path / "symlink")
    assert not report.ok
    assert [r["status"] for r in report.rows if r["bids_path"] == rel] == ["refused"]
    assert report.problems[0].startswith(f"{rel}: refused: exists and is not")
    assert os.readlink(sym) == str(src / "s2" / "s2_1.edf")  # left as it was


def test_a_link_that_fails_is_reported_and_the_others_written(
    src, tmp_path, monkeypatch
):
    import eegdash.to_bids.core as core  # noqa: PLC0415

    real = core.make_link

    def cross_device(source, path, link):
        if "sub-s2" in str(path):
            raise OSError(18, "Invalid cross-device link")
        real(source, path, link)

    monkeypatch.setattr(core, "make_link", cross_device)
    report = _run(src, tmp_path / "cache", link="hard")
    status = {
        r["subject"]: r["status"] for r in report.rows if r["status"] != "excluded"
    }
    assert status == {"s1": "written", "s2": "failed"}
    assert report.problems == [
        "sub-s2/eeg/sub-s2_task-rest_run-1_eeg.edf: failed: "
        "[Errno 18] Invalid cross-device link"
    ]
    root = tmp_path / "cache" / DSID
    assert [r["participant_id"] for r in _tsv(root / "participants.tsv")] == ["sub-s1"]
    assert list((root / "sub-s2").rglob("*")) in ([], [root / "sub-s2" / "eeg"])


# --------------------------------------------------------------- the sidecars


def test_channels_tsv_lists_the_readers_channels_in_order(src, tmp_path):
    report = _run(src, tmp_path / "cache")
    eeg = tmp_path / "cache" / DSID / "sub-s1" / "eeg"
    rows = _tsv(eeg / "sub-s1_task-rest_run-1_channels.tsv")
    raw = mne.io.read_raw_edf(src / "s1" / "s1_1.edf", verbose="ERROR")
    assert len(rows) == len(raw.ch_names) == len(LABELS)  # the annotation signal out
    assert [r["name"] for r in rows] == NAMES
    assert [r["description"] for r in rows] == raw.ch_names
    assert [r["type"] for r in rows] == ["EEG"] * 5 + ["ECG", "MISC"]
    assert {r["units"] for r in rows} == {"µV"}
    side = json.loads((eeg / "sub-s1_task-rest_run-1_eeg.json").read_text())
    assert side["TaskName"] == "rest"  # the recording's, not the adapter's
    assert side["SamplingFrequency"] == 256 and side["RecordingDuration"] == 12.0
    assert side["PowerLineFrequency"] == 60 and side["EEGReference"] == "average"
    assert side["SoftwareFilters"] == "n/a" and side["RecordingType"] == "continuous"
    assert (side["EEGChannelCount"], side["ECGChannelCount"]) == (5, 1)
    assert (side["MiscChannelCount"], side["TriggerChannelCount"]) == (1, 0)
    events = _tsv(eeg / "sub-s1_task-rest_run-1_events.tsv")
    assert events == [{"onset": "1.0", "duration": "0.5", "trial_type": "blink"}]
    summary = report.to_dict()
    assert summary["unmapped"] == {"ODD 7": 2}
    assert summary["labels"]["EEG CZ-REF"] == {
        "recordings": 2,
        "name": "CZ",
        "type": "EEG",
        "mapped": True,
    }
    assert "UNMAPPED 1 label(s) typed MISC: ODD 7" in str(report)


def test_labels_that_give_one_name_twice_are_refused(src, tmp_path):
    write_edf(src / "s2" / "s2_1.edf", ["EEG FP1-REF", "EEG FP1-LE", "EEG CZ-REF"])
    report = _run(src, tmp_path / "cache")
    (row,) = [r for r in report.rows if r["subject"] == "s2"]
    assert row["status"] == "invalid" and "not unique: FP1" in row["reason"]
    assert not report.ok and not (tmp_path / "cache" / DSID / "sub-s2").exists()


# --------------------------------------------------------------- exclusions


def test_held_out_recordings_and_subjects_are_left_out_and_reported(src, tmp_path):
    exclude = tmp_path / "exclude.csv"
    exclude.write_text("key,match,reason\ns2,subject,asked to leave\n")
    report = _run(src, tmp_path / "cache", exclude=exclude)
    root = tmp_path / "cache" / DSID
    assert not (root / "sub-s9").exists() and not (root / "sub-s2").exists()
    assert not list(root.glob("sub-s1/eeg/*run-2*"))
    excluded = {e["source"]: e["reason"] for e in report.to_dict()["excluded"]}
    assert excluded == {
        "s1/s1_2.edf": "benchmark recording",
        "s2/s2_1.edf": "asked to leave",
        "s9/s9_1.edf": "benchmark subject",
    }
    listed = {
        r["source"]: r for r in _tsv(root / ".eegdash/to_bids/map.csv", delimiter=",")
    }
    assert listed["s9/s9_1.edf"]["status"] == "excluded"
    assert listed["s9/s9_1.edf"]["link"] == ""
    assert [r["participant_id"] for r in _tsv(root / "participants.tsv")] == ["sub-s1"]
    kept = _run(src, tmp_path / "all", keep_held_out=True)
    assert kept.held_out_kept == 2 and kept.options["keep_held_out"] is True
    assert kept.to_dict()["excluded"] == [] and kept.counts["written"] == 4
    report = json.loads(
        (tmp_path / "all" / DSID / ".eegdash/to_bids/report.json").read_text()
    )
    assert report["held_out_kept"] == 2 and report["options"]["keep_held_out"] is True
    again = _run(src, tmp_path / "all")  # held out now, but in the tree from before
    assert any("in the tree from an earlier run" in p for p in again.problems)


# ------------------------------------------------------------ resume, shards


def test_a_second_run_writes_nothing(src, tmp_path):
    first = _run(src, tmp_path / "cache")
    root = tmp_path / "cache" / DSID
    before = _tree(root)
    second = _run(src, tmp_path / "cache")
    assert second.ok and _tree(root) == before
    assert first.counts["written"] == second.counts["kept"] == 2
    assert second.counts["written"] == 0
    # a sidecar lost since: written again, the link kept
    victim = root / "sub-s2" / "eeg" / "sub-s2_task-rest_run-1_channels.tsv"
    victim.unlink()
    third = _run(src, tmp_path / "cache")
    assert third.counts["written"] == 1 and victim.is_file()
    assert _tree(root).keys() == before.keys()


def test_shards_and_a_final_call_equal_one_full_run(src, tmp_path):
    write_edf(src / "s3" / "s3_1.edf", LABELS, seed=9)
    full = _run(src, tmp_path / "full", n_jobs=2)
    assert full.ok and full.counts["written"] == 3
    parts = tmp_path / "parts"
    for k in (0, 1):
        assert _run(src, parts, shard=k, n_shards=2).ok
        assert not (parts / DSID / "README.md").exists()  # root files: the final call
    state = parts / DSID / ".eegdash" / "to_bids"
    assert sorted(p.name for p in state.iterdir()) == ["scan-0.jsonl", "scan-1.jsonl"]
    final = _run(src, parts)
    assert final.counts["kept"] == 3 and final.counts["written"] == 0
    lists = []
    for out in (tmp_path / "full", parts):
        n = corpus_mod.write_file_list(out / DSID, tmp_path / f"{out.name}.csv")
        lists.append((tmp_path / f"{out.name}.csv").read_text())
        assert n == 19  # 3 x (edf, channels, eeg.json, events) + 7 root files
    assert lists[0] == lists[1]
    maps = [
        (o / DSID / ".eegdash/to_bids/map.csv").read_text()
        for o in (tmp_path / "full", parts)
    ]
    assert maps[0] == maps[1]
    with pytest.raises(ValueError, match="shard 2 of 2"):
        _run(src, parts, shard=2, n_shards=2)
    with pytest.raises(ValueError, match="go together"):
        _run(src, parts, shard=0)


def test_two_runs_into_two_folders_give_the_same_tree(src, tmp_path):
    for name in ("a", "b"):
        _run(src, tmp_path / name)
        corpus_mod.write_file_list(tmp_path / name / DSID, tmp_path / f"{name}.csv")
    assert (tmp_path / "a.csv").read_text() == (tmp_path / "b.csv").read_text()
    reports = [
        (tmp_path / n / DSID / ".eegdash/to_bids/report.json").read_text()
        for n in ("a", "b")
    ]
    assert reports[0] == reports[1]
    assert str(src) not in reports[0]  # no absolute path of the source
    root = tmp_path / "a" / DSID
    desc = json.loads((root / "dataset_description.json").read_text())
    assert desc["GeneratedBy"] == [{"Name": "eegdash.to_bids", "Version": "fake/1"}]
    assert desc["BIDSVersion"] and desc["DatasetType"] == "raw"
    assert (root / ".bidsignore").read_text() == ".eegdash/\n"
    assert (root / "sourcedata" / "FAKE").read_text() == "fake release\n"
    assert (root / "sourcedata" / "kept.txt").read_text() == "s1,s2\n"  # not s9
    assert (
        json.loads((root / "task-rest_events.json").read_text())["onset"]["Units"]
        == "s"
    )
    assert (root / "README.md").read_text() == "# Fake\n\n2 recordings, 0.007 h\n"
    assert (root / "participants.tsv").read_text() == (
        "participant_id\tage\nsub-s1\t30\nsub-s2\t30\n"
    )
    done = json.loads((root / ".eegdash/to_bids/DONE").read_text())
    assert done["written"] == 2 and done["problems"] == 0


# ----------------------------------------------------------------- dry run


def test_a_dry_run_writes_nothing_under_the_cache(src, tmp_path, capsys):
    cache = tmp_path / "cache"
    out = tmp_path / "reports" / "dry.json"
    args = ["bids", DSID, "--src", str(src), "--cache-dir", str(cache)]
    args += ["--adapter", f"{__name__}:Fake", "--dry-run", "--json", str(out)]
    assert corpus_mod.main(args) == 0
    assert not cache.exists()
    printed = capsys.readouterr().out
    assert (
        "2 planned, 2 excluded" in printed and "(dry run, nothing written)" in printed
    )
    report = json.loads(out.read_text())
    assert report["counts"]["planned"] == 2 and report["counts"]["excluded"] == 2
    assert report["sfreq"] == {"256": 2} and report["unmapped"] == {"ODD 7": 2}
    assert report["configurations"][0] == {
        "n_channels": 7,
        "recordings": 2,
        "names": NAMES,
    }
    rows = _tsv(out.with_suffix(".csv"), delimiter=",")
    assert [(r["source"], r["status"]) for r in rows] == [
        ("s1/s1_1.edf", "planned"),
        ("s1/s1_2.edf", "excluded"),
        ("s2/s2_1.edf", "planned"),
        ("s9/s9_1.edf", "excluded"),
    ]


def test_an_unreadable_header_is_reported_and_its_subject_left_out(src, tmp_path):
    (src / "s3").mkdir()
    (src / "s3" / "s3_1.edf").write_bytes(b"not an EDF file" * 40)
    report = _run(src, tmp_path / "cache")
    root = tmp_path / "cache" / DSID
    (row,) = [r for r in report.rows if r["subject"] == "s3"]
    assert row["status"] == "unreadable" and not report.ok
    assert report.problems == [f"s3/s3_1.edf: unreadable: {row['reason']}"]
    assert not (root / "sub-s3").exists()
    assert [r["participant_id"] for r in _tsv(root / "participants.tsv")] == [
        "sub-s1",
        "sub-s2",
    ]


# ----------------------------------------------------------- command line


def test_cli_bids(src, tmp_path, capsys):
    base = ["--src", str(src), "--cache-dir", str(tmp_path / "cache")]
    base += ["--adapter", f"{__name__}:Fake", "--n-jobs", "1"]
    assert corpus_mod.main(["bids", DSID, *base]) == 0
    assert "2 written, 2 excluded" in capsys.readouterr().out
    rel = "sub-s1/eeg/sub-s1_task-rest_run-1_eeg.edf"
    assert (tmp_path / "cache" / DSID / rel).is_symlink()  # the default link
    for wrong in ("ds000001", f"{DSID},other-1.0"):
        with pytest.raises(SystemExit) as exc:
            corpus_mod.main(["bids", wrong, *base])
        assert exc.value.code == 2
    with pytest.raises(SystemExit):
        corpus_mod.main(["bids", DSID, "--cache-dir", str(tmp_path)])  # no --src
    capsys.readouterr()
    assert corpus_mod.main(["bids", DSID, *base[:2], "--cache-dir", str(src)]) == 1
    assert "BIDS REFUSED" in capsys.readouterr().out


# ------------------------------------------------------------------ chain


def test_bids_then_files_verify_shards_convert(src, tmp_path, capsys):
    pytest.importorskip("zarr")
    cache = tmp_path / "cache"
    pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({k: list(map(float, v)) for k, v in pos.items()}))
    base = [DSID, "--cache-dir", str(cache)]
    adapter = ["--adapter", f"{__name__}:Fake"]
    assert corpus_mod.main(["bids", *base, "--src", str(src), *adapter]) == 0
    listing = cache / f"{DSID}.files.csv"
    assert corpus_mod.main(["files", *base, "--out", str(listing)]) == 0
    assert corpus_mod.main(["verify", *base, "--level", "hash"]) == 0
    assert corpus_mod.main(["shards", *base]) == 0
    assert "shards: 1" in capsys.readouterr().out
    out = tmp_path / "out"
    assert (
        corpus_mod.main(["convert", *base, "--out", str(out), "--positions", str(bank)])
        == 0
    )
    part = out / "parts" / f"{DSID}_000"
    summary = json.loads((part / "summary.json").read_text())
    assert summary["skipped_by_reason"] == {} and summary["n_sessions"] == 2
    zarr = pytest.importorskip("zarr")
    group = zarr.open_group(part / "samples.zarr", mode="r")["0"]
    assert group.attrs["channels"] == ["C3", "C4", "Cz", "Fp1", "Fp2"]
    assert group["data"].shape == (2, 2000, 5)
    assert np.isfinite(group["data"][:]).all()


# ------------------------------------------------------------ review fixes


@pytest.mark.parametrize("link", ["hard", "copy"])
def test_another_source_on_a_hard_link_or_a_copy_is_refused(src, tmp_path, link):
    first = _run(src, tmp_path / "cache", link=link)
    assert first.ok
    root = tmp_path / "cache" / DSID
    rel = "sub-s1/eeg/sub-s1_task-rest_run-1_eeg.edf"
    before = (root / rel).read_bytes()
    # s2's first run, of the same size, now planned on s1's path (and back)
    swapped = to_bids(
        DSID, src, cache_dir=tmp_path / "cache", adapter=Swapped, link=link
    )
    status = {r["bids_path"]: r["status"] for r in swapped.rows}
    assert status[rel] == "refused" and not swapped.ok
    assert (root / rel).read_bytes() == before
    # without the earlier run's rows, a copy is told by its bytes, a hard link by its inode
    for scan in (root / ".eegdash" / "to_bids").glob("scan-*.jsonl"):
        scan.unlink()
    again = to_bids(DSID, src, cache_dir=tmp_path / "cache", adapter=Swapped, link=link)
    assert {r["bids_path"]: r["status"] for r in again.rows}[rel] == "refused"
    listed = _tsv(root / ".eegdash/to_bids/map.csv", delimiter=",")
    assert {r["bids_path"]: r["source"] for r in listed}[rel] == "s2/s2_1.edf"
    assert {r["bids_path"]: r["status"] for r in listed}[rel] == "refused"


def _dat_source(root):
    root.mkdir()
    (root / "FAKE").write_text("fake release\n")
    for rel in ("s1/s1_1.dat", "s2/s2_1.dat"):
        (root / rel).parent.mkdir()
        (root / rel).write_text("\n".join(LABELS))
    return root


def test_a_file_the_adapter_writes_is_read_and_written_again_after_a_new_revision(
    tmp_path,
):
    src = _dat_source(tmp_path / "release")
    cache = tmp_path / "cache"
    dry = to_bids(DSID, src, cache_dir=cache, adapter=Written, dry_run=True)
    assert dry.counts["planned"] == 2 and not cache.exists()
    report = to_bids(DSID, src, cache_dir=cache, adapter=Written)
    assert report.ok, report.problems
    eeg = cache / DSID / "sub-s1" / "eeg"
    edf = eeg / "sub-s1_task-rest_run-1_eeg.edf"
    assert edf.is_file() and not edf.is_symlink()
    rows = _tsv(eeg / "sub-s1_task-rest_run-1_channels.tsv")
    raw = mne.io.read_raw_edf(edf, verbose="ERROR")
    assert [r["description"] for r in rows] == raw.ch_names == LABELS
    assert to_bids(DSID, src, cache_dir=cache, adapter=Written).counts["kept"] == 2
    again = to_bids(DSID, src, cache_dir=cache, adapter=WrittenAgain)
    assert again.ok and again.counts["written"] == 2
    raw = mne.io.read_raw_edf(edf, verbose="ERROR")
    rows = _tsv(eeg / "sub-s1_task-rest_run-1_channels.tsv")
    assert raw.ch_names == [*LABELS, "PHOTIC"] == [r["description"] for r in rows]
    assert rows[-1]["type"] == "TRIG"
    assert not [p for p in (cache / DSID).rglob(".*") if ".tmp" in p.name]


def test_a_recording_that_left_the_plan_is_reported_not_deleted(src, tmp_path):
    assert _run(src, tmp_path / "cache").ok
    report = to_bids(DSID, src, cache_dir=tmp_path / "cache", adapter=WithoutS2)
    rel = "sub-s2/eeg/sub-s2_task-rest_run-1_eeg.edf"
    assert report.problems == [f"{rel}: in the tree but not in the plan: remove it"]
    assert (tmp_path / "cache" / DSID / rel).is_symlink()


def test_stale_temporary_files_are_removed_and_never_listed(src, tmp_path):
    assert _run(src, tmp_path / "cache").ok
    root = tmp_path / "cache" / DSID
    stale = [
        root / "sub-s1" / "eeg" / ".sub-s1_task-rest_run-1_channels.999.tmp.tsv",
        root / "sub-s1" / "eeg" / ".sub-s1_task-rest_run-1_eeg.999.tmp.edf",
        root / ".README.999.tmp.md",
    ]
    for p in stale:
        p.write_text("cut")
    keep = root / "sub-s1" / "eeg" / ".notes"  # not a name of ours
    keep.write_text("mine")
    listing = tmp_path / "list.csv"
    corpus_mod.write_file_list(root, listing)
    assert ".tmp" not in listing.read_text()
    assert _run(src, tmp_path / "cache").ok
    assert not any(p.exists() for p in stale) and keep.exists()


def test_a_cut_run_resumes_without_reading_the_finished_subjects_again(
    src, tmp_path, monkeypatch
):
    import eegdash.to_bids.core as core  # noqa: PLC0415

    real_convert, real_read = core.convert_recording, core.read_header

    def cut_at_s2(ctx, rec, prior=None):
        if rec.subject == "s2":
            raise KeyboardInterrupt("job cut")
        return real_convert(ctx, rec, prior)

    monkeypatch.setattr(core, "convert_recording", cut_at_s2)
    with pytest.raises(KeyboardInterrupt):
        _run(src, tmp_path / "cache")
    monkeypatch.setattr(core, "convert_recording", real_convert)
    read = []
    monkeypatch.setattr(core, "read_header", lambda p: read.append(p) or real_read(p))
    report = _run(src, tmp_path / "cache")
    assert report.ok and report.counts["kept"] == 1 and report.counts["written"] == 1
    assert [p.name for p in read] == ["s2_1.edf"]


@pytest.mark.parametrize(("seconds", "appends"), [(3600.0, 1), (0.0, 3)])
def test_the_scan_file_is_appended_to_in_batches_not_once_per_subject(
    src, tmp_path, monkeypatch, seconds, appends
):
    import eegdash.to_bids.core as core  # noqa: PLC0415

    write_edf(src / "s3" / "s3_1.edf", LABELS, seed=9)
    real_append, calls = core._append_rows, []
    monkeypatch.setattr(core, "FLUSH_SECONDS", seconds)
    monkeypatch.setattr(
        core,
        "_append_rows",
        lambda p, rows: calls.append(len(rows)) or real_append(p, rows),
    )
    report = _run(src, tmp_path / "cache")
    assert report.ok and len(calls) == appends and sum(calls) == 3
    scan = tmp_path / "cache" / DSID / ".eegdash" / "to_bids" / "scan-all.jsonl"
    assert len(scan.read_text().splitlines()) == 3


def test_shard_reports_add_up_to_the_full_run(src, tmp_path):
    write_edf(src / "s3" / "s3_1.edf", LABELS, seed=9)
    full = _run(src, tmp_path / "full")
    for n in (2, 3):
        shards = [_run(src, tmp_path / f"of{n}", shard=k, n_shards=n) for k in range(n)]
        for key in ("written", "excluded", "recordings"):
            assert sum(r.counts[key] for r in shards) == full.counts[key], (n, key)


def test_the_plan_refuses_a_format_without_a_reader_or_a_source_of_another_format(
    src, tmp_path
):
    with pytest.raises(ValueError, match="the data file must be one of"):
        to_bids(DSID, src, cache_dir=tmp_path / "cache", adapter=AsGdf)
    with pytest.raises(ValueError, match="linked as .bdf are not .bdf files"):
        to_bids(DSID, src, cache_dir=tmp_path / "cache", adapter=AsBdf)
    assert not (tmp_path / "cache").exists()


def test_channels_tsv_in_another_order_than_the_reader_is_refused(src, tmp_path):
    report = to_bids(DSID, src, cache_dir=tmp_path / "cache", adapter=Reordered)
    assert {r["status"] for r in report.rows if r["status"] != "excluded"} == {
        "invalid"
    }
    assert "row 1 describes 'ODD 7'" in report.problems[0]
    assert not list((tmp_path / "cache" / DSID).glob("sub-*"))


def test_the_default_link_is_hard_for_merge_and_symbolic_for_bids(
    tmp_path, monkeypatch
):
    import eegdash.to_bids  # noqa: PLC0415

    seen = {}

    def merge(self, out, name, **kwargs):
        seen["merge"] = kwargs["link"]
        raise ValueError("stop here")

    def bids(dataset_id, src, **kwargs):
        seen["bids"] = kwargs["link"]
        raise ValueError("stop here")

    monkeypatch.setattr(corpus_mod.EEGDashCorpus, "merge", merge)
    monkeypatch.setattr(eegdash.to_bids, "to_bids", bids)
    base = [DSID, "--cache-dir", str(tmp_path)]
    assert corpus_mod.main(["merge", *base, "--out", str(tmp_path), "--name", "x"]) == 1
    assert corpus_mod.main(["bids", *base, "--src", str(tmp_path)]) == 1
    assert seen == {"merge": "hard", "bids": "symlink"}
