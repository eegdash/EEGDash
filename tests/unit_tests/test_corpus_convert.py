"""EEGDashCorpus.shards and .convert: local datasets, the shard plan, one part per shard.

The datasets are tiny BIDS trees written with MNE-BIDS (EEGLAB files, which
need no extra writer). Nothing touches the network: the catalogue client and
``socket.create_connection`` raise. Tests that write a part need ``zarr``.
"""

from __future__ import annotations

import hashlib
import json
import socket

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, write_raw_bids

import eegdash.api
import eegdash.corpus as corpus_mod
import eegdash.corpus_convert as convert_mod
import eegdash.corpus_recipes as recipes_mod
import eegdash.local_bids
from eegdash import EEGDashDataset
from eegdash.corpus import EEGDashCorpus
from eegdash.corpus_convert import (
    Bank,
    Shard,
    convert_shard,
    discover_records,
    is_local_id,
    keep_listed,
    plan_shards,
    sample_offsets,
)
from eegdash.corpus_recipes import RECIPES, Recipe

NAMES = ["Fp1", "Fp2", "Cz", "Pz", "O1", "O2"]
SORTED = sorted(NAMES)


def _refuse(*args, **kwargs):
    raise AssertionError("the network was used")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(eegdash.api, "EEGDash", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)


@pytest.fixture(scope="module")
def bank():
    pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    return Bank(list(pos), np.array(list(pos.values())))


def _recording(root, subject, *, names=NAMES, seconds=30.0, sfreq=250, run=1):
    """One EEGLAB recording: the named EEG channels and an EOG channel."""
    rng = np.random.default_rng(int(subject) * 10 + run)
    data = rng.standard_normal((len(names) + 1, int(seconds * sfreq))) * 2e-5
    info = mne.create_info([*names, "EOG1"], sfreq, ["eeg"] * len(names) + ["eog"])
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.info["line_freq"] = 50
    path = BIDSPath(root=root, subject=subject, task="rest", run=run, datatype="eeg")
    write_raw_bids(
        raw, path, format="EEGLAB", allow_preload=True, overwrite=True, verbose="ERROR"
    )


def _tree(root):
    """Three subjects: two runs; four of the six channels; one of 8 s."""
    _recording(root, "01", seconds=30)  # 6000 timesteps at 200 Hz: 2 samples
    _recording(root, "01", seconds=45, run=2)  # 9000: 4 samples
    _recording(root, "02", names=NAMES[:4], seconds=30, sfreq=500)
    _recording(root, "03", seconds=8)  # 10 s or less
    return root


@pytest.fixture
def cache(tmp_path):
    _tree(tmp_path / "cache" / "demo-1.0")
    return tmp_path / "cache"


def _open(part):
    zarr = pytest.importorskip("zarr")
    return zarr.open_group(part / "samples.zarr", mode="r")


# ----------------------------------------------------------------- the rules


def test_local_ids():
    for dsid in ("tdbrain-3.1", "tueg-2.0.1", "icare-bdsp-2.0", "demo-1"):
        assert is_local_id(dsid), dsid
    for dsid in ("ds004840", "nm000115", "on003810", "EEG2025r7", "eegdash-corpus-v0"):
        assert not is_local_id(dsid), dsid


def test_samples_are_cut_as_the_loader_cuts_them():
    assert sample_offsets(2000).tolist() == []
    assert sample_offsets(4000).tolist() == [0]  # k * window gives k - 1
    assert sample_offsets(4001).tolist() == [0, 2000]
    assert sample_offsets(900, window=300).tolist() == [0, 300]


def test_bank_names(bank):
    names = ["EEG Fp1", "P03", "CZ", "Cz", "Nose", "O1"]
    kept, canonical, dropped = bank.resolve(names)
    assert [names[i] for i in kept] == ["EEG Fp1", "P03", "CZ", "O1"]
    assert [c.casefold() for c in canonical] == ["fp1", "po3", "cz", "o1"]
    assert dropped == ["Cz", "Nose"]  # the same electrode twice; an unknown name
    rings = [f"{r}{k}" for r in "AB" for k in range(1, 33)]
    cap = Bank([f"biosemi128_{n}" for n in rings] + ["A1"], np.ones((65, 3)))
    assert cap.resolve(rings)[1][0] == "biosemi128_A1"  # 64 ring labels: a cap
    assert cap.resolve(["A1", "A2"])[1] == ["A1"]  # two: the earlobes


def test_recipes_are_written_out_in_one_place(capsys):
    with pytest.raises(TypeError):
        Recipe()  # no default values: a recipe states every number
    assert all(name == r.name for name, r in RECIPES.items())
    reve = RECIPES["reve"].to_dict()
    assert reve == {
        "name": "reve",
        "version": 1,
        "sfreq": 200,
        "l_freq": 0.5,
        "h_freq": 99.5,
        "h_freq_max_ratio": 0.45,
        "resample_method": "polyphase",
        "window": 2000,
        "min_channels": 3,
        "units": "uV",
        "refused": sorted(recipes_mod.HELD_OUT),
    }
    assert recipes_mod.main() == 0
    assert json.loads(capsys.readouterr().out) == {"reve": reve}


def test_benchmark_sources_are_refused(tmp_path, bank):
    recipe = RECIPES["reve"]
    assert recipe.refuses("ds004362") and recipe.refuses("on004362")
    assert recipe.refuses("nm000110") and not recipe.refuses("ds004840")
    with pytest.raises(ValueError, match="benchmark source"):
        EEGDashCorpus(["demo-1.0", "nm000139"], cache_dir=tmp_path).convert(tmp_path)
    with pytest.raises(ValueError, match="must not be pretrained on"):
        convert_shard("on004362", None, tmp_path, tmp_path / "part", bank=bank)
    assert not (tmp_path / "part").exists()
    with pytest.raises(ValueError, match="unknown recipe"):
        EEGDashCorpus(["demo-1.0"], cache_dir=tmp_path).convert(tmp_path, recipe="x")


# ------------------------------------------------------------ the shard plan


def test_shard_plan_is_deterministic_and_covers_every_subject():
    sizes = {"03": 40, "01": 30, "10": 10, "02": 100, "04": 5}
    shards = plan_shards("ds1", sizes, max_bytes=60)
    assert shards == plan_shards("ds1", dict(reversed(sizes.items())), max_bytes=60)
    assert [s.subjects for s in shards] == [("01",), ("02",), ("03", "04", "10")]
    assert [s.name for s in shards] == ["ds1_000", "ds1_001", "ds1_002"]
    assert [s.n_bytes for s in shards] == [30, 100, 55]  # one above the limit: alone
    assert sorted(x for s in shards for x in s.subjects) == sorted(sizes)
    assert plan_shards("ds1", sizes, max_bytes=10**9) == [
        Shard("ds1", 0, ("01", "02", "03", "04", "10"), 185)
    ]
    assert plan_shards("ds1", {}, max_bytes=60) == []


def test_shards_from_a_local_tree_and_from_a_manifest(cache):
    corpus = EEGDashCorpus(["demo-1.0", "ds9"], cache_dir=cache)
    one = corpus.shards()
    assert [(s.name, s.subjects) for s in one] == [("demo-1.0_000", ("01", "02", "03"))]
    assert one == EEGDashCorpus(["demo-1.0", "ds9"], cache_dir=cache).shards()
    each = corpus.shards(max_bytes=1)
    assert [s.subjects for s in each] == [("01",), ("02",), ("03",)]
    # a catalogue dataset: from its manifest, without looking at the files
    manifest = cache / "ds9" / ".eegdash" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    rows = [
        {"dataset": "ds9"},
        {"path": "participants.tsv", "kind": "dataset", "size": 9},
        {"path": "sub-b/eeg/sub-b_task-x_eeg.edf", "kind": "recording", "size": 70},
        {"path": "sub-b/eeg/sub-b_task-x_channels.tsv", "kind": "sidecar", "size": 5},
        {"path": "sub-a/eeg/sub-a_task-x_eeg.edf", "kind": "recording", "size": 50},
        {"path": "sub-c/sub-c_scans.tsv", "kind": "sidecar", "size": 3},
    ]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    planned = [s for s in corpus.shards(max_bytes=100) if s.dataset_id == "ds9"]
    assert [(s.subjects, s.n_bytes) for s in planned] == [(("a",), 50), (("b",), 75)]


def _no_walk(*args, **kwargs):
    raise AssertionError("the tree was walked again")


def test_a_to_bids_tree_is_scanned_once_and_the_converting_processes_read_the_scan(
    cache, tmp_path, monkeypatch
):
    pytest.importorskip("zarr")
    root, scan = cache / "demo-1.0", cache / "demo-1.0.local.json"
    corpus = EEGDashCorpus(["demo-1.0"], cache_dir=cache)
    walked = corpus.shards(max_bytes=1)
    assert not scan.exists()  # not a to_bids tree: nothing saved, as before
    done = root / ".eegdash" / "to_bids" / "DONE"
    done.parent.mkdir(parents=True)
    done.write_text("{}\n")
    assert corpus.verify().ok and corpus.shards(max_bytes=1) == walked
    assert set(json.loads(scan.read_text())) == {"entries", "key", "subject_bytes"}
    for name in ("local_subjects", "_local_files", "_check"):
        monkeypatch.setattr(corpus_mod, name, _no_walk)
    again = EEGDashCorpus(["demo-1.0"], cache_dir=cache)  # another process
    assert again.shards(max_bytes=1) == walked  # the same plan: part 000 still holds
    assert again.status().ok
    assert again.convert(tmp_path / "out", shard=1, n_shards=3).ok
    done.write_text('{"run": 2}\n')  # the tree written again: the scan is stale
    with pytest.raises(AssertionError, match="walked again"):
        EEGDashCorpus(["demo-1.0"], cache_dir=cache).shards()


def test_a_local_tree_scanned_in_parallel_gives_what_one_lookup_at_a_time_gives(cache):
    root = cache / "demo-1.0"
    sizes = convert_mod.local_subjects(root, n_jobs=1)
    assert list(sizes) == ["01", "02", "03"]  # in order: the shard plan reads it so
    assert convert_mod.local_subjects(root, n_jobs=8) == sizes
    assert list(convert_mod.local_subjects(root, n_jobs=8)) == list(sizes)
    one, many = (
        EEGDashCorpus(["demo-1.0"], cache_dir=cache).verify(n_jobs=n).datasets[0]
        for n in (1, 8)
    )
    assert one.ok and one.entries == many.entries


def test_shard_and_n_shards_select_the_parts(cache, tmp_path, monkeypatch):
    calls = []

    def fake(dataset, subjects, root, out, **kwargs):
        calls.append((out.name, tuple(subjects)))
        return {"n_sessions": 1, "n_records_seen": 1}

    monkeypatch.setattr(corpus_mod, "convert_shard", fake)
    corpus = EEGDashCorpus(["demo-1.0"], cache_dir=cache)
    kwargs = dict(max_bytes=1, positions=Bank(["Cz"], np.zeros((1, 3))))
    assert corpus.convert(tmp_path, shard=1, **kwargs).ok
    assert calls == [("demo-1.0_001", ("02",))]
    calls.clear()
    report = corpus.convert(tmp_path, shard=0, n_shards=2, **kwargs)
    assert [c[0] for c in calls] == ["demo-1.0_000", "demo-1.0_002"]
    assert [s.state for s in report.shards] == ["converted"] * 2
    calls.clear()
    assert len(corpus.convert(tmp_path, **kwargs).shards) == 3 and len(calls) == 3
    with pytest.raises(ValueError, match="shard 3 of 3"):
        corpus.convert(tmp_path, shard=3, **kwargs)


class _Record:
    """A record as the converter reads it: its path inside the dataset."""

    def __init__(self, path):
        self.record = {"bids_relpath": path}


def test_only_a_datasets_own_recordings_are_kept():
    recs = [
        _Record("sub-1/ses-0/eeg/sub-1_ses-0_task-x_acq-a_run-0_eeg.bdf"),
        _Record("sub-1/ses-0/eeg/sub-1_ses-0_task-x_run-0_recording-a_eeg.bdf"),
        _Record("sourcedata/sub-01/eeg/sub-01_task-x_eeg.edf"),
        _Record("derivatives/clean/sub-1/eeg/sub-1_task-x_eeg.bdf"),
    ]
    kept, left = keep_listed(recs, None)  # no list: everything under sub-*
    assert kept == recs[:2]
    assert [x["detail"] for x in left] == ["sourcedata", "derivatives"]
    kept, left = keep_listed(recs, {recs[0].record["bids_relpath"]})
    assert kept == recs[:1]  # the old name of the same file is not listed
    assert [x["reason"] for x in left] == [
        "not_in_file_list",
        "outside_subject_folders",
        "outside_subject_folders",
    ]


def test_a_file_the_list_does_not_name_is_not_converted(tmp_path, bank):
    pytest.importorskip("zarr")
    root = tmp_path / "cache" / "ds000002"
    _recording(root, "01")
    _recording(root, "01", run=2)
    one = "sub-01/eeg/sub-01_task-rest_run-1_eeg.set"
    summary = convert_shard(
        "ds000002", None, root, tmp_path / "part", bank=bank, files={one}
    )
    assert summary["skipped_by_reason"] == {"not_in_file_list": 1}
    assert summary["n_records_seen"] == 2 and summary["n_sessions"] == 1
    sessions = _open(tmp_path / "part")["0"].attrs["sessions"]
    assert [s["file"] for s in sessions] == [one]


def test_a_shard_lists_its_own_subjects_and_reads_them_as_the_offline_mode_does(
    tmp_path, monkeypatch
):
    root = _tree(tmp_path / "cache" / "ds000003")
    (root / "sourcedata" / "sub-01" / "eeg").mkdir(parents=True)
    (root / "sourcedata" / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set").write_text("")
    offline = EEGDashDataset(
        cache_dir=str(root.parent), dataset="ds000003", download=False, subject=["01"]
    ).datasets
    offline = sorted(
        (r for r in offline if r.record["bids_relpath"].startswith("sub-")),
        key=lambda r: r.record["bids_relpath"],
    )
    # the whole-dataset walk is not used: a shard lists its subject folders only
    monkeypatch.setattr(eegdash.local_bids, "find_matching_paths", _refuse)
    recs = discover_records(root.parent, "ds000003", ["01"])
    assert [r.record["bids_relpath"] for r in recs] == [
        "sub-01/eeg/sub-01_task-rest_run-1_eeg.set",
        "sub-01/eeg/sub-01_task-rest_run-2_eeg.set",
    ]
    filled = {"digested_at", "sampling_frequency", "nchans", "ntimes", "ch_names"}
    for mine, theirs in zip(recs, offline, strict=True):
        assert {k: v for k, v in mine.record.items() if k not in filled} == {
            k: v for k, v in theirs.record.items() if k not in filled
        }
        keys = ("subject", "session", "run", "task")
        assert [mine.description.get(k) for k in keys] == [
            theirs.description.get(k) for k in keys
        ]
        assert mine.raw.ch_names == theirs.raw.ch_names
        np.testing.assert_array_equal(mine.raw.get_data(), theirs.raw.get_data())
    assert discover_records(root.parent, "ds000003", ["09"]) == []
    assert len(discover_records(root.parent, "ds000003", None)) == 4


def test_a_loaded_recording_is_released_and_a_listed_one_not_on_disk_is_reported(
    tmp_path, bank, monkeypatch
):
    pytest.importorskip("zarr")
    root = tmp_path / "cache" / "ds000004"
    _recording(root, "01")
    _recording(root, "01", run=2)
    _recording(root, "02")
    held = []

    def keeping(*args):
        held.extend(discover_records(*args))
        return list(held)

    monkeypatch.setattr(convert_mod, "discover_records", keeping)
    listed = {
        "sub-01/eeg/sub-01_task-rest_run-1_eeg.set",
        "sub-01/eeg/sub-01_task-rest_run-2_eeg.set",
        "sub-01/eeg/sub-01_task-rest_run-3_eeg.set",  # never fetched
        "sub-02/eeg/sub-02_task-rest_run-1_eeg.set",  # another shard's
    }
    summary = convert_shard(
        "ds000004", ["01"], root, tmp_path / "part", bank=bank, files=listed
    )
    assert summary["n_sessions"] == 2 and summary["n_records_seen"] == 2
    assert summary["skipped_by_reason"] == {"not_on_disk": 1}
    assert len(held) == 2 and all(rec._raw is None for rec in held)


def test_convert_passes_the_manifests_recordings(cache, tmp_path, monkeypatch):
    seen = {}

    def fake(dataset, subjects, root, out, **kwargs):
        seen[dataset] = kwargs["files"]
        return {"n_sessions": 1, "n_records_seen": 1}

    monkeypatch.setattr(corpus_mod, "convert_shard", fake)
    manifest = cache / "ds9" / ".eegdash" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    rows = [
        {"dataset": "ds9"},
        {"path": "sub-a/eeg/sub-a_task-x_eeg.edf", "kind": "recording", "size": 50},
        {"path": "sub-a/eeg/sub-a_task-x_channels.tsv", "kind": "sidecar", "size": 5},
    ]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    corpus = EEGDashCorpus(["demo-1.0", "ds9"], cache_dir=cache)
    corpus.convert(
        tmp_path, positions=Bank(["Cz"], np.zeros((1, 3))), require_complete=False
    )
    assert seen["ds9"] == {"sub-a/eeg/sub-a_task-x_eeg.edf"}
    assert len(seen["demo-1.0"]) == 4  # a local tree: the recordings under sub-*


def test_cli_converts_an_unverified_dataset_only_when_told(tmp_path, bank, capsys):
    pytest.importorskip("zarr")
    _recording(tmp_path / "cache" / "ds000003", "01")  # a catalogue id, never planned
    positions = tmp_path / "bank.json"
    positions.write_text(json.dumps(dict(zip(bank.names, bank.xyz.tolist()))))
    args = ["convert", "ds000003", "--cache-dir", str(tmp_path / "cache")]
    args += ["--out", str(tmp_path / "out"), "--positions", str(positions)]
    assert corpus_mod.main(args) == 1
    assert "no manifest" in capsys.readouterr().out
    assert corpus_mod.main([*args, "--unverified"]) == 0
    assert " converted " in capsys.readouterr().out


# ------------------------------------------------------------ local datasets


def test_a_local_dataset_is_read_in_place_and_never_downloaded(cache, tmp_path):
    corpus = EEGDashCorpus(["demo-1.0", "other-2.0"], cache_dir=cache)
    for report in (corpus.plan(), corpus.download(), corpus.verify(), corpus.status()):
        demo, other = report.datasets
        assert demo.ok and demo.state == "local" and demo.row()["present"] == 4
        assert demo.subjects == 3
        assert not other.ok and "obtain it under its terms" in other.problems[0]
        assert str(cache / "other-2.0") in other.problems[0]
    assert not (cache / "demo-1.0" / ".eegdash").exists()  # the tree is not written to
    assert len(list(EEGDashCorpus(["demo-1.0"], cache_dir=cache).records())) == 4
    report = EEGDashCorpus(["other-2.0"], cache_dir=cache).convert(tmp_path / "out")
    assert report.problems == ["other-2.0: nothing on disk: download or place it first"]
    assert not report.ok and not (tmp_path / "out").exists()


def test_a_local_copy_is_checked_against_its_file_list(cache, tmp_path, monkeypatch):
    root = cache / "demo-1.0"
    files = sorted(p for p in root.rglob("*") if p.is_file())
    listing = tmp_path / "corpora" / "files" / "demo-1.0.csv"
    listing.parent.mkdir(parents=True)
    rows = [
        f"{p.relative_to(root).as_posix()},{p.stat().st_size},"
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}"
        for p in files
    ]
    listing.write_text("path,size,sha256\n" + "\n".join(rows) + "\n")
    monkeypatch.setattr(corpus_mod, "_CORPORA", tmp_path / "corpora")
    corpus = EEGDashCorpus(["demo-1.0"], cache_dir=cache)
    report = corpus.verify(level="hash")
    assert report.ok and report.datasets[0].row()["present"] == len(files)
    victim = root / "sub-02" / "eeg" / "sub-02_task-rest_run-1_channels.tsv"
    victim.write_bytes(victim.read_bytes()[::-1])  # the same size, other bytes
    assert corpus.verify(level="size").ok
    report = corpus.verify(level="hash")
    assert not report.ok and report.datasets[0].row()["corrupt"] == 1
    victim.unlink()
    assert corpus.status().datasets[0].row()["missing"] == 1
    listing.write_text("path,size,sha256\n../x.edf,1,00\n")
    assert "outside the dataset folder" in corpus.status().problems[0]


def test_a_file_list_is_written_from_a_tree_behind_a_link(tmp_path):
    held = tmp_path / "held" / "v1"
    (held / "sub-01" / "eeg").mkdir(parents=True)
    (held / "sub-01" / "eeg" / "b.txt").write_bytes(b"bb")
    (held / "a.txt").write_bytes(b"a")
    (held / "sub-01" / "z.txt").write_bytes(b"zzz")
    (held / ".git").mkdir()
    (held / ".git" / "HEAD").write_bytes(b"hidden folder")
    (held / "sub-01" / ".DS_Store").write_bytes(b"hidden file")
    (tmp_path / "cache").mkdir()
    root = tmp_path / "cache" / "demo-1.0"
    root.symlink_to(held, target_is_directory=True)
    out = tmp_path / "lists" / "demo-1.0.csv"
    assert corpus_mod.write_file_list(root, out, n_jobs=2) == 3
    lines = out.read_text().splitlines()
    assert lines[0] == "path,size,sha256"
    rows = [line.split(",") for line in lines[1:]]
    assert [r[0] for r in rows] == ["a.txt", "sub-01/eeg/b.txt", "sub-01/z.txt"]
    assert [int(r[1]) for r in rows] == [1, 2, 3]
    assert [r[2] for r in rows] == [
        hashlib.sha256(b).hexdigest() for b in (b"a", b"bb", b"zzz")
    ]
    with pytest.raises(ValueError, match="inside"):
        corpus_mod.write_file_list(root, root / "list.csv")
    assert not (held / "list.csv").exists()


def test_a_local_copy_is_checked_against_the_file_list_beside_it(
    cache, tmp_path, monkeypatch
):
    monkeypatch.setattr(corpus_mod, "_CORPORA", tmp_path / "corpora")  # none packaged
    root = cache / "demo-1.0"
    n = corpus_mod.write_file_list(root, cache / "demo-1.0.files.csv")
    corpus = EEGDashCorpus(["demo-1.0"], cache_dir=cache)
    report = corpus.verify(level="hash")
    assert report.ok and report.datasets[0].row()["present"] == n
    assert n > 4  # every file of the tree, not the four recordings alone
    victim = root / "sub-01" / "eeg" / "sub-01_task-rest_run-1_channels.tsv"
    victim.write_bytes(victim.read_bytes()[::-1])  # the same size, other bytes
    assert corpus.verify(level="size").ok
    report = corpus.verify(level="hash")
    assert not report.ok and report.datasets[0].row()["corrupt"] == 1
    (cache / "demo-1.0.files.csv").unlink()  # no list: the recordings, present
    assert corpus.verify(level="hash").datasets[0].row()["present"] == 4


def _field(value, width):
    return str(value).ljust(width).encode("latin-1")[:width]


def _write_edf(path, labels, *, sfreq=256, seconds=12):
    """An EDF written byte by byte: *labels*, 1-s records of int16 samples in uV."""
    ns = len(labels)
    header = b"".join(
        [
            _field("0", 8),
            _field("X X X X", 80),
            _field("X", 80),
            _field("01.01.20", 8),
            _field("10.00.00", 8),
            _field(256 * (ns + 1), 8),
            _field("", 44),
            _field(seconds, 8),
            _field(1, 8),
            _field(ns, 4),
        ]
    )
    header += b"".join(_field(label, 16) for label in labels)
    for width, value in [
        (80, ""),  # transducer
        (8, "uV"),
        (8, -3200),
        (8, 3200),
        (8, -32768),
        (8, 32767),
        (80, ""),  # prefiltering
        (8, sfreq),
        (32, ""),
    ]:
        header += _field(value, width) * ns
    rng = np.random.default_rng(0)
    data = rng.integers(-1000, 1000, (seconds, ns, sfreq)).astype("<i2").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + data)
    return path


def _clean_names_tree(root, names):
    """One EDF labelled ``EEG FP1-REF``-style, its channels.tsv listing *names*."""
    eeg = root / "sub-01" / "eeg"
    stem = "sub-01_task-rest"
    labels = ["EEG FP1-REF", "EEG FP2-REF", "EEG C3-REF", "EEG C4-REF", "EEG CZ-REF"]
    _write_edf(eeg / f"{stem}_eeg.edf", [*labels, "EKG1"])
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "x", "BIDSVersion": "1.9.0"})
    )
    (eeg / f"{stem}_eeg.json").write_text(
        json.dumps(
            {"TaskName": "rest", "SamplingFrequency": 256, "PowerLineFrequency": 60}
        )
    )
    rows = [f"{n}\t{'ECG' if n == 'ECG' else 'EEG'}\tuV" for n in names]
    (eeg / f"{stem}_channels.tsv").write_text(
        "name\ttype\tunits\n" + "\n".join(rows) + "\n"
    )
    return root


def test_a_local_tree_whose_channels_tsv_cleans_the_labels_is_read(tmp_path, bank):
    pytest.importorskip("zarr")
    names = ["Fp1", "Fp2", "C3", "C4", "Cz", "ECG"]
    root = _clean_names_tree(tmp_path / "cache" / "demo-2.0", names)
    summary = convert_shard("demo-2.0", None, root, tmp_path / "part", bank=bank)
    assert summary["skipped_by_reason"] == {} and summary["n_sessions"] == 1
    assert _open(tmp_path / "part")["0"].attrs["channels"] == sorted(names[:5])
    # one row short: no name is set, the raw labels resolve to nothing, a skip
    root = _clean_names_tree(tmp_path / "cache" / "demo-3.0", names[:5])
    summary = convert_shard("demo-3.0", None, root, tmp_path / "short", bank=bank)
    assert summary["n_sessions"] == 0
    assert summary["skipped_by_reason"] == {"lt3_resolvable": 1}


def test_cli_writes_the_file_list_of_one_local_dataset(cache, tmp_path, capsys):
    out = tmp_path / "demo.csv"
    args = ["--cache-dir", str(cache), "--out", str(out), "--n-jobs", "2"]
    assert corpus_mod.main(["files", "demo-1.0", *args]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("files: ") and f"-> {out}" in printed
    n = len(out.read_text().splitlines()) - 1
    assert f"files: {n} files, " in printed
    for wrong in ("ds000001", "demo-1.0,other-2.0", "eegdash-corpus-v0"):
        with pytest.raises(SystemExit) as exc:
            corpus_mod.main(["files", wrong, *args])
        assert exc.value.code == 2
    with pytest.raises(SystemExit):
        corpus_mod.main(["files", "demo-1.0", "--cache-dir", str(cache)])  # no --out


# ------------------------------------------------------------------ one part


def test_convert_writes_one_part_per_shard(cache, tmp_path, bank):
    pytest.importorskip("zarr")
    corpus = EEGDashCorpus(["demo-1.0"], cache_dir=cache)
    report = corpus.convert(tmp_path / "out", positions=bank)
    assert report.ok, report.problems
    part = tmp_path / "out" / "parts" / "demo-1.0_000"
    root = _open(part)
    assert sorted(k for k, _ in root.groups()) == ["0", "1"]
    assert root.attrs["recipe"]["name"] == "reve" and root.attrs["sfreq"] == 200
    assert root.attrs["band_pass_hz"] == [0.5, 99.5] and root.attrs["units"] == "uV"
    full, four = root["0"], root["1"]
    assert full.attrs["channels"] == SORTED  # sorted names; the EOG channel is out
    assert full["data"].shape == (6, 2000, 6) and full["data"].dtype == "float32"
    assert full["session"][:].tolist() == [0, 0, 1, 1, 1, 1]
    assert full["stats"].shape == (2, 2, 6) and full["positions"].shape == (6, 3)
    assert [s["duration"] for s in full.attrs["sessions"]] == [6000, 9000]
    assert np.allclose(full["positions"][:], bank.positions(SORTED))
    assert four.attrs["channels"] == sorted(NAMES[:4])
    assert four["data"].shape == (2, 2000, 4)  # 500 Hz resampled: 6000 timesteps
    assert 5 < float(np.abs(full["data"][:]).mean()) < 50  # microvolts
    summary = json.loads((part / "summary.json").read_text())
    assert summary["skipped_by_reason"] == {"too_short": 1}
    assert summary["n_sessions"] == 3 and summary["n_samples"] == 8
    assert summary["recipe"] == json.loads((part / "recipe.json").read_text())
    manifest = (part / "manifest_part.csv").read_text().splitlines()
    assert (
        len(manifest) == 4
        and "sub-01/eeg/sub-01_task-rest_run-1_eeg.set" in manifest[1]
    )
    assert (part / "_DONE").exists()
    row = report.shards[0].row()
    assert (row["state"], row["sessions"], row["samples"]) == ("converted", 3, 8)
    assert report.to_dict()["ok"] is True and "1 shard(s)" in str(report)


def test_a_finished_part_is_skipped_and_an_unfinished_one_redone(
    cache, tmp_path, bank, monkeypatch
):
    pytest.importorskip("zarr")
    corpus = EEGDashCorpus(["demo-1.0"], cache_dir=cache)
    out = tmp_path / "out"
    first = corpus.convert(out, max_bytes=1, positions=bank)
    assert [s.state for s in first.shards] == ["converted"] * 3
    assert first.problems == ["demo-1.0_002: no session written: {'too_short': 1}"]
    done = out / "parts" / "demo-1.0_001" / "_DONE"
    done.unlink()  # an interrupted run: the folder is there, its marker is not
    (done.parent / "junk").write_text("left behind")
    calls = []
    real = corpus_mod.convert_shard
    monkeypatch.setattr(
        corpus_mod,
        "convert_shard",
        lambda *a, **k: calls.append(a[3].name) or real(*a, **k),
    )
    second = corpus.convert(out, max_bytes=1, positions=bank)
    assert [s.state for s in second.shards] == ["done", "converted", "done"]
    assert calls == ["demo-1.0_001"]
    assert done.exists() and not (done.parent / "junk").exists()
    assert [s.row()["samples"] for s in second.shards] == [6, 2, 0]


def test_a_tree_behind_a_link_and_a_relative_cache(tmp_path, bank, monkeypatch):
    pytest.importorskip("zarr")
    _recording(tmp_path / "held" / "v1", "01")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "demo-1.0").symlink_to(tmp_path / "held" / "v1", target_is_directory=True)
    (cache / "ds000001").symlink_to(tmp_path / "held" / "v1", target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    corpus = EEGDashCorpus(["demo-1.0", "ds000001"], cache_dir="cache")
    assert [s.name for s in corpus.shards()] == ["demo-1.0_000", "ds000001_000"]
    report = corpus.convert("out", positions=bank)
    # a catalogue id needs its files checked first, or require_complete=False
    assert report.problems == ["ds000001_000: no manifest: run plan or download first"]
    report = corpus.convert("out", positions=bank, require_complete=False)
    assert report.ok and [s.state for s in report.shards] == ["done", "converted"]
    for name in ("demo-1.0", "ds000001"):
        part = tmp_path / "out" / "parts" / f"{name}_000"
        root = _open(part)
        assert root.attrs["dataset"] == root["0"].attrs["dataset"] == name
        file = root["0"].attrs["sessions"][0]["file"]
        assert file == "sub-01/eeg/sub-01_task-rest_run-1_eeg.set"
    relpaths = [r.record["bids_relpath"] for r in corpus.records()]
    assert relpaths == ["sub-01/eeg/sub-01_task-rest_run-1_eeg.set"] * 2


def test_cli(cache, tmp_path, bank, capsys):
    pytest.importorskip("zarr")
    positions = tmp_path / "bank.json"
    positions.write_text(json.dumps(dict(zip(bank.names, bank.xyz.tolist()))))
    base = ["demo-1.0", "--cache-dir", str(cache)]
    assert corpus_mod.main(["shards", *base, "--max-gb", "1e-9"]) == 0
    assert "shards: 3" in capsys.readouterr().out
    args = ["--out", str(tmp_path / "out"), "--positions", str(positions)]
    report = tmp_path / "r.json"
    assert corpus_mod.main(["convert", *base, *args, "--json", str(report)]) == 0
    assert json.loads(report.read_text())["shards"][0]["samples"] == 8
    assert corpus_mod.main(["convert", *base, *args, "--shard", "0"]) == 0
    assert " done " in capsys.readouterr().out
    assert (
        corpus_mod.main(["status", "eegdash-corpus-v0", "--cache-dir", str(cache)]) == 1
    )
    assert "77 datasets" in capsys.readouterr().out  # a corpus by its name
