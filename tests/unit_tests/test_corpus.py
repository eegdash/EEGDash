"""EEGDashCorpus: plan, download (via EEGDashDataset.download_all), verify, status, CLI.

The catalogue is a dict of documents and records; S3 is the real ``S3Client``
over a fake boto3 client serving bytes from a dict, so the downloader's
staging and key checks run for real. Position sidecars and NEMAR root/session
metadata (separate network paths with their own tests) are stubbed.
"""

from __future__ import annotations

import hashlib
import json
import threading

import pytest
from botocore.exceptions import ClientError

import eegdash.corpus as corpus_mod
import eegdash.downloader as downloader
from eegdash.corpus import EEGDashCorpus, _read_manifest
from eegdash.dataset.base import EEGDashRaw

DATA = b"0123456789abcdefghij"  # 20 bytes
CHANNELS = b"name\ttype\nCz\tEEG\n"
PARTICIPANTS = b"participant_id\nsub-01\nsub-02\n"
RAW = "sub-{s}/eeg/sub-{s}_task-rest_eeg.edf"
DEP = "sub-{s}/eeg/sub-{s}_task-rest_channels.tsv"


def _key(data: bytes, ext: str) -> str:
    return f"SHA256E-s{len(data)}--{hashlib.sha256(data).hexdigest()}.{ext}"


class FakeBoto:
    def __init__(self, objects: dict[str, bytes]):
        self.objects, self.gets, self.lock = objects, [], threading.Lock()

    @staticmethod
    def _error(status: int, op: str) -> ClientError:
        return ClientError(
            {
                "Error": {"Code": str(status)},
                "ResponseMetadata": {"HTTPStatusCode": status},
            },
            op,
        )

    def head_object(self, Bucket, Key, **_):
        if f"{Bucket}/{Key}" not in self.objects:
            raise self._error(404, "HeadObject")
        return {"ContentLength": len(self.objects[f"{Bucket}/{Key}"])}

    def get_object(self, Bucket, Key, **_):
        path = f"{Bucket}/{Key}"
        with self.lock:
            self.gets.append(path)
        if path not in self.objects:
            raise self._error(404, "GetObject")
        data = self.objects[path]

        class _Body:
            def iter_chunks(self, chunk_size):
                yield data

        return {"Body": _Body(), "ContentLength": len(data)}


class FakeCatalogue:
    def __init__(self, docs: dict, records: dict):
        self.docs, self.records, self.queries = docs, records, 0

    def get_dataset(self, dataset_id):
        self.queries += 1
        return self.docs.get(dataset_id)

    def find(self, query=None, /, **kw):
        self.queries += 1
        recs = [
            dict(r, storage=dict(r["storage"]))
            for r in self.records.get(kw["dataset"], [])
        ]
        if "subject" in kw:
            recs = [r for r in recs if r["subject"] in kw["subject"]]
        return recs


class Archive:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.docs: dict[str, dict] = {}
        self.records: dict[str, list[dict]] = {}

    def dataset(self, dsid, subjects, *, size_bytes=None, nemar=False):
        bucket = "nemar" if nemar else "openneuro.org"
        base = f"s3://{bucket}/{dsid}"
        backend = "nemar" if nemar else "s3"
        self.docs[dsid] = {
            "dataset_id": dsid,
            "size_bytes": size_bytes,
            "ingestion_fingerprint": f"fp-{dsid}",
            "storage": {
                "backend": backend,
                "base": base,
                "raw_key": "participants.tsv",
                "dep_keys": [],
                "annex_keys": {"participants.tsv": _key(PARTICIPANTS, "tsv")},
            },
        }
        if not nemar:
            self.objects[f"{bucket}/{dsid}/participants.tsv"] = PARTICIPANTS
        self.records[dsid] = []
        for i, sub in enumerate(subjects):
            self.recording(dsid, sub, DATA + bytes(range(i)), nemar=nemar)
        return self

    def recording(self, dsid, sub, data, *, nemar=False, raw=None, ext="edf"):
        bucket = "nemar" if nemar else "openneuro.org"
        raw = raw or RAW.format(s=sub)
        dep = DEP.format(s=sub)
        raw_key, dep_key = _key(data, ext), _key(CHANNELS, "tsv")
        self.records[dsid].append(
            {
                "dataset": dsid,
                "bids_relpath": raw,
                "subject": sub,
                "extension": f".{ext}",
                "storage": {
                    "backend": "nemar" if nemar else "s3",
                    "base": f"s3://{bucket}/{dsid}",
                    "raw_key": raw,
                    "dep_keys": [dep],
                    "annex_keys": {raw: raw_key, dep: dep_key},
                },
            }
        )
        if nemar:
            self.objects[f"nemar/{dsid}/objects/{raw_key}"] = data
            self.objects[f"nemar/{dsid}/objects/{dep_key}"] = CHANNELS
        else:
            self.objects[f"openneuro.org/{dsid}/{raw}"] = data
            self.objects[f"openneuro.org/{dsid}/{dep}"] = CHANNELS
        return self.records[dsid][-1]


@pytest.fixture(autouse=True)
def offline_metadata(monkeypatch):
    for name in (
        "_download_position_sidecars",
        "_fetch_nemar_root_metadata",
        "_fetch_nemar_session_metadata",
    ):
        monkeypatch.setattr(EEGDashRaw, name, lambda *a, **k: None)


@pytest.fixture
def archive():
    return (
        Archive()
        .dataset("ds1", ("01", "02"), size_bytes=200)
        .dataset("nm2", ("03",), size_bytes=100, nemar=True)
    )


@pytest.fixture
def s3(archive, monkeypatch):
    client = downloader.S3Client.__new__(downloader.S3Client)
    client._client = FakeBoto(archive.objects)
    monkeypatch.setattr(downloader, "get_s3_filesystem", lambda **_: client)
    return client._client


def _corpus(tmp_path, archive, ids=("ds1", "nm2")):
    return EEGDashCorpus(
        list(ids),
        cache_dir=tmp_path,
        eeg_dash_instance=FakeCatalogue(archive.docs, archive.records),
    )


# ---------------------------------------------------------------- construction


def test_table_and_ids(tmp_path):
    table = tmp_path / "my.csv"
    table.write_text("dataset_id,n_subjects,note\nds1,2,x\nnm2,,\n")
    corpus = EEGDashCorpus(table, cache_dir=tmp_path)
    assert corpus.ids == ["ds1", "nm2"]
    assert corpus._table_subjects("ds1") == 2 and corpus._table_subjects("nm2") is None
    (tmp_path / "bad.csv").write_text("id\nds1\n")
    with pytest.raises(ValueError, match="dataset_id"):
        EEGDashCorpus(tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="duplicate"):
        EEGDashCorpus(["ds1", "ds1"])
    with pytest.raises(ValueError, match="not in the corpus"):
        corpus.plan(datasets="ds9")


# ---------------------------------------------------------------- plan


def test_plan_lists_files_from_the_records(tmp_path, archive, s3):
    report = _corpus(tmp_path, archive).plan()
    assert report.ok and s3.gets == []
    ds1, nm2 = report.datasets
    assert set(ds1.entries) == {
        "participants.tsv",
        *(f.format(s=s) for s in ("01", "02") for f in (RAW, DEP)),
    }
    assert ds1.entries[RAW.format(s="02")]["size"] == 21
    assert ds1.entries[RAW.format(s="02")]["required"] is True
    assert ds1.subjects == 2 and ds1.size_bytes == 200
    assert nm2.entries[RAW.format(s="03")]["key"].startswith("SHA256E-s20--")
    header, entries = _read_manifest(tmp_path / "ds1")
    assert header["fingerprint"] == "fp-ds1" and set(entries) == set(ds1.entries)


def test_plan_reports_unserved_short_and_broken(tmp_path, archive, s3):
    table = tmp_path / "t.csv"
    table.write_text("dataset_id,n_subjects\nds1,3\nds404,\n")
    corpus = EEGDashCorpus(
        table,
        cache_dir=tmp_path,
        eeg_dash_instance=FakeCatalogue(archive.docs, archive.records),
    )
    report = corpus.plan()
    assert report.problems == [
        "ds1: the catalogue lists 2 subjects, the table 3",
        "ds404: not served by the catalogue",
    ]
    assert not (tmp_path / "ds404").exists()

    class Broken(FakeCatalogue):
        def get_dataset(self, dataset_id):
            raise ConnectionError("api down")

    broken = EEGDashCorpus(
        ["ds1"], cache_dir=tmp_path, eeg_dash_instance=Broken({}, {})
    )
    assert "catalogue query failed" in broken.plan().problems[0]


def test_catalogue_paths_cannot_escape_the_cache(tmp_path, archive, s3):
    """The catalogue is a trust boundary: a '../' path is refused, nothing written."""
    cache = tmp_path / "cache"
    archive.records["ds1"][0]["storage"]["dep_keys"].append("../../escaped.tsv")
    archive.objects["openneuro.org/ds1/../../escaped.tsv"] = b"x"
    report = _corpus(cache, archive, ids=("ds1",)).download(n_jobs=1)
    assert "outside the dataset folder" in report.problems[0]
    assert not (tmp_path / "escaped.tsv").exists() and s3.gets == []


# ---------------------------------------------------------------- download


def test_download_smallest_first_then_verify_and_resume(tmp_path, archive, s3):
    archive.dataset("ds3", ("05",), size_bytes=50)
    corpus = _corpus(tmp_path, archive, ids=("ds1", "nm2", "ds3"))
    report = corpus.download(n_jobs=1, n_datasets=1)
    assert report.ok, report.problems
    assert [g.split("/")[1] for g in s3.gets][0] == "ds3"  # smallest first
    assert {d.state for d in report.datasets} == {"complete"}
    assert (tmp_path / "ds1" / RAW.format(s="01")).read_bytes() == DATA
    assert not list(tmp_path.rglob("*.part"))
    gets = len(s3.gets)
    assert corpus.download(n_jobs=4).ok and len(s3.gets) == gets  # no network
    (tmp_path / "ds1" / RAW.format(s="02")).unlink()  # deleted since: refetched
    assert corpus.download(n_jobs=1).ok and len(s3.gets) == gets + 1


def test_short_archive_object_is_refused(tmp_path, archive, s3):
    """S3's own length can be short (nm000289): the key governs, nothing is accepted."""
    raw = RAW.format(s="01")
    archive.objects[f"openneuro.org/ds1/{raw}"] = DATA[:10]
    report = _corpus(tmp_path, archive, ids=("ds1",)).download(n_jobs=1)
    assert not report.ok and "1 recording(s) failed" in report.problems[0]
    assert report.datasets[0].entries[raw]["state"] == "missing"
    assert not (tmp_path / "ds1" / raw).exists()
    archive.objects[f"openneuro.org/ds1/{raw}"] = DATA
    assert _corpus(tmp_path, archive, ids=("ds1",)).download(n_jobs=1).ok


def test_nemar_objects_route(tmp_path, archive, s3):
    report = _corpus(tmp_path, archive, ids=("nm2",)).download(n_jobs=1)
    assert report.ok, report.problems
    assert any("nemar/nm2/objects/SHA256E" in g for g in s3.gets)


def test_keyless_companion_the_archive_lacks_is_absent(tmp_path, archive, s3):
    """on002680 lists an .fdt the archive never had: reported, never required."""
    ghost = "sub-01/eeg/sub-01_task-rest_eeg.fdt"
    archive.records["ds1"][0]["storage"]["dep_keys"].append(ghost)
    corpus = _corpus(tmp_path, archive, ids=("ds1",))
    report = corpus.download(n_jobs=1)
    assert report.ok, report.problems
    assert report.datasets[0].entries[ghost]["state"] == "absent"
    gets = len(s3.gets)
    assert corpus.download(n_jobs=1).ok and len(s3.gets) == gets  # stays complete


def test_time_budget_defers(tmp_path, archive, s3):
    corpus = _corpus(tmp_path, archive)
    report = corpus.download(n_jobs=1, n_datasets=1, time_budget=0)
    assert {d.state for d in report.datasets} == {"deferred"} and s3.gets == []
    assert corpus.download(n_jobs=1).ok


def test_subjects_restrict_the_plan(tmp_path, archive, s3):
    corpus = _corpus(tmp_path, archive, ids=("ds1",))
    assert corpus.download(n_jobs=1, subjects=["02"]).ok
    assert not (tmp_path / "ds1/sub-01").exists()
    assert _read_manifest(tmp_path / "ds1")[0]["subjects"] == ["02"]
    full = corpus.plan().datasets[0]  # the full plan withdraws COMPLETE
    assert (
        full.state == "partial"
        and full.entries[RAW.format(s="01")].get("state") is None
    )


def test_retired_openneuro_id_follows_its_nemar_twin(tmp_path, archive, s3):
    archive.dataset("on000003", ("04",), size_bytes=30, nemar=True)
    archive.docs["ds000003"] = archive.docs["on000003"]
    archive.records["ds000003"] = archive.records["on000003"]
    report = _corpus(tmp_path, archive, ids=("ds000003",)).download(n_jobs=1)
    assert report.ok, report.problems
    assert report.datasets[0].served_as == "on000003"
    assert (tmp_path / "on000003" / RAW.format(s="04")).read_bytes() == DATA
    assert not (tmp_path / "ds000003").exists()
    offline = EEGDashCorpus(["ds000003"], cache_dir=tmp_path)  # no client
    assert offline.verify(level="hash").ok
    assert offline.status().datasets[0].served_as == "on000003"


def test_two_corpora_share_one_cache(tmp_path, archive, s3):
    a, b = _corpus(tmp_path, archive), _corpus(tmp_path, archive)
    reports = {}
    threads = [
        threading.Thread(
            target=lambda c=c, k=k: reports.__setitem__(k, c.download(n_jobs=2))
        )
        for k, c in (("a", a), ("b", b))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert reports["a"].ok and reports["b"].ok
    assert not list(tmp_path.rglob("*.part"))


# ---------------------------------------------------------------- verify and status


def test_verify_finds_missing_and_corrupt(tmp_path, archive, s3):
    corpus = _corpus(tmp_path, archive, ids=("ds1",))
    assert corpus.download(n_jobs=1).ok
    raw1, raw2 = (tmp_path / "ds1" / RAW.format(s=s) for s in ("01", "02"))
    raw1.write_bytes(DATA[:-1] + b"!")  # right size, wrong bytes
    raw2.write_bytes(DATA)  # wrong size (21 expected)
    states = lambda r: {p: e["state"] for p, e in r.datasets[0].entries.items()}  # noqa: E731
    size = corpus.verify()
    assert states(size)[RAW.format(s="01")] == "present"
    assert states(size)[RAW.format(s="02")] == "corrupt" and not size.ok
    assert states(corpus.verify(level="hash"))[RAW.format(s="01")] == "corrupt"
    assert corpus.status().datasets[0].state == "partial"
    assert corpus.download(n_jobs=1).ok and corpus.verify(level="hash").ok


def test_partial_directory_recording_is_incomplete(tmp_path):
    ds = tmp_path / "x.ds"
    ds.mkdir()
    (ds / "BadChannels").write_bytes(b"x")
    entry = {
        "path": "x.ds",
        "kind": "recording",
        "size": None,
        "key": None,
        "required": True,
    }
    corpus_mod._check(tmp_path, entry, "hash")
    assert entry["state"] == "incomplete"
    (ds / "x.meg4").write_bytes(b"x")
    (ds / "x.res4").write_bytes(b"x")
    corpus_mod._check(tmp_path, entry, "size")
    assert entry["state"] == "present"


def test_brainvision_samples_file_is_required(tmp_path, archive, s3):
    """dep_keys may not list the .eeg: COMPLETE must still imply the samples are on disk."""
    raw = "sub-09/eeg/sub-09_task-rest_eeg.vhdr"
    header = b"Brain Vision Data Exchange Header File Version 1.0\nDataFile=sub-09_task-rest_eeg.eeg\n"
    archive.recording("ds1", "09", header, raw=raw, ext="vhdr")
    corpus = _corpus(tmp_path, archive, ids=("ds1",))
    report = corpus.download(n_jobs=1)  # the archive has no .eeg
    assert not report.ok and report.datasets[0].entries[raw]["state"] == "incomplete"
    assert report.datasets[0].state != "complete"


def test_status_is_offline(tmp_path, archive, s3):
    corpus = _corpus(tmp_path, archive)
    report = corpus.status()
    assert [d.state for d in report.datasets] == ["unplanned"] * 2 and not report.ok
    corpus.download(n_jobs=1, datasets="nm2")
    corpus.plan(datasets="ds1")
    queries = corpus._client.queries
    by_id = {
        d.dataset_id: d
        for d in EEGDashCorpus(["ds1", "nm2"], cache_dir=tmp_path).status().datasets
    }
    assert corpus._client.queries == queries
    assert by_id["nm2"].state == "complete" and by_id["ds1"].state == "planned"
    assert by_id["ds1"].row()["subjects"] == 2


def test_datasets_and_records_offline(tmp_path, archive, s3):
    corpus = _corpus(tmp_path, archive, ids=("ds1",))
    assert corpus.download(n_jobs=1).ok
    assert sorted(r.record["bids_relpath"] for r in corpus.records()) == [
        RAW.format(s="01"),
        RAW.format(s="02"),
    ]


def test_cli(tmp_path, archive, s3, monkeypatch, capsys):
    monkeypatch.setattr(
        EEGDashCorpus,
        "client",
        property(lambda self: FakeCatalogue(archive.docs, archive.records)),
    )
    cache = str(tmp_path / "cache")
    assert corpus_mod.main(["status", "ds1,nm2", "--cache-dir", cache]) == 1
    out = tmp_path / "r.json"
    assert (
        corpus_mod.main(
            ["download", "ds1,nm2", "--cache-dir", cache, "--json", str(out)]
        )
        == 0
    )
    assert json.loads(out.read_text())["ok"] is True
    assert (
        corpus_mod.main(["verify", "ds1,nm2", "--cache-dir", cache, "--level", "hash"])
        == 0
    )
    assert "complete" in capsys.readouterr().out
    table = tmp_path / "c.csv"
    table.write_text("dataset_id\nds1\nds404\n")
    assert corpus_mod.main(["status", str(table), "--cache-dir", cache]) == 1
    assert "PROBLEM ds404" in capsys.readouterr().out
    assert (
        corpus_mod.main(
            ["status", str(table), "--cache-dir", cache, "--datasets", "ds1"]
        )
        == 0
    )
