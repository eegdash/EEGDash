"""EEGDashCorpus: plan, download, verify, status and the CLI, offline.

The catalogue is a dict of dataset documents and records; S3 is the real
``S3Client`` over a fake boto3 client that serves bytes from a dict, so the
staging, key checks and error mapping of the downloader run for real.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import subprocess
import threading
import time

import pytest
from botocore.exceptions import ClientError

import eegdash.corpus as corpus_mod
import eegdash.downloader as downloader
from eegdash.corpus import EEGDashCorpus, _FileLock, _Manifest

DATA = b"0123456789abcdefghij"  # 20 bytes
CHANNELS = b"name\ttype\nCz\tEEG\n"
PARTICIPANTS = b"participant_id\nsub-01\nsub-02\n"


def _key(data: bytes, ext: str) -> str:
    return f"SHA256E-s{len(data)}--{hashlib.sha256(data).hexdigest()}.{ext}"


class FakeBoto:
    """boto3 stand-in: GET serves ``objects[bucket/key]``; HEAD too unless denied."""

    def __init__(
        self, objects: dict[str, bytes], *, head: bool = True, delay: float = 0.0
    ):
        self.objects = objects
        self.head = head
        self.delay = delay
        self.gets: list[str] = []
        self.lock = threading.Lock()

    def _error(self, status: int, op: str) -> ClientError:
        return ClientError(
            {
                "Error": {"Code": str(status)},
                "ResponseMetadata": {"HTTPStatusCode": status},
            },
            op,
        )

    def head_object(self, Bucket, Key, **_):
        if not self.head:
            raise self._error(403, "HeadObject")
        if f"{Bucket}/{Key}" not in self.objects:
            raise self._error(404, "HeadObject")
        return {"ContentLength": len(self.objects[f"{Bucket}/{Key}"])}

    def get_object(self, Bucket, Key, **_):
        path = f"{Bucket}/{Key}"
        with self.lock:
            self.gets.append(path)
        if path not in self.objects:
            raise self._error(404, "GetObject")
        if self.delay:
            threading.Event().wait(self.delay)
        data = self.objects[path]

        class _Body:
            def iter_chunks(self, chunk_size):
                yield data

        return {"Body": _Body(), "ContentLength": len(data)}


class FakeCatalogue:
    """``get_dataset`` and ``find`` over plain dicts; counts the queries."""

    def __init__(self, docs: dict, records: dict):
        self.docs, self.records, self.queries = docs, records, 0

    def get_dataset(self, dataset_id):
        self.queries += 1
        return self.docs.get(dataset_id)

    def find(self, query=None, /, **kw):
        self.queries += 1
        recs = list(self.records.get(kw["dataset"], []))
        if "subject" in kw:
            recs = [r for r in recs if r["subject"] in kw["subject"]]
        return recs


class Fixture:
    """A fake archive: datasets with records on S3 (OpenNeuro layout) or NEMAR."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.docs: dict[str, dict] = {}
        self.records: dict[str, list[dict]] = {}

    def dataset(
        self, dsid: str, subjects: tuple[str, ...], *, size_bytes=None, nemar=False
    ):
        bucket = "nemar" if nemar else "openneuro.org"
        base = f"s3://{bucket}/{dsid}"
        self.docs[dsid] = {
            "dataset_id": dsid,
            "size_bytes": size_bytes,
            "ingestion_fingerprint": f"fp-{dsid}",
            "storage": {
                "backend": "nemar" if nemar else "s3",
                "base": base,
                "raw_key": "participants.tsv",
                "dep_keys": [],
                **(
                    {"sidecar_inline": {"participants.tsv": PARTICIPANTS.decode()}}
                    if nemar
                    else {}
                ),
            },
        }
        if not nemar:
            self.objects[f"{bucket}/{dsid}/participants.tsv"] = PARTICIPANTS
        self.records[dsid] = []
        for i, sub in enumerate(subjects):
            data = DATA + bytes(range(i))  # every recording a different size
            raw = f"sub-{sub}/eeg/sub-{sub}_task-rest_eeg.edf"
            dep = f"sub-{sub}/eeg/sub-{sub}_task-rest_channels.tsv"
            raw_key, dep_key = _key(data, "edf"), _key(CHANNELS, "tsv")
            rec = {
                "dataset": dsid,
                "bids_relpath": raw,
                "subject": sub,
                "entities": {"subject": sub},
                "storage": {
                    "backend": "nemar" if nemar else "s3",
                    "base": base,
                    "raw_key": raw,
                    "dep_keys": [dep],
                    "annex_keys": {raw: raw_key, dep: dep_key},
                },
            }
            if nemar:
                self.objects[f"nemar/{dsid}/objects/{raw_key}"] = data
                self.objects[f"nemar/{dsid}/objects/{dep_key}"] = CHANNELS
            else:
                self.objects[f"openneuro.org/{dsid}/{raw}"] = data
                self.objects[f"openneuro.org/{dsid}/{dep}"] = CHANNELS
            self.records[dsid].append(rec)
        return self

    def catalogue(self) -> FakeCatalogue:
        return FakeCatalogue(self.docs, self.records)

    def s3(self, **kw) -> tuple[downloader.S3Client, FakeBoto]:
        client = downloader.S3Client.__new__(downloader.S3Client)
        client._client = FakeBoto(self.objects, **kw)
        return client, client._client


@pytest.fixture(autouse=True)
def short_sleep(monkeypatch):
    """The retry backoff (5, 20, 60 s) shrinks to 20 ms per round."""
    real_sleep = time.sleep
    monkeypatch.setattr(corpus_mod.time, "sleep", lambda s: real_sleep(min(s, 0.02)))


@pytest.fixture
def archive():
    return (
        Fixture()
        .dataset("ds1", ("01", "02"), size_bytes=200)
        .dataset("nm2", ("03",), size_bytes=100, nemar=True)
    )


def _corpus(tmp_path, archive, ids=("ds1", "nm2"), monkeypatch=None, **s3_kw):
    client, boto = archive.s3(**s3_kw)
    if monkeypatch is not None:
        monkeypatch.setattr(downloader, "get_s3_filesystem", lambda **_: client)
    corpus = EEGDashCorpus(
        list(ids), cache_dir=tmp_path, eeg_dash_instance=archive.catalogue()
    )
    return corpus, boto


# ---------------------------------------------------------------- construction


def test_named_table_missing_names_the_file():
    with pytest.raises(FileNotFoundError, match="eegdash-corpus-v1.csv"):
        EEGDashCorpus("eegdash-corpus-v1")


def test_from_table_reads_dataset_id_and_subjects(tmp_path):
    table = tmp_path / "my.csv"
    table.write_text("dataset_id,n_subjects,note\nds1,2,x\nnm2,,\n")
    corpus = EEGDashCorpus.from_table(table, cache_dir=tmp_path)
    assert corpus.ids == ["ds1", "nm2"]
    assert corpus.name == "my"
    assert corpus._table_subjects("ds1") == 2 and corpus._table_subjects("nm2") is None
    (tmp_path / "bad.csv").write_text("id\nds1\n")
    with pytest.raises(ValueError, match="dataset_id"):
        EEGDashCorpus.from_table(tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="not in the corpus"):
        corpus.plan(datasets="ds9")


# ---------------------------------------------------------------- plan


def test_plan_lists_files_and_bytes_from_the_keys(tmp_path, archive):
    corpus, boto = _corpus(tmp_path, archive)
    report = corpus.plan()
    assert report.ok and boto.gets == []
    ds1 = report.datasets[0]
    assert ds1.state == "planned" and ds1.subjects == 2 and ds1.size_bytes == 200
    paths = set(ds1.entries)
    assert (
        "participants.tsv" in paths and "sub-01/eeg/sub-01_task-rest_eeg.edf" in paths
    )
    assert len(paths) == 1 + 2 * 2  # participants + (raw, channels) per subject
    assert ds1.entries["sub-02/eeg/sub-02_task-rest_eeg.edf"].size == 21
    assert ds1.entries["sub-01/eeg/sub-01_task-rest_channels.tsv"].kind == "sidecar"
    assert ds1.entries["participants.tsv"].kind == "dataset"
    manifest = _Manifest(tmp_path / "ds1")
    header, entries = manifest.read()
    assert header["dataset"] == "ds1" and header["fingerprint"] == "fp-ds1"
    assert set(entries) == paths and all(e.state == "planned" for e in entries.values())
    nm2 = report.datasets[1]
    assert nm2.entries["participants.tsv"].kind == "dataset"
    assert nm2.entries["sub-03/eeg/sub-03_task-rest_eeg.edf"].key.startswith(
        "SHA256E-s20--"
    )


def test_plan_reports_a_dataset_the_catalogue_does_not_serve(
    tmp_path, archive, monkeypatch
):
    corpus, boto = _corpus(
        tmp_path, archive, ids=("ds1", "ds404"), monkeypatch=monkeypatch
    )
    report = corpus.plan()
    assert not report.ok
    assert report.problems == [
        "ds404: not served by the catalogue (no dataset document)"
    ]
    assert report.datasets[1].state == "not served"
    assert not (tmp_path / "ds404").exists()
    report = corpus.download()
    assert boto.gets and not any("ds404" in g for g in boto.gets)
    assert report.datasets[1].state == "not served" and not report.ok


def test_plan_reports_a_subject_shortfall_against_the_table(tmp_path, archive):
    table = tmp_path / "t.csv"
    table.write_text("dataset_id,n_subjects\nds1,3\nnm2,1\n")
    corpus = EEGDashCorpus.from_table(
        table, cache_dir=tmp_path, eeg_dash_instance=archive.catalogue()
    )
    report = corpus.plan()
    assert report.problems == ["ds1: the catalogue lists 2 subjects, the table 3"]


def test_plan_survives_a_failing_catalogue(tmp_path, archive):
    class Broken(FakeCatalogue):
        def get_dataset(self, dataset_id):
            raise ConnectionError("api down")

    corpus = EEGDashCorpus(
        ["ds1"], cache_dir=tmp_path, eeg_dash_instance=Broken({}, {})
    )
    report = corpus.plan()
    assert report.datasets[0].state == "problem"
    assert "catalogue query failed" in report.problems[0]


# ---------------------------------------------------------------- download


def test_download_order_smallest_dataset_then_dataset_files_then_largest(
    tmp_path, archive, monkeypatch
):
    archive.dataset("ds3", ("05", "06", "07"), size_bytes=50)
    corpus, boto = _corpus(
        tmp_path, archive, ids=("ds1", "nm2", "ds3"), monkeypatch=monkeypatch
    )
    report = corpus.download(n_jobs=1, n_datasets=1)
    assert report.ok, report.problems
    datasets = [g.split("/")[1] for g in boto.gets]
    # smallest first: ds3 (50), nm2 (100), ds1 (200)
    assert datasets == ["ds3"] * 7 + ["nm2"] * 2 + ["ds1"] * 5
    ds3 = boto.gets[:7]
    assert ds3[0].endswith("participants.tsv")  # dataset-level first
    sizes = [len(archive.objects[g]) for g in ds3[1:]]
    assert sizes == sorted(sizes, reverse=True)  # then largest first
    assert (tmp_path / "ds1/sub-01/eeg/sub-01_task-rest_eeg.edf").read_bytes() == DATA
    assert (tmp_path / "nm2/participants.tsv").read_bytes() == PARTICIPANTS  # inline
    assert not list(tmp_path.rglob("*.part")) and not list(tmp_path.rglob("*.lock"))


def test_done_file_of_the_right_size_is_skipped_without_network(
    tmp_path, archive, monkeypatch
):
    corpus, boto = _corpus(tmp_path, archive, monkeypatch=monkeypatch)
    assert corpus.download().ok
    first = len(boto.gets)
    report = corpus.download()
    assert report.ok and len(boto.gets) == first
    assert report.datasets[0].state == "fetched"
    entries = report.datasets[0].entries.values()
    assert all(e.state == "done" and e.bytes == (e.size or e.bytes) for e in entries)
    assert (
        report.datasets[0].entries["participants.tsv"].size is None
    )  # keyless: the size seen


def test_resume_after_a_kill(tmp_path, archive, monkeypatch):
    """Half the manifest written, a stale .part left: only the rest is fetched."""
    corpus, boto = _corpus(tmp_path, archive, ids=("ds1",), monkeypatch=monkeypatch)
    corpus.plan()
    manifest = _Manifest(tmp_path / "ds1")
    header, entries = manifest.read()
    raw = "sub-01/eeg/sub-01_task-rest_eeg.edf"
    (tmp_path / "ds1" / raw).parent.mkdir(parents=True)
    (tmp_path / "ds1" / raw).write_bytes(DATA)
    entries[raw].state, entries[raw].bytes = "done", 20
    manifest.write(header, {raw: entries[raw]})  # the killed job flushed one line
    stale = tmp_path / "ds1/sub-02/eeg/sub-02_task-rest_eeg.edf.1234.5678.part"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"xx")
    os.utime(stale, (time.time() - 7200, time.time() - 7200))
    report = corpus.download(n_jobs=1)
    assert report.ok
    assert not any(g.endswith(raw) for g in boto.gets)
    assert len(boto.gets) == 4  # participants, sub-02 raw, two channels
    assert not stale.exists()


def test_short_file_is_refused_as_integrity_then_repaired(
    tmp_path, archive, monkeypatch
):
    corpus, boto = _corpus(
        tmp_path, archive, ids=("ds1",), monkeypatch=monkeypatch, head=False
    )
    raw = "sub-01/eeg/sub-01_task-rest_eeg.edf"
    good = archive.objects[f"openneuro.org/ds1/{raw}"]
    archive.objects[f"openneuro.org/ds1/{raw}"] = good[:10]
    report = corpus.download(n_jobs=1)
    entry = report.datasets[0].entries[raw]
    assert not report.ok
    assert entry.state == "failed" and entry.error == "integrity"
    assert "expected 20 bytes, got 10" in entry.message
    assert not (tmp_path / "ds1" / raw).exists() and not list(tmp_path.rglob("*.part"))
    assert boto.gets.count(f"openneuro.org/ds1/{raw}") == 1 + corpus_mod._RETRY_ROUNDS
    assert report.datasets[0].errors == {"integrity": 1}
    assert report.datasets[0].state == "partial"
    archive.objects[f"openneuro.org/ds1/{raw}"] = good
    report = corpus.download(n_jobs=1)
    assert report.ok and report.datasets[0].entries[raw].state == "done"
    assert (tmp_path / "ds1" / raw).read_bytes() == good


def test_catalogue_404_is_classed_catalogue_and_not_retried(
    tmp_path, archive, monkeypatch
):
    corpus, boto = _corpus(tmp_path, archive, ids=("ds1",), monkeypatch=monkeypatch)
    raw = "sub-02/eeg/sub-02_task-rest_eeg.edf"
    del archive.objects[f"openneuro.org/ds1/{raw}"]
    report = corpus.download(n_jobs=2)
    entry = report.datasets[0].entries[raw]
    assert entry.state == "failed" and entry.error == "catalogue"
    assert boto.gets.count(f"openneuro.org/ds1/{raw}") == 1
    assert report.datasets[0].counts["present"] == 4
    assert "catalogue:1" in str(report)


def test_classify_errors():
    classify = corpus_mod._classify
    assert classify(FileNotFoundError("x")) == "catalogue"
    assert classify(PermissionError("x")) == "catalogue"
    assert classify(downloader.DownloadIntegrityError("x")) == "integrity"
    assert classify(OSError(28, "No space left on device")) == "storage"
    assert classify(OSError(errno.ECONNRESET, "reset")) == "network"
    assert classify(ConnectionError("x")) == "network"
    assert classify(TimeoutError()) == "network"
    assert classify(RuntimeError("x")) == "unknown"
    busy = ClientError(
        {"Error": {"Code": "503"}, "ResponseMetadata": {"HTTPStatusCode": 503}},
        "GetObject",
    )
    assert classify(busy) == "network"
    wrapped = RuntimeError("outer")
    wrapped.__cause__ = downloader.DownloadIntegrityError("inner")
    assert classify(wrapped) == "integrity"


def test_lock_per_target_no_double_fetch(tmp_path, archive, monkeypatch):
    corpus, boto = _corpus(tmp_path, archive, ids=("ds1",), monkeypatch=monkeypatch)
    raw = "sub-01/eeg/sub-01_task-rest_eeg.edf"
    dest = tmp_path / "ds1" / raw
    dest.parent.mkdir(parents=True)
    live = _FileLock(dest)
    assert live.acquire()  # another job of this host holds it
    report = corpus.download(n_jobs=1)
    entry = report.datasets[0].entries[raw]
    assert entry.state == "locked" and "pid" in entry.message
    assert not any(g.endswith(raw) for g in boto.gets)
    assert report.datasets[0].counts["present"] == 4 and not report.ok
    live.release()
    report = corpus.download(n_jobs=1)
    assert report.ok and boto.gets.count(f"openneuro.org/ds1/{raw}") == 1


def test_stale_lock_of_a_dead_process_is_broken(tmp_path, archive, monkeypatch):
    corpus, boto = _corpus(tmp_path, archive, ids=("ds1",), monkeypatch=monkeypatch)
    raw = "sub-01/eeg/sub-01_task-rest_eeg.edf"
    dest = tmp_path / "ds1" / raw
    dest.parent.mkdir(parents=True)
    lock = dest.with_name(dest.name + ".lock")
    lock.write_text(
        json.dumps({"host": corpus_mod.socket.gethostname(), "pid": _dead_pid()})
    )
    assert corpus.download(n_jobs=1).ok
    assert not lock.exists()


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def test_two_jobs_share_one_cache(tmp_path, archive, monkeypatch):
    """Two corpora downloading the same datasets at once fetch every file once."""
    corpus_a, boto = _corpus(tmp_path, archive, monkeypatch=monkeypatch, delay=0.02)
    corpus_b = EEGDashCorpus(
        ["ds1", "nm2"], cache_dir=tmp_path, eeg_dash_instance=archive.catalogue()
    )
    reports = {}
    threads = [
        threading.Thread(
            target=lambda c=c, k=k: reports.__setitem__(k, c.download(n_jobs=2))
        )
        for k, c in (("a", corpus_a), ("b", corpus_b))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert reports["a"].ok and reports["b"].ok
    assert len(boto.gets) == len(set(boto.gets)) == 7
    assert corpus_a.status().ok and not list(tmp_path.rglob("*.lock"))


def test_time_budget_defers_the_next_dataset(tmp_path, archive, monkeypatch):
    corpus, boto = _corpus(tmp_path, archive, monkeypatch=monkeypatch, delay=0.05)
    report = corpus.download(n_jobs=1, n_datasets=1, time_budget=0.02)
    by_id = {d.dataset_id: d for d in report.datasets}
    assert by_id["nm2"].state == "fetched"  # the smallest, started within the budget
    assert (
        by_id["ds1"].state == "deferred" and "time budget" in by_id["ds1"].problems[0]
    )
    assert not any("ds1" in g for g in boto.gets) and not report.ok
    assert corpus.download(n_jobs=1).ok  # the next job resumes


def test_subjects_restricts_the_plan_and_keeps_the_rest(tmp_path, archive, monkeypatch):
    corpus, boto = _corpus(tmp_path, archive, ids=("ds1",), monkeypatch=monkeypatch)
    report = corpus.download(n_jobs=1, subjects=["02"])
    assert report.ok
    entries = report.datasets[0].entries
    assert set(entries) == {
        "participants.tsv",
        "sub-02/eeg/sub-02_task-rest_eeg.edf",
        "sub-02/eeg/sub-02_task-rest_channels.tsv",
    }
    assert not (tmp_path / "ds1/sub-01").exists()
    assert _Manifest(tmp_path / "ds1").read()[0]["subjects"] == ["02"]
    assert corpus.verify().ok  # complete for the planned subset
    assert _Manifest(tmp_path / "ds1").complete_info()["subjects"] == ["02"]
    full = corpus.plan()
    assert len(full.datasets[0].entries) == 5
    assert (
        full.datasets[0].entries["sub-02/eeg/sub-02_task-rest_eeg.edf"].state == "done"
    )
    assert (
        full.datasets[0].entries["sub-01/eeg/sub-01_task-rest_eeg.edf"].state
        == "planned"
    )
    assert not _Manifest(tmp_path / "ds1").is_complete()  # new files reset it
    assert corpus.status().datasets[0].state == "partial"


def test_stale_record_is_accepted_and_reported(tmp_path, archive, monkeypatch):
    """The record's key says 20 bytes, the archive serves 30 and says so: stale-record."""
    corpus, boto = _corpus(tmp_path, archive, ids=("ds1",), monkeypatch=monkeypatch)
    raw = "sub-01/eeg/sub-01_task-rest_eeg.edf"
    archive.objects[f"openneuro.org/ds1/{raw}"] = DATA + b"0123456789"
    report = corpus.download(n_jobs=1)
    entry = report.datasets[0].entries[raw]
    assert report.ok
    assert entry.state == "stale-record" and entry.bytes == 30 and entry.size == 20
    assert "record 20 bytes, archive 30" in entry.message
    assert (tmp_path / "ds1" / raw).stat().st_size == 30
    assert report.datasets[0].counts["stale"] == 1
    assert (
        corpus.download(n_jobs=1).ok
        and boto.gets.count(f"openneuro.org/ds1/{raw}") == 2
    )
    status = corpus.status().datasets[0]
    assert status.state == "fetched" and status.counts["stale"] == 1
    verified = corpus.verify(level="hash")
    assert verified.ok and verified.datasets[0].entries[raw].state == "stale-record"


def test_nemar_objects_route_refuses_a_mismatch(tmp_path, archive, monkeypatch):
    """On objects/<key> the key names the object: no stale-record, refused."""
    corpus, boto = _corpus(tmp_path, archive, ids=("nm2",), monkeypatch=monkeypatch)
    rec = archive.records["nm2"][0]
    key = rec["storage"]["annex_keys"][rec["bids_relpath"]]
    archive.objects[f"nemar/nm2/objects/{key}"] = DATA + b"0123456789"
    report = corpus.download(n_jobs=1)
    entry = report.datasets[0].entries[rec["bids_relpath"]]
    assert entry.state == "failed" and entry.error == "integrity"


# ---------------------------------------------------------------- verify and status


def test_complete_is_written_by_verify_only(tmp_path, archive, monkeypatch):
    corpus, boto = _corpus(tmp_path, archive, ids=("ds1",), monkeypatch=monkeypatch)
    manifest = _Manifest(tmp_path / "ds1")
    assert corpus.download(n_jobs=1).ok
    assert not manifest.is_complete() and corpus.status().datasets[0].state == "fetched"
    report = corpus.verify()
    assert report.ok and manifest.is_complete()
    info = manifest.complete_info()
    assert (
        info["level"] == "size"
        and info["files"] == 5
        and info["fingerprint"] == "fp-ds1"
    )
    assert corpus.status().datasets[0].state == "complete"
    # a file removed: download finds something to fetch and drops COMPLETE
    raw = tmp_path / "ds1/sub-02/eeg/sub-02_task-rest_eeg.edf"
    raw.unlink()
    assert corpus.download(n_jobs=1).ok and not manifest.is_complete()
    assert raw.exists()
    # nothing to fetch: download leaves COMPLETE alone
    assert corpus.verify(level="hash").ok and manifest.is_complete()
    assert corpus.download(n_jobs=1).ok and manifest.is_complete()


def test_verify_finds_missing_and_corrupt_files(tmp_path, archive, monkeypatch):
    corpus, boto = _corpus(tmp_path, archive, ids=("ds1",), monkeypatch=monkeypatch)
    assert corpus.download(n_jobs=1).ok and corpus.verify().ok
    manifest = _Manifest(tmp_path / "ds1")
    raw1 = tmp_path / "ds1/sub-01/eeg/sub-01_task-rest_eeg.edf"
    raw2 = tmp_path / "ds1/sub-02/eeg/sub-02_task-rest_eeg.edf"
    raw1.write_bytes(DATA[:-1] + b"!")  # right size, wrong bytes
    raw2.write_bytes(DATA)  # wrong size (21 expected)
    report = corpus.verify(level="size")
    entries = report.datasets[0].entries
    assert entries["sub-01/eeg/sub-01_task-rest_eeg.edf"].state == "done"
    assert entries["sub-02/eeg/sub-02_task-rest_eeg.edf"].state == "corrupt"
    assert (
        "expected 21 bytes, got 20"
        in entries["sub-02/eeg/sub-02_task-rest_eeg.edf"].message
    )
    assert not report.ok and not manifest.is_complete()
    report = corpus.verify(level="hash")
    assert (
        report.datasets[0].entries["sub-01/eeg/sub-01_task-rest_eeg.edf"].state
        == "corrupt"
    )
    assert report.datasets[0].counts["corrupt"] == 2
    raw1.unlink()
    report = corpus.verify()
    assert (
        report.datasets[0].entries["sub-01/eeg/sub-01_task-rest_eeg.edf"].state
        == "missing"
    )
    assert report.datasets[0].counts["missing"] == 1
    # the next download repairs both, no catalogue change needed
    assert corpus.download(n_jobs=1).ok and corpus.verify(level="hash").ok


def test_status_offline_from_the_manifests(tmp_path, archive, monkeypatch):
    corpus, boto = _corpus(tmp_path, archive, monkeypatch=monkeypatch)
    report = corpus.status()
    assert [d.state for d in report.datasets] == [
        "unplanned",
        "unplanned",
    ] and not report.ok
    assert "no manifest" in report.problems[0]
    corpus.download(n_jobs=1, datasets="nm2")
    corpus.plan(datasets="ds1")
    queries = corpus._client.queries
    offline = EEGDashCorpus(
        ["ds1", "nm2"], cache_dir=tmp_path
    )  # no client: must not need one
    report = offline.status()
    assert corpus._client.queries == queries
    by_id = {d.dataset_id: d for d in report.datasets}
    assert by_id["nm2"].state == "fetched" and by_id["nm2"].counts["present"] == 3
    assert by_id["ds1"].state == "planned" and by_id["ds1"].counts["missing"] == 5
    assert by_id["nm2"].bytes_present == 20 + len(CHANNELS) + len(PARTICIPANTS)
    assert not report.ok


def test_status_online_reports_catalogue_drift(tmp_path, archive, monkeypatch):
    corpus, boto = _corpus(tmp_path, archive, ids=("ds1",), monkeypatch=monkeypatch)
    assert corpus.download(n_jobs=1).ok and corpus.verify().ok
    assert corpus.status(online=True).ok  # nothing changed: no drift
    # a re-curation: sub-02 renamed, a new subject added
    archive.records["ds1"] = [archive.records["ds1"][0]]
    archive.dataset("ds1b", ("09",))
    moved = archive.records["ds1b"][0]
    moved["dataset"] = "ds1"
    archive.records["ds1"].append(moved)
    report = corpus.status(online=True)
    ds1 = report.datasets[0]
    assert ds1.drift["added"] == [
        "sub-09/eeg/sub-09_task-rest_channels.tsv",
        "sub-09/eeg/sub-09_task-rest_eeg.edf",
    ]
    assert ds1.drift["removed"] == [
        "sub-02/eeg/sub-02_task-rest_channels.tsv",
        "sub-02/eeg/sub-02_task-rest_eeg.edf",
    ]
    assert "catalogue drift: 2 added, 2 removed" in ds1.problems
    assert ds1.state == "partial" and not report.ok  # COMPLETE withdrawn
    assert ds1.entries["sub-01/eeg/sub-01_task-rest_eeg.edf"].state == "done"


# ---------------------------------------------------------------- holding and CLI


def test_datasets_and_records_offline(tmp_path, archive, monkeypatch):
    corpus, boto = _corpus(tmp_path, archive, ids=("ds1",), monkeypatch=monkeypatch)
    assert corpus.download(n_jobs=1).ok
    datasets = corpus.datasets
    assert list(datasets) == ["ds1"] and datasets["ds1"].download is False
    assert sorted(r.record["bids_relpath"] for r in corpus.records()) == [
        "sub-01/eeg/sub-01_task-rest_eeg.edf",
        "sub-02/eeg/sub-02_task-rest_eeg.edf",
    ]
    assert corpus.datasets is datasets


def test_cli(tmp_path, archive, monkeypatch, capsys):
    client, boto = archive.s3()
    monkeypatch.setattr(downloader, "get_s3_filesystem", lambda **_: client)
    monkeypatch.setattr(
        EEGDashCorpus, "client", property(lambda self: archive.catalogue())
    )
    cache = str(tmp_path / "cache")
    assert corpus_mod.main(["status", "ds1,nm2", "--cache-dir", cache]) == 1
    assert "unplanned" in capsys.readouterr().out
    assert corpus_mod.main(["plan", "ds1,nm2", "--cache-dir", cache]) == 0
    out = tmp_path / "report.json"
    assert (
        corpus_mod.main(
            [
                "download",
                "ds1,nm2",
                "--cache-dir",
                cache,
                "--n-jobs",
                "2",
                "--json",
                str(out),
            ]
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
    assert (tmp_path / "cache/.eegdash/corpus-ids.json").exists()


def test_retired_openneuro_id_follows_its_nemar_twin(tmp_path, archive, monkeypatch):
    """ds003810 is served as on003810 (eegdash #404): files and manifest go where EEGDashDataset looks."""
    archive.dataset("on000003", ("04",), size_bytes=30, nemar=True)
    archive.docs["ds000003"] = archive.docs["on000003"]
    archive.records["ds000003"] = archive.records["on000003"]
    corpus, boto = _corpus(
        tmp_path, archive, ids=("ds000003",), monkeypatch=monkeypatch
    )
    report = corpus.download(n_jobs=1)
    assert report.ok, report.problems
    row = report.datasets[0]
    assert row.dataset_id == "ds000003" and row.served_as == "on000003"
    assert row.row()["served_as"] == "on000003"
    twin = tmp_path / "on000003"
    assert (twin / "sub-04/eeg/sub-04_task-rest_eeg.edf").read_bytes() == DATA
    assert (twin / "participants.tsv").read_bytes() == PARTICIPANTS
    assert not (tmp_path / "ds000003").exists()
    header, _ = _Manifest(twin).read()
    assert header["dataset"] == "on000003" and header["requested"] == "ds000003"
    offline = EEGDashCorpus(["ds000003"], cache_dir=tmp_path)  # no client
    assert offline.verify(level="hash").ok and _Manifest(twin).is_complete()
    status = offline.status().datasets[0]
    assert status.state == "complete" and status.served_as == "on000003"
    assert offline._folder("ds000003") == "on000003" and offline._folder("ds1") == "ds1"


def test_a_keyless_companion_the_archive_lacks_is_absent_not_failed(
    tmp_path, archive, monkeypatch
):
    """on002680 lists an .fdt the archive never had (2026-10-27): reported, the dataset still completes."""
    rec = archive.records["ds1"][0]
    ghost = "sub-01/eeg/sub-01_task-rest_eeg.fdt"
    rec["storage"]["dep_keys"].append(ghost)  # no annex key, no object
    corpus, boto = _corpus(tmp_path, archive, ids=("ds1",), monkeypatch=monkeypatch)
    report = corpus.download(n_jobs=2)
    ds = report.datasets[0]
    assert report.ok, report.problems
    assert (
        ds.entries[ghost].state == "absent" and ds.entries[ghost].error == "catalogue"
    )
    assert (
        ds.counts["absent"] == 1
        and ds.counts["failed"] == 0
        and ds.row()["absent"] == 1
    )
    assert "1 listed but not on the archive" in report.summary()
    verified = corpus.verify(level="hash", n_jobs=4)
    assert verified.ok and _Manifest(tmp_path / "ds1").is_complete()
    assert (
        EEGDashCorpus(["ds1"], cache_dir=tmp_path).status().datasets[0].state
        == "complete"
    )


def test_a_keyed_file_the_archive_lacks_still_fails(tmp_path, archive, monkeypatch):
    """A recording carries a key: if the archive does not serve it, the dataset is not whole."""
    rec = archive.records["ds1"][0]
    del archive.objects[f"openneuro.org/ds1/{rec['bids_relpath']}"]
    corpus, boto = _corpus(tmp_path, archive, ids=("ds1",), monkeypatch=monkeypatch)
    report = corpus.download(n_jobs=1)
    e = report.datasets[0].entries[rec["bids_relpath"]]
    assert (
        not report.ok
        and e.state == "failed"
        and report.datasets[0].counts["absent"] == 0
    )
