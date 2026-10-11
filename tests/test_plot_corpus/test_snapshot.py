"""The committed data copy: hashes match its manifest, row counts are the corpus's, every page key exists."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
from plot_corpus import sync_data
from plot_corpus.data import (
    numbers,
    read_buckets,
    read_electrodes,
    read_rows,
    v0_rows,
    v1_rows,
)
from plot_corpus.fragments import build_fragments

from .conftest import DOCS, write_csv

DATA = DOCS / "plot_corpus" / "data"
RST = DOCS / "source" / "corpus"


def test_committed_data_matches_manifest():
    assert sync_data.check(DATA) == []


def test_committed_row_counts():
    rows = read_rows(DATA / "atlas.csv")
    assert (len(rows), len(v0_rows(rows)), len(v1_rows(rows))) == (93, 66, 78)
    assert len(read_buckets(DATA / "reve-public-nc-channels.csv")) == 968
    assert (
        len({b["dataset"] for b in read_buckets(DATA / "eegdash-channel-sets.csv")})
        == 10
    )  # 11 to convert, SparrKULee has no channel metadata


def test_every_rst_key_exists():
    rows = read_rows(DATA / "atlas.csv")
    frags = build_fragments(
        rows,
        read_buckets(DATA / "reve-public-nc-channels.csv"),
        read_electrodes(DATA / "electrodes.json"),
        "2026-01-01",
        read_buckets(DATA / "eegdash-channel-sets.csv"),
    )
    used = {
        k
        for p in RST.glob("*.rst")
        for k in re.findall(r"^\s*\.\. corpus-figure:: (\S+)", p.read_text(), re.M)
    }
    assert used, "no corpus-figure directive found"
    assert used <= set(frags), sorted(used - set(frags))
    subs = {
        s for p in RST.glob("*.rst") for s in re.findall(r"\|(\w+)\|", p.read_text())
    }
    known = set(
        numbers(
            rows,
            read_buckets(DATA / "reve-public-nc-channels.csv"),
            read_electrodes(DATA / "electrodes.json"),
            "x",
            read_buckets(DATA / "eegdash-channel-sets.csv"),
        )
    )
    assert subs <= known, sorted(subs - known)


def test_scan_catches_internal_names():
    hits = sync_data.scan(
        {"a.csv": "ok\n$PROJECT/raw\n/home/someone/x\nme@example.org"}, sync_data.DENY
    )
    assert len(hits) == 3


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def source(tmp_path, rows, buckets, bank, added):
    """A throwaway source repository laid out as the real one."""
    src = tmp_path / "src"
    (src / "corpus/eeg-atlas").mkdir(parents=True)
    (src / "results/conversion/zarr").mkdir(parents=True)
    write_csv(src / "corpus/eeg-atlas/atlas.csv", rows, list(rows[0]))
    write_csv(
        src / "results/conversion/zarr/reve-public-nc-channels.csv",
        buckets,
        list(buckets[0]),
    )
    write_csv(
        src / "results/conversion/eegdash-channel-sets.csv", added, list(added[0])
    )
    (src / "corpus/reve-positions.json").write_text(json.dumps(bank))
    _git(src, "init", "-q")
    _git(src, "add", ".")
    _git(
        src, "-c", "user.name=Ada Tester", "-c", "user.email=t@t", "commit", "-qm", "x"
    )
    return src


def test_sync_copies_scans_and_refuses_dirty(source, tmp_path):
    out = tmp_path / "data"
    m = sync_data.sync(
        source, out, allow_dirty=False, synced="2026-01-02", extra_deny=[]
    )
    assert (
        m["files"]["atlas.csv"]["rows"] == 8
        and m["files"]["electrodes.json"]["rows"] == 6
    )
    assert sync_data.check(out) == []
    (source / "corpus/eeg-atlas/atlas.csv").write_text(
        (source / "corpus/eeg-atlas/atlas.csv")
        .read_text()
        .replace("Dataset ds000001", "Dataset by Tester")
    )
    with pytest.raises(SystemExit, match="uncommitted"):
        sync_data.sync(
            source, out, allow_dirty=False, synced="2026-01-02", extra_deny=[]
        )
    with pytest.raises(
        SystemExit, match="Tester"
    ):  # an author name of the source repository
        sync_data.sync(
            source, out, allow_dirty=True, synced="2026-01-02", extra_deny=[]
        )


def test_every_data_file_is_tracked_by_git():
    """The pages are built from docs/plot_corpus/data in CI: a file on disk that git ignores (the repository has
    blanket data/ and *.json rules) passes every other test locally and fails the build on GitHub.
    """
    try:
        tracked = subprocess.run(
            ["git", "-C", str(DATA), "ls-files", "--", "."],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("not a git checkout")
    on_disk = sorted(p.name for p in DATA.iterdir() if p.is_file())
    assert sorted(Path(t).name for t in tracked) == on_disk
