"""The committed data copy and the pages: hashes match the manifest, row counts are the corpus's, the pages describe one
release (v0) on two pages under a landing page, every page key and substitution exists, nothing names an unreleased store.
"""

from __future__ import annotations

import collections
import json
import re
import subprocess
from pathlib import Path

import pytest
from plot_corpus import sync_data
from plot_corpus.data import (
    BLOCKED,
    CONVERTED,
    EXCLUDED,
    GATED,
    RECORDING_COLUMNS,
    STATUSES,
    TABLE,
    kpis,
    numbers,
    read_buckets,
    read_electrodes,
    read_optional,
    read_rows,
    read_table,
    read_unplaced,
    reason,
    v0_rows,
)
from plot_corpus.fragments import build_fragments

from .conftest import DOCS, REPO_ROOT, source_rows, table_rows, write_csv

DATA = DOCS / "plot_corpus" / "data"
RST = DOCS / "source" / "corpus"
PAGES = ["index", "introduction", "v0"]
HF_URL = "https://huggingface.co/datasets/EEGDash/eegdash-corpus-v0"
CODE_URL = "https://github.com/eegdash/eegdash-corpus"
TABLE_LINK = ":download:`eegdash-corpus-v0.csv </_static/corpus_generated/eegdash-corpus-v0.csv>`"
# The ten paradigm groups of v0, as atlas.csv's paradigm column reads for its rows.
PARADIGMS = {
    "perception & emotion": 20,
    "BCI (MI & P300)": 15,
    "development (rest & tasks)": 9,
    "resting & consciousness": 9,
    "learning & reward": 9,
    "auditory & language": 6,
    "memory & cognition": 5,
    "SSVEP": 2,
    "clinical routine EEG": 1,
    "seizure": 1,
}
# A mention of a store that is not released (``v1.0.2`` versions and a ``(V3)`` agreement do not match).
DATE = r"20\d\d-\d\d-\d\d"  # a date of the work
V1 = re.compile(
    r"\bv1\b(?![.\d])|eegdash-corpus-v1|in_v1|development \(HBN\)|reve-public-nc"
)


def committed():
    """The committed data copy, read as the build reads it."""
    rows = read_rows(DATA / "atlas.csv")
    buckets = read_buckets(DATA / "eegdash-corpus-v0-channels.csv")
    electrodes = read_electrodes(DATA / "electrodes.json")
    unplaced = read_unplaced(DATA / "electrodes.json")
    recordings = read_optional(
        DATA / "eegdash-corpus-v0-recordings.csv", RECORDING_COLUMNS
    )
    return rows, buckets, electrodes, unplaced, recordings


def test_committed_data_matches_manifest():
    assert sync_data.check(DATA) == []


def test_committed_row_counts():
    rows, buckets, _, _, recordings = committed()
    assert (len(rows), len(v0_rows(rows)), len(buckets)) == (93, 77, 632)
    # every row counts: the union of REVE's paper (89) and its release card (70, 4 not in the paper)
    k = kpis(rows)
    assert (k["datasets"], k["paper"], k["card"], k["card_only"]) == (93, 89, 70, 4)
    # converted and released, blocked by the licence, gated by an access agreement, excluded
    assert {s: k[s] for s in STATUSES} == {
        CONVERTED: 77,
        BLOCKED: 8,
        GATED: 4,
        EXCLUDED: 4,
    }
    assert (k["out"], k["blocked_nd"]) == (12, 6)
    assert sorted(r["dataset_id"] for r in rows if r["status"] == GATED) == [
        "hms",
        "icare",
        "tdbrain",
        "tuh-eeg",
    ]
    assert all(
        reason(r) for r in rows if r["status"] != CONVERTED
    )  # every dataset out of the corpus says why
    assert {b["dataset"] for b in buckets} == {r["dataset_id"] for r in v0_rows(rows)}
    assert (
        sum(b["dataset"] == "ds003004" for b in buckets) == 34
    )  # PEERS's BioSemi-256 sets
    assert len(recordings) == 77 and all(int(r["tokens"]) > 0 for r in recordings)
    nums = numbers(rows, buckets, committed()[2], "x", recordings)
    assert nums["v0_tokens"] == "6.21 B"
    # NMT, Shin2017A and Weibo2014, recorded at 200 Hz: the converter caps their band at 0.45 x 200 Hz
    assert (nums["v0_capped"], nums["v0_capped_rates"], nums["v0_capped_band"]) == (
        "three",
        "200",
        "0.5–90",
    )
    assert (nums["v0_tag"], nums["v0_files"], nums["v0_size"]) == (
        "v1.0",
        "44,801",
        "3.60 TB",
    )


def test_committed_corpus_table_is_v0():
    """The downloadable corpus table: v0's 77 datasets (56 OpenNeuro, 21 NEMAR), no internal note, every OpenNeuro
    id paired with the id EEGDash serves it under.
    """
    table = read_table(DATA / TABLE)
    rows = committed()[0]
    assert len(table) == 77
    assert {r["dataset_id"] for r in table} == {r["dataset_id"] for r in v0_rows(rows)}
    assert collections.Counter(r["dataset_id"][:2] for r in table) == {
        "ds": 56,
        "nm": 21,
    }
    assert all(r["eegdash_id"][2:] == r["dataset_id"][2:] and r["eegdash_id"][:2] in ("ds", "on", "nm")
               for r in table)  # fmt: skip
    assert "note" not in table[0] and list(table[0])[0] == "dataset_id"


def test_the_ten_paradigms_of_v0():
    """v0's paradigms are exactly the ten groups of the data; an unknown or renamed one fails here first."""
    rows, buckets, electrodes, unplaced, recordings = committed()
    v0 = v0_rows(rows)
    assert dict(collections.Counter(r["paradigm"] for r in v0)) == PARADIGMS
    frags = build_fragments(
        rows, buckets, electrodes, "x", recordings, unplaced=unplaced
    )
    chart = frags["paradigms"][0]
    shown = re.findall(r'text-anchor="end">([^<]+)</text>', chart)
    assert sorted(s.replace("&amp;", "&") for s in shown) == sorted(PARADIGMS)
    key = re.search(
        r'<ul class="corpus-key"[^>]*>(.*?)</ul>', frags["tokens"][0]
    ).group(1)
    assert sorted(re.findall(r"<b>([^<]+)</b>", key.replace("&amp;", "&"))) == sorted(
        PARADIGMS
    )
    assert frags["paradigms"][1].startswith("10 paradigms;")


def test_committed_electrodes_cover_the_buckets():
    """Every channel of the store is drawn or listed as placed from its dataset's own coordinates (the 12 names the
    bank lacks, ds002680's and the CB1/CB2 it shares with ds003690); the BioSemi-256 template names render.
    """
    from plot_corpus.electrodes import check_coverage, unplaced_note

    _, buckets, els, unplaced, _ = committed()
    check_coverage(els, buckets, unplaced)
    assert len(unplaced) == 12 and {d for u in unplaced for d in u["datasets"]} == {
        "ds002680",
        "ds003690",
    }
    assert unplaced_note(unplaced).startswith(
        " 12 electrodes of two datasets were placed"
    )
    assert sum(n.startswith("biosemi256_") for e in els for n in e["names"]) == 256
    assert all(set(e) == {"names", "xyz", "hours", "datasets", "buckets"} for e in els)


def test_nothing_names_an_unreleased_store():
    """One release: no v1 in the data files, their names and columns, the RST pages, the fragments or the numbers."""
    assert not [p.name for p in DATA.iterdir() if "v1" in p.name]
    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    assert sorted(manifest["files"]) == sorted(
        ["atlas.csv", "eegdash-corpus-v0-channels.csv", "eegdash-corpus-v0-recordings.csv",
         "eegdash-corpus-demographics.csv", "eegdash-corpus-v0.csv", "electrodes.json"]
    )  # fmt: skip
    texts = {p.name: p.read_text() for p in DATA.iterdir() if p.is_file()}
    texts |= {f"rst/{p.name}": p.read_text() for p in RST.glob("*.rst")}
    rows, buckets, electrodes, unplaced, recordings = committed()
    frags = build_fragments(
        rows, buckets, electrodes, "x", recordings, unplaced=unplaced
    )
    texts |= {f"fragment/{k}": body + cap for k, (body, cap) in frags.items()}
    texts["numbers"] = json.dumps(numbers(rows, buckets, electrodes, "x", recordings))
    hits = [
        f"{name}: {m.group(0)!r}" for name, t in texts.items() for m in V1.finditer(t)
    ]
    assert hits == []


def test_no_date_or_note_is_published():
    """The published data, the fragments and the numbers carry no date of the work and no internal note; the one date
    is the manifest's sync date (``synced``, the pages' "as of"), exempt by name.
    """
    texts = {
        p.name: p.read_text()
        for p in DATA.iterdir()
        if p.is_file() and p.name != "MANIFEST.json"
    }
    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    texts["MANIFEST.json"] = json.dumps(
        {k: v for k, v in manifest.items() if k != "synced"}
    )
    rows, buckets, electrodes, unplaced, recordings = committed()
    frags = build_fragments(
        rows, buckets, electrodes, "x", recordings, unplaced=unplaced
    )
    texts |= {f"fragment/{k}": body + cap for k, (body, cap) in frags.items()}
    nums = numbers(rows, buckets, electrodes, "synced-date", recordings)
    texts["numbers"] = json.dumps(nums)
    assert [
        f"{n}: {m.group(0)}" for n, t in texts.items() for m in re.finditer(DATE, t)
    ] == []
    header = (DATA / "atlas.csv").read_text().splitlines()[0].split(",")
    assert "note" not in header and "in_v1" not in header and "reason" in header
    assert "Note" not in frags["table-all"][0] and "Note" not in frags["table-v0"][0]


def test_reason_on_exactly_the_sixteen_rows_out_of_the_corpus():
    rows = committed()[0]
    with_reason = [r["dataset_id"] for r in rows if r["reason"]]
    assert len(with_reason) == 16 and all(
        r["status"] != CONVERTED for r in rows if r["reason"]
    )
    assert all(r["reason"] for r in rows if r["status"] != CONVERTED)
    assert collections.Counter(r["reason"] for r in rows if r["reason"]) == {
        "No Derivatives licence (CC-BY-NC-ND-4.0)": 5,
        "No Derivatives licence (CC-BY-ND-4.0)": 1,
        "no licence stated": 2,
        "gated (application): redistribution forbidden": 2,
        "gated (registration): redistribution forbidden": 1,
        "gated (competition): redistribution forbidden": 1,
        "held out for evaluation (an OpenEEGBench test task)": 1,
        "an fMRI accession with no EEG": 1,
        "recorded at 100 Hz, below REVE's 200 Hz input": 1,
        "a release-card entry that names no dataset": 1,
    }


def test_two_pages_and_their_links():
    """Introduction and EEGDash Corpus v0 under the landing page; no page links a removed one (Coverage, whose table
    moved into the Introduction).
    """
    assert sorted(p.stem for p in RST.glob("*.rst")) == sorted(PAGES)
    index = (RST / "index.rst").read_text()
    toc = re.search(r"\.\. toctree::\n(?:\s+:\w+:.*\n)*\n((?:\s+.+\n?)+)", index).group(
        1
    )
    assert [line.strip() for line in toc.strip().splitlines()] == [
        "Introduction <introduction>", "EEGDash Corpus v0 <v0>"]  # fmt: skip
    assert re.findall(r":link: (\w+)", index) == ["introduction", "v0"]
    assert ".. grid:: 1 1 2 2" in index
    sources = list((DOCS / "source").rglob("*.rst")) + [DOCS / "source" / "conf.py"]
    stale = [str(p) for p in sources if re.search(r"corpus/(v1|progress|coverage)\b|<(progress|coverage)>|"
                                                     r"corpus-(progress|coverage)|corpus-v1",
                                                     p.read_text(encoding="utf-8"))]  # fmt: skip
    assert stale == []


def test_every_rst_key_and_substitution_exists():
    rows, buckets, electrodes, unplaced, recordings = committed()
    frags = build_fragments(
        rows, buckets, electrodes, "x", recordings, unplaced=unplaced
    )
    used = {
        k
        for p in RST.glob("*.rst")
        for k in re.findall(r"^\s*\.\. corpus-figure:: (\S+)", p.read_text(), re.M)
    }
    assert used == set(frags), (
        sorted(used - set(frags)),
        sorted(set(frags) - used),
    )  # every view is on a page
    keys = re.findall(
        r"^\s*\.\. corpus-figure:: (\S+)", (RST / "v0.rst").read_text(), re.M
    )
    assert keys.index("scatter") < keys.index("lengths") < keys.index("table-v0")
    intro = re.findall(
        r"^\.\. corpus-figure:: (\S+)", (RST / "introduction.rst").read_text(), re.M
    )
    assert intro[-2:] == [
        "licences",
        "table-all",
    ]  # the table of every dataset follows the licences
    assert not {"status", "flow"} & set(frags)
    # no page links the removed Coverage page ("electrode coverage" on the v0 page is prose, not a link)
    links = (
        r"<coverage>|:link: coverage|:doc:`[^`]*coverage|corpus-coverage|coverage\.rst"
    )
    assert [p.name for p in RST.glob("*.rst") if re.search(links, p.read_text())] == []
    known = set(numbers(rows, buckets, electrodes, "x", recordings)) | {"hf_corpus"}
    # every file a page offers with :download: from the generated folder is one the build writes
    targets = {t for p in RST.glob("*.rst")
               for t in re.findall(r":download:`[^`<]*</_static/corpus_generated/([^>]+)>`", p.read_text())}  # fmt: skip
    assert targets == {TABLE}
    subs = {
        s for p in RST.glob("*.rst") for s in re.findall(r"\|(\w+)\|", p.read_text())
    }
    assert subs <= known, sorted(subs - known)
    assert {
        "universe",
        "excluded_prose",
        "montage_rows",
        "not_redistributable",
        "v0_licences",
    } <= subs


def test_download_section():
    """The Hugging Face link is set once (a substitution) and the section carries the corpus table's download, the
    store's commands, the loading example, the integrity check, the code repository and the raw download with the
    corpus object; the introduction carries the three ways in short.
    """
    sources = {
        p.name: p.read_text()
        for p in RST.rglob("*")
        if p.suffix in (".rst", ".inc", ".txt")
    }
    assert sum(t.count(HF_URL) for t in sources.values()) == 1
    assert sum(t.count("PLACEHOLDER") for t in sources.values()) == 0
    v0 = sources["v0.rst"]
    assert re.search(r"\n\.\. \|hf_corpus\| replace:: .*" + re.escape(HF_URL), v0)
    section = v0.split("\nDownload\n========\n")[1].split(
        "\nThe |v0_datasets| datasets\n"
    )[0]
    assert (
        "|hf_corpus|" in section
        and "hf auth login" in section
        and "LICENSES.csv" in section
    )
    for line in (
        "hf download EEGDash/eegdash-corpus-v0 --repo-type dataset --local-dir eegdash-corpus-v0          # everything, 3.6 TB",
        '--include "eegdash-corpus-v0.zarr/zarr.json" "eegdash-corpus-v0.zarr/0/**"',
        'root = zarr.open_group("eegdash-corpus-v0/eegdash-corpus-v0.zarr", mode="r")',
        "z = np.clip((x - mean) / sd, -15, 15)",
        'without_nc_gpl = {"CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0"}',
        "sha256sum -c eegdash-corpus-v0-SHA256SUMS",
        TABLE_LINK,
        CODE_URL,
        "python -m eegdash.corpus plan eegdash-corpus-v0.csv --cache-dir eegdash-cache",
        "--n-jobs 16 --n-datasets 4 --time-budget 21600",
        "python -m eegdash.corpus verify eegdash-corpus-v0.csv --cache-dir eegdash-cache --level hash",
        "python -m eegdash.corpus status eegdash-corpus-v0.csv --cache-dir eegdash-cache",
        'corpus = EEGDashCorpus("eegdash-corpus-v0", cache_dir="eegdash-cache")',
        "|v0_tag|",
    ):
        assert line in section, line
    assert section.index("\nThe converted store\n---") < section.index(
        "\nThe raw datasets\n---"
    )
    intro = (
        sources["introduction.rst"]
        .split("\nHow to get it\n=============\n")[1]
        .split("\n=====")[0]
    )
    for line in (
        "hf download EEGDash/eegdash-corpus-v0 --repo-type dataset --local-dir eegdash-corpus-v0",
        'root = zarr.open_group("eegdash-corpus-v0/eegdash-corpus-v0.zarr", mode="r")',
        'from eegdash import EEGDashCorpus',
        'corpus = EEGDashCorpus("eegdash-corpus-v0", cache_dir="eegdash-cache")',
        'EEGDashDataset(dataset="nm000138", cache_dir="eegdash-cache")',
        ":ref:`corpus-download`",
    ):
        assert line in intro, line
    assert CODE_URL in sources["introduction.rst"]
    assert (
        ".. _corpus-download:" in v0
        and ":ref:`corpus-download`" in sources["introduction.rst"]
    )


def test_scan_catches_internal_dated_and_unreleased_text():
    patterns = sync_data.DENY + sync_data.PUBLIC
    hits = sync_data.scan(
        {
            "a.csv": "ok\n$PROJECT/raw\n/home/someone/x\nme@example.org\na (V3) agreement, year 2021"
        },
        patterns,
    )
    assert len(hits) == 3
    hits = sync_data.scan(
        {
            "a.csv": "in eegdash-corpus-v1\nreve-public-nc\nfound 2026-10-09\ndevelopment (HBN)\nNEMAR v1.0.2"
        },
        patterns,
    )
    assert len(hits) == 5


def test_public_table_drops_the_note(rows):
    """The published corpus table keeps the source's columns in order, without the dated ``note``."""
    import csv
    import io

    src = io.StringIO()
    w = csv.DictWriter(src, fieldnames=list(table_rows(rows)[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(table_rows(rows))
    text = sync_data.public_table(src.getvalue())
    out = list(csv.DictReader(io.StringIO(text)))
    assert list(out[0]) == [c for c in table_rows(rows)[0] if c != "note"]
    assert [r["dataset_id"] for r in out] == ["ds000001", "ds000002"] and not re.search(
        DATE, text
    )


def test_public_atlas_derives_reason_and_drops_note(rows):
    """The published atlas.csv has a derived ``reason`` in place of the internal ``note``, and no ``in_v1``."""
    import csv
    import io

    from .conftest import source_rows

    src = io.StringIO()
    w = csv.DictWriter(src, fieldnames=list(source_rows(rows)[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(source_rows(rows))
    text = sync_data.public_atlas(src.getvalue())
    out = list(csv.DictReader(io.StringIO(text)))
    assert (
        "note" not in out[0]
        and "in_v1" not in out[0]
        and list(out[0])[-2:] == ["reason", "link"]
    )
    assert [r["reason"] for r in out] == [r["reason"] for r in rows]
    assert not re.search(DATE, text)
    gated = dict(out[4], note="gated (competition): rules forbid it")
    assert (
        sync_data.derive_reason(gated)
        == "gated (competition): redistribution forbidden"
    )
    for note, words in (
        (
            "excluded from the corpus: OpenEEGBench test task (PhysioNet MI)",
            "held out for evaluation (an OpenEEGBench test task)",
        ),
        ("excluded from the corpus: fMRI, no EEG", "an fMRI accession with no EEG"),
        (
            "excluded from the corpus: names no dataset",
            "a release-card entry that names no dataset",
        ),
    ):
        assert sync_data.derive_reason(dict(out[5], note=note)) == words  # fmt: skip
    assert (
        sync_data.derive_reason(dict(out[2], licence="CC-BY-ND-4.0"))
        == "No Derivatives licence (CC-BY-ND-4.0)"
    )
    assert (
        sync_data.derive_reason(dict(out[2], licence="Custom"))
        == "licence Custom forbids a converted copy"
    )


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def source(tmp_path, rows, buckets, bank):
    """A throwaway source repository laid out as the real one."""
    src = tmp_path / "src"
    (src / "corpus/corpus-atlas").mkdir(parents=True)
    srows = source_rows(rows)
    write_csv(src / "corpus/corpus-atlas/atlas.csv", srows, list(srows[0]))
    write_csv(
        src / "corpus/corpus-atlas/eegdash-corpus-v0-channels.csv",
        buckets,
        list(buckets[0]),
    )
    (src / "corpus/corpus-atlas/electrode-positions.json").write_text(json.dumps(bank))
    write_csv(
        src / "corpus/eegdash-corpus-v0.csv",
        table_rows(rows),
        list(table_rows(rows)[0]),
    )
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
        m["files"]["atlas.csv"]["rows"] == 6
        and m["files"]["electrodes.json"]["rows"] == 5
    )
    assert m["files"][TABLE] == {"source": "corpus/eegdash-corpus-v0.csv", "rows": 2,
                                 "sha256": sync_data.sha256(out / TABLE)}  # fmt: skip
    assert (
        "note" not in (out / TABLE).read_text().splitlines()[0]
    )  # the dated note is dropped, the scan passes
    assert sync_data.check(out) == []
    atlas = source / "corpus/corpus-atlas/atlas.csv"
    atlas.write_text(atlas.read_text().replace("Dataset ds000001", "Dataset by Tester"))
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
    atlas.write_text(
        atlas.read_text().replace("Dataset by Tester", "Dataset in reve-public-nc")
    )
    with pytest.raises(SystemExit, match="reve-public-nc"):
        sync_data.sync(
            source, out, allow_dirty=True, synced="2026-01-02", extra_deny=[]
        )


def test_sync_copies_optional_tables_when_present(source, tmp_path, recordings):
    write_csv(
        source / "corpus/corpus-atlas/eegdash-corpus-v0-recordings.csv",
        recordings,
        RECORDING_COLUMNS,
    )
    _git(source, "add", ".")
    _git(
        source,
        "-c",
        "user.name=Ada Tester",
        "-c",
        "user.email=t@t",
        "commit",
        "-qm",
        "y",
    )
    out = tmp_path / "data"
    m = sync_data.sync(
        source, out, allow_dirty=False, synced="2026-01-02", extra_deny=[]
    )
    assert m["files"]["eegdash-corpus-v0-recordings.csv"]["rows"] == 2
    assert (
        "eegdash-corpus-demographics.csv" not in m["files"]
    )  # absent in the source: skipped
    assert sync_data.check(out) == []


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


def test_3d_script_on_the_v0_page_only():
    text = (DOCS / "source/_extensions/corpus_figure.py").read_text()
    assert 'if pagename == "corpus/v0":' in text and "corpus/v1" not in text
    assert REPO_ROOT.is_dir()
