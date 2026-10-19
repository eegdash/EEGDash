"""Copy the EEGDash Corpus tables from their source of truth into ``plot_corpus/data/``.

Usage::

    python docs/plot_corpus/sync_data.py --from ../replicating-reve            # copy, scan, write MANIFEST.json
    python docs/plot_corpus/sync_data.py --from ../replicating-reve --allow-dirty
    python docs/plot_corpus/sync_data.py --check                              # re-hash against MANIFEST.json

The source repository holds the corpus tables and the scripts that derive
``corpus/corpus-atlas/atlas.csv`` (one row per dataset) and the converted
stores' channel sets (one row per bucket: v0's, and the v1 store's that v0
does not hold), and the per-dataset recording, token and demographic tables
when it carries them. This script copies those files and derives
``electrodes.json`` (one entry per electrode position used by the buckets, and
the channel names the bank lacks) from the converter's position bank. Every cell is scanned for
cluster names, internal paths, e-mail addresses and the source repository's
author names; a hit stops the sync. A dirty source tree is refused unless
``--allow-dirty`` (the manifest then records ``"dirty": true``). No network.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from plot_corpus.data import (  # noqa: E402
    DEMOGRAPHIC_COLUMNS,
    RECORDING_COLUMNS,
    read_buckets,
    read_optional,
    read_rows,
)
from plot_corpus.electrodes import aggregate_positions, unplaced_names  # noqa: E402

SOURCES = {  # data file -> path in the source repository
    "atlas.csv": "corpus/corpus-atlas/atlas.csv",
    "eegdash-corpus-v0-channels.csv": "corpus/corpus-atlas/eegdash-corpus-v0-channels.csv",
    "eegdash-corpus-v1-channels.csv": "corpus/corpus-atlas/eegdash-corpus-v1-channels.csv",
    "eegdash-corpus-v0-recordings.csv": "corpus/corpus-atlas/eegdash-corpus-v0-recordings.csv",
    "eegdash-corpus-v1-recordings.csv": "corpus/corpus-atlas/eegdash-corpus-v1-recordings.csv",
    "eegdash-corpus-demographics.csv": "corpus/corpus-atlas/eegdash-corpus-demographics.csv",
}
# Tables the pages can do without: copied when the source carries them, skipped (and absent from the manifest)
# otherwise; the chart that reads one says so on the page.
OPTIONAL = {
    "eegdash-corpus-v0-recordings.csv",
    "eegdash-corpus-v1-recordings.csv",
    "eegdash-corpus-demographics.csv",
}
BANK = "corpus/corpus-atlas/electrode-positions.json"  # the converter's position bank (REVE's public one and MNE's BioSemi templates), {name: [x, y, z]} in metres
DERIVED = "electrodes.json"
FILES = [*SOURCES, DERIVED]
# Words and paths that must never reach the public site.
DENY = [
    r"\b(expanse|delta|deltaai|sdsc|ncsa|sdp190|sdp207|csd403|bbnv|bixu)\b",
    r"/expanse/", r"/work/hdd", r"/Users/", r"/home/", r"\$PROJECT", r"\$OUTROOT",
    r"[\w.+-]+@[\w-]+\.[\w.-]+",
]  # fmt: skip


def git(src: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(src), *args], check=True, capture_output=True, text=True
    ).stdout


def author_patterns(src: Path) -> list[str]:
    """The source repository's author names (each word of three letters or more), so no name is typed here."""
    try:
        names = set(git(src, "log", "--format=%an").split("\n"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    words = {w for n in names for w in re.split(r"[\s.]+", n) if len(w) >= 3}
    return [rf"\b{re.escape(w)}\b" for w in sorted(words)]


def scan(texts: dict[str, str], patterns: list[str]) -> list[str]:
    """Every (file, line, match) hit of a deny pattern, case-insensitive."""
    rx = re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)
    hits = []
    for name, text in texts.items():
        for k, line in enumerate(text.splitlines(), 1):
            for m in rx.finditer(line):
                hits.append(f"{name}:{k}: {m.group(0)!r}")
    return hits


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_of(path: Path) -> int:
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as fh:
            return sum(1 for _ in csv.DictReader(fh))
    return len(json.loads(path.read_text(encoding="utf-8"))["electrodes"])


def derive_electrodes(buckets_csv: Path, added_csv: Path, bank_json: Path) -> dict:
    bank = json.loads(bank_json.read_text(encoding="utf-8"))
    buckets, added = read_buckets(buckets_csv), read_buckets(added_csv)
    return {
        "about": "Electrode positions used by the v0 Zarr store's channel sets (hours, datasets, buckets) and by the "
        "channel sets the v1 store adds to v0 (added_*), derived from the converter's position bank (REVE's public "
        "bank and MNE's BioSemi-160/256 templates; coordinates in metres, head-centred, x right, y anterior, z up, "
        "rounded to 1e-4 m); case-fold spellings sharing a coordinate are merged. 'unplaced' lists the channel names "
        "the bank lacks, which the converter placed from their dataset's own coordinates (not drawn). Generated by "
        "docs/plot_corpus/sync_data.py, do not hand-edit.",
        "electrodes": aggregate_positions(buckets, bank, added),
        "unplaced": unplaced_names(buckets, bank, added),
    }


def dump_electrodes(derived: dict) -> str:
    """One electrode per line, so a resync diffs per position."""
    lines = ",\n".join(
        "  " + json.dumps(e, ensure_ascii=False) for e in derived["electrodes"]
    )
    unplaced = ",\n".join(
        "  " + json.dumps(u, ensure_ascii=False) for u in derived.get("unplaced", [])
    )
    return (f'{{\n "about": {json.dumps(derived["about"])},\n "electrodes": [\n{lines}\n ],\n'
            f' "unplaced": [\n{unplaced}\n ]\n}}\n')  # fmt: skip


def sync(
    src: Path, out: Path, allow_dirty: bool, synced: str, extra_deny: list[str]
) -> dict:
    src = src.resolve()
    dirty = bool(git(src, "status", "--porcelain").strip())
    if dirty and not allow_dirty:
        raise SystemExit(
            f"{src} has uncommitted changes: commit them or pass --allow-dirty"
        )
    commit = git(src, "rev-parse", "HEAD").strip()
    texts = {
        name: (src / rel).read_text(encoding="utf-8")
        for name, rel in SOURCES.items()
        if name not in OPTIONAL or (src / rel).is_file()
    }
    derived = derive_electrodes(
        src / SOURCES["eegdash-corpus-v0-channels.csv"],
        src / SOURCES["eegdash-corpus-v1-channels.csv"],
        src / BANK,
    )
    texts[DERIVED] = dump_electrodes(derived)
    hits = scan(texts, DENY + author_patterns(src) + extra_deny)
    if hits:
        raise SystemExit(
            "refused, internal names or paths in the data:\n  " + "\n  ".join(hits)
        )
    out.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (out / name).write_text(text, encoding="utf-8")
    read_rows(
        out / "atlas.csv"
    )  # refuses an unknown status before the manifest is written
    for name in (
        "eegdash-corpus-v0-recordings.csv",
        "eegdash-corpus-v1-recordings.csv",
    ):
        read_optional(
            out / name, RECORDING_COLUMNS
        )  # refuses a header that misses a column
    read_optional(out / "eegdash-corpus-demographics.csv", DEMOGRAPHIC_COLUMNS)
    manifest = {
        "about": "Generated by docs/plot_corpus/sync_data.py, do not hand-edit.",
        "source_commit": commit,
        "dirty": dirty,
        "synced": synced,
        "files": {name: {"source": SOURCES.get(name, f"derived from {BANK} and the channel sets"),
                         "sha256": sha256(out / name), "rows": rows_of(out / name)} for name in FILES
                  if name in texts},
    }  # fmt: skip
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def check(out: Path) -> list[str]:
    """Mismatches between the files and MANIFEST.json (empty = clean)."""
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    errs = []
    for name, meta in manifest["files"].items():
        p = out / name
        if not p.is_file():
            errs.append(f"{name}: missing")
        elif sha256(p) != meta["sha256"]:
            errs.append(
                f"{name}: sha256 differs from MANIFEST.json (hand-edited? resync)"
            )
        elif rows_of(p) != meta["rows"]:
            errs.append(f"{name}: {rows_of(p)} rows, MANIFEST.json says {meta['rows']}")
    return errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--from",
        dest="src",
        type=Path,
        help="the source repository (holds corpus/ and results/)",
    )
    ap.add_argument(
        "--out", type=Path, default=DATA, help="data folder (default plot_corpus/data)"
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="re-hash the data files against MANIFEST.json",
    )
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="sync from a source tree with uncommitted changes",
    )
    ap.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="sync date recorded in MANIFEST.json",
    )
    ap.add_argument(
        "--deny-file", type=Path, help="extra deny patterns (regex), one per line"
    )
    a = ap.parse_args(argv)
    if a.check:
        errs = check(a.out)
        print("\n".join(errs) if errs else f"{a.out}: clean")
        return 1 if errs else 0
    if not a.src:
        ap.error("--from is required unless --check")
    extra = (
        [ln.strip() for ln in a.deny_file.read_text().splitlines() if ln.strip()]
        if a.deny_file
        else []
    )
    m = sync(a.src, a.out, a.allow_dirty, a.date, extra)
    for name, meta in m["files"].items():
        print(f"{name}: {meta['rows']} rows, sha256 {meta['sha256'][:12]}")
    print(
        f"source {m['source_commit'][:12]}{' (dirty)' if m['dirty'] else ''}, synced {m['synced']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
