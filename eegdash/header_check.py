"""Consistency checks on the EDF/BDF channel headers of a BIDS dataset.

Only headers are read: from local files, or as byte ranges of NEMAR objects,
so a whole dataset is checked without downloading its signals. The checks:

- ``unit``: a channel's header unit differs across recordings;
- ``sidecar``: a header unit is another quantity than ``channels.tsv`` says;
- ``range``: a channel reads at an implausible scale for its quantity;
- ``prefix`` (minor): header and ``channels.tsv`` differ by an SI prefix only.

MNE scales samples by the header unit and ignores ``channels.tsv``, so the
first three flag channels that a reader gets at the wrong scale. Run::

    python -m eegdash.header_check nm000281 [--local ROOT] [--per-subject N]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

# Unit as written -> (SI unit, factor to SI).
UNITS = {
    "V": ("V", 1.0),
    "mV": ("V", 1e-3),
    "uV": ("V", 1e-6),
    "µV": ("V", 1e-6),
    "μV": ("V", 1e-6),
    "nV": ("V", 1e-9),
    "rad": ("rad", 1.0),
    "deg": ("rad", 0.017453292519943295),
}
# Largest plausible absolute physical value, in SI units.
MAX_ABS = {"V": 1.0, "rad": 100.0}
EXTENSIONS = (".bdf", ".edf")
_TREE_URL = "https://api.github.com/repos/NEMARDatasets/{}/git/trees/{}"


@dataclass(frozen=True)
class Channel:
    name: str
    unit: str
    physical_min: float
    physical_max: float


@dataclass(frozen=True)
class Finding:
    check: str
    channel: str
    message: str
    files: tuple[str, ...]


def header_size(main: bytes) -> int:
    """Bytes of the full header, from the 256-byte main header."""
    return 256 * (int(main[252:256]) + 1)


def parse_header(buf: bytes) -> list[Channel]:
    """Channels of an EDF/BDF header (main and signal headers)."""
    n = int(buf[252:256])

    def field(offset, width):
        start = 256 + offset * n
        return [
            buf[start + width * i : start + width * (i + 1)].decode("latin-1").strip()
            for i in range(n)
        ]

    # Signal header fields: label 16, transducer 80, dimension 8, then
    # physical min 8 and max 8.
    names, units = field(0, 16), field(96, 8)
    pmin, pmax = field(104, 8), field(112, 8)
    return [
        Channel(names[i], units[i], float(pmin[i]), float(pmax[i])) for i in range(n)
    ]


def sidecar_units(tsv: str) -> dict[str, str]:
    """``name -> units`` from a ``channels.tsv``."""
    rows = csv.DictReader(io.StringIO(tsv), delimiter="\t")
    return {row["name"]: row.get("units", "n/a") for row in rows}


def check(recordings: dict[str, tuple[list[Channel], dict[str, str]]]) -> list[Finding]:
    """Findings over ``path -> (header channels, channels.tsv units)``."""
    findings = []
    by_channel = defaultdict(lambda: defaultdict(list))
    for path, (channels, sidecar) in recordings.items():
        for ch in channels:
            by_channel[ch.name][ch.unit].append(path)
            expected = sidecar.get(ch.name)
            header_si = UNITS.get(ch.unit, (None, None))
            sidecar_si = UNITS.get(expected, (None, None))
            if expected not in (None, "n/a") and header_si != sidecar_si:
                kind = "prefix" if header_si[0] == sidecar_si[0] else "sidecar"
                findings.append(
                    Finding(
                        kind,
                        ch.name,
                        f"header unit {ch.unit!r}, channels.tsv {expected!r}",
                        (path,),
                    )
                )
            # What a reader gets (header scale) in the quantity the channel
            # is (sidecar unit, else the header's).
            _, factor = UNITS.get(ch.unit, (None, None))
            si, _ = UNITS.get(expected, UNITS.get(ch.unit, (None, None)))
            read = max(abs(ch.physical_min), abs(ch.physical_max)) * (factor or 0)
            if si in MAX_ABS and read > MAX_ABS[si]:
                findings.append(
                    Finding(
                        "range",
                        ch.name,
                        f"reads up to {read:g} {si}, "
                        f"implausible beyond {MAX_ABS[si]:g} {si}",
                        (path,),
                    )
                )
    for name, units in by_channel.items():
        if len(units) > 1:
            counts = ", ".join(f"{u!r} x{len(p)}" for u, p in units.items())
            files = tuple(p[0] for p in units.values())
            findings.append(Finding("unit", name, f"units differ: {counts}", files))
    return findings


def summarize(findings: list[Finding]) -> str:
    """One line per (check, message), with how many channels and files share it."""
    groups = defaultdict(lambda: [set(), Counter()])
    for f in findings:
        group = groups[(f.check, re.sub(r"\d[\d.]*(e[+-]?\d+)?", "#", f.message))]
        group[0].add(f.channel)
        group[1].update(f.files)
    lines = []
    for (name, message), (channels, files) in sorted(groups.items()):
        example = next(iter(files))
        lines.append(
            f"[{name}] {message}: {len(channels)} channels, "
            f"{len(files)} files (e.g. {example})"
        )
    return "\n".join(lines)


def _pick(paths: list[str], per_subject: int | None) -> list[str]:
    if per_subject is None:
        return paths
    kept, seen = [], Counter()
    for path in sorted(paths):
        subject = path.split("/", 1)[0]
        if seen[subject] < per_subject:
            kept.append(path)
            seen[subject] += 1
    return kept


def _sidecar_path(path: str) -> str:
    return re.sub(r"_[a-z]+\.(bdf|edf)$", "_channels.tsv", path)


def read_local(root: Path, per_subject: int | None = None):
    """``path -> (channels, sidecar units)`` for the EDF/BDF files under *root*."""
    paths = _pick(
        [
            str(p.relative_to(root))
            for p in root.rglob("sub-*/**/*")
            if p.suffix in EXTENSIONS
        ],
        per_subject,
    )
    recordings = {}
    for path in paths:
        with open(root / path, "rb") as f:
            main = f.read(256)
            buf = main + f.read(header_size(main) - 256)
        tsv = root / _sidecar_path(path)
        sidecar = sidecar_units(tsv.read_text()) if tsv.exists() else {}
        recordings[path] = (parse_header(buf), sidecar)
    return recordings


def _tree(dataset_id: str, sha: str, recursive: bool) -> dict:
    url = _TREE_URL.format(dataset_id, sha) + ("?recursive=1" if recursive else "")
    headers = {"Accept": "application/vnd.github+json"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as r:
        return json.load(r)


def nemar_files(dataset_id: str) -> list[str]:
    """Every file path of a NEMAR dataset, one subtree at a time when GitHub
    truncates the recursive listing (large datasets).
    """
    tree = _tree(dataset_id, "HEAD", recursive=True)
    if not tree.get("truncated"):
        return [e["path"] for e in tree["tree"] if e["type"] == "blob"]
    paths = []
    for entry in _tree(dataset_id, "HEAD", recursive=False)["tree"]:
        if entry["type"] == "blob":
            paths.append(entry["path"])
        elif entry["type"] == "tree":
            sub = _tree(dataset_id, entry["sha"], recursive=True)
            paths += [
                f"{entry['path']}/{e['path']}"
                for e in sub["tree"]
                if e["type"] == "blob"
            ]
    return paths


def read_nemar(dataset_id: str, per_subject: int | None = None, workers: int = 32):
    """Same as :func:`read_local` for a NEMAR dataset, from header byte ranges."""
    # Lazy: the local check does not need eegdash's reader stack.
    from eegdash.dataset.base import _resolve_nemar_pointer  # noqa: PLC0415

    s3 = boto3.client(
        "s3", config=Config(signature_version=UNSIGNED, max_pool_connections=workers)
    )
    files = nemar_files(dataset_id)
    paths = _pick([p for p in files if p.endswith(EXTENSIONS)], per_subject)

    def fetch(key, start, stop):
        rng = f"bytes={start}-{stop - 1}"
        return s3.get_object(
            Bucket="nemar", Key=f"{dataset_id}/objects/{key}", Range=rng
        )["Body"].read()

    def one(path):
        key, _ = _resolve_nemar_pointer(dataset_id, path)
        main = fetch(key, 0, 256)
        buf = main + fetch(key, 256, header_size(main))
        annex, raw = _resolve_nemar_pointer(dataset_id, _sidecar_path(path))
        sidecar = {} if annex else sidecar_units(raw.decode("utf-8"))
        return path, (parse_header(buf), sidecar)

    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(one, paths))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", help="NEMAR dataset id, e.g. nm000281")
    parser.add_argument("--local", type=Path, help="BIDS root of a local copy")
    parser.add_argument(
        "--per-subject", type=int, help="check at most N recordings per subject"
    )
    args = parser.parse_args(argv)
    recordings = (
        read_local(args.local, args.per_subject)
        if args.local
        else read_nemar(args.dataset, args.per_subject)
    )
    findings = check(recordings)
    flagged = {p for f in findings if f.check != "prefix" for p in f.files}
    print(
        f"{args.dataset}: {len(recordings)} recordings checked, "
        f"{len(flagged)} with channel unit or scale errors"
    )
    if findings:
        print(summarize(findings))
    return int(bool(flagged))


if __name__ == "__main__":
    raise SystemExit(main())
