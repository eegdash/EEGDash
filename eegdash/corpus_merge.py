# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Merge converted parts into one store, or extend a finished store with them.

The last stage of :class:`~eegdash.corpus.EEGDashCorpus`::

    corpus.convert("/data/out")                  # <out>/parts/<dataset>_<kkk>/
    corpus.merge("/data/out", "my-corpus")       # <out>/my-corpus.zarr
    corpus.merge("/data/out", "v0-plus", base="/data/eegdash-corpus-v0.zarr")

A store ``<out>/<name>.zarr`` is a Zarr group whose bucket groups ``0`` to
``n-1`` are the groups of the parts (see :mod:`eegdash.corpus_convert`),
numbered in the order of the parts. Its root attributes record the recipe and
index the buckets (``buckets``). Beside it go ``<name>-buckets.csv``, that
index, and ``<name>-manifest.csv``, one row per session, whose
``big_recording_index`` is the session's bucket.

With *base*, a finished store (the published ``eegdash-corpus-v0``, say) is
extended and never written to: its buckets keep their numbers and the parts'
follow. The parts must have been converted with the recipe the base records,
and a dataset the base holds is refused: no dataset is stored twice.

Only the small ``zarr.json`` files are copied, so attributes can be set in the
new store without reaching its sources. The chunk files are hard links by
default (``link="hard"``, one file system), else symbolic links or copies.
The store is built as ``<name>.zarr.partial`` and renamed once it has passed
its check. ``zarr`` is imported only when a merge runs.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from .corpus_convert import PART_STORE, is_local_id, write_csv
from .corpus_recipes import Recipe

LINK_MODES = ("hard", "symlink", "copy")
# The root attributes a store takes from its parts; the parts must agree on them.
STORE_KEYS = ("format", "version", "sfreq", "window", "units", "band_pass_hz")
BUCKET_FIELDS = ["bucket", "dataset", "shard", "n_samples", "n_sessions", "n_chans"]
META = "zarr.json"


# --------------------------------------------------------------------- checks


def _attrs(node: Path) -> dict[str, Any]:
    """The attributes of a Zarr v3 node, read from its ``zarr.json``: nothing is opened for writing."""
    meta = node / META
    if not meta.is_file():
        raise ValueError(f"{node}: not a Zarr v3 store (no {META})")
    return json.loads(meta.read_text()).get("attributes", {})


def _differ(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """The keys whose values differ between *a* and *b*."""
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def _refuser(recipe: Mapping[str, Any]) -> Recipe:
    """A :class:`Recipe` from a part's ``recipe.json``, for its ``refuses`` rule."""
    values = {f.name: recipe.get(f.name) for f in fields(Recipe)}
    values["refused"] = frozenset(recipe.get("refused") or ())
    return Recipe(**values)


def _aliases(dataset_id: str) -> set[str]:
    """*dataset_id* and its other archive id (``ds003810`` and ``on003810``), as ``Recipe.refuses`` reads them."""
    if re.fullmatch(r"(ds|on)\d+", dataset_id):
        return {"ds" + dataset_id[2:], "on" + dataset_id[2:]}
    return {dataset_id}


def _groups(store: Any) -> list[tuple[str, Any]]:
    """The bucket groups of a part or a store, by number."""
    return sorted(store.groups(), key=lambda kv: int(kv[0]))


def bucket_problems(name: str, group: Any, window: int) -> list[str]:
    """What is wrong with one bucket; empty when it agrees with its record of sessions.

    Samples, session ids, stats, channels and positions must agree with the
    ``sessions`` attribute: a part two workers wrote at once does not
    (2026-09-19, an older layout).
    """
    try:
        sessions, channels = group.attrs["sessions"], group.attrs["channels"]
        data, positions, stats = group["data"], group["positions"], group["stats"]
        session = group["session"][:]
        if len(data.shape) != 3:
            return [f"bucket {name}: data of shape {data.shape}"]
        n, w, c = data.shape
        expected = [s for s, x in enumerate(sessions) for _ in range(x["n_samples"])]
        short = any(x["duration"] <= window for x in sessions)
    except (KeyError, TypeError) as exc:
        return [f"bucket {name}: unreadable ({exc!r})"]
    out = []
    if w != window or c != len(channels) or positions.shape != (c, 3):
        out.append(
            f"bucket {name}: data {data.shape} for {len(channels)} channels, "
            f"positions {positions.shape}"
        )
    if n != len(session) or session.tolist() != expected:
        out.append(
            f"bucket {name}: {n} samples, {len(session)} session ids, "
            f"{len(expected)} per the record of sessions"
        )
    if stats.shape != (len(sessions), 2, c):
        out.append(f"bucket {name}: stats {stats.shape} for {len(sessions)} sessions")
    if short:
        out.append(f"bucket {name}: a session of {window} timesteps or fewer")
    return out


def _one_recipe(parts: list[Path]) -> dict[str, Any]:
    """The recipe every part was converted with; refused when two differ."""
    first = json.loads((parts[0] / "recipe.json").read_text())
    for part in parts[1:]:
        other = json.loads((part / "recipe.json").read_text())
        if other != first:
            raise ValueError(
                f"part {part.name} was converted with another recipe than "
                f"{parts[0].name}: {', '.join(_differ(first, other))} differ"
            )
    return first


def _one_store_attrs(parts: list[Path], sources: list[Any]) -> dict[str, Any]:
    """The root attributes the parts share (format, rate, window, units, band)."""
    first = {k: sources[0].attrs.get(k) for k in STORE_KEYS}
    lacking = [k for k, v in first.items() if v is None]
    if lacking:
        raise ValueError(f"part {parts[0].name}: no {', '.join(lacking)} attribute")
    for part, src in zip(parts[1:], sources[1:]):
        diff = _differ(first, {k: src.attrs.get(k) for k in STORE_KEYS})
        if diff:
            raise ValueError(
                f"part {part.name} differs from {parts[0].name}: {', '.join(diff)}"
            )
    return first


def _check_base(
    base: Path, recipe: dict[str, Any], store_attrs: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The base's root attributes and bucket index, once it is known to take the parts.

    A store written before the recipe was recorded (the published
    ``eegdash-corpus-v0``) is compared on what it does record. Its arrays are
    not read: each listed bucket must only be on disk.
    """
    attrs = _attrs(base)
    if "recipe" in attrs:
        diff = _differ(attrs["recipe"], recipe)
        diff += ["format"] if attrs.get("format") != store_attrs["format"] else []
    else:
        recorded = {k: store_attrs[k] for k in ("format", "sfreq", "window", "units")}
        recorded["band_pass_hz"] = [recipe.get("l_freq"), recipe.get("h_freq")]
        diff = [k for k, v in recorded.items() if attrs.get(k) != v]
    if diff:
        raise ValueError(
            f"base {base.name} was not converted as the parts were: "
            f"{', '.join(diff)} differ"
        )
    rows = [dict(r) for r in attrs.get("buckets") or []]
    k = len(rows)
    if [r.get("bucket") for r in rows] != list(range(k)) or attrs.get(
        "n_buckets", k
    ) != k:
        raise ValueError(f"base {base.name}: its buckets are not numbered 0 to {k - 1}")
    total = sum(int(r["n_samples"]) for r in rows)
    if attrs.get("n_samples", total) != total:
        raise ValueError(
            f"base {base.name}: n_samples {attrs['n_samples']}, its buckets {total}"
        )
    missing = [
        str(r["bucket"]) for r in rows if not (base / str(r["bucket"]) / META).is_file()
    ]
    if missing:
        raise ValueError(
            f"base {base.name}: {len(missing)} listed bucket(s) not on disk, "
            f"first: {', '.join(missing[:5])}"
        )
    return attrs, rows


def _check_datasets(
    parts: list[Path],
    sources: list[Any],
    recipe: dict[str, Any],
    base_attrs: dict[str, Any],
    base_rows: list[dict[str, Any]],
) -> None:
    """Refuse a benchmark source anywhere, and a dataset of the parts the base holds."""
    mine: set[str] = set()
    for src in sources:
        mine |= {src.attrs.get("dataset")} | {
            g.attrs.get("dataset") for _, g in src.groups()
        }
    mine.discard(None)
    held = set(base_attrs.get("datasets") or []) | {r.get("dataset") for r in base_rows}
    held.discard(None)
    refuser = _refuser(recipe)
    refused = sorted(d for d in mine | held if refuser.refuses(d))
    if refused:
        raise ValueError(
            f"benchmark source(s) of recipe {recipe.get('name')!r}, never "
            f"pretrained on: {refused}"
        )
    held_aliases = set().union(*(_aliases(d) for d in held)) if held else set()
    twice = sorted(d for d in mine if _aliases(d) & held_aliases)
    if twice:
        raise ValueError(
            f"dataset(s) already in the base: {twice}; a dataset is never stored twice"
        )


def _part_problems(source: Any, window: int) -> list[str]:
    return [p for name, g in _groups(source) for p in bucket_problems(name, g, window)]


# -------------------------------------------------------------------- writing


class _Placer:
    """Puts a bucket's files into the new store: ``zarr.json`` copied, the rest as *link* says."""

    def __init__(self, link: str):
        self.link = link
        self.linked = (
            False  # a first hard link worked: a later failure is not the location's
        )

    def tree(self, src: Path, dst: Path) -> None:
        for dirpath, _, files in os.walk(src):
            target = dst / Path(dirpath).relative_to(src)
            target.mkdir(parents=True, exist_ok=True)
            for name in sorted(files):
                self.file(Path(dirpath) / name, target / name)

    def file(self, src: Path, dst: Path) -> None:
        if src.name == META:  # a new file: attributes set here never reach the source
            shutil.copyfile(src, dst)
        elif self.link == "copy":
            shutil.copy2(src, dst)
        elif self.link == "symlink":
            dst.symlink_to(src.resolve())
        else:
            try:
                os.link(src, dst)
            except OSError as exc:
                if self.linked:
                    raise
                raise ValueError(
                    f"hard links are not possible from {src.parent} to {dst.parent} "
                    f"({exc.strerror or exc}): pass link='symlink' or link='copy'"
                ) from exc
            self.linked = True


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Columns and rows of a CSV file."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _union(*columns: Sequence[str]) -> list[str]:
    """Every column once, in order of first appearance."""
    return list(dict.fromkeys(c for cols in columns for c in cols))


def _final_check(
    zarr: Any, path: Path, rows: list[dict[str, Any]], n_base: int, window: int
) -> None:
    """Re-open the new store: buckets ``0..n-1``, each new one sound, each base one of its indexed shape."""
    root = zarr.open_group(path, mode="r")
    names = sorted(int(k) for k, _ in root.groups())
    problems = []
    if names != list(range(len(rows))):
        problems.append(f"buckets {names[:5]}... are not 0 to {len(rows) - 1}")
    else:
        for row in rows:
            g = root[str(row["bucket"])]
            shape = (int(row["n_samples"]), window, int(row["n_chans"]))
            if g["data"].shape != shape:  # metadata only: no chunk is read
                problems.append(
                    f"bucket {row['bucket']}: data {g['data'].shape}, index {shape}"
                )
            if row["bucket"] >= n_base:
                problems += bucket_problems(str(row["bucket"]), g, window)
    if problems:
        raise ValueError(
            "the merged store failed its check: " + "; ".join(problems[:10])
        )


# --------------------------------------------------------------------- report


@dataclass
class MergeReport:
    """The result of :func:`merge_parts`."""

    store: Path
    name: str
    n_buckets: int = 0
    n_samples: int = 0
    n_base_buckets: int = 0
    datasets: list[str] = field(default_factory=list)
    new_datasets: list[str] = field(default_factory=list)
    link: str = "hard"
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "store": str(self.store),
            "name": self.name,
            "n_buckets": self.n_buckets,
            "n_samples": self.n_samples,
            "n_base_buckets": self.n_base_buckets,
            "datasets": self.datasets,
            "new_datasets": self.new_datasets,
            "link": self.link,
            "seconds": round(self.seconds, 1),
        }

    def __str__(self) -> str:
        lines = [f"store: {self.store}"]
        if self.n_base_buckets:
            lines.append(
                f"base: {self.n_base_buckets} bucket(s) kept with their numbers"
            )
        lines.append(
            f"new datasets ({len(self.new_datasets)}): {', '.join(self.new_datasets) or '-'}"
        )
        lines.append(
            f"merge: {self.name}: {self.n_buckets} buckets, {self.n_samples} samples, "
            f"{len(self.datasets)} datasets ({len(self.new_datasets)} new); "
            f"link={self.link}; {self.seconds:.0f} s"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------- merge


def merge_parts(
    parts: Sequence[str | Path],
    out: str | Path,
    name: str,
    *,
    base: str | Path | None = None,
    link: str = "hard",
    licences: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> MergeReport:
    """Merge finished parts into the store ``<out>/<name>.zarr``, after a base's buckets if given.

    Parameters
    ----------
    parts : sequence of str | Path
        Part folders (each with ``_DONE``, ``recipe.json`` and
        ``samples.zarr``), in the order their buckets take.
    out : str | Path
        Where the store and its side files ``<name>-buckets.csv`` and
        ``<name>-manifest.csv`` go.
    name : str
        The store's name.
    base : str | Path | None
        A finished store to extend; never written to. Its buckets keep their
        numbers, its side file ``<base name>-manifest.csv`` (when beside it)
        heads the new manifest, its licences are carried over.
    link : {"hard", "symlink", "copy"}
        How the chunk files enter the new store.
    licences : mapping of str to str | None
        Dataset id -> licence id, set on the new buckets and added to the
        root's ``licences``.
    overwrite : bool
        Replace an existing store of that name (once the new one has passed
        its check).

    Returns
    -------
    MergeReport

    Raises
    ------
    ValueError
        Nothing is written when the merge is refused: an unfinished part,
        parts of different recipes, a base of another recipe, a benchmark
        source, a dataset the base holds, a damaged part, an existing store,
        a store inside the base, an unknown *link*.

    """
    import zarr  # noqa: PLC0415 (optional dependency)

    t0 = time.monotonic()
    if link not in LINK_MODES:
        raise ValueError(f"link must be one of {LINK_MODES}, got {link!r}")
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"a store's name is a plain file name, got {name!r}")
    parts = [Path(p) for p in parts]
    if not parts:
        raise ValueError("a merge needs at least one part")
    for part in parts:
        lacking = [
            f for f in ("_DONE", "recipe.json", PART_STORE) if not (part / f).exists()
        ]
        if lacking:
            raise ValueError(
                f"part {part}: no {', '.join(lacking)} (not a finished part)"
            )
    out = Path(out)
    target, partial = out / f"{name}.zarr", out / f"{name}.zarr.partial"
    base = None if base is None else Path(base)
    if base is not None:
        held = base.resolve()
        for path in (target, partial):
            mine = path.resolve()
            if mine.is_relative_to(held) or held.is_relative_to(mine):
                raise ValueError(
                    f"the new store {target} and the base {base} must lie outside each other"
                )
    if target.exists() and not overwrite:
        raise ValueError(f"{target} exists: pass overwrite=True to replace it")

    recipe = _one_recipe(parts)
    sources = [zarr.open_group(p / PART_STORE, mode="r") for p in parts]
    store_attrs = _one_store_attrs(parts, sources)
    window = int(store_attrs["window"])
    base_attrs, base_rows = (
        ({}, []) if base is None else _check_base(base, recipe, store_attrs)
    )
    _check_datasets(parts, sources, recipe, base_attrs, base_rows)
    damaged = [
        f"{p.name}: {m}"
        for p, s in zip(parts, sources)
        for m in _part_problems(s, window)
    ]
    if damaged:
        raise ValueError(
            f"{len(damaged)} problem(s) in the parts; delete them and convert again: "
            + "; ".join(damaged[:10])
        )

    licences = dict(licences or {})
    out.mkdir(parents=True, exist_ok=True)
    if partial.exists():  # left by an interrupted merge
        shutil.rmtree(partial)
    try:
        zarr.open_group(partial, mode="w")
        placer = _Placer(link)
        for row in base_rows:
            placer.tree(base / str(row["bucket"]), partial / str(row["bucket"]))
        rows, new = list(base_rows), []
        manifest_cols, manifest = [], []
        base_manifest = (
            None
            if base is None
            else base.parent / f"{base.name.removesuffix('.zarr')}-manifest.csv"
        )
        if base_manifest is not None and base_manifest.is_file():
            manifest_cols, manifest = _read_csv(base_manifest)
        for part, src in zip(parts, sources):
            remap = {}
            for key, g in _groups(src):
                number = len(rows)
                placer.tree(part / PART_STORE / key, partial / str(number))
                row = dict(
                    bucket=number,
                    dataset=g.attrs["dataset"],
                    shard=part.name,
                    n_samples=g["data"].shape[0],
                    n_sessions=len(g.attrs["sessions"]),
                    n_chans=g["data"].shape[2],
                )
                rows.append(row)
                new.append(row)
                remap[int(key)] = number
            cols, part_rows = _read_csv(part / "manifest_part.csv")
            manifest_cols = _union(manifest_cols, cols)
            for r in part_rows:
                old = int(r["big_recording_index"])
                if old not in remap:
                    raise ValueError(
                        f"{part.name}: manifest_part.csv names a bucket {old} it lacks"
                    )
                r["big_recording_index"] = remap[old]
                manifest.append(r)

        store = zarr.open_group(partial, mode="r+")
        for row in new:
            extra = (
                {"licence": licences[row["dataset"]]}
                if row["dataset"] in licences
                else {}
            )
            store[str(row["bucket"])].attrs.update(
                bucket=row["bucket"], source_shard=row["shard"], **extra
            )
        new_datasets = sorted({r["dataset"] for r in new})
        datasets = sorted(
            set(base_attrs.get("datasets") or []) | {r["dataset"] for r in rows}
        )
        attrs: dict[str, Any] = {
            **store_attrs,
            "recipe": recipe,
            "datasets": datasets,
            "n_buckets": len(rows),
            "n_samples": sum(int(r["n_samples"]) for r in rows),
            "buckets": rows,
            "local_datasets": [d for d in datasets if is_local_id(d)],
        }
        if base is not None:
            attrs["base"] = {
                "name": base.name.removesuffix(".zarr"),
                "n_buckets": len(base_rows),
                "n_samples": base_attrs.get(
                    "n_samples", sum(int(r["n_samples"]) for r in base_rows)
                ),
            }
            attrs.update(
                {
                    k: base_attrs[k]
                    for k in ("licence", "licences", "licence_file")
                    if k in base_attrs
                }
            )
        known = {d: licences[d] for d in new_datasets if d in licences}
        if known:
            attrs["licences"] = {**attrs.get("licences", {}), **known}
        store.attrs.update(attrs)
        _final_check(zarr, partial, rows, len(base_rows), window)
        if (
            target.exists()
        ):  # overwrite: the old store goes only once the new one is sound
            shutil.rmtree(target)
        partial.rename(target)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    write_csv(
        out / f"{name}-buckets.csv",
        rows,
        _union(BUCKET_FIELDS, *(r.keys() for r in rows)),
    )
    write_csv(
        out / f"{name}-manifest.csv", manifest, manifest_cols or ["dataset_id", "file"]
    )
    return MergeReport(
        store=target,
        name=name,
        n_buckets=len(rows),
        n_samples=attrs["n_samples"],
        n_base_buckets=len(base_rows),
        datasets=datasets,
        new_datasets=new_datasets,
        link=link,
        seconds=time.monotonic() - t0,
    )


__all__ = ["LINK_MODES", "MergeReport", "bucket_problems", "merge_parts"]
