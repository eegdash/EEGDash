# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Convert the recordings of a corpus into training samples, one part per shard.

The conversion stage of :class:`~eegdash.corpus.EEGDashCorpus`::

    corpus = EEGDashCorpus("eegdash-corpus-v0")
    corpus.shards()                                  # the parts, the same in every process
    corpus.convert("/data/out", shard=3, n_shards=64)  # this process's parts

A shard is one dataset, or some of its subjects; it becomes one part folder
``<out>/parts/<dataset>_<kkk>/`` holding::

    samples.zarr/<g>/data       (N, window, C) float32, one row per training sample
    samples.zarr/<g>/session    (N,) int32, the session each sample came from
    samples.zarr/<g>/stats      (S, 2, C) float32, mean and std per session
    samples.zarr/<g>/positions  (C, 3) float32, metres
    manifest_part.csv           one row per session written
    summary.json                counts, skipped recordings, dropped channels, the recipe
    recipe.json                 the recipe alone
    _DONE                       written last; a part that has it is never redone

A group ``<g>`` is one channel set: every session of the shard whose resolved
channels are the same, names sorted. Signals are microvolts and are not
normalised; the per-session statistics are for the loader.

The recipes are defined in :mod:`eegdash.corpus_recipes`, and nowhere else. The
one named ``reve`` is that of the REVE paper (section 3.1.1): EEG channels with a known 3-D position, band-pass 0.5-99.5 Hz,
resampled to 200 Hz, cut into samples of 2000 timesteps, recordings of 10 s or
less dropped. ``zarr`` is needed to write a part and is imported only then.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Sequence

import numpy as np
import pandas as pd

from .corpus_recipes import HELD_OUT, RECIPES, Recipe, get_recipe
from .logging import logger

PART_STORE = "samples.zarr"
SHARD_SAMPLES = 128  # samples per Zarr shard file: about 130 MB at 129 channels
DEFAULT_SHARD_BYTES = 20 * 1024**3  # raw bytes of one shard
# A gated or otherwise local dataset: <name>-<version>, lower case (tdbrain-3.1).
LOCAL_ID_RE = re.compile(r"[a-z][a-z0-9]*(-[a-z][a-z0-9]*)*-\d[a-z0-9.]*")
# Data files; .eeg, .fdt and .vmrk are companions.
PRIMARY_EXT = (".set", ".vhdr", ".edf", ".bdf", ".fif", ".gdf", ".cnt")
# the datatype folders and suffixes EEGDash's offline discovery reads by default
DATATYPES = ("eeg", "meg", "ieeg", "nirs", "emg")

BIOSEMI_RE = re.compile(r"[A-D]([1-9]|[12]\d|3[0-2])")  # A1..D32
# E1..E32: the fifth ring of a BioSemi-160 or -256 cap (and GSN's first 32 names)
RING_E_RE = re.compile(r"E([1-9]|[12]\d|3[0-2])")
RING_FGH_RE = re.compile(r"[F-H]([1-9]|[12]\d|3[0-2])")  # the outer rings of a 256 cap
RING_EH_RE = re.compile(r"[E-H]([1-9]|[12]\d|3[0-2])")
# A 256 cap needs many outer labels, not just some: a 10-10 cap has F1..F10
# beside C1..C6, a BioSemi-160 cap one ring E.
BIOSEMI_MIN_INNER = 32  # labels from rings A-D: a BioSemi cap
BIOSEMI256_MIN_FGH = 4  # then any real presence of rings F-H makes it a 256 cap
BIOSEMI160_MIN_E = 8  # else a ring E makes it a 160 cap; else the 128 cap
# P03, P0z: a zero typed for the letter O
ZERO_FOR_O_RE = re.compile(r"^([A-Za-z]+)0(\d+|[zZ])$")
ALIGN_MIN_SHARED = 8  # electrodes an electrodes.tsv must share with the bank
# RMS residual of the fit, metres. A file that projects the electrodes onto a
# unit sphere fits the template head to 16 mm; a mirrored or random file sits
# far above.
ALIGN_MAX_RMS_M = 0.020
FILTER_ONE_CORE_BYTES = 1_000_000_000
# Channel names that are not scalp EEG, never placed from a dataset's
# coordinates. A bare I is an inion electrode.
NON_EEG_RE = re.compile(
    r"(EXG|EOG|HEO|VEO|E[CK]G|EMG|GSR|Erg\d*|Resp|Plet|Pulse|Optic|Temp|Status|Trig"
    r"|Audio|Mastoid|Eye|SpO2|Dia-|\bHR\b"
    r"|^(Empty|M[12]|REF\w*|GND|Ground|MK|\d+|[A-HJ-Z])$)",
    re.IGNORECASE,
)
# A modality prefix some EDF writers put on every label: "EEG Fp1", "EEG-Cz".
EEG_PREFIX_RE = re.compile(r"^EEG[\s_-]+", re.IGNORECASE)
ENTITY_RE = re.compile(r"(?:^|_)(sub|ses)-([^_]+)")


def is_local_id(dataset_id: str) -> bool:
    """Whether *dataset_id* names a local dataset, ``<name>-<version>``.

    Such a dataset is not in the catalogue (a gated one, or the user's own):
    its BIDS tree is read from ``<cache_dir>/<dataset_id>/`` and is never
    queried or downloaded.
    """
    return bool(LOCAL_ID_RE.fullmatch(dataset_id))


# --------------------------------------------------------------- position bank


def strip_eeg_prefix(name: str) -> str:
    """``EEG Fp1`` -> ``Fp1``; a name that would become empty is returned unchanged."""
    bare = EEG_PREFIX_RE.sub("", name.strip())
    return bare or name


def zero_for_o(name: str) -> str | None:
    """``P03`` -> ``PO3``, ``P0z`` -> ``POz``; None without a zero after the letters."""
    m = ZERO_FOR_O_RE.match(name)
    return f"{m[1]}O{m[2]}" if m else None


def with_biosemi_templates(
    names: list[str], xyz: np.ndarray
) -> tuple[list[str], np.ndarray]:
    """Add MNE's biosemi160 and biosemi256 templates to a bank.

    They enter as ``biosemi160_A1..E32`` and ``biosemi256_A1..H32`` (metres,
    the frame in which the bank's ``biosemi128_*`` equal MNE's). A bank with
    the 128 cap alone gave rings A-D of a bigger cap the 128 cap's positions.
    Names already present are not added twice.
    """
    import mne  # noqa: PLC0415 (heavy import)

    have = set(names)
    extra = []
    for cap in ("biosemi160", "biosemi256"):
        ch_pos = mne.channels.make_standard_montage(cap).get_positions()["ch_pos"]
        extra += [
            (f"{cap}_{k}", v) for k, v in ch_pos.items() if f"{cap}_{k}" not in have
        ]
    if not extra:
        return list(names), np.asarray(xyz, dtype=np.float32)
    return (
        list(names) + [n for n, _ in extra],
        np.concatenate(
            [
                np.asarray(xyz, dtype=np.float32),
                np.array([v for _, v in extra], dtype=np.float32),
            ]
        ),
    )


class Bank:
    """Electrode name -> 3-D position (metres), with the naming rules of the corpus.

    The rules, applied by :meth:`resolve`:

    - the bank is case-sensitive, but its case-fold duplicates share
      coordinates, so a case-insensitive match is safe;
    - a modality prefix (``EEG Fp1``) is dropped when the bare name fails;
    - bare ``A1..D32`` are 10-20 names, but 32 or more of them are the rings
      of a BioSemi cap: ``biosemi128_*``, or ``biosemi160_*`` / ``biosemi256_*``
      when the outer rings are there;
    - a zero typed for the letter O (``P03``) is tried last;
    - the same electrode twice (``Cz`` and ``CZ``) is kept once.
    """

    def __init__(self, names: Sequence[str], xyz: np.ndarray):
        xyz = np.asarray(xyz, dtype=np.float32)
        if len(names) != len(xyz) or xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("a position bank needs one (x, y, z) per name")
        self.names = list(names)
        self.xyz = xyz
        self.exact = {n: i for i, n in enumerate(self.names)}
        self.fold: dict[str, int] = {}
        for i, n in enumerate(self.names):
            self.fold.setdefault(n.casefold(), i)

    @classmethod
    def from_braindecode(cls) -> Bank:
        """REVE's public position bank, through braindecode's cached copy."""
        from braindecode.models.reve import RevePositionBank  # noqa: PLC0415

        bank = RevePositionBank()
        return cls(
            *with_biosemi_templates(
                list(bank.position_names), bank.embedding.detach().cpu().numpy()
            )
        )

    @classmethod
    def from_json(cls, path: str | Path) -> Bank:
        """A ``{name: [x, y, z]}`` file in metres, plus the BioSemi templates."""
        d = json.loads(Path(path).read_text())
        return cls(
            *with_biosemi_templates(
                list(d), np.array(list(d.values()), dtype=np.float32)
            )
        )

    def lookup(self, name: str) -> int | None:
        """Row of *name*, exact spelling first, then ignoring case."""
        i = self.exact.get(name)
        return i if i is not None else self.fold.get(name.casefold())

    def resolve(self, ch_names: list[str]) -> tuple[list[int], list[str], list[str]]:
        """Return (indices kept, their bank names in the same order, names dropped)."""
        inner = sum(bool(BIOSEMI_RE.fullmatch(n)) for n in ch_names)
        cap = None  # the BioSemi cap of the ring labels, decided once per recording
        if inner >= BIOSEMI_MIN_INNER:
            n_fgh = sum(bool(RING_FGH_RE.fullmatch(n)) for n in ch_names)
            n_e = sum(bool(RING_E_RE.fullmatch(n)) for n in ch_names)
            if n_fgh >= BIOSEMI256_MIN_FGH:
                cap = "biosemi256"
            elif n_e >= BIOSEMI160_MIN_E:
                cap = "biosemi160"
        biosemi = inner >= BIOSEMI_MIN_INNER and cap is None
        kept, canonical, dropped, seen = [], [], [], set()
        for i, n in enumerate(ch_names):
            bare = strip_eeg_prefix(n)
            if cap and (BIOSEMI_RE.fullmatch(n) or RING_EH_RE.fullmatch(n)):
                # never the 128 cap's ring, the 10-20 name or GSN's E1..E32
                cands = [f"{cap}_{n}"]
            else:
                cands = (
                    [f"biosemi128_{n}"] if biosemi and BIOSEMI_RE.fullmatch(n) else []
                )
                cands += [n] + ([bare] if bare != n else [])
                # a fallback only: after every exact candidate
                cands += [o for o in map(zero_for_o, cands) if o]
            hits = (self.lookup(c) for c in cands)
            hit = next((h for h in hits if h is not None), None)
            # an unknown name, or the same electrode twice (Cz and CZ)
            if hit is None or self.names[hit].casefold() in seen:
                dropped.append(n)
                continue
            seen.add(self.names[hit].casefold())
            kept.append(i)
            canonical.append(self.names[hit])
        return kept, canonical, dropped

    def positions(self, canonical: list[str]) -> np.ndarray:
        """The (C, 3) positions of bank names, in the order given."""
        return np.stack([self.xyz[self.exact[n]] for n in canonical]).astype(np.float32)


# ------------------------------------------------- positions from the dataset


def find_electrodes_tsv(root: Path, relpath: str) -> Path | None:
    """The ``*_electrodes.tsv`` of the recording at ``root/relpath``, by BIDS inheritance.

    Its own folder first, then the session and subject folders. Within a
    folder: a file whose sub/ses entities match the recording's, then one
    without ``space-``; another subject's file never.
    """
    root = Path(root)
    rec = root / relpath
    ents = dict(ENTITY_RE.findall(rec.name))
    for d in rec.parents:
        if d == root or root not in d.parents:
            break
        ranked = []
        for f in sorted(d.glob("*_electrodes.tsv")):
            fe = dict(ENTITY_RE.findall(f.name))
            if "sub" in fe and "sub" in ents and fe["sub"] != ents["sub"]:
                continue
            mismatch = not all(ents.get(k) == v for k, v in fe.items())
            ranked.append((mismatch, "space-" in f.name, f.name, f))
        if ranked:
            return min(ranked)[-1]
    return None


def read_electrodes_tsv(path: Path) -> dict[str, np.ndarray]:
    """Name -> (x, y, z) in the file's own units and frame.

    Rows with n/a or all-zero coordinates are skipped (some datasets write
    0 0 0 for an electrode they did not digitise). The first of two spellings
    wins.
    """
    out: dict[str, np.ndarray] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                xyz = np.array([float(row[k]) for k in ("x", "y", "z")])
            except (KeyError, TypeError, ValueError):
                continue
            name = str(row.get("name") or "").strip()
            if name and np.all(np.isfinite(xyz)) and np.any(xyz != 0):
                out.setdefault(name, xyz)
    return out


def similarity_fit(
    src: np.ndarray, dst: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Scale s, rotation R, translation t minimising ``|dst - (s R src + t)|^2``.

    Umeyama (1991). R is a proper rotation (det +1): a mirrored file must fail
    the residual test, not be silently flipped.
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    x, y = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(y.T @ x / len(src))
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2] = -1
    r = u @ np.diag(sign) @ vt
    var = (x**2).sum() / len(src)
    s = float((d * sign).sum() / var) if var > 0 else 0.0
    return s, r, mu_d - s * r @ mu_s


def _tsv_name(name: str, coords: dict, folded: dict[str, str]) -> str | None:
    """The spelling of *name* in an electrodes.tsv, with or without its EEG prefix."""
    for c in dict.fromkeys((name, strip_eeg_prefix(name))):
        if c in coords:
            return c
        if c.casefold() in folded:
            return folded[c.casefold()]
    return None


def dataset_positions(
    path: Path, names: list[str], kept: list[int], canonical: list[str], bank: Bank
) -> tuple[list[int], list[str], np.ndarray, float | None]:
    """Place the channels the bank does not know from the dataset's electrodes.tsv.

    The file is fitted to the bank on the channels :meth:`Bank.resolve` kept
    (rotation, uniform scale and translation, so its units and frame need not
    be trusted), and a channel keeps the file's spelling.

    Returns
    -------
    tuple
        Indices into *names*, their names, their (k, 3) positions in metres in
        the bank's frame, and the RMS residual of the fit in metres (None when
        no fit was made). Empty when the file is unreadable, shares fewer than
        ``ALIGN_MIN_SHARED`` channels with the bank or fits worse than
        ``ALIGN_MAX_RMS_M``.

    """
    none = ([], [], np.zeros((0, 3), dtype=np.float32))
    try:
        coords = read_electrodes_tsv(path)
    except (OSError, UnicodeDecodeError, csv.Error):
        return (*none, None)
    folded: dict[str, str] = {}
    for k in coords:
        folded.setdefault(k.casefold(), k)
    pairs = []
    for i, c in zip(kept, canonical):
        k = _tsv_name(names[i], coords, folded)
        if k:
            pairs.append((coords[k], bank.xyz[bank.exact[c]]))
    if len(pairs) < ALIGN_MIN_SHARED:
        return (*none, None)
    src = np.array([p[0] for p in pairs])
    dst = np.array([p[1] for p in pairs], dtype=np.float64)
    s, r, t = similarity_fit(src, dst)
    rms = float(np.sqrt(((s * src @ r.T + t - dst) ** 2).sum(1).mean()))
    if not np.isfinite(rms) or s <= 0 or rms > ALIGN_MAX_RMS_M:
        return (*none, rms)
    taken = {c.casefold() for c in canonical}
    idx, out, pos = [], [], []
    keep = set(kept)
    for i, n in enumerate(names):
        # an electrode the bank knows (a duplicate, or a ring label of a cap
        # the bank lacks) is not re-placed
        if (
            i in keep
            or NON_EEG_RE.search(n.strip())
            or bank.lookup(n) is not None
            or bank.lookup(strip_eeg_prefix(n)) is not None
        ):
            continue
        k = _tsv_name(n, coords, folded)
        if k is None or k.casefold() in taken:
            continue
        taken.add(k.casefold())
        idx.append(i)
        out.append(k)
        pos.append(s * r @ coords[k] + t)
    if not idx:
        return (*none, rms)
    return idx, out, np.array(pos, dtype=np.float32), rms


# -------------------------------------------------------------- preprocessing


@dataclass
class Prepared:
    """One recording after the recipe."""

    data: np.ndarray  # (T, C) float32, channels in sorted name order
    names: list[str]
    dropped: list[str]
    sfreq_native: float
    typed_from_names: bool = False  # channel types came from the bank, not the file
    positions: np.ndarray | None = None  # (C, 3) metres, same order as names
    aligned: list[str] = field(default_factory=list)  # placed from electrodes.tsv


@dataclass
class Skipped:
    """A recording the recipe leaves out, and why."""

    reason: str
    detail: str = ""


def filter_jobs(n_bytes: int, n_jobs: int) -> int:
    """Workers for MNE's filter and resample: one above ``FILTER_ONE_CORE_BYTES``.

    Each joblib worker holds its own copy of the data: a 3 GB recording on ten
    cores is 30 GB.
    """
    return 1 if n_bytes > FILTER_ONE_CORE_BYTES else max(1, n_jobs)


def preprocess_recording(
    raw: Any,
    bank: Bank,
    recipe: Recipe = RECIPES["reve"],
    *,
    n_jobs: int = 1,
    electrodes: Path | None = None,
) -> Prepared | Skipped:
    """Apply *recipe* to one MNE ``Raw`` (not yet loaded).

    EEG channels only, those with a position (the bank, then *electrodes*, the
    recording's ``*_electrodes.tsv``: see :func:`dataset_positions`), band-pass,
    resample, microvolts, float32.
    """
    sfreq = float(raw.info["sfreq"])
    if sfreq < recipe.sfreq - 1e-6:
        return Skipped(f"sfreq_lt_{recipe.sfreq}", f"{sfreq:g} Hz")
    if raw.n_times / sfreq <= recipe.window / recipe.sfreq:
        return Skipped("too_short", f"{raw.n_times / sfreq:.1f} s")
    types = raw.get_channel_types()
    eeg = [n for n, t in zip(raw.ch_names, types) if t == "eeg"]
    typed_from_names = False
    if not eeg:
        # A channels.tsv with type "n/a" leaves every channel "misc". Fall back
        # to all channels and let the bank decide: a name that resolves to a
        # scalp electrode is EEG.
        eeg, typed_from_names = list(raw.ch_names), True
    kept, canonical, dropped = bank.resolve(eeg)
    if typed_from_names and not kept:
        return Skipped(
            "no_eeg", "no EEG-typed channels and no name resolves in the position bank"
        )
    if len(kept) < recipe.min_channels:
        return Skipped(
            f"lt{recipe.min_channels}_resolvable", f"{len(kept)} of {len(eeg)}"
        )
    a_idx, a_names, a_pos = [], [], np.zeros((0, 3), dtype=np.float32)
    if electrodes is not None and dropped:
        a_idx, a_names, a_pos, _ = dataset_positions(
            Path(electrodes), eeg, kept, canonical, bank
        )
        placed = {eeg[i] for i in a_idx}
        dropped = [n for n in dropped if n not in placed]
    idx, canonical = kept + a_idx, canonical + a_names
    pos = np.concatenate([bank.positions(canonical[: len(kept)]), a_pos])
    # pick in file order, so the data rows follow idx whatever pick does
    by_file = np.argsort(idx, kind="stable")
    idx, canonical = [idx[k] for k in by_file], [canonical[k] for k in by_file]
    pos = pos[by_file]
    raw = raw.copy().pick([eeg[i] for i in idx]).load_data(verbose="ERROR")
    if typed_from_names:  # MNE filters "data" channels only
        raw.set_channel_types({n: "eeg" for n in raw.ch_names}, verbose="ERROR")
    jobs = filter_jobs(len(raw.ch_names) * raw.n_times * 8, n_jobs)  # float64
    raw.filter(
        recipe.l_freq,
        min(recipe.h_freq, recipe.h_freq_max_ratio * sfreq),
        n_jobs=jobs,
        verbose="ERROR",
    )
    if abs(sfreq - recipe.sfreq) > 1e-6:
        raw.resample(
            recipe.sfreq, method=recipe.resample_method, n_jobs=jobs, verbose="ERROR"
        )
    order = np.argsort(canonical)
    data = raw.get_data(units=recipe.units)[order].T  # (T, C)
    return Prepared(
        np.ascontiguousarray(data, dtype="<f4"),
        [canonical[i] for i in order],
        dropped,
        sfreq,
        typed_from_names=typed_from_names,
        positions=np.ascontiguousarray(pos[order], dtype=np.float32),
        aligned=sorted(a_names),
    )


# --------------------------------------------------------------------- writer


def sample_offsets(n_times: int, window: int = 2000) -> np.ndarray:
    """Start of every training sample in a session of *n_times* timesteps.

    As REVE's loader cuts them: ``np.arange(start, end - window, window)``.
    The exclusive end means a session of exactly ``k * window`` timesteps
    yields ``k - 1`` samples. Reproduced on purpose, so a store holds
    precisely the samples REVE trains on.
    """
    return np.arange(0, n_times - window, window)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    """Write *rows* with *fields* as the header."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


@dataclass
class ZarrGroup:
    """One channel set of a part while it is written."""

    index: int
    names: list[str]
    data: Any  # the zarr array (N, window, C)
    positions: np.ndarray  # (C, 3) metres, of the group's first session
    n_written: int = 0
    pending: list = field(default_factory=list)  # blocks not yet written
    n_pending: int = 0
    session_of_sample: list = field(default_factory=list)
    stats: list = field(default_factory=list)  # (2, C) per session
    sessions: list = field(default_factory=list)  # provenance per session


class ZarrGroupWriter:
    """Stream sessions into one part: a Zarr store of samples, a group per channel set.

    One chunk per sample, ``SHARD_SAMPLES`` chunks to a file. Positions are
    those of a group's first session: a bank name has one position; a name
    placed from an electrodes.tsv may differ by a few millimetres between the
    subjects of one shard, and the first wins.
    """

    def __init__(
        self, out: Path, dataset: str, bank: Bank, recipe: Recipe, compress: bool = True
    ):
        import zarr  # noqa: PLC0415 (optional dependency)
        from zarr.codecs import BloscCodec  # noqa: PLC0415

        self.out, self.dataset, self.bank, self.recipe = (
            Path(out),
            dataset,
            bank,
            recipe,
        )
        self.out.mkdir(parents=True, exist_ok=True)
        self.root = zarr.open_group(self.out / PART_STORE, mode="w")
        self.root.attrs.update(
            format="reve-samples",
            version=1,
            sfreq=recipe.sfreq,
            window=recipe.window,
            units=recipe.units,
            band_pass_hz=[recipe.l_freq, recipe.h_freq],
            dataset=dataset,
            recipe=recipe.to_dict(),
        )
        self.compressors = (
            BloscCodec(cname="zstd", clevel=5, shuffle="shuffle") if compress else None
        )
        self.groups: dict[tuple, ZarrGroup] = {}
        self.manifest: list[dict] = []

    def _group(self, prep: Prepared) -> ZarrGroup:
        key = tuple(prep.names)
        if key not in self.groups:
            idx, n_chans, window = len(self.groups), len(prep.names), self.recipe.window
            data = self.root.create_group(str(idx)).create_array(
                "data",
                shape=(0, window, n_chans),
                chunks=(1, window, n_chans),
                shards=(SHARD_SAMPLES, window, n_chans),
                dtype="float32",
                compressors=self.compressors,
            )
            positions = (
                prep.positions
                if prep.positions is not None
                else self.bank.positions(prep.names)
            )
            self.groups[key] = ZarrGroup(idx, list(prep.names), data, positions)
        return self.groups[key]

    def _flush(self, grp: ZarrGroup, final: bool = False) -> None:
        """Write pending samples in whole shards, a partial one only at the end."""
        if not grp.n_pending:
            return
        block = np.concatenate(grp.pending)
        n = len(block) if final else (len(block) // SHARD_SAMPLES) * SHARD_SAMPLES
        if n:
            grp.data.resize((grp.n_written + n, self.recipe.window, len(grp.names)))
            grp.data[grp.n_written : grp.n_written + n] = block[:n]
            grp.n_written += n
        grp.pending = [block[n:]] if n < len(block) else []
        grp.n_pending = len(block) - n

    def append(self, prep: Prepared, file: str, meta: dict) -> int:
        """Add one session; returns its group's index."""
        n_times, n_chans = prep.data.shape
        window = self.recipe.window
        if n_times <= window:  # the loader drops it: stats rows would misalign
            raise ValueError(f"session {file} has {n_times} <= {window} timesteps")
        grp = self._group(prep)
        s = len(grp.sessions)
        n = len(sample_offsets(n_times, window))
        # offsets are 0, window, 2 * window, ... so this is exact
        samples = prep.data[: n * window].reshape(n, window, n_chans)
        first = grp.n_written + grp.n_pending
        grp.pending.append(samples)
        grp.n_pending += n
        grp.session_of_sample += [s] * n
        # over the whole session, not only its samples
        grp.stats.append(np.stack([prep.data.mean(0), prep.data.std(0)]).astype("<f4"))
        grp.sessions.append(
            dict(file=file, duration=n_times, n_samples=n, first_sample=first)
        )
        self.manifest.append(
            dict(
                dataset_id=self.dataset,
                file=file,
                big_recording_index=grp.index,
                session_index=s,
                n_chans=n_chans,
                duration_s=round(n_times / self.recipe.sfreq, 2),
                sfreq_native=prep.sfreq_native,
                n_dropped_no_position=len(prep.dropped),
                n_positions_aligned=len(prep.aligned),
                n_samples=n,
                **meta,
            )
        )
        self._flush(grp)
        return grp.index

    def finalize(self, summary: dict) -> None:
        """Write what is pending, the side files and, last, ``_DONE``."""
        groups = sorted(self.groups.values(), key=lambda g: g.index)
        for grp in groups:
            self._flush(grp, final=True)
            g = self.root[str(grp.index)]
            g.create_array(
                "session", data=np.asarray(grp.session_of_sample, dtype="int32")
            )
            g.create_array("stats", data=np.stack(grp.stats))
            g.create_array("positions", data=grp.positions)
            g.attrs.update(
                channels=grp.names, dataset=self.dataset, sessions=grp.sessions
            )
        fields = (
            list(self.manifest[0].keys()) if self.manifest else ["dataset_id", "file"]
        )
        write_csv(self.out / "manifest_part.csv", self.manifest, fields)
        sfreq, window = self.recipe.sfreq, self.recipe.window
        timesteps = sum(s["duration"] for g in groups for s in g.sessions)
        n_samples = sum(g.n_written for g in groups)
        summary.update(
            format="zarr",
            n_big_recordings=len(groups),
            n_sessions=sum(len(g.sessions) for g in groups),
            n_samples=n_samples,
            hours_written=round(timesteps / sfreq / 3600, 3),
            hours_in_samples=round(n_samples * window / sfreq / 3600, 3),
            recipe=self.recipe.to_dict(),
        )
        (self.out / "summary.json").write_text(
            json.dumps(summary, indent=1, default=str)
        )
        (self.out / "recipe.json").write_text(
            json.dumps(self.recipe.to_dict(), indent=1) + "\n"
        )
        (self.out / "_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


# -------------------------------------------------------------------- records


@dataclass
class LocalRecord:
    """A recording of a local BIDS tree, shaped like the records the converter reads."""

    path: Path
    root: Path

    @property
    def record(self) -> dict:
        return {"bids_relpath": self.path.relative_to(self.root).as_posix()}

    @property
    def description(self) -> dict:
        from mne_bids import get_entities_from_fname  # noqa: PLC0415

        e = get_entities_from_fname(self.path.name)
        return {k: e.get(k) for k in ("subject", "session", "run", "task")}

    @property
    def raw(self) -> Any:
        from mne_bids import (  # noqa: PLC0415
            BIDSPath,
            get_entities_from_fname,
            read_raw_bids,
        )

        e = get_entities_from_fname(self.path.name)
        bp = BIDSPath(
            root=self.root,
            datatype="eeg",
            suffix="eeg",
            extension=self.path.suffix,
            check=False,
            **{k: v for k, v in e.items() if v is not None},
        )
        return read_raw_bids(bp, verbose="ERROR")


def local_subjects(root: Path) -> dict[str, int]:
    """Subject label -> bytes of its EEG folders, for the subjects of a BIDS tree with a recording."""
    out: dict[str, int] = {}
    for d in sorted(Path(root).glob("sub-*")):
        files = [p for p in d.glob("**/eeg/*") if p.is_file()]
        if any(p.name.endswith(tuple(f"_eeg{e}" for e in PRIMARY_EXT)) for p in files):
            out[d.name[4:]] = sum(p.stat().st_size for p in files)
    return out


def discover_bids(root: Path, subjects: Sequence[str] | None) -> list[LocalRecord]:
    """The recordings of a local BIDS tree, listed per subject folder.

    One glob per subject of the shard, not a walk of the whole tree for every
    shard; paths stay relative to *root* as given, so a tree behind a symbolic
    link reads the same.
    """
    root = Path(root)
    if subjects:
        subs = [f"sub-{s}" for s in subjects]
    else:
        subs = sorted(d.name for d in root.glob("sub-*") if d.is_dir())
    return [
        LocalRecord(p, root)
        for s in subs
        for p in sorted((root / s).glob("**/eeg/*_eeg.*"))
        if p.suffix in PRIMARY_EXT
    ]


def record_file(rec: Any) -> str:
    """The path of a record inside its dataset."""
    r = rec.record
    return str(r.get("bids_relpath") or r.get("bidspath") or r.get("data_name") or "?")


def _record_order(rec: Any) -> tuple[str, ...]:
    d = rec.description
    keys = tuple(str(d.get(k) or "") for k in ("subject", "session", "run"))
    return (*keys, record_file(rec))


def subject_recordings(root: Path, subjects: Sequence[str] | None) -> list[str]:
    """The recordings on disk under the subject folders of *root*, as paths inside it.

    One listing per subject folder of the shard. EEGDash's offline discovery
    lists the whole dataset whichever subjects are asked for, three times over:
    on a parallel file system that is about 17 files a second, half an hour for
    a dataset of 32,000 files, for every shard, before a recording is read.
    """
    from mne_bids.config import ALLOWED_DATATYPE_EXTENSIONS  # noqa: PLC0415

    exts = {e for d in DATATYPES for e in ALLOWED_DATATYPE_EXTENSIONS.get(d, [])}
    root = Path(root)
    if subjects:
        folders = [root / f"sub-{s}" for s in subjects]
    else:
        folders = sorted(d for d in root.glob("sub-*") if d.is_dir())
    found = set()
    for folder in folders:
        for datatype in DATATYPES:
            for path in folder.glob(f"**/{datatype}/*"):
                suffix = path.name.split(".")[0].rsplit("_", 1)[-1]
                if suffix in DATATYPES and path.suffix.lower() in exts:
                    found.add(path.relative_to(root).as_posix())
    return sorted(found)


def _entities(name: str) -> dict[str, str]:
    """The ``key-value`` entities of a BIDS file name."""
    pairs = (piece.partition("-") for piece in name.split(".")[0].split("_")[:-1])
    return {key: value for key, sep, value in pairs if sep}


def discover_records(
    cache_dir: Path, folder: str, subjects: Sequence[str] | None
) -> list[Any]:
    """The recordings of a dataset in the cache, offline, in a fixed order.

    The files are listed per subject (:func:`subject_recordings`) and handed to
    :class:`~eegdash.EEGDashDataset` as records, the same record its offline
    discovery builds, so each is read by the same loader.
    """
    from .dataset import EEGDashDataset  # noqa: PLC0415 (heavy import)
    from .schemas import create_record  # noqa: PLC0415

    # resolved: the loader reads sidecars against the real folder
    root = (Path(cache_dir) / folder).resolve()
    records = []
    for rel in subject_recordings(root, subjects):
        e = _entities(Path(rel).name)
        records.append(
            create_record(
                dataset=root.name,
                storage_base=str(root),
                bids_relpath=rel,
                subject=e.get("sub"),
                session=e.get("ses"),
                task=e.get("task"),
                run=e.get("run"),
                acquisition=e.get("acq"),
                dep_keys=[],
                datatype=Path(rel).parent.name,
                suffix=Path(rel).name.split(".")[0].rsplit("_", 1)[-1],
                storage_backend="local",
            )
        )
    if not records:
        return []
    dataset = EEGDashDataset(
        cache_dir=str(root.parent),
        dataset=root.name,
        records=records,
        download=False,
        n_jobs=1,
    )
    return sorted(dataset.datasets, key=_record_order)


def release(rec: Any) -> None:
    """Forget the recording a record has loaded.

    An EEGDash record keeps its ``Raw``, and an EEGLAB file that holds its own
    data is in memory once opened: a shard of 270 such recordings grew by a
    recording per recording until the job was killed.
    """
    if getattr(rec, "_raw", None) is not None:
        rec._raw = None


def keep_listed(
    recs: Sequence[Any], files: Collection[str] | None
) -> tuple[list[Any], list[dict]]:
    """Split *recs* into the recordings to convert and those left out, with the reason.

    A recording is a dataset's own only under a subject folder of its root:
    ``sourcedata/`` holds the files a dataset was made from (the same signal a
    second time), ``derivatives/`` what was computed from it. When *files* is
    given (the recordings the catalogue or a shipped file list names), a file on
    disk that is not in it is left out too: a cache keeps a file after the
    archive renames it, and both names would be converted.
    """
    kept, left = [], []
    for rec in recs:
        file = record_file(rec)
        top = file.split("/")[0]
        if not top.startswith("sub-"):
            left.append(dict(file=file, reason="outside_subject_folders", detail=top))
        elif files is not None and file not in files:
            left.append(dict(file=file, reason="not_in_file_list", detail=""))
        else:
            kept.append(rec)
    return kept, left


def record_electrodes(rec: Any, root: Path) -> Path | None:
    """The ``*_electrodes.tsv`` that applies to a record of the dataset at *root*."""
    if isinstance(rec, LocalRecord):
        return find_electrodes_tsv(rec.root, rec.path.relative_to(rec.root).as_posix())
    rel = rec.record.get("bids_relpath")
    if not rel:
        return None
    return find_electrodes_tsv(Path(getattr(rec, "bids_root", None) or root), str(rel))


def record_meta(rec: Any) -> dict[str, str]:
    """Subject, session, run and task of a record, as text."""
    d = rec.description
    keys = ("subject", "session", "run", "task")
    return {k: ("" if d.get(k) is None else str(d.get(k))) for k in keys}


# ---------------------------------------------------------------------- shard


@dataclass(frozen=True)
class Shard:
    """One part to convert: a dataset, or some of its subjects."""

    dataset_id: str
    index: int
    subjects: tuple[str, ...]
    n_bytes: int = 0

    @property
    def name(self) -> str:
        """The part's folder name, ``<dataset>_<kkk>``."""
        return f"{self.dataset_id}_{self.index:03d}"


def plan_shards(
    dataset_id: str, subject_bytes: dict[str, int], max_bytes: int
) -> list[Shard]:
    """Pack the subjects of a dataset, in sorted order, into shards of at most *max_bytes*.

    A subject larger than *max_bytes* is a shard of its own. The plan depends
    only on its arguments, so every process computes the same one.
    """
    shards: list[Shard] = []
    current: list[str] = []
    size = 0
    for subject in sorted(subject_bytes):
        n = subject_bytes[subject]
        if current and size + n > max_bytes:
            shards.append(Shard(dataset_id, len(shards), tuple(current), size))
            current, size = [], 0
        current.append(subject)
        size += n
    if current:
        shards.append(Shard(dataset_id, len(shards), tuple(current), size))
    return shards


def convert_shard(
    dataset: str,
    subjects: Sequence[str] | None,
    root: Path,
    out: Path,
    *,
    recipe: str | Recipe = "reve",
    bank: Bank | None = None,
    local: bool | None = None,
    n_jobs: int = 1,
    compress: bool = True,
    use_dataset_positions: bool = True,
    max_records: int | None = None,
    files: Collection[str] | None = None,
) -> dict:
    """Convert one shard into the part folder *out*; returns its summary.

    Parameters
    ----------
    dataset : str
        The id written into the part.
    subjects : sequence of str | None
        The subject labels of this shard; None for every subject.
    root : Path
        The dataset's BIDS tree (``<cache_dir>/<folder>``).
    out : Path
        The part folder. One that holds ``_DONE`` is left alone and its
        summary returned; one without is an interrupted run and is removed
        first.
    recipe : str | Recipe
        The preprocessing; ``"reve"`` by default.
    bank : Bank | None
        The position bank; REVE's public one by default.
    local : bool | None
        Read the tree per subject folder with MNE-BIDS (a local dataset)
        instead of through :class:`~eegdash.EEGDashDataset`'s offline mode.
        By default, whether *dataset* is a local id (:func:`is_local_id`).
    n_jobs : int
        Workers for MNE's filter and resample.
    compress : bool
        Blosc-zstd chunks; False stores them as they are.
    use_dataset_positions : bool
        Place channels the bank does not know from the dataset's
        electrodes.tsv; False drops them.
    max_records : int | None
        Stop after this many recordings (a dry run).
    files : collection of str | None
        The dataset's recordings, as paths inside it. A recording on disk that
        is not one of them is left out and listed in the summary
        (:func:`keep_listed`); None converts every recording under the
        subject folders.

    """
    recipe = get_recipe(recipe)
    if recipe.refuses(dataset):
        raise ValueError(
            f"{dataset} is a benchmark source of recipe {recipe.name!r} "
            "and must not be pretrained on"
        )
    root, out = Path(root), Path(out)
    if (out / "_DONE").exists():
        return json.loads((out / "summary.json").read_text())
    if out.exists():  # no _DONE: an interrupted run
        shutil.rmtree(out)
    t0 = time.time()
    bank = bank or Bank.from_braindecode()
    if local is None:
        local = is_local_id(dataset)
    if local:
        recs = discover_bids(root, subjects)
    else:
        recs = discover_records(root.parent, root.name, subjects)
    if not recs:
        raise ValueError(
            f"no recordings found for {dataset} subjects={subjects or 'all'} under {root}"
        )
    recs, skipped = keep_listed(recs, files)
    n_left_out = len(skipped)
    if files is not None and subjects:  # a listed recording the disk does not hold
        on_disk = {record_file(r) for r in recs}
        mine = tuple(f"sub-{s}/" for s in subjects)
        skipped += [
            dict(file=f, reason="not_on_disk", detail="")
            for f in sorted(files)
            if f.startswith(mine) and f not in on_disk
        ]
    if skipped:
        reasons = dict(Counter(s["reason"] for s in skipped))
        logger.warning(
            "%s: %d recording(s) left out: %s", dataset, len(skipped), reasons
        )
    if max_records:
        recs = recs[:max_records]
    logger.info("%s subjects=%s: %d recordings", dataset, subjects or "all", len(recs))

    writer = ZarrGroupWriter(out, dataset, bank, recipe, compress)
    dropped_names: Counter = Counter()
    aligned_names: Counter = Counter()
    sfreqs: Counter = Counter()
    n_typed_from_names = 0
    for k, rec in enumerate(recs, 1):
        file = record_file(rec)
        try:
            electrodes = record_electrodes(rec, root) if use_dataset_positions else None
            res = preprocess_recording(
                rec.raw, bank, recipe, n_jobs=n_jobs, electrodes=electrodes
            )
        except Exception as err:  # keep the shard going; the summary lists it
            res = Skipped("load_error", f"{err.__class__.__name__}: {err}"[:300])
        finally:
            release(rec)
        if isinstance(res, Skipped):
            skipped.append(dict(file=file, reason=res.reason, detail=res.detail))
            logger.info(
                "[%d/%d] skip %s: %s %s", k, len(recs), file, res.reason, res.detail
            )
            continue
        sfreqs[str(int(res.sfreq_native))] += 1
        dropped_names.update(res.dropped)
        aligned_names.update(res.aligned)
        n_typed_from_names += res.typed_from_names
        g = writer.append(res, file, record_meta(rec))
        logger.info(
            "[%d/%d] %s: %d ch, %.0f s, %d Hz -> group %d",
            k,
            len(recs),
            file,
            res.data.shape[1],
            res.data.shape[0] / recipe.sfreq,
            int(res.sfreq_native),
            g,
        )

    summary = dict(
        dataset=dataset,
        subjects=list(subjects) if subjects else None,
        offline=True,
        n_records_seen=len(recs) + n_left_out,
        n_records_typed_from_names=n_typed_from_names,
        skipped=skipped,
        skipped_by_reason=dict(Counter(s["reason"] for s in skipped)),
        channels_dropped_no_position=dict(dropped_names.most_common()),
        channels_aligned_from_dataset=dict(aligned_names.most_common()),
        sfreq_native_hist=dict(sfreqs),
        elapsed_s=round(time.time() - t0, 1),
    )
    writer.finalize(summary)
    return summary


# --------------------------------------------------------------------- report


@dataclass
class ShardReport:
    """What :meth:`EEGDashCorpus.convert` did with one shard."""

    shard: str
    dataset_id: str
    state: str = "problem"  # converted, done (already there), problem
    summary: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def row(self) -> dict[str, Any]:
        s = self.summary
        return {
            "shard": self.shard,
            "state": self.state,
            "recordings": s.get("n_records_seen", 0),
            "sessions": s.get("n_sessions", 0),
            "skipped": len(s.get("skipped", [])),
            "groups": s.get("n_big_recordings", 0),
            "samples": s.get("n_samples", 0),
            "hours": s.get("hours_written", 0.0),
        }


@dataclass
class ConvertReport:
    """The result of :meth:`EEGDashCorpus.convert`."""

    shards: list[ShardReport] = field(default_factory=list)
    seconds: float = 0.0
    action: str = "convert"

    @property
    def problems(self) -> list[str]:
        return [f"{s.shard}: {p}" for s in self.shards for p in s.problems]

    @property
    def ok(self) -> bool:
        """Every shard converted, now or before, and none without a session."""
        return all(s.ok for s in self.shards)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "ok": self.ok,
            "seconds": round(self.seconds, 1),
            "shards": [
                {
                    **s.row(),
                    "dataset_id": s.dataset_id,
                    "problems": s.problems,
                    "skipped_by_reason": s.summary.get("skipped_by_reason", {}),
                }
                for s in self.shards
            ],
        }

    def __str__(self) -> str:
        rows = [s.row() for s in self.shards]
        lines = [pd.DataFrame(rows).to_string(index=False)] if rows else []
        total = {k: sum(r[k] for r in rows) for k in ("sessions", "samples", "hours")}
        return "\n".join(
            [
                *lines,
                *(f"PROBLEM {p}" for p in self.problems),
                f"{self.action}: {len(rows)} shard(s), {total['sessions']} sessions, "
                f"{total['samples']} samples, {total['hours']:.1f} h; "
                f"{len(self.problems)} problem(s); {self.seconds:.0f} s",
            ]
        )


def shard_report(shard: Shard, state: str, summary: dict) -> ShardReport:
    """A :class:`ShardReport` from a part's summary; a part without a session is a problem."""
    rep = ShardReport(shard.name, shard.dataset_id, state, summary)
    if not summary.get("n_sessions"):
        why = summary.get("skipped_by_reason") or "no recording found"
        rep.problems.append(f"no session written: {why}")
    return rep


__all__ = [
    "Bank",
    "ConvertReport",
    "HELD_OUT",
    "Recipe",
    "RECIPES",
    "Shard",
    "ShardReport",
    "convert_shard",
    "get_recipe",
    "is_local_id",
    "keep_listed",
    "plan_shards",
    "preprocess_recording",
    "sample_offsets",
]
