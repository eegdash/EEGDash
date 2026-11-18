"""Channel positions of an fNIRS recording from its dataset's montage derivative.

MNE-BIDS 0.20 reads no ``optodes.tsv`` for NIRS (``read_raw_bids`` applies
electrodes.tsv for MEG, EEG and iEEG only), so MNE has only the SNIRF probe's
positions, in whatever frame the file uses. The fNIRS deposits on NEMAR carry a
derivative with every montage mapped to one space (fsaverage MRI, mm)::

    derivatives/fnirs-montage/
        montage_recordings.tsv              recording -> montage_id (written yes/no)
        montages.tsv                        montage_id -> tier, method, channels_table
        montages/<montage_id>_channels.tsv  name, source, detector, x, y, z, ...
        montages/channels.json              column descriptions and units

:func:`read_fnirs_montage` returns a recording's channel table aligned to
``raw.ch_names``. It only reads: ``raw.info`` is not touched, MNE's channel
locations stay the SNIRF probe's, in the probe's frame, and the derivative's
positions stay in the derivative's space.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger("eegdash")
_NAME = re.compile(r"^S(\d+)_D(\d+)\s+(\S+)$", re.IGNORECASE)
_XYZ = ["x", "y", "z"]


def _wavelength(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return int(value) if math.isfinite(value) else None


def _key(name, wavelength=None):
    """Matching key of an fNIRS channel name: ``S1_D1 760``, ``S1_D1 hbo``.

    MNE names a channel with the integer part of its wavelength (``int(w)``); a
    known ``wavelength`` (the table's ``wavelength_nominal``) is used the same
    way, so a table that rounded the name still matches. Other names stay as is.
    """
    name = str(name).strip()
    match = _NAME.match(name)
    if match is None:
        return name
    source, detector, kind = match.groups()
    nominal = _wavelength(kind)
    if nominal is not None and _wavelength(wavelength) is not None:
        nominal = _wavelength(wavelength)
    kind = kind.lower() if nominal is None else str(nominal)
    return f"S{int(source)}_D{int(detector)} {kind}"


def _tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _relpath(root, recording):
    path, root = Path(recording), Path(root)
    if path.is_absolute():
        try:
            path = path.relative_to(root)
        except ValueError:
            path = path.resolve().relative_to(root.resolve())
    return path.as_posix()


def read_fnirs_montage(root, recording, ch_names, *, derivative="fnirs-montage"):
    """Return the montage derivative's channel table for ``recording``, one row per channel.

    Parameters
    ----------
    root : path-like
        The BIDS dataset root, the folder holding ``derivatives/``.
    recording : path-like
        The recording's data file, relative to ``root`` (``record["bids_relpath"]``)
        or an absolute path inside it.
    ch_names : list of str
        The names to align to, usually ``raw.ch_names``. Channels are matched by
        MNE's fNIRS name, ``S<i>_D<j> <wavelength|hbo|hbr>``, ignoring case.
    derivative : str
        The derivative's folder name under ``derivatives/``.

    Returns
    -------
    pandas.DataFrame | None
        Indexed by ``ch_names``, in their order, with the table's columns;
        ``x``, ``y``, ``z`` (channel midpoints) are floats. A channel the table
        does not have, or has as ``n/a``, gets NaN. ``attrs`` holds
        ``montage_id``, ``tier``, ``heuristic``, ``method``, ``units`` and
        ``table`` (the file read). None when the derivative has no montage
        written for the recording.

    """
    tree = Path(root) / "derivatives" / derivative
    index = tree / "montage_recordings.tsv"
    if not index.is_file():
        return None
    rel = _relpath(root, recording)
    rows = _tsv(index)
    rows = rows[rows["recording"].str.strip().str.removeprefix("./") == rel]
    if rows.empty or rows.iloc[0].get("written", "yes").strip().lower() != "yes":
        return None
    montage_id = rows.iloc[0]["montage_id"].strip()
    montage = {}
    if (tree / "montages.tsv").is_file():
        montages = _tsv(tree / "montages.tsv")
        found = montages[montages["montage_id"].str.strip() == montage_id]
        montage = found.iloc[0].to_dict() if len(found) else {}
    table_rel = montage.get("channels_table", "").strip()
    if table_rel in ("", "n/a"):
        table_rel = f"montages/{montage_id}_channels.tsv"
    table_path = tree / table_rel
    table = _tsv(table_path)
    wavelengths = table.get("wavelength_nominal", pd.Series([None] * len(table)))
    keys = pd.Series(
        [_key(n, w) for n, w in zip(table["name"], wavelengths)], index=table.index
    )
    if keys.duplicated().any():
        logger.warning(
            "%s: %d repeated channel names; the first row of each is used.",
            table_path.name,
            int(keys.duplicated().sum()),
        )
        table, keys = table[~keys.duplicated()], keys[~keys.duplicated()]
    table = table.copy()
    table.index = keys.to_numpy()
    out = table.reindex([_key(ch) for ch in ch_names])
    out.index = pd.Index(list(ch_names), name="ch_name")
    for col in _XYZ:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")  # n/a -> NaN
    if set(_XYZ) <= set(out) and (missing := int(out[_XYZ].isna().all(axis=1).sum())):
        logger.warning(
            "%s: no position for %d of %d channels of %s.",
            table_path.name,
            missing,
            len(out),
            rel,
        )
    units = None
    sidecar = table_path.with_name("channels.json")
    if sidecar.is_file():
        units = (
            json.loads(sidecar.read_text(encoding="utf-8")).get("x", {}).get("Units")
        )
    out.attrs = {
        "montage_id": montage_id,
        "tier": montage.get("tier"),
        "heuristic": montage.get("heuristic"),
        "method": montage.get("method"),
        "units": units,
        "table": str(table_path),
    }
    return out
