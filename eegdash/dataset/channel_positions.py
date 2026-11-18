"""Positions for iEEG channels whose names are not rows of electrodes.tsv.

A recording's channels are often named differently from its contacts:

- an EDF type word: ``POL A3`` is the contact ``A3`` (on004551);
- a reference suffix: ``EEG A1-Ref`` is contact ``A1`` against a reference
  that is not a contact (on004551);
- a bipolar pair: ``RAFA1-RAFA2`` (on004789, on004809 ``acq-bipolar``) or
  ``POL B10-B18`` (on006910, on006914, on005545) is the difference of two
  contacts.

MNE-BIDS places only channels named exactly like an electrodes.tsv row, so
these channels had no position although both contacts are positioned.

:func:`resolve_channel_positions` is the one rule the digest (stored
montages, ``scripts/ingestions/_montage.py``) and the runtime
(:func:`fill_missing_positions`) share:

1. A channel named exactly like a contact is that contact.
2. Otherwise a name names one contact by lookup only: case and white space
   are ignored, and a leading ``POL``/``EEG`` type word may be dropped. Two
   contacts that match the same name this way make it ambiguous: no position.
3. ``<contact>-Ref`` is that contact, when ``Ref`` is not itself a contact.
4. A pair ``<A>-<B>`` sits at the midpoint of A and B. Every ``-`` is tried
   as the separator; exactly one split must name two distinct contacts, both
   positioned. An unknown, ``n/a`` or ambiguous endpoint, or several possible
   splits, leave the channel unpositioned.

The midpoint is in the electrodes' own coordinate system and units, and the
distance between the two contacts is kept with it. Nothing is written back
to electrodes.tsv, no contact or anatomy is invented, channel names are not
changed and the signal is not re-referenced.
"""

from __future__ import annotations

import logging
import math
import re
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: ``position_source`` of a channel named exactly like a contact.
MEASURED = "measured"
#: ``position_source`` of a channel that names one contact under the lookup rules.
ELECTRODE_ALIAS = "electrode_alias"
#: ``position_source`` of a channel placed at the midpoint of two contacts.
BIPOLAR_MIDPOINT = "bipolar_midpoint"

# EDF signal labels may lead with a type word (EDF+ spec: "EEG Fpz-Cz"); Nihon
# Kohden exports write "POL" (on004551: "POL A3", "EEG A1-Ref").
_TYPE_PREFIX = re.compile(r"(?:POL|EEG)\s+", re.IGNORECASE)
_AMBIGUOUS = object()

Point = tuple[float, float, float]


@dataclass(frozen=True)
class ChannelPosition:
    """Where a channel sits, and which contacts that position comes from."""

    xyz: Point
    source: str
    contacts: tuple[str, ...]
    contact_distance: float | None = None

    def provenance(self) -> dict[str, Any]:
        """``position_source``, ``contacts`` and, for a pair, ``contact_distance``."""
        out: dict[str, Any] = {
            "position_source": self.source,
            "contacts": list(self.contacts),
        }
        if self.contact_distance is not None:
            out["contact_distance"] = self.contact_distance
        return out


def _key(name: str) -> str:
    return "".join(name.split()).casefold()


def _finite(xyz: Any) -> Point | None:
    if xyz is None:
        return None
    try:
        point = tuple(float(v) for v in xyz)
    except (TypeError, ValueError):
        return None
    if len(point) != 3 or not all(math.isfinite(v) for v in point):
        return None
    return point  # type: ignore[return-value]


class _Contacts:
    """Lookup over every electrodes.tsv name, positioned or not."""

    def __init__(self, electrodes: Mapping[str, Any]):
        self.xyz = {str(k): _finite(v) for k, v in electrodes.items()}
        self.by_key: dict[str, set[str]] = defaultdict(set)
        for name in self.xyz:
            self.by_key[_key(name)].add(name)

    def find(self, token: str) -> Any:
        """The contact ``token`` names: a name, ``None``, or ``_AMBIGUOUS``."""
        token = token.strip()
        if not token:
            return None
        if token in self.xyz:
            return token
        found = set(self.by_key.get(_key(token), ()))
        if m := _TYPE_PREFIX.match(token):
            rest = token[m.end() :].strip()
            if rest:
                found |= self.by_key.get(_key(rest), set())
        if len(found) > 1:
            return _AMBIGUOUS
        return next(iter(found)) if found else None


def _pair_splits(
    channel: str, contacts: _Contacts
) -> list[tuple[str, str | None]] | None:
    """Every ``-`` split naming contacts on both sides; ``None`` if one is ambiguous."""
    splits: list[tuple[str, str | None]] = []
    for i, char in enumerate(channel):
        if char != "-":
            continue
        left, right = channel[:i], channel[i + 1 :]
        a, b = contacts.find(left), contacts.find(right)
        if a is _AMBIGUOUS or b is _AMBIGUOUS:
            return None
        if a is None:
            continue
        if b is not None:
            splits.append((a, b))
        elif right.strip().casefold() == "ref":
            splits.append((a, None))  # "<contact>-Ref": the contact, monopolar
    return splits


def _resolve(channel: str, contacts: _Contacts) -> ChannelPosition | None:
    if channel in contacts.xyz:  # exact electrodes.tsv name: always that contact
        xyz = contacts.xyz[channel]
        return None if xyz is None else ChannelPosition(xyz, MEASURED, (channel,))

    found = contacts.find(channel)
    splits = _pair_splits(channel, contacts)
    if found is _AMBIGUOUS or splits is None:
        return None
    if found is not None:
        if splits:  # one contact by lookup and a pair as well: ambiguous
            return None
        xyz = contacts.xyz[found]
        return None if xyz is None else ChannelPosition(xyz, ELECTRODE_ALIAS, (found,))

    if len(splits) != 1:
        return None
    a, b = splits[0]
    if b is None:
        xyz = contacts.xyz[a]
        return None if xyz is None else ChannelPosition(xyz, ELECTRODE_ALIAS, (a,))
    pa, pb = contacts.xyz[a], contacts.xyz[b]
    if a == b or pa is None or pb is None:
        return None
    mid = tuple((u + v) / 2 for u, v in zip(pa, pb))
    return ChannelPosition(mid, BIPOLAR_MIDPOINT, (a, b), math.dist(pa, pb))  # type: ignore[arg-type]


def resolve_channel_positions(
    ch_names: Sequence[str], electrodes: Mapping[str, Any]
) -> dict[str, ChannelPosition]:
    """Position of every channel the module rules place, keyed by channel name.

    Parameters
    ----------
    ch_names : sequence of str
        Channel names as the data file names them.
    electrodes : mapping
        Every electrodes.tsv name to its ``(x, y, z)``, or to ``None`` for an
        ``n/a`` row. Unpositioned names still count when telling whether a
        name is ambiguous.

    Returns
    -------
    dict
        Channel name to :class:`ChannelPosition`; channels left out have no
        position.

    """
    contacts = _Contacts(electrodes)
    out: dict[str, ChannelPosition] = {}
    for channel in ch_names:
        position = _resolve(str(channel), contacts)
        if position is not None:
            out[str(channel)] = position
    return out


def derived_positions(
    ch_names: Sequence[str], electrodes: Mapping[str, Any]
) -> dict[str, ChannelPosition]:
    """:func:`resolve_channel_positions` without the exactly named (measured) channels."""
    return {
        name: pos
        for name, pos in resolve_channel_positions(ch_names, electrodes).items()
        if pos.source != MEASURED
    }


def read_electrode_table(
    path: Path, encoding: str | None = None
) -> dict[str, Point | None]:
    """Every name of an electrodes.tsv with its ``(x, y, z)``, ``None`` when ``n/a``.

    A name listed twice has no single position, so it maps to ``None``. A file
    without ``name``/``x``/``y``/``z`` columns gives ``{}``.
    """
    df = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, encoding=encoding
    )
    if not {"name", "x", "y", "z"} <= set(df.columns):
        return {}
    table: dict[str, Point | None] = {}
    for name, x, y, z in zip(df["name"], df["x"], df["y"], df["z"]):
        name = str(name).strip()
        if not name:
            continue
        xyz = _finite(pd.to_numeric(pd.Series([x, y, z]), errors="coerce"))
        table[name] = None if name in table else xyz
    return table


def _has_position(ch: Mapping[str, Any]) -> bool:
    loc = np.asarray(ch["loc"][:3], dtype=float)
    return bool(np.all(np.isfinite(loc)) and np.any(loc != 0))


def _contacts_as_mne_places_them(electrodes_tsv, coordsystem_json, datatype, names):
    """Positions MNE-BIDS gives the contacts, and their coordinate frame.

    The contacts are read onto a one-sample stand-in recording through the same
    MNE-BIDS reader ``read_raw_bids`` uses, so the units, the frame and any
    transform are the ones the recording's measured channels got.
    """
    import mne
    from mne_bids.read import _read_dig_bids

    from ..io import tolerant

    ch_type = "seeg" if datatype == "ieeg" else datatype
    stand_in = mne.io.RawArray(
        np.zeros((len(names), 1)), mne.create_info(names, 1.0, ch_type), verbose="ERROR"
    )
    with tolerant(), warnings.catch_warnings(), mne.utils.use_log_level("ERROR"):
        warnings.simplefilter("ignore")
        _read_dig_bids(electrodes_tsv, coordsystem_json, datatype, stand_in)
    montage = stand_in.get_montage()
    if montage is None:
        return {}, None
    placed = {
        ch["ch_name"]: (
            tuple(float(v) for v in ch["loc"][:3]) if _has_position(ch) else None
        )
        for ch in stand_in.info["chs"]
    }
    pos = montage.get_positions()
    frame = {
        "chs": stand_in.info["chs"][0]["coord_frame"],
        "dig": pos["coord_frame"],
        **{k: pos[k] for k in ("nasion", "lpa", "rpa") if pos.get(k) is not None},
    }
    return placed, frame


def fill_missing_positions(
    raw, electrodes_tsv: Path, coordsystem_json: Path, datatype: str = "ieeg"
) -> dict[str, dict[str, Any]]:
    """Give channels without a position the one the module rules derive.

    Only channels with no position are touched; a measured position is never
    replaced. The contacts are placed as MNE-BIDS places them (same frame,
    units and transform). The provenance of every filled channel goes to
    ``raw.info["temp"]["derived_positions"]``.

    Returns
    -------
    dict
        Channel name to its provenance, for the channels this filled.

    """
    import mne
    from mne.io.constants import FIFF

    missing = [ch["ch_name"] for ch in raw.info["chs"] if not _has_position(ch)]
    if not missing:
        return {}
    encoding = None
    try:
        from ..io.bids import tsv_encoding

        encoding = tsv_encoding(electrodes_tsv)
    except (OSError, ImportError):
        pass
    names = list(read_electrode_table(electrodes_tsv, encoding=encoding))
    if not names:
        return {}
    derived_candidates = derived_positions(
        missing, dict.fromkeys(names, (1.0, 1.0, 1.0))
    )
    if not derived_candidates:
        return {}  # no channel names a contact by alias or pair: skip the read
    placed, frame = _contacts_as_mne_places_them(
        electrodes_tsv, coordsystem_json, datatype, names
    )
    derived = derived_positions(missing, placed) if placed else {}
    if not derived:
        return {}
    filled: dict[str, dict[str, Any]] = {}
    for ch in raw.info["chs"]:
        if (pos := derived.get(ch["ch_name"])) is None:
            continue
        ch["loc"][:3] = pos.xyz
        ch["coord_frame"] = frame["chs"]
        filled[ch["ch_name"]] = pos.provenance()
    # Digitization points, as set_montage writes them, so get_montage() shows them.
    fiducials = {k: frame[k] for k in ("nasion", "lpa", "rpa") if k in frame}
    points = mne.channels.make_dig_montage(
        ch_pos={name: np.asarray(derived[name].xyz) for name in filled},
        coord_frame=frame["dig"],
        **({} if raw.info["dig"] else fiducials),
    ).dig
    with raw.info._unlock():
        dig = list(raw.info["dig"] or [])
        ident = max(
            (d["ident"] for d in dig if d["kind"] == FIFF.FIFFV_POINT_EEG), default=0
        )
        for d in points:
            if d["kind"] == FIFF.FIFFV_POINT_EEG:
                ident += 1
                d["ident"] = ident
            dig.append(d)
        raw.info["dig"] = dig
        temp = raw.info.get("temp")
        temp = dict(temp) if isinstance(temp, dict) else {}
        temp["derived_positions"] = {**temp.get("derived_positions", {}), **filled}
        raw.info["temp"] = temp
    logger.info(
        "%d channels placed from their contacts (%s).",
        len(filled),
        ", ".join(sorted({p["position_source"] for p in filled.values()})),
    )
    return filled
