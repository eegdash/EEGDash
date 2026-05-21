"""Sensor-layout extraction at digestion time.

One module, four modality-specific extractors, plus a dispatcher. Each
extractor returns ``(hash, layout_doc)`` or ``None`` with a consistent
document shape so the downstream Mongo collection stays polymorphic.

Supported BIDS datatypes
------------------------

- **EEG** (``datatype == "eeg"``): parses ``*_electrodes.tsv`` +
  ``*_coordsystem.json``. Positions are on a scalp sphere. Renders in
  the existing 2D azimuthal-equidistant viewer.

- **iEEG** (``datatype == "ieeg"``): same ``*_electrodes.tsv`` file,
  but positions live in MRI / ACPC / MNI brain space. Hash is stored so
  cross-subject identical grids still collapse, but the 2D sphere viewer
  can't render them — a future glass-brain viewer will consume the same
  documents.

- **MEG** (``datatype == "meg"``): sensor positions live inside the raw
  file header (FIF / CTF `.ds` / KIT). We read the header only (no data
  samples) via ``mne.io.read_info`` and pull ``info['chs'][i]['loc'][:3]``.
  Projection works with the same spherical viewer because MEG helmets
  approximate a sphere at the sensor layer.

- **fNIRS** (``datatype == "nirs"``): ``*_optodes.tsv`` defines source
  and detector positions (not electrodes). ``*_channels.tsv`` defines
  which source+detector pairs constitute a measurement channel. We store
  both, keyed on the combined hash.

Shared document shape
---------------------

All four extractors return a dict with these keys::

    {
      "hash": str,                       # 16-char sha1 prefix, per-modality
      "modality": "eeg"|"ieeg"|"meg"|"nirs",
      "n_sensors": int,
      "space_declared": str | None,      # raw value from BIDS metadata
      "units_declared": str | None,
      "sensors": list[dict],             # [{"name","x","y","z", ...}]
    }

Hashing convention
------------------

SHA1 prefix (16 hex chars) over a canonical, sorted representation of
``(modality, name, x_mm, y_mm, z_mm, type?)`` tuples. Modality is part
of the hash input so an EEG cap can't accidentally collide with a MEG
helmet in the same hash space.

MNE dependency
--------------

EEG / iEEG / fNIRS extractors are pure — stdlib + pandas only. The MEG
extractor requires ``mne`` because sensor positions are only accessible
via the FIF reader. Import is done lazily inside the function so the
rest of the module stays MNE-free.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("digest.montage")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_COORDSYS_PREFIXES = ("EEG", "iEEG", "MEG", "EMG", "NIRS")


def _round_mm(v: float) -> int:
    return round(v * 1000)


def _hash_sensors(modality: str, sensors: list[dict[str, Any]]) -> str:
    """Stable short hash over a canonical (modality, name, mm-rounded coords) form.

    Including ``modality`` in the hash prevents a MEG helmet and an EEG
    cap from aliasing on name + position (vanishingly unlikely, but free
    to prevent).
    """
    canonical = [
        modality,
        *sorted(
            (
                s.get("name", ""),
                _round_mm(s.get("x", 0.0)),
                _round_mm(s.get("y", 0.0)),
                _round_mm(s.get("z", 0.0)),
                s.get("type", ""),
            )
            for s in sensors
        ),
    ]
    payload = repr(canonical).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _parse_coordsystem_json(path: Path) -> tuple[str | None, str | None]:
    """(space, units) from a BIDS coordsystem.json; both None if missing."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None
    for prefix in _COORDSYS_PREFIXES:
        space = doc.get(f"{prefix}CoordinateSystem")
        units = doc.get(f"{prefix}CoordinateUnits")
        if space or units:
            return (
                str(space) if space else None,
                str(units).lower() if units else None,
            )
    return None, None


def _walk_up_find(
    data_file: Path,
    bids_root: Path,
    pattern: str,
) -> Path | None:
    """BIDS-inheritance lookup: walk up from data_file's parent to bids_root,
    returning the first match of ``pattern`` (alphabetically).
    """
    try:
        current = data_file.parent.resolve()
        root = bids_root.resolve()
    except OSError:
        return None
    if not str(current).startswith(str(root)):
        return None
    while True:
        matches = sorted(current.glob(pattern))
        if matches:
            return matches[0]
        if current == root:
            return None
        current = current.parent


def _companion_coords_for(tsv: Path, bids_suffix: str = "_electrodes.tsv") -> Path:
    """Given sub-XX_…_electrodes.tsv, return sub-XX_…_coordsystem.json."""
    name = tsv.name
    if name.endswith(bids_suffix):
        coord_name = name[: -len(bids_suffix)] + "_coordsystem.json"
    else:
        coord_name = re.sub(re.escape(bids_suffix) + r"$", "_coordsystem.json", name)
    return tsv.with_name(coord_name)


def _parse_sensor_tsv(
    path: Path,
    required: Iterable[str] = ("name", "x", "y", "z"),
    extras: Iterable[str] = ("type", "material", "impedance"),
) -> list[dict[str, Any]]:
    """Parse a BIDS sensor TSV (_electrodes.tsv or _optodes.tsv).

    Drops rows where any required column is ``n/a`` / NaN. Preserves the
    listed ``extras`` when non-empty.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, na_values=["n/a", "N/A", ""])
    required = list(required)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")

    for col in ("x", "y", "z"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[c for c in required if c in df.columns])

    rows: list[dict[str, Any]] = []
    for r in df.itertuples(index=False):
        entry: dict[str, Any] = {}
        for col in required:
            val = getattr(r, col)
            if col in ("x", "y", "z"):
                entry[col] = float(val)
            else:
                entry[col] = str(val).strip()
        if not entry.get("name"):
            continue
        for col in extras:
            val = getattr(r, col, None)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            s = str(val).strip()
            if s and s.lower() not in {"n/a", "nan"}:
                entry[col] = s
        rows.append(entry)
    return rows


# ---------------------------------------------------------------------------
# Generic TSV-based layout extractor
# ---------------------------------------------------------------------------
#
# BIDS scalp EEG, iEEG (brain space), EMG (body landmarks) and fNIRS
# (optodes) all share the same sidecar pattern: a sensor TSV next to (or
# up-tree from) the data file, plus an optional coordsystem.json. The
# per-modality extractors below differ only in: TSV glob, required /
# extras columns, minimum-sensor threshold, and modality tag. MEG is the
# exception — it needs raw-header reading via MNE, so it gets its own
# implementation below.


def _extract_tsv_layout(
    data_file: Path,
    bids_root: Path,
    *,
    modality: str,
    tsv_pattern: str = "*_electrodes.tsv",
    extras: tuple[str, ...] = ("type", "material", "impedance"),
    min_sensors: int = 4,
    coord_suffix: str = "_electrodes.tsv",
) -> tuple[str, dict[str, Any]] | None:
    """Shared pipeline for BIDS TSV-based sensor layouts.

    Returns ``(hash, layout_doc)`` or ``None`` if the TSV is missing,
    unparsable, or under ``min_sensors``. The returned doc has the
    canonical shape documented in this module's top-level docstring.
    """
    tsv = _walk_up_find(data_file, bids_root, tsv_pattern)
    if tsv is None:
        return None
    try:
        sensors = _parse_sensor_tsv(tsv, extras=extras)
    except (OSError, ValueError, KeyError, UnicodeDecodeError) as exc:
        # _parse_sensor_tsv hits filesystem and pandas. ValueError covers
        # NaN/Inf coord rejection; KeyError covers missing required
        # columns; UnicodeDecodeError covers non-UTF8 TSV files.
        LOGGER.warning("[layout.%s] parse error on %s: %s", modality, tsv, exc)
        return None
    if len(sensors) < min_sensors:
        LOGGER.info(
            "[layout.%s] %s: only %d sensors with finite coords; skipping",
            modality,
            tsv.name,
            len(sensors),
        )
        return None
    space, units = _parse_coordsystem_json(
        _companion_coords_for(tsv, bids_suffix=coord_suffix)
    )
    h = _hash_sensors(modality, sensors)
    return h, {
        "hash": h,
        "modality": modality,
        "n_sensors": len(sensors),
        "space_declared": space,
        "units_declared": units,
        "sensors": sensors,
    }


# ---------------------------------------------------------------------------
# EEG — scalp electrodes on a sphere
# ---------------------------------------------------------------------------


# extract_eeg_layout removed (Phase 8 P2.1). EEG's "TSV first, then
# template-match fallback" logic is now expressed by
# ``_TSV_MODALITY_CONFIGS["eeg"]`` having ``template_fallback=True``;
# :func:`_extract_layout_for_config` runs the cascade.


@dataclass(frozen=True)
class _ModalityConfig:
    """Per-modality parametrization of :func:`_extract_tsv_layout`.

    The 4 TSV-based modalities (EEG, iEEG, EMG, fNIRS) differ only in
    these knobs; collecting them as a dataclass makes "what varies per
    modality" a single named place instead of 4 functions with bespoke
    kwarg patterns.

    Per ROADMAP P2.1 — the implicit Seam between the 4 TSV-based
    extractors becomes explicit: each modality is a named config, the
    dispatcher reads from a dict, ``_extract_tsv_layout`` is called once.
    """

    modality: str
    tsv_pattern: str = "*_electrodes.tsv"
    extras: tuple[str, ...] = ("type", "material", "impedance")
    min_sensors: int = 4
    coord_suffix: str = "_electrodes.tsv"
    # EEG-only: when *_electrodes.tsv is absent, fall back to
    # matching channel-name list against MNE's standard montages.
    template_fallback: bool = False


_TSV_MODALITY_CONFIGS: dict[str, _ModalityConfig] = {
    "eeg": _ModalityConfig(
        modality="eeg",
        template_fallback=True,
    ),
    "ieeg": _ModalityConfig(
        modality="ieeg",
        extras=("type", "hemisphere", "material", "impedance", "group", "size"),
        min_sensors=1,
    ),
    "emg": _ModalityConfig(
        modality="emg",
        extras=("type", "material", "impedance", "coordinate_system", "group"),
        min_sensors=2,
    ),
    "nirs": _ModalityConfig(
        modality="nirs",
        tsv_pattern="*_optodes.tsv",
        extras=("type", "template_x", "template_y", "template_z"),
        min_sensors=2,
        coord_suffix="_optodes.tsv",
    ),
}


def _extract_layout_for_config(
    data_file: Path, bids_root: Path, config: _ModalityConfig
) -> tuple[str, dict[str, Any]] | None:
    """Run the TSV-based pipeline for one ``_ModalityConfig``.

    Phase 8 P2.1 — replaces 4 thin ``extract_<modality>_layout``
    wrapper functions with one config-driven helper. The dispatcher
    in :func:`extract_layout` selects the right config by datatype
    name and calls this helper.

    For configs with ``template_fallback=True`` (EEG only today), the
    helper tries ``_extract_tsv_layout`` first, then falls back to
    template-matching when no ``*_electrodes.tsv`` is published.
    """
    direct = _extract_tsv_layout(
        data_file,
        bids_root,
        modality=config.modality,
        tsv_pattern=config.tsv_pattern,
        extras=config.extras,
        min_sensors=config.min_sensors,
        coord_suffix=config.coord_suffix,
    )
    if direct is not None:
        return direct
    if config.template_fallback:
        return _extract_template_from_channels(
            data_file, bids_root, modality=config.modality
        )
    return None


# ---------------------------------------------------------------------------
# Template-matched fallback (EEG only, channels.tsv → MNE canonical montage)
# ---------------------------------------------------------------------------
#
# Many public datasets publish ``*_channels.tsv`` (channel name + type +
# sampling frequency) without an ``*_electrodes.tsv`` (3D positions).
# When the cap is a standard one (10-20 / 10-10 / BioSemi / EasyCap /
# HydroCel / …), MNE already ships the canonical positions — we just
# need to match the channel-name list to one of those templates.
#
# Outcome:
#  * Returns a sensor layout whose positions come from MNE's built-in
#    ``make_standard_montage(...)`` call, filtered to the subset of
#    channels actually present in this dataset.
#  * The returned doc's ``source`` is ``"template-matched"`` (vs the
#    default ``"subject-tsv"`` implied by ``_extract_tsv_layout``) so
#    the viewer can flag "canonical, not subject-fitted" in the UI
#    later if we want.


# MNE lists ~28 built-in montages; we materialise them lazily on the
# first call and cache the name → {name_uppercase: (x, y, z)} mapping so
# repeated template scoring stays cheap. Key is the MNE montage string;
# value is a dict (upper-case channel name → tuple[float, float, float]
# in metres).
_MNE_TEMPLATE_CACHE: dict[str, dict[str, tuple[float, float, float]]] | None = None
_MNE_TEMPLATE_KEYSETS: dict[str, frozenset[str]] | None = None
_MNE_TEMPLATE_CACHE_LOCK = threading.Lock()


_ELECTRODE_EXPLORER_MONTAGES_JSON = (
    Path(__file__).resolve().parents[2]
    / "eeg_eletrodes"
    / "electrode-explorer"
    / "montages.json"
)


def _load_mne_templates() -> dict[str, dict[str, tuple[float, float, float]]]:
    """Merge two template pools:

    1. Everything MNE ships via ``get_builtin_montages()`` (≈28 entries).
    2. The electrode-explorer viewer's pre-built catalog
       (``montages.json``): ANT Waveguard 32/64/128/256, BrainProducts
       ActiCap 65/68/97/128, Neuroscan Quik-cap 64/68/123/128, EGI
       classic GSN 64v1/64v2/128/256, EGI infant/adult average nets,
       BESA 254, Wearable Sensing DSI-24, BioSemi label variants, etc.
       — ~40 additional templates imported from Brainstorm / DIPFIT /
       EEGLAB.

    MNE is the authoritative source when a name collides; our pool only
    adds templates MNE doesn't ship.
    """
    global _MNE_TEMPLATE_CACHE, _MNE_TEMPLATE_KEYSETS
    if _MNE_TEMPLATE_CACHE is not None:
        return _MNE_TEMPLATE_CACHE
    with _MNE_TEMPLATE_CACHE_LOCK:
        if _MNE_TEMPLATE_CACHE is not None:
            return _MNE_TEMPLATE_CACHE
        cache: dict[str, dict[str, tuple[float, float, float]]] = {}

        # Pool 1 — MNE built-ins. These land keyed by MNE's canonical name.
        try:
            import mne  # type: ignore

            for name in mne.channels.get_builtin_montages():
                try:
                    m = mne.channels.make_standard_montage(name)
                    cache[name] = {
                        k.upper(): (float(v[0]), float(v[1]), float(v[2]))
                        for k, v in m.get_positions()["ch_pos"].items()
                    }
                except (ValueError, KeyError, TypeError, AttributeError):
                    # MNE can raise on a misnamed standard montage or
                    # an unexpected positions dict shape. Skip that
                    # template; the other 50+ will still populate.
                    continue
        except (ImportError, AttributeError) as exc:
            # ImportError: MNE not installed. AttributeError: a future
            # MNE refactor removed get_builtin_montages.
            LOGGER.warning("[template] MNE import failed (%s); pool 1 empty", exc)

        # Pool 2 — the electrode-explorer extras. montages.json stores each
        # position in meters already (xyz is centred on the fitted head
        # sphere), identical to what make_standard_montage produces.
        try:
            raw = _ELECTRODE_EXPLORER_MONTAGES_JSON.read_text()
        except FileNotFoundError:
            raw = None
        except OSError as exc:
            LOGGER.warning("[template] electrode-explorer catalog unreadable (%s)", exc)
            raw = None
        if raw is not None:
            try:
                doc = json.loads(raw)
                for key, entry in doc.items():
                    if key == "_meta" or not isinstance(entry, dict):
                        continue
                    if key in cache:  # MNE already provided this exact template
                        continue
                    sensors = entry.get("electrodes") or []
                    if not sensors:
                        continue
                    cache[key] = {
                        s["name"].upper(): (float(s["x"]), float(s["y"]), float(s["z"]))
                        for s in sensors
                        if s.get("name") and "x" in s and "y" in s and "z" in s
                    }
            except (ValueError, TypeError) as exc:
                LOGGER.warning(
                    "[template] electrode-explorer catalog parse failed (%s)", exc
                )

        _MNE_TEMPLATE_CACHE = cache
        _MNE_TEMPLATE_KEYSETS = {name: frozenset(pos) for name, pos in cache.items()}
        LOGGER.info(
            "[template] loaded %d templates (MNE + electrode-explorer)", len(cache)
        )
        return cache


_CHANNELS_TSV_TYPE_EEG = {"EEG", "EEGREF", "REF"}
_CHANNELS_TSV_TYPE_SKIP = {
    "MISC",
    "TRIG",
    "STIM",
    "STATUS",
    "EOG",
    "VEOG",
    "HEOG",
    "EMG",
    "ECG",
    "EKG",
    "GSR",
    "RESP",
    "TEMP",
    "PPG",
    "AUDIO",
    "PHOTO",
    "EYEGAZE",
    "PUPIL",
    "OTHER",
    "BAD",
    "N/A",
    "DC",
}


def _parse_channels_tsv_for_eeg(path: Path) -> list[str]:
    """Return EEG channel names from a BIDS ``_channels.tsv``.

    Rules:
      * If the TSV has a ``type`` column, only keep rows whose type is in
        ``EEG`` / ``EEGREF`` / ``REF`` (or empty — some datasets omit the
        type). Explicit non-EEG types (EOG, TRIG, etc.) are dropped.
      * If there's no ``type`` column, keep every row (the caller may
        still drop the dataset when no canonical match is found).
    """
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, na_values=["n/a", "N/A", ""])
    except (
        OSError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
        ValueError,
    ):
        # channels.tsv absent, malformed, non-UTF8, or empty. Empty
        # channel list lets the caller fall back to inferring from the
        # binary header.
        return []
    if "name" not in df.columns:
        return []
    has_type = "type" in df.columns
    out: list[str] = []
    for r in df.itertuples(index=False):
        name = str(getattr(r, "name", "") or "").strip()
        if not name:
            continue
        if has_type:
            typ = str(getattr(r, "type", "") or "").strip().upper()
            if typ in _CHANNELS_TSV_TYPE_SKIP:
                continue
            # Empty type is accepted — many datasets omit it.
            if typ and typ not in _CHANNELS_TSV_TYPE_EEG:
                continue
        out.append(name)
    return out


def _score_template_match(
    channels: list[str],
    templates: dict[str, dict[str, tuple[float, float, float]]],
    *,
    min_hits: int = 4,
    min_ratio: float = 0.8,
) -> tuple[str, dict[str, tuple[float, float, float]]] | None:
    """Pick the MNE built-in montage whose name set best covers the dataset's
    channels. Ties broken by picking the *smallest* template (most specific
    fit: a 64-channel dataset should map to ``biosemi64``, not to the
    343-channel ``standard_1005`` superset).
    """
    if not channels:
        return None
    channels_up = {c.upper() for c in channels}
    # Keysets are pre-frozen at template-load time (see _load_mne_templates).
    keysets = _MNE_TEMPLATE_KEYSETS or {}
    best: tuple[int, int, str, dict[str, tuple[float, float, float]]] | None = None
    for tname, tpos in templates.items():
        hits = channels_up & keysets.get(tname, frozenset(tpos))
        if len(hits) < min_hits:
            continue
        ratio = len(hits) / len(channels_up)
        if ratio < min_ratio:
            continue
        # Rank primarily on hit count; tiebreak on smaller template size.
        # `best` stores (hits, -template_size, name, matched_subset) so
        # max(...) picks highest hits then smallest template.
        matched = {k: tpos[k] for k in hits}
        key = (len(hits), -len(tpos))
        if best is None or key > (best[0], best[1]):
            best = (len(hits), -len(tpos), tname, matched)
    if best is None:
        return None
    return best[2], best[3]


def _extract_template_from_channels(
    data_file: Path, bids_root: Path, *, modality: str = "eeg"
) -> tuple[str, dict[str, Any]] | None:
    """Produce a synthetic electrode layout from channels.tsv + a standard
    MNE montage. Returns ``None`` when no ``*_channels.tsv`` exists, when
    fewer than 4 EEG channel names are present, or when no MNE template
    matches ≥80% of the channel list.
    """
    channels_tsv = _walk_up_find(data_file, bids_root, pattern="*_channels.tsv")
    if channels_tsv is None:
        return None
    names = _parse_channels_tsv_for_eeg(channels_tsv)
    if len(names) < 4:
        return None
    templates = _load_mne_templates()
    if not templates:
        return None
    match = _score_template_match(names, templates)
    if match is None:
        LOGGER.info(
            "[template.%s] %s: no canonical match for %d channels; skipping fallback",
            modality,
            channels_tsv.name,
            len(names),
        )
        return None
    template_name, matched_positions = match
    # Emit sensors in the loader-native shape: mm, {name, x, y, z, type}.
    # Use MNE's canonical casing (Fp1, not FP1) — that's what the viewer's
    # label-based region heuristics already key on.
    names_up = {n.upper(): n for n in names}
    sensors: list[dict[str, Any]] = []
    for _, canonical_name, pos in sorted(
        ((k, k, v) for k, v in matched_positions.items()),
        key=lambda t: t[0],
    ):
        # ``canonical_name`` is MNE's uppercase key; recover the original
        # casing the dataset's channels.tsv used so downstream tools can
        # cross-reference without a case-normalisation step.
        display = names_up.get(canonical_name, canonical_name)
        sensors.append(
            {
                "name": display,
                "x": round(float(pos[0]) * 1000, 5),
                "y": round(float(pos[1]) * 1000, 5),
                "z": round(float(pos[2]) * 1000, 5),
                "type": "EEG",
            }
        )
    match_ratio = len(sensors) / max(1, len(names))
    h = _hash_sensors(modality, sensors)
    return h, {
        "hash": h,
        "modality": modality,
        "n_sensors": len(sensors),
        "space_declared": "CapTrak",  # MNE's canonical frame is RAS+
        "units_declared": "mm",
        "sensors": sensors,
        "source": "template-matched",
        "template": template_name,
        "template_match_ratio": round(match_ratio, 3),
        "channels_tsv": str(channels_tsv.relative_to(bids_root))
        if channels_tsv.is_relative_to(bids_root)
        else channels_tsv.name,
    }


# ---------------------------------------------------------------------------
# iEEG — electrodes in brain space (ACPC / MNI152 / other)
# ---------------------------------------------------------------------------


# extract_ieeg_layout removed (Phase 8 P2.1) — its body was a thin
# wrapper over ``_extract_tsv_layout``. The kwargs now live in
# ``_TSV_MODALITY_CONFIGS["ieeg"]`` and the dispatcher in
# :func:`extract_layout` invokes ``_extract_layout_for_config``.


# ---------------------------------------------------------------------------
# MEG — sensor positions inside the raw file header
# ---------------------------------------------------------------------------

# Map from MNE channel "kind" integer to human label. Sourced from
# mne/io/constants.py::FIFF.FIFFV_*_CH.
_MNE_CH_KIND_LABEL = {
    1: "MEG",  # FIFFV_MEG_CH
    2: "EEG",  # FIFFV_EEG_CH
    3: "MCG",  # FIFFV_MCG_CH
    101: "REF_MEG",  # FIFFV_REF_MEG_CH
    202: "EOG",
    302: "EMG",
    402: "ECG",
    502: "MISC",
    602: "RESP",
    702: "IAS",
    1000: "FIFFV_STIM_CH",
}

# Map from MNE coord_frame integer to human label. Crucial for MEG
# deduplication: device-frame positions are invariant across subjects
# (the hash stays stable for any recording on the same helmet), while
# head-frame positions change per subject (head-in-scanner pose).
# Sourced from mne/io/constants.py::FIFF.FIFFV_COORD_*.
_MNE_COORD_FRAME_LABEL = {
    0: "unknown",
    1: "device",  # FIFFV_COORD_DEVICE — MEG scanner-relative
    2: "isotrak",
    3: "hpi",
    4: "head",  # FIFFV_COORD_HEAD — LPA/RPA/nasion frame
    5: "mri",  # FIFFV_COORD_MRI
}


def _fetch_fif_metadata_streaming(data_file: Path, url: str, total: int) -> str | None:
    """Reconstruct a parseable FIF for **streaming-format** files (no
    central directory). Walks tags sequentially from the file's start
    until it finds ``FIFFB_MEAS_INFO`` end, then drops everything after.

    The trick: MNE's ``read_info`` is happy when the file reports its
    true size but only has valid FIF tags through the end of
    ``MEAS_INFO``. We write a sparse tempfile of the correct size,
    populate just the metadata prefix, and let MNE seek-but-not-read
    the raw-data tail (it's zeros, which MNE skips).
    """
    import struct
    import tempfile

    from _parser_utils import fetch_bytes_from_s3

    # Try progressively larger buffers. 306-channel Neuromag files end
    # their MEAS_INFO around 3-8 MB; 4 MB covers most, 16 MB is a safe
    # backstop for dense HPI + isotrak + long comments.
    for buf_size in (4_194_304, 16_777_216, 67_108_864):
        data = fetch_bytes_from_s3(url, max_bytes=min(buf_size, total), timeout=90.0)
        if data is None:
            LOGGER.debug("[layout.meg] streaming fetch %s failed", url)
            return None

        # Sequentially walk tags; track FIFFB_MEAS_INFO (kind 101)
        # nesting to find its BLOCK_END position.
        pos = 0
        info_depth = 0
        info_end: int | None = None
        while pos + 16 <= len(data):
            kind, _t, size, nxt = struct.unpack(">iiii", data[pos : pos + 16])
            if kind == 104 and size == 4:  # FIFF_BLOCK_START
                blk = struct.unpack(">i", data[pos + 16 : pos + 20])[0]
                if blk == 101:  # FIFFB_MEAS_INFO
                    info_depth += 1
            elif kind == 105 and size == 4:  # FIFF_BLOCK_END
                blk = struct.unpack(">i", data[pos + 16 : pos + 20])[0]
                if blk == 101 and info_depth > 0:
                    info_depth -= 1
                    if info_depth == 0:
                        info_end = pos + 16 + size
                        break
            # Advance: `next = -1 or 0` → sequential; else jump to offset
            if nxt in (-1, 0):
                new_pos = pos + 16 + size
            else:
                new_pos = nxt
            if new_pos <= pos or new_pos > len(data):
                break
            pos = new_pos

        if info_end is None:
            continue  # escalate buffer size and retry

        # Truncate the tempfile at info_end — do NOT pad to full size.
        # Streaming-format FIF has no directory, so MNE's tag walker
        # reads sequentially from byte 0. A sparse tail of zeros
        # parses as (kind=0, type=0, size=0, next=0) tag headers
        # forever, effectively hanging the walker for multi-GB files.
        # MEAS_INFO ends at info_end; everything after is raw samples
        # MNE doesn't need.
        tmp = tempfile.NamedTemporaryFile(
            suffix=data_file.suffix, prefix="megstr_", delete=False
        )
        try:
            tmp.write(data[:info_end])
            tmp.flush()
            LOGGER.info(
                "[layout.meg] %s: streaming mode, truncated at "
                "MEAS_INFO end (%.1f MB; original file %.1f MB)",
                data_file.name,
                info_end / (1024 * 1024),
                total / (1024 * 1024),
            )
            return tmp.name
        except (OSError, ValueError) as exc:
            # Filesystem failure on the tempfile write, or malformed
            # input bytes triggering the truncation logic.
            LOGGER.debug("[layout.meg] streaming stitch %s failed: %s", url, exc)
            try:
                Path(tmp.name).unlink()
            except OSError:
                pass
            return None
        finally:
            tmp.close()

    LOGGER.info(
        "[layout.meg] %s: MEAS_INFO end not found within 64 MB; skipping",
        url,
    )
    return None


def _fetch_fif_metadata_via_directory(data_file: Path, url: str) -> str | None:
    """Reconstruct a parseable FIF from S3 by fetching only metadata tags.

    OpenNeuro / NEMAR clone datasets leave FIF files as broken git-annex
    symlinks after our shallow ``git clone --depth 1 GIT_LFS_SKIP_SMUDGE=1``.
    ``mne.io.read_info`` then fails. Calling ``git annex get`` would
    work but pulls multi-GB raw data just to read a header.

    FIF files written by Neuromag/MaxFilter carry a central directory
    (``FIFF_DIR`` tag, offset pointed to by ``FIFF_DIR_POINTER`` at
    file start) that lists every tag's byte position. That directory
    is small (~25 KB for a 306-channel recording). Of the ~1600 tags
    it indexes, one kind (``FIFF_DATA_BUFFER`` = 300) accounts for
    >99% of the file's bulk; the remaining ~7 MB is channel info,
    HPI, isotrak, and block structure — exactly what ``read_info``
    walks.

    Flow:
      1. HEAD → total size. Range-fetch last 64 KB → locate the
         directory tag at ``FIFF_DIR_POINTER``.
      2. Parse directory entries. Skip kind=300. Merge remaining
         ranges (adjacent blocks get coalesced with an 8 KB slack to
         cut HTTP round trips).
      3. Write a sparse tempfile: seek to file-size, write one zero
         byte (so the file reports the true size to MNE), then drop
         every captured range into its original byte offset. Holes
         read as zeros on macOS/Linux — MNE never touches them
         because the directory drives all seeks.
      4. Return the tempfile path. Caller deletes after parsing.

    Typical cost: ~7 MB per file, 2-3 Range requests, ~3-7 seconds.
    Returns ``None`` when the file lacks a usable directory (streaming-
    only FIF) or the fetch fails.
    """
    import struct
    import tempfile

    from _parser_utils import fetch_bytes_from_s3, head_content_length

    # 1. Total size
    total = head_content_length(url, timeout=30.0)
    if total is None or total <= 0:
        return None

    # 2. Read FIFF_FILE_ID + FIFF_DIR_POINTER from the first 512 bytes.
    #    FILE_ID sits at pos 0 with a 20-byte body (16 header + 20 data);
    #    DIR_POINTER tag header is at pos 36, i32 data at pos 52.
    head = fetch_bytes_from_s3(url, max_bytes=512, timeout=30.0)
    if head is None or len(head) < 56:
        return None
    # DIR_POINTER value: signed i32 at offset 52
    dir_pointer = struct.unpack(">i", head[52:56])[0]
    if dir_pointer <= 0 or dir_pointer >= total:
        # Streaming-format FIF (no central directory). Fall back to a
        # sequential tag walk from the start: grab an initial chunk,
        # scan for BLOCK_END(FIFFB_MEAS_INFO=101), truncate the buffer
        # there. Older MEG recordings (Neuromag MaxFilter pre-2010ish)
        # use streaming mode exclusively.
        return _fetch_fif_metadata_streaming(data_file, url, total)

    # 3. Fetch the directory tag (we don't know its full size, so grab
    #    from dir_pointer to EOF — typical directories are <100 KB).
    dir_bytes = fetch_bytes_from_s3(
        url, start=dir_pointer, max_bytes=total - dir_pointer, timeout=60.0
    )
    if dir_bytes is None or len(dir_bytes) < 16:
        return None
    dir_kind, _, dir_size, _ = struct.unpack(">iiii", dir_bytes[:16])
    if dir_kind != 102:  # FIFF_DIR
        LOGGER.debug("[layout.meg] %s: expected FIFF_DIR (102), got %d", url, dir_kind)
        return None

    RAW_DATA_KINDS = {300}  # FIFF_DATA_BUFFER — the huge payload
    ranges: list[tuple[int, int]] = [(0, 4096)]
    for i in range(dir_size // 16):
        off = 16 + i * 16
        k, _t, s, pos = struct.unpack(">iiii", dir_bytes[off : off + 16])
        if k in RAW_DATA_KINDS or pos < 0:
            continue
        ranges.append((pos, pos + 16 + s))
    # Always include the directory itself + everything after.
    ranges.append((dir_pointer, total))
    ranges.sort()

    # Merge adjacent ranges with an 8 KB coalesce slack.
    merged: list[list[int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1] + 8192:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    total_bytes = sum(e - s for s, e in merged)
    if total_bytes > 128 * 1024 * 1024:  # pathological — bail out
        LOGGER.info(
            "[layout.meg] %s: metadata fetch would exceed 128 MB (%d bytes); skipping",
            url,
            total_bytes,
        )
        return None

    # 4. Build sparse tempfile
    tmp = tempfile.NamedTemporaryFile(
        suffix=data_file.suffix, prefix="megpart_", delete=False
    )
    try:
        # Create a sparse file at the true size by seeking past the end
        # and writing a single byte. macOS/Linux fill the hole with
        # zeros on read, which is exactly what MNE will skip over.
        tmp.seek(total - 1)
        tmp.write(b"\0")
        for start, end in merged:
            data = fetch_bytes_from_s3(
                url, start=start, max_bytes=end - start, timeout=90.0
            )
            if data is None:
                LOGGER.debug(
                    "[layout.meg] range stitch %s: fetch %d-%d failed",
                    url,
                    start,
                    end,
                )
                return None
            tmp.seek(start)
            tmp.write(data)
        tmp.flush()
        LOGGER.info(
            "[layout.meg] %s: reconstructed %.1f MB from %d range requests",
            data_file.name,
            total_bytes / (1024 * 1024),
            len(merged),
        )
        return tmp.name
    except (OSError, ValueError, KeyError) as exc:
        # Range-request stitch: OSError on tempfile write, ValueError on
        # truncation arithmetic, KeyError on malformed range metadata.
        LOGGER.debug("[layout.meg] range stitch %s failed: %s", url, exc)
        try:
            Path(tmp.name).unlink()
        except OSError:
            pass
        return None
    finally:
        tmp.close()


def extract_meg_layout(
    data_file: Path, _bids_root: Path | None = None
) -> tuple[str, dict[str, Any]] | None:
    """MEG: read sensor positions from the raw file header via MNE.

    Supports FIF (``.fif``), CTF (``.ds`` directory), and KIT
    (``.sqd`` / ``.con``). Only the header is read, so this is cheap
    even for multi-GB recordings. MEG/EEG reference channels and
    stimulus/misc channels are filtered out — only channels with a
    valid sensor location are kept.

    When the FIF file is a broken git-annex symlink (shallow clone),
    attempts a directory-aware S3 Range-fetch via
    ``_fetch_fif_metadata_via_directory``: only metadata tags come
    over the wire (~7 MB for a 2 GB recording).
    """
    # Import MNE lazily so this module's other entry points keep working
    # even when MNE isn't installed (rare, but possible in minimal envs).
    try:
        import mne
    except ImportError:
        LOGGER.warning("[layout.meg] mne not available; skipping %s", data_file)
        return None

    suffix = data_file.suffix.lower()
    name = data_file.name.lower()

    info = None
    raw = None
    tmp_path: str | None = None
    try:
        try:
            if suffix == ".fif":
                info = mne.io.read_info(str(data_file), verbose="error")
            elif data_file.is_dir() and suffix == ".ds":
                # CTF datasets are directories; read_raw_ctf takes the .ds path.
                raw = mne.io.read_raw_ctf(
                    str(data_file), preload=False, verbose="error"
                )
                info = raw.info
            elif suffix in {".sqd", ".con"} or name.endswith(".kit"):
                raw = mne.io.read_raw_kit(
                    str(data_file), preload=False, verbose="error"
                )
                info = raw.info
            else:
                LOGGER.info("[layout.meg] unrecognized MEG format: %s", data_file)
                return None
        except (
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
            AttributeError,
        ) as direct_exc:
            # MNE raises RuntimeError on unsupported format variant,
            # ValueError on truncated/malformed header, OSError on file
            # not found / permission, KeyError on missing required field.
            LOGGER.debug(
                "[layout.meg] direct read %s failed: %s", data_file, direct_exc
            )
            # S3 directory-aware reconstruction — FIF only. CTF .ds is
            # a directory of files, not range-fetchable as one blob;
            # KIT .sqd uses a different binary layout we haven't
            # investigated yet.
            if suffix != ".fif":
                return None
            from _parser_utils import (
                build_s3_url,
                extract_dataset_info,
                is_broken_symlink,
            )

            # Only chase S3 for files that really are broken annex pointers.
            if not is_broken_symlink(data_file) and data_file.exists():
                LOGGER.warning(
                    "[layout.meg] read failed on present file %s: %s",
                    data_file,
                    direct_exc,
                )
                return None
            info_tuple = extract_dataset_info(data_file)
            if info_tuple is None:
                return None
            source, ds_id, rel = info_tuple
            try:
                url = build_s3_url(ds_id, rel, source=source)
            except ValueError:
                return None
            tmp_path = _fetch_fif_metadata_via_directory(data_file, url)
            if tmp_path is None:
                return None
            try:
                info = mne.io.read_info(tmp_path, verbose="error")
            except (OSError, ValueError, RuntimeError, KeyError) as partial_exc:
                # Same MNE failure classes; the reconstructed FIF stub
                # may still be malformed for various reasons.
                LOGGER.info(
                    "[layout.meg] %s: directory-reconstructed FIF still "
                    "unparsable (%s)",
                    data_file.name,
                    partial_exc,
                )
                return None
    finally:
        if raw is not None:
            try:
                raw.close()
            except (OSError, AttributeError):
                # OSError: already-closed. AttributeError: not an MNE
                # Raw (defensive guard).
                pass
        if tmp_path:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass
    if info is None:
        return None

    sensors: list[dict[str, Any]] = []
    coord_frames_seen: set[int] = set()
    for ch in info["chs"]:
        kind = ch.get("kind")
        # Only MEG channels (and MEG references) carry meaningful sensor
        # positions. Skip EEG channels on MEG datasets — those are
        # handled by the EEG extractor via _electrodes.tsv.
        if kind not in (1, 101):  # MEG, REF_MEG
            continue
        loc = np.asarray(ch.get("loc", []), dtype=float)
        if loc.size < 3:
            continue
        x, y, z = float(loc[0]), float(loc[1]), float(loc[2])
        if not all(np.isfinite([x, y, z])):
            continue
        coord_frame = int(ch.get("coord_frame", 0))
        coord_frames_seen.add(coord_frame)
        sensors.append(
            {
                "name": str(ch.get("ch_name") or "").strip(),
                "x": x,
                "y": y,
                "z": z,
                "type": _MNE_CH_KIND_LABEL.get(kind, "MEG"),
                "coil_type": int(ch.get("coil_type", 0)),
            }
        )

    if len(sensors) < 4:
        return None

    # Deduplication relies on the frame being consistent across a
    # recording. MEG channels are typically all in DEVICE or all in HEAD
    # frame; mixed frames would make the hash subject-dependent, which
    # defeats catalogue deduplication — flag that case.
    if len(coord_frames_seen) == 1:
        frame_label = _MNE_COORD_FRAME_LABEL.get(
            next(iter(coord_frames_seen)), "unknown"
        )
    else:
        frame_label = "mixed"
        LOGGER.info(
            "[layout.meg] %s: channels span multiple coord frames %s; hash may be subject-specific",
            data_file,
            coord_frames_seen,
        )

    h = _hash_sensors("meg", sensors)
    return h, {
        "hash": h,
        "modality": "meg",
        "n_sensors": len(sensors),
        # Honest reporting of the actual frame. DEVICE frame means the
        # hash is stable across subjects wearing the same helmet (ideal
        # for the catalogue); HEAD frame means it's subject-specific
        # (subject's head pose in the scanner).
        "space_declared": frame_label,
        "units_declared": "m",
        "sensors": sensors,
    }


# ---------------------------------------------------------------------------
# EMG — surface electrodes on muscles, body-landmark coordinate frames
# ---------------------------------------------------------------------------


# extract_emg_layout removed (Phase 8 P2.1). EMG kwargs documented:
# anatomical coord systems (HySER ExtensorDistal etc.), per-row
# coordinate_system column, often percent units. All preserved in
# ``_TSV_MODALITY_CONFIGS["emg"]``. See the config above for the
# extras tuple including ``coordinate_system`` and ``group``.


# ---------------------------------------------------------------------------
# fNIRS — optodes (sources + detectors)
# ---------------------------------------------------------------------------


# extract_fnirs_layout removed (Phase 8 P2.1). fNIRS reads
# ``*_optodes.tsv`` (not _electrodes.tsv) — that's the only thing
# that distinguishes it from the other TSV modalities. See
# ``_TSV_MODALITY_CONFIGS["nirs"]`` for the per-modality kwargs.


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def extract_layout(
    data_file: Path,
    bids_root: Path,
    datatype: str,
) -> tuple[str, dict[str, Any]] | None:
    """Dispatch to the right per-modality extractor.

    Phase 8 P2.1 — the dispatch now reads from
    :data:`_TSV_MODALITY_CONFIGS` for the 4 TSV-based modalities
    (EEG, iEEG, EMG, fNIRS). MEG remains a special case (sensor
    positions in the FIF header, not a sidecar — ~190 LOC of header
    streaming logic in :func:`extract_meg_layout`).

    Returns ``None`` for unsupported datatypes (beh, etc.) so the
    caller can simply record ``layout_hash = None`` and move on.

    Note: ``"fnirs"`` is accepted as an alias for ``"nirs"`` — some
    older datasets use it as the datatype name.
    """
    dt = (datatype or "").lower()
    if dt == "meg":
        return extract_meg_layout(data_file, bids_root)
    if dt == "fnirs":  # alias for nirs in some older datasets
        dt = "nirs"
    config = _TSV_MODALITY_CONFIGS.get(dt)
    if config is None:
        return None
    return _extract_layout_for_config(data_file, bids_root, config)
