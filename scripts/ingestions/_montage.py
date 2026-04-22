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
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

LOGGER = logging.getLogger("digest.montage")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_COORDSYS_PREFIXES = ("EEG", "iEEG", "MEG", "EMG", "NIRS")


def _round_mm(v: float) -> int:
    return int(round(v * 1000))


def _hash_sensors(modality: str, sensors: list[dict[str, Any]]) -> str:
    """Stable short hash over a canonical (modality, name, mm-rounded coords) form.

    Including ``modality`` in the hash prevents a MEG helmet and an EEG
    cap from aliasing on name + position (vanishingly unlikely, but free
    to prevent).
    """
    canonical = [modality] + sorted(
        (
            s.get("name", ""),
            _round_mm(s.get("x", 0.0)),
            _round_mm(s.get("y", 0.0)),
            _round_mm(s.get("z", 0.0)),
            s.get("type", ""),
        )
        for s in sensors
    )
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
    except Exception as exc:  # noqa: BLE001
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


def extract_eeg_layout(
    data_file: Path, bids_root: Path
) -> tuple[str, dict[str, Any]] | None:
    """Scalp EEG: parse ``*_electrodes.tsv`` → hash-identified doc.

    When no ``*_electrodes.tsv`` sidecar was published for the dataset,
    fall back to matching ``*_channels.tsv`` names against MNE's
    built-in standard montages and emit a template-derived layout. The
    fallback doc is tagged ``source: "template-matched"`` so downstream
    consumers can tell subject-specific positions apart from canonical
    vendor-cap positions.
    """
    direct = _extract_tsv_layout(data_file, bids_root, modality="eeg")
    if direct is not None:
        return direct
    return _extract_template_from_channels(data_file, bids_root, modality="eeg")


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
    global _MNE_TEMPLATE_CACHE
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
            except Exception:  # noqa: BLE001 — skip anything MNE can't build here
                continue
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[template] MNE import failed (%s); pool 1 empty", exc)

    # Pool 2 — the electrode-explorer extras. montages.json stores each
    # position in meters already (xyz is centred on the fitted head
    # sphere), identical to what make_standard_montage produces.
    try:
        if _ELECTRODE_EXPLORER_MONTAGES_JSON.exists():
            doc = json.loads(_ELECTRODE_EXPLORER_MONTAGES_JSON.read_text())
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
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[template] electrode-explorer catalog failed (%s)", exc)

    _MNE_TEMPLATE_CACHE = cache
    LOGGER.info("[template] loaded %d templates (MNE + electrode-explorer)", len(cache))
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
    except Exception:
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
    best: tuple[int, int, str, dict[str, tuple[float, float, float]]] | None = None
    for tname, tpos in templates.items():
        hits = channels_up & set(tpos.keys())
        if len(hits) < min_hits:
            continue
        ratio = len(hits) / len(channels_up)
        if ratio < min_ratio:
            continue
        # Rank primarily on hit count; tiebreak on smaller template size.
        # `best` stores (hits, -template_size, name, matched_subset) so
        # max(...) picks highest hits then smallest template.
        matched = {k: tpos[k] for k in tpos if k in channels_up}
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


def extract_ieeg_layout(
    data_file: Path, bids_root: Path
) -> tuple[str, dict[str, Any]] | None:
    """Intracranial EEG: positions in brain space (no radius sanity).

    The 2D sphere viewer can't render these today, but cataloguing the
    hash still deduplicates identical grids across subjects for future
    glass-brain viewers.
    """
    return _extract_tsv_layout(
        data_file,
        bids_root,
        modality="ieeg",
        extras=("type", "hemisphere", "material", "impedance", "group", "size"),
        min_sensors=1,
    )


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


def extract_meg_layout(
    data_file: Path, _bids_root: Path | None = None
) -> tuple[str, dict[str, Any]] | None:
    """MEG: read sensor positions from the raw file header via MNE.

    Supports FIF (``.fif``), CTF (``.ds`` directory), and KIT
    (``.sqd`` / ``.con``). Only the header is read, so this is cheap
    even for multi-GB recordings. MEG/EEG reference channels and
    stimulus/misc channels are filtered out — only channels with a
    valid sensor location are kept.

    The underscore-prefixed ``_bids_root`` parameter exists so the
    extractor matches the shared dispatcher signature; unused here.
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

    try:
        if suffix == ".fif":
            info = mne.io.read_info(str(data_file), verbose="error")
        elif data_file.is_dir() and suffix == ".ds":
            # CTF datasets are directories; read_raw_ctf takes the .ds path.
            info = mne.io.read_raw_ctf(
                str(data_file), preload=False, verbose="error"
            ).info
        elif suffix in {".sqd", ".con"} or name.endswith(".kit"):
            info = mne.io.read_raw_kit(
                str(data_file), preload=False, verbose="error"
            ).info
        else:
            # Not a MEG format we recognise; the caller shouldn't have
            # routed us here, but degrade gracefully.
            LOGGER.info("[layout.meg] unrecognised MEG format: %s", data_file)
            return None
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[layout.meg] failed to read header for %s: %s", data_file, exc)
        return None

    import numpy as np  # noqa: PLC0415 — localised import

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


def extract_emg_layout(
    data_file: Path, bids_root: Path
) -> tuple[str, dict[str, Any]] | None:
    """Surface EMG: parse ``*_electrodes.tsv`` with body-landmark frames.

    EMG differs from the scalp modalities in three ways worth knowing:

    1. **Anatomical coordinate systems** — positions live in body-part
       frames (e.g. HySER's ``ExtensorDistal`` / ``forearm``), not on a
       head sphere. Hash dedupes across subjects; the 2D sphere viewer
       can't render.
    2. **Non-length units** — ``EMGCoordinateUnits`` is often
       ``"percent"`` (normalised to anatomical landmarks). Preserved
       verbatim; viewers that need mm must supply calibration.
    3. **Per-row coordinate system** — one TSV can mix frames via the
       optional ``coordinate_system`` column (HySER: 4 muscles × 64
       electrodes in one file). Preserved on each sensor.
    """
    return _extract_tsv_layout(
        data_file,
        bids_root,
        modality="emg",
        extras=("type", "material", "impedance", "coordinate_system", "group"),
        min_sensors=2,
    )


# ---------------------------------------------------------------------------
# fNIRS — optodes (sources + detectors)
# ---------------------------------------------------------------------------


def extract_fnirs_layout(
    data_file: Path, bids_root: Path
) -> tuple[str, dict[str, Any]] | None:
    """fNIRS: parse ``*_optodes.tsv`` for source/detector positions.

    ``*_channels.tsv`` declares source+detector pairings per measurement
    channel; we don't decode the pairing here (keeps hash stable across
    upstream channel-name churn). The optode ``type`` column is kept so
    the viewer can distinguish sources from detectors.
    """
    return _extract_tsv_layout(
        data_file,
        bids_root,
        modality="nirs",
        tsv_pattern="*_optodes.tsv",
        extras=("type", "template_x", "template_y", "template_z"),
        min_sensors=2,
        coord_suffix="_optodes.tsv",
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_DISPATCH = {
    "eeg": extract_eeg_layout,
    "ieeg": extract_ieeg_layout,
    "meg": extract_meg_layout,
    "emg": extract_emg_layout,
    "nirs": extract_fnirs_layout,
    "fnirs": extract_fnirs_layout,  # some older datasets use this token
}


def extract_layout(
    data_file: Path,
    bids_root: Path,
    datatype: str,
) -> tuple[str, dict[str, Any]] | None:
    """Dispatch to the right per-modality extractor.

    Returns ``None`` for unsupported datatypes (EMG, beh, etc.) so the
    caller can simply record ``layout_hash = None`` and move on.
    """
    fn = _DISPATCH.get((datatype or "").lower())
    if fn is None:
        return None
    return fn(data_file, bids_root)


# ---------------------------------------------------------------------------
# Backwards-compat alias (previous name of the single EEG extractor)
# ---------------------------------------------------------------------------


def extract_montage(
    data_file: Path,
    bids_root: Path,
    datatype: str = "eeg",
) -> tuple[str, dict[str, Any]] | None:
    """Deprecated alias — use :func:`extract_layout` instead."""
    return extract_layout(data_file, bids_root, datatype)
