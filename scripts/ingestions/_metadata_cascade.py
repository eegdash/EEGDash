"""Technical-metadata cascade for the BIDS digest pipeline.

Resolves ``(sampling_frequency, nchans, ntimes, ch_names)`` plus a
per-field provenance dict by trying up to five sources in order with
first-writer-wins semantics: MneBidsStep → ModalitySidecarStep →
ChannelsTsvStep → BinaryParserStep → MneFallbackStep.

Public surface: :class:`MetadataCascade.run` takes a
:class:`CascadeContext` and returns a :class:`CascadeResult`.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

import mne
from _format_parser_registry import get_parser_for_extension

# Provenance source names persisted into ``_metadata_provenance``; values are
# snapshot-tested for byte-level stability — do not rename.
PROV_MNE_BIDS = "mne_bids"
PROV_MODALITY_SIDECAR = "modality_sidecar"
PROV_CHANNELS_TSV = "channels_tsv"
PROV_BINARY_PARSER = "binary_parser"
PROV_MNE_FALLBACK = "mne_fallback"

_METADATA_FIELDS: tuple[str, ...] = (
    "sampling_frequency",
    "nchans",
    "ntimes",
    "ch_names",
)


# ─── BIDS sidecar helpers ─────────────────────────────────────────────────


# Per-modality JSON sidecar suffixes. MEG before EEG so a co-recorded
# MEG+EEG study finds the MEG sidecar first (authoritative for n_megchans).
_MODALITY_SIDECAR_SUFFIXES = (
    "_meg.json",
    "_eeg.json",
    "_ieeg.json",
    "_nirs.json",
)

_BIDS_CHANNEL_COUNT_FIELDS: tuple[str, ...] = (
    "MEGChannelCount",
    "EEGChannelCount",
    "EOGChannelCount",
    "ECGChannelCount",
    "EMGChannelCount",
    "MiscChannelCount",
    "TriggerChannelCount",
    "iEEGChannelCount",
    "SEEGChannelCount",
    "ECOGChannelCount",
    "NIRSChannelCount",
    "ACCELChannelCount",
)


def sum_bids_channel_counts(sidecar_data: dict[str, Any]) -> int | None:
    """Sum all known BIDS ``*ChannelCount`` fields; return ``None`` when zero."""
    total = sum(sidecar_data.get(field, 0) or 0 for field in _BIDS_CHANNEL_COUNT_FIELDS)
    return total if total > 0 else None


def _build_bids_search_paths(
    bids_file_path: Path, bids_root: Path
) -> tuple[list[str], list[Path]]:
    """Return base-name variants and ancestor directories for a BIDS inheritance walk."""
    stem = bids_file_path.stem
    parts = stem.split("_")

    base_names_to_try: list[str] = []
    full_base = "_".join(parts[:-1]) if len(parts) > 1 else stem
    base_names_to_try.append(full_base)

    if "_run-" in full_base:
        without_run = "_".join(p for p in parts[:-1] if not p.startswith("run-"))
        base_names_to_try.append(without_run)

    if "_acq-" in full_base:
        without_acq_run = "_".join(
            p
            for p in parts[:-1]
            if not p.startswith("run-") and not p.startswith("acq-")
        )
        base_names_to_try.append(without_acq_run)

    task_part = next((p for p in parts if p.startswith("task-")), None)
    if task_part and task_part not in base_names_to_try:
        base_names_to_try.append(task_part)

    parent_dir = bids_file_path.parent
    dirs_to_try: list[Path] = [parent_dir]
    current = parent_dir
    while current != bids_root and current.parent != current:
        current = current.parent
        dirs_to_try.append(current)
        if current == bids_root:
            break

    return base_names_to_try, dirs_to_try


def extract_sfreq_nchans_from_modality_sidecar(
    bids_file_path: Path,
    bids_root: Path,
    sampling_frequency: float | None,
    nchans: int | None,
) -> tuple[float | None, int | None]:
    """Fill missing sfreq/nchans from a modality JSON sidecar using BIDS inheritance."""
    if sampling_frequency and nchans:
        return sampling_frequency, nchans

    base_names_to_try, dirs_to_try = _build_bids_search_paths(bids_file_path, bids_root)

    for search_dir in dirs_to_try:
        if sampling_frequency and nchans:
            break
        for base_name in base_names_to_try:
            if sampling_frequency and nchans:
                break
            for sidecar_suffix in _MODALITY_SIDECAR_SUFFIXES:
                sidecar_path = search_dir / f"{base_name}{sidecar_suffix}"
                if not sidecar_path.exists():
                    continue
                try:
                    with open(sidecar_path) as f:
                        sidecar_data = json.load(f)
                except (OSError, json.JSONDecodeError, ValueError, TypeError):
                    # Unreadable or malformed sidecar — try next candidate.
                    continue
                if not sampling_frequency and "SamplingFrequency" in sidecar_data:
                    try:
                        sampling_frequency = float(sidecar_data["SamplingFrequency"])
                    except (TypeError, ValueError):
                        pass
                if not nchans:
                    summed = sum_bids_channel_counts(sidecar_data)
                    if summed is not None:
                        nchans = summed
                break  # only consult one sidecar variant per (dir, base)

    return sampling_frequency, nchans


def extract_sfreq_nchans_from_channels_tsv(
    bids_file_path: Path,
    bids_root: Path,
    sampling_frequency: float | None,
    nchans: int | None,
) -> tuple[float | None, int | None]:
    """Fill missing sfreq/nchans from a ``*_channels.tsv`` using BIDS inheritance."""
    if sampling_frequency and nchans:
        return sampling_frequency, nchans

    base_names_to_try, dirs_to_try = _build_bids_search_paths(bids_file_path, bids_root)

    for search_dir in dirs_to_try:
        if sampling_frequency and nchans:
            break
        for base_name in base_names_to_try:
            if sampling_frequency and nchans:
                break
            channels_path = search_dir / f"{base_name}_channels.tsv"
            if not channels_path.exists():
                continue
            try:
                channels_df = pd.read_csv(
                    channels_path,
                    sep="\t",
                    dtype="string",
                    keep_default_na=False,
                )
            except (
                pd.errors.ParserError,
                pd.errors.EmptyDataError,
                OSError,
                UnicodeDecodeError,
                ValueError,
                KeyError,
            ):
                # channels.tsv malformed; try next search dir.
                continue
            if not nchans:
                nchans = len(channels_df)
            if not sampling_frequency:
                for col in ("sampling_frequency", "SamplingFrequency"):
                    if col not in channels_df.columns:
                        continue
                    for val in channels_df[col]:
                        try:
                            sfreq_val = float(val)
                        except (TypeError, ValueError):
                            continue
                        if sfreq_val > 0:
                            sampling_frequency = sfreq_val
                            break
                    if sampling_frequency:
                        break
            break  # one channels.tsv per (dir, base) is authoritative

    return sampling_frequency, nchans


# ─── FIF metadata helper ──────────────────────────────────────────────────


def _parse_fif_with_mne(fif_path: Path) -> tuple[dict[str, Any] | None, bool]:
    """Parse FIF metadata via MNE; ``on_split_missing="warn"`` lets git-annex hash-named files yield header data."""
    # Broken git-annex symlinks resolve to non-existent targets.
    if not fif_path.exists():
        return None, False
    try:
        resolved = fif_path.resolve()
        if not resolved.exists():
            return None, False
    except (OSError, RuntimeError):
        return None, False

    try:
        is_split = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            raw = mne.io.read_raw_fif(
                str(fif_path),
                preload=False,
                on_split_missing="warn",
                verbose=False,
            )
        for w in caught:
            if "split" in str(w.message).lower():
                is_split = True
                break

        try:
            result: dict[str, Any] = {}

            sfreq = raw.info.get("sfreq")
            if sfreq:
                result["sampling_frequency"] = float(sfreq)

            ch_names = raw.info.get("ch_names")
            if ch_names:
                result["ch_names"] = list(ch_names)
                result["nchans"] = len(ch_names)

            if raw.n_times and raw.n_times > 0:
                result["n_times"] = int(raw.n_times)

            return (result if result else None), is_split
        finally:
            try:
                raw.close()
            except (OSError, AttributeError):
                pass

    except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError):
        # AttributeError: mne >=1.12 raises this from _fiff/open.py:151
        # when reading a malformed FIF (tag is None → tag.kind crashes).
        return None, False


# ─── Dataclasses ──────────────────────────────────────────────────────────


@dataclass
class CascadeContext:
    """Input bundle for one cascade run; derived fields populated in ``__post_init__``."""

    bids_dataset: Any
    bids_file: str

    # Derived in __post_init__.
    bids_file_path: Path = field(init=False)
    bids_root: Path = field(init=False)
    ext: str = field(init=False)

    def __post_init__(self) -> None:
        self.bids_file_path = Path(self.bids_file)
        self.bids_root = Path(self.bids_dataset.bidsdir)
        self.ext = self.bids_file_path.suffix.lower()


@dataclass
class CascadeResult:
    """Accumulated output of one cascade run; ``provenance[field]`` is the first source that filled it, or ``None``."""

    sampling_frequency: float | None = None
    nchans: int | None = None
    ntimes: int | None = None
    ch_names: list[str] | None = None
    fif_is_split: bool = False
    fif_continuations_ok: bool = True
    provenance: dict[str, str | None] = field(
        default_factory=lambda: {f: None for f in _METADATA_FIELDS}
    )

    def stamp(self, source: str, field_name: str, old: Any, new: Any) -> None:
        """Record ``source`` as the provenance for ``field_name`` on first write."""
        if old is None and new is not None and self.provenance[field_name] is None:
            self.provenance[field_name] = source


# ─── Step Protocol + concrete implementations ─────────────────────────────


class CascadeStep(Protocol):
    """Protocol for cascade steps; each mutates ``result`` in-place."""

    def fill(self, ctx: CascadeContext, result: CascadeResult) -> None: ...


class MneBidsStep:
    """Step 1: ``EEGBIDSDataset`` attribute getters (mne_bids backend)."""

    def fill(self, ctx: CascadeContext, result: CascadeResult) -> None:
        bd = ctx.bids_dataset
        try:
            sf = bd.get_bids_file_attribute("sfreq", ctx.bids_file)
            nc = bd.get_bids_file_attribute("nchans", ctx.bids_file)
            nt = bd.get_bids_file_attribute("ntimes", ctx.bids_file)
        except (FileNotFoundError, OSError):
            # Broken git-annex symlink on the BIDS sidecar JSON.
            sf = nc = nt = None

        if sf:
            result.sampling_frequency = float(sf)
            result.provenance["sampling_frequency"] = PROV_MNE_BIDS
        if nc:
            result.nchans = int(nc)
            result.provenance["nchans"] = PROV_MNE_BIDS
        if nt:
            result.ntimes = int(nt)
            result.provenance["ntimes"] = PROV_MNE_BIDS

        # channel_labels reads channels.tsv via mne_bids — same provenance bucket.
        try:
            ch_names = bd.channel_labels(ctx.bids_file)
        except (FileNotFoundError, OSError, ValueError, KeyError, AttributeError):
            ch_names = None
        if ch_names:
            result.ch_names = ch_names
            result.provenance["ch_names"] = PROV_MNE_BIDS
            # channel_labels count is more complete than sidecar *ChannelCount sums;
            # overwrite the value but only claim provenance if not yet stamped.
            if result.provenance["nchans"] is None:
                result.provenance["nchans"] = PROV_MNE_BIDS
            result.nchans = len(ch_names)


class ModalitySidecarStep:
    """Step 2: modality JSON sidecar with BIDS-inheritance walk."""

    def fill(self, ctx: CascadeContext, result: CascadeResult) -> None:
        sf_before = result.sampling_frequency
        n_before = result.nchans
        sf, nc = extract_sfreq_nchans_from_modality_sidecar(
            ctx.bids_file_path, ctx.bids_root, sf_before, n_before
        )
        result.sampling_frequency = sf
        result.nchans = nc
        result.stamp(PROV_MODALITY_SIDECAR, "sampling_frequency", sf_before, sf)
        result.stamp(PROV_MODALITY_SIDECAR, "nchans", n_before, nc)


class ChannelsTsvStep:
    """Step 3: ``channels.tsv`` with BIDS-inheritance walk."""

    def fill(self, ctx: CascadeContext, result: CascadeResult) -> None:
        sf_before = result.sampling_frequency
        n_before = result.nchans
        sf, nc = extract_sfreq_nchans_from_channels_tsv(
            ctx.bids_file_path, ctx.bids_root, sf_before, n_before
        )
        result.sampling_frequency = sf
        result.nchans = nc
        result.stamp(PROV_CHANNELS_TSV, "sampling_frequency", sf_before, sf)
        result.stamp(PROV_CHANNELS_TSV, "nchans", n_before, nc)


class BinaryParserStep:
    """Step 4: per-extension binary parser via the format registry."""

    def fill(self, ctx: CascadeContext, result: CascadeResult) -> None:
        parser = get_parser_for_extension(ctx.ext)
        if parser is None:
            return

        # Skip when all four fields are already populated.
        if (
            result.sampling_frequency
            and result.nchans
            and result.ntimes
            and result.ch_names
        ):
            return

        md = parser(ctx.bids_file_path)
        if not md:
            return

        sf_before = result.sampling_frequency
        n_before = result.nchans
        nt_before = result.ntimes
        ch_before = result.ch_names

        result.sampling_frequency = sf_before or md.get("sampling_frequency")
        result.nchans = n_before or md.get("nchans")
        result.ntimes = nt_before or md.get("n_times") or md.get("n_samples")
        result.ch_names = ch_before or md.get("ch_names")

        for fname, old, new in (
            ("sampling_frequency", sf_before, result.sampling_frequency),
            ("nchans", n_before, result.nchans),
            ("ntimes", nt_before, result.ntimes),
            ("ch_names", ch_before, result.ch_names),
        ):
            result.stamp(PROV_BINARY_PARSER, fname, old, new)


class MneFallbackStep:
    """Step 5: MNE fallbacks for VHDR ntimes and FIF metadata + split detection."""

    def fill(self, ctx: CascadeContext, result: CascadeResult) -> None:
        # --- VHDR: try MNE for n_times specifically (needs binary companion). ---
        if ctx.ext == ".vhdr" and not result.ntimes:
            raw = None
            try:
                raw = mne.io.read_raw_brainvision(
                    str(ctx.bids_file_path), preload=False, verbose=False
                )
                if raw.n_times and raw.n_times > 0:
                    result.ntimes = int(raw.n_times)
                    if result.provenance["ntimes"] is None:
                        result.provenance["ntimes"] = PROV_MNE_FALLBACK
            except (OSError, ValueError, RuntimeError, KeyError):
                # Missing companion file, malformed header, or unsupported variant.
                pass
            finally:
                if raw is not None:
                    try:
                        raw.close()
                    except (OSError, AttributeError):
                        pass

        # --- FIF: full-record fallback + split detection. ---
        if ctx.ext == ".fif" and (
            not result.sampling_frequency or not result.nchans or not result.ntimes
        ):
            fif_metadata, fif_is_split = _parse_fif_with_mne(ctx.bids_file_path)
            result.fif_is_split = fif_is_split
            if fif_metadata:
                sf_before = result.sampling_frequency
                n_before = result.nchans
                nt_before = result.ntimes
                ch_before = result.ch_names

                result.sampling_frequency = sf_before or fif_metadata.get(
                    "sampling_frequency"
                )
                result.nchans = n_before or fif_metadata.get("nchans")
                result.ntimes = nt_before or fif_metadata.get("n_times")
                result.ch_names = ch_before or fif_metadata.get("ch_names")

                for fname, old, new in (
                    ("sampling_frequency", sf_before, result.sampling_frequency),
                    ("nchans", n_before, result.nchans),
                    ("ntimes", nt_before, result.ntimes),
                    ("ch_names", ch_before, result.ch_names),
                ):
                    result.stamp(PROV_MNE_FALLBACK, fname, old, new)


# ─── Cascade runner ───────────────────────────────────────────────────────


class MetadataCascade:
    """Runs cascade steps in order; first writer per field wins. Inject ``steps`` in tests."""

    def __init__(self, steps: tuple[CascadeStep, ...] | None = None) -> None:
        self.steps = steps or (
            MneBidsStep(),
            ModalitySidecarStep(),
            ChannelsTsvStep(),
            BinaryParserStep(),
            MneFallbackStep(),
        )

    def run(self, ctx: CascadeContext) -> CascadeResult:
        result = CascadeResult()
        for step in self.steps:
            step.fill(ctx, result)
        return result
