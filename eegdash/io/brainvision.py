"""MNE's BrainVision reader, told by ``overrides=`` (MNE 1.13, mne-python#13884) which files to read.

A ``.vhdr`` whose ``DataFile=``/``MarkerFile=`` name pre-BIDS files or git-annex
keys is read as it is: :func:`overrides` names the recording's own files and MNE
reads them. Rewriting the cached header made every session download it again
and let a second process read it half written.
"""

import logging
import re
from difflib import SequenceMatcher
from pathlib import Path

import mne

logger = logging.getLogger("eegdash")

_ANNEX_KEY_RE = re.compile(r"^(SHA256E|MD5E)-s\d+--[0-9a-f]+\.", flags=re.IGNORECASE)
_POINTER_RE = re.compile(
    r"^\s*(DataFile|MarkerFile)\s*=\s*(.*?)\s*$", flags=re.IGNORECASE | re.MULTILINE
)


def _find_best_matching_file(
    directory: Path, target_name: str, extension: str, threshold: float = 0.5
) -> str | None:
    """Return the file of ``directory`` with ``extension`` closest to ``target_name``.

    One candidate is returned whatever its name; with several, the most similar
    stem above ``threshold`` (:class:`difflib.SequenceMatcher` ratio), else None.
    """
    candidates = list(directory.glob(f"*{extension}"))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0].name
    target_stem = Path(target_name).stem.lower()
    best_match, best_ratio = None, threshold
    for candidate in candidates:
        ratio = SequenceMatcher(None, target_stem, candidate.stem.lower()).ratio()
        if ratio > best_ratio:
            best_ratio, best_match = ratio, candidate.name
    return best_match


def _resolve(vhdr: Path, value: str | None, ext: str) -> str | None:
    """Return the file to read instead of ``value``, or None to keep the header's."""
    folder = vhdr.parent
    own = vhdr.with_suffix(ext).name
    is_annex_key = value is not None and bool(_ANNEX_KEY_RE.match(value))
    if value is not None and not is_annex_key and (folder / value).is_file():
        return None
    if ext == ".vmrk":
        # Missing key, stale name or annex key: the recording's own .vmrk;
        # without one MNE warns and reads no annotations.
        return own if (folder / own).is_file() else None
    # A key-named file only resolves inside the annex, so an annex key always
    # gives way to the BIDS name (MNE then names that file if it is missing).
    if is_annex_key or (folder / own).is_file():
        return own if own != value else None
    # A data file with neither the header's name nor the BIDS name: the most
    # similar .eeg (typos, pre-BIDS names), but never one that has its own
    # .vhdr, which is another recording's data.
    # ponytail: name similarity; wrong if two runs' .eeg differ by a character and
    # neither has a .vhdr. Map names from the DB record if that shows up.
    match = _find_best_matching_file(folder, value, ext)
    if (
        match
        and not _ANNEX_KEY_RE.match(match)
        and not (folder / match).with_suffix(".vhdr").is_file()
    ):
        return match
    return None


def overrides(vhdr_fname) -> dict:
    """Return the ``overrides`` that point ``vhdr_fname`` at its own files.

    ``{}`` when the header's ``DataFile=`` and ``MarkerFile=`` are right.
    Only names are returned; MNE resolves them against the header's folder.
    The header is read, never written.
    """
    vhdr = Path(vhdr_fname)
    try:
        text = vhdr.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    found = {key.lower(): value for key, value in _POINTER_RE.findall(text)}
    out = {}
    for key, name, ext in (
        ("datafile", "data_fname", ".eeg"),
        ("markerfile", "marker_fname", ".vmrk"),
    ):
        if key == "datafile" and key not in found:
            continue  # MNE raises its own error
        target = _resolve(vhdr, found.get(key), ext)
        if target is not None:
            logger.info(
                "%s: %s=%s -> %s (read-time override, header unchanged)",
                vhdr.name,
                "DataFile" if ext == ".eeg" else "MarkerFile",
                found.get(key, "(missing)"),
                target,
            )
            out[name] = target
    return out


def read_raw_brainvision(vhdr_fname, **kwargs):
    """Read with :func:`mne.io.read_raw_brainvision`, passing :func:`overrides`."""
    kwargs.setdefault("overrides", overrides(vhdr_fname) or None)
    return mne.io.read_raw_brainvision(vhdr_fname, **kwargs)
