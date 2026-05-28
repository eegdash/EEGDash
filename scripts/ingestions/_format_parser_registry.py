"""Format metadata parser registry — explicit Seam for binary-header readers.

. Each neuroimaging file format (``.edf`` / ``.bdf`` /
``.set`` / ``.vhdr`` / ``.snirf`` / ``.mefd``) needs its own binary
header reader because sidecar JSON may be absent. Before this module
the 5 parsers shared an implicit contract — same return shape, same
``None`` semantics, but documented only in their individual docstrings.

This file names the Seam: a :class:`FormatParser` Protocol, a
:class:`FormatParserResult` TypedDict, and a registry that maps
extensions to parsers.

The 4 standalone parsers (``_set_parser.py`` / ``_vhdr_parser.py`` /
``_snirf_parser.py`` / ``_mef3_parser.py``) already conform; this
module is the seam they share. The MNE-based EDF/BDF parser
(``_parse_edf_with_mne``) lives here too — it's just an MNE wrapper,
small enough that a dedicated file would be overkill.

FIF stays special-cased in ``3_digest.py``: ``_parse_fif_with_mne``
returns ``(dict, is_split_bool)`` rather than just ``dict``. The
extra flag feeds the split-FIF integrity check downstream and
doesn't fit the shared shape. Documented as such; do not "regularise"
it into this registry without changing the downstream check.

See Also
--------
  individual parsers (``_vhdr_parser.py`` at 69.5% kill ratio).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypedDict

import mne

from _mef3_parser import parse_mef3_metadata
from _set_parser import parse_set_metadata
from _snirf_parser import parse_snirf_metadata
from _vhdr_parser import parse_vhdr_metadata


class FormatParserResult(TypedDict, total=False):
    """The shape returned by every :class:`FormatParser`.

    All fields optional — a parser fills only what the binary header
    actually carries. The cascade in
    ``3_digest.py:_extract_technical_metadata`` reads ``result.get(k)``
    rather than ``result[k]``, so absence is fine.

    Field semantics:

    sampling_frequency
        Hz, floating-point. The parser converts whatever the header
        uses (sampling interval in microseconds for VHDR, sampling
        rate for EDF / BDF / SET) to a unified Hz.
    nchans
        Number of channels. Some headers carry it directly
        (``NumberOfChannels`` in VHDR); others derive from
        ``len(ch_names)``.
    ch_names
        Channel labels as published in the header. The cascade may
        override ``nchans`` to ``len(ch_names)`` when both are
        present and disagree (``channels.tsv`` is authoritative).
    n_times
        Number of samples in the recording (NOT seconds). VHDR's
        header doesn't include this; MNE computes it from the
        binary companion at read time.
    n_samples
        Alias of ``n_times`` used by some older parsers. The
        cascade accepts either key.
    """

    sampling_frequency: float
    nchans: int
    ch_names: list[str]
    n_times: int
    n_samples: int


class FormatParser(Protocol):
    """Callable: ``path -> FormatParserResult | None``.

    Every registered parser must:
    - Accept a single :class:`pathlib.Path` argument.
    - Return a (possibly empty) :class:`FormatParserResult` dict or
      ``None`` when the file can't be parsed (broken, missing, wrong
      format).
    - **Never raise** on recoverable parse failures — the cascade
      relies on ``None`` as the signal. Programmer errors (e.g.,
      wrong argument type) propagate per Phase 9 F1.
    - Be safe to call on broken git-annex symlinks (return ``None``).

    The contract is documented here so future parsers (e.g., a
    BrainVision .ahdr variant, .nirs binary format) can be added
    consistently.
    """

    def __call__(self, path: Path) -> FormatParserResult | None: ...


def _parse_edf_with_mne(path: Path) -> FormatParserResult | None:
    """MNE-based parser for EDF / BDF files.

    Reads the file header (``preload=False``) and returns the standard
    :class:`FormatParserResult` with all 4 fields populated when MNE
    succeeds. Returns ``None`` on any recoverable MNE failure
    (RuntimeError on unsupported variants, OSError on filesystem
    issues, ValueError on truncated header, KeyError on missing
    required fields).

    Previously inlined inside ``3_digest.py``; moved here in P2.2 so
    all 5 single-dict parsers live in one Module.
    """
    if not path.exists():
        return None
    try:
        resolved = path.resolve()
        if not resolved.exists():
            return None
    except (OSError, RuntimeError):
        return None

    try:
        raw = mne.io.read_raw_edf(str(path), preload=False, verbose=False)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        return None

    try:
        result: FormatParserResult = {}
        sfreq = raw.info.get("sfreq")
        if sfreq:
            result["sampling_frequency"] = float(sfreq)
        ch_names = raw.info.get("ch_names")
        if ch_names:
            result["ch_names"] = list(ch_names)
            result["nchans"] = len(ch_names)
        if raw.n_times and raw.n_times > 0:
            result["n_times"] = int(raw.n_times)
        return result if result else None
    finally:
        try:
            raw.close()
        except (OSError, AttributeError):
            pass


def _build_registry() -> dict[str, FormatParser]:
    """Construct the extension → parser map."""
    return {
        ".edf": _parse_edf_with_mne,
        ".bdf": _parse_edf_with_mne,
        ".set": parse_set_metadata,
        ".vhdr": parse_vhdr_metadata,
        ".snirf": parse_snirf_metadata,
        ".mefd": parse_mef3_metadata,
    }


_REGISTRY: dict[str, FormatParser] | None = None


def get_parser_for_extension(ext: str) -> FormatParser | None:
    """Return the registered :class:`FormatParser` for ``ext``, or None.

    Parameters
    ----------
    ext : str
        File extension *with* the leading dot, lower-cased
        (e.g., ``".edf"``). The cascade in
        ``3_digest.py:_extract_technical_metadata`` reads
        ``bids_file_path.suffix.lower()`` and passes it directly.

    Returns
    -------
    FormatParser or None
        ``None`` for any extension this Module doesn't handle (the
        caller falls through to MNE-only paths for FIF, or skips).
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY.get(ext)


def registered_extensions() -> tuple[str, ...]:
    """The set of extensions this Module knows how to parse.

    Returns extensions sorted for stable ordering — used by tests
    that need to enumerate all registered formats.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return tuple(sorted(_REGISTRY.keys()))


__all__ = [
    "FormatParser",
    "FormatParserResult",
    "_parse_edf_with_mne",
    "get_parser_for_extension",
    "registered_extensions",
]
