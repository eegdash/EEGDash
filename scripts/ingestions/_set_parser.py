"""Parse metadata from EEGLAB .set files.

This module provides a fallback mechanism to extract EEG metadata
(sampling_frequency, nchans, ch_names) from EEGLAB .set header files
when BIDS sidecar files (JSON/TSV) are unavailable or when the
companion .fdt data file is missing.

EEGLAB .set files are MATLAB .mat files that can store data in two ways:
1. Embedded: Data stored directly in the .set file
2. External: Data stored in a companion .fdt file, .set contains metadata only
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from _parser_utils import validate_file_path

logger = logging.getLogger(__name__)

# Above this size, an embedded-data .set (no .fdt) is NOT loadmat-ed: scipy
# would materialize the whole EEG.data array. The sidecar-arithmetic tier
# supplies n_times for those instead. Bounds worst-case memory/time.
_SET_EMBEDDED_LOADMAT_CEILING = 50 * 1024 * 1024  # 50 MB

#: EEG fields an EEGLAB "onefile" save stores as top-level MAT variables.
_ROOT_FIELDS = ("srate", "nbchan", "pnts", "trials", "chanlocs", "datfile")


def _set_exceeds_embedded_ceiling(set_path: Path) -> bool:
    """True when *set_path* is larger than the embedded-data loadmat ceiling."""
    try:
        return set_path.stat().st_size > _SET_EMBEDDED_LOADMAT_CEILING
    except OSError:
        return False


# scipy's MatReadError isn't in the public io namespace, but it's the
# canonical class raised for truncated/corrupt MAT files. Importing it
# lazily so an environment without scipy still loads this module —
# falling back to a no-op base class so the `except` below is harmless.
try:
    from scipy.io.matlab._miobase import MatReadError as _MatReadError
except ImportError:  # pragma: no cover — scipy unavailable

    class _MatReadError(Exception):
        pass


def _fdt_n_times(fdt_path: Path, nchans: int | None) -> int | None:
    """``n_times`` for an external EEGLAB ``.fdt`` from its SIZE alone — 0 bytes read.

    The external ``.fdt`` is a raw ``float32`` ``[nchans x n_times]`` block with no
    header, so ``n_times = fdt_size / (nchans x 4)``. The size comes from the
    git-annex key on a shallow clone (``_sizing.data_file_size`` reads the pointer,
    never the signal). Returns ``None`` when *nchans* is falsy/non-positive, the
    size is unavailable, or the size is not a whole multiple of ``nchans x 4`` —
    never a guessed value. Never raises (FormatParser contract).
    """
    try:
        if not nchans or nchans <= 0:
            return None
        from _sizing import data_file_size

        size = data_file_size(fdt_path)
        if not size or size % (nchans * 4) != 0:
            return None
        return size // (nchans * 4)
    except Exception as e:  # noqa: BLE001 — never raise on a recoverable parse failure
        logger.debug("_fdt_n_times failed: %s", e)
        return None


def _maybe_fill_ntimes_from_fdt(
    result: dict[str, Any], set_path: Path, trials: int | None
) -> None:
    """Set ``result['n_times']`` from the companion ``.fdt`` size when safe.

    Only fires for CONTINUOUS data: if the parsed EEG struct exposes
    ``trials > 1`` (epoched, 3-D), the flat ``size / (nchans x 4)`` divide would
    be wrong, so we skip it. When *trials* is unknown (``None``) we allow it —
    continuous is EEGLAB's default. No-op unless ``has_fdt`` is True, an
    ``n_times``/``n_samples`` is still absent, and ``nchans`` is known. Mutates
    *result* in place; never raises.
    """
    try:
        # A 0 (EEG.pnts==0 on a header-only external .set) counts as absent —
        # zero samples is not a real recording; the .fdt size is the truth.
        if result.get("n_times") or result.get("n_samples"):
            return
        if not result.get("has_fdt"):
            return
        if trials is not None and trials > 1:
            return  # epoched -> flat divide unsafe
        nchans = result.get("nchans")
        n_times = _fdt_n_times(set_path.with_suffix(".fdt"), nchans)
        if n_times is not None:
            result["n_times"] = n_times
    except Exception as e:  # noqa: BLE001 — never raise on a recoverable parse failure
        logger.debug("_maybe_fill_ntimes_from_fdt failed: %s", e)


def _h5_scalar(group: Any, name: str) -> float | None:
    """A 1x1 MATLAB scalar dataset as float, ``None`` when absent or not scalar."""
    try:
        return float(group[name][()].ravel()[0]) if name in group else None
    except (TypeError, ValueError, IndexError, KeyError, AttributeError):
        return None


def set_fields_from_h5(f: Any) -> dict[str, Any]:
    """``sampling_frequency``/``nchans``/``n_samples``/``trials`` of a MAT-v7.3 ``.set``.

    EEGLAB writes the EEG fields either under an ``EEG`` group or, for its own
    v7.3 "onefile" saves, at the file root (on004105: ``/srate``, ``/pnts``,
    ``/data``). Reads only the 1x1 scalars, never ``data``; works on any
    h5py-openable file-like, so a ranged remote reader fetches metadata only.
    """
    eeg = f["EEG"] if "EEG" in f else f
    out: dict[str, Any] = {}
    srate, nbchan = _h5_scalar(eeg, "srate"), _h5_scalar(eeg, "nbchan")
    pnts, trials = _h5_scalar(eeg, "pnts"), _h5_scalar(eeg, "trials")
    if srate:
        out["sampling_frequency"] = srate
    if nbchan:
        out["nchans"] = int(nbchan)
    if pnts:
        out["n_samples"] = int(pnts)
    if trials:
        out["trials"] = int(trials)
    if srate and pnts and srate > 0:
        out["duration"] = pnts / srate
    return out


def parse_set_metadata(set_path: Path | str) -> dict[str, Any] | None:
    """Parse metadata from EEGLAB .set file.

    Extracts sampling frequency, number of channels, and channel names
    from a .set file without loading the actual data. This works even
    when the companion .fdt file is missing.

    Parameters
    ----------
    set_path : Path | str
        Path to the .set file.

    Returns
    -------
    dict[str, Any] | None
        Dictionary with keys:
        - sampling_frequency: float (in Hz)
        - nchans: int
        - ch_names: list[str] (if available)
        - n_samples: int (if available)
        - duration: float in seconds (if available)
        - has_fdt: bool - whether companion .fdt file exists
        Returns None if file cannot be parsed.

    Notes
    -----
    Uses scipy.io.loadmat for MATLAB v5 format files.
    For newer HDF5-based .set files, uses h5py.

    """
    set_path = Path(set_path)

    # Use the shared validate_file_path so .set files behind broken
    # git-annex symlinks are handled consistently
    # with the other parsers (previously this just called .exists()
    # which returns True for broken symlinks pointing into git-annex).
    if not validate_file_path(set_path):
        return None

    result: dict[str, Any] = {}

    # Check for companion .fdt file
    fdt_path = set_path.with_suffix(".fdt")
    result["has_fdt"] = fdt_path.exists()

    # Try scipy.io first (for MATLAB v5 format)
    try:
        import scipy.io

        # Header-only discipline: for embedded-data .set (no .fdt companion),
        # scipy loadmat materializes EEG.data. Skip *only the loadmat* for
        # oversized embedded files (mat = {}), then fall through to the cheap
        # h5py branch below — for MAT-v7.3 (HDF5) .set, EEG/srate, EEG/nbchan
        # and EEG/pnts are 1x1 scalar datasets readable without loading EEG.data.
        if not result["has_fdt"] and _set_exceeds_embedded_ceiling(set_path):
            mat: dict[str, Any] = {}
        else:
            # variable_names=['EEG'] avoids loading unrelated top-level MAT variables.
            mat = scipy.io.loadmat(
                str(set_path),
                struct_as_record=False,
                squeeze_me=True,
                variable_names=["EEG"],
            )
            if "EEG" not in mat:
                # EEGLAB "onefile" saves put the fields at the file root
                # (on007822); load the scalars only, never ``data``.
                root = scipy.io.loadmat(
                    str(set_path),
                    struct_as_record=False,
                    squeeze_me=True,
                    variable_names=list(_ROOT_FIELDS),
                )
                if "srate" in root:
                    mat = {
                        "EEG": SimpleNamespace(
                            **{k: root[k] for k in _ROOT_FIELDS if k in root}
                        )
                    }

        if "EEG" in mat:
            eeg = mat["EEG"]

            # Extract sampling rate
            if hasattr(eeg, "srate"):
                srate = eeg.srate
                if hasattr(srate, "item"):
                    srate = srate.item()
                result["sampling_frequency"] = float(srate)

            # Extract number of channels
            if hasattr(eeg, "nbchan"):
                nbchan = eeg.nbchan
                if hasattr(nbchan, "item"):
                    nbchan = nbchan.item()
                result["nchans"] = int(nbchan)

            # Extract number of samples/points
            if hasattr(eeg, "pnts"):
                pnts = eeg.pnts
                if hasattr(pnts, "item"):
                    pnts = pnts.item()
                result["n_samples"] = int(pnts)

            # Calculate duration if we have both
            if "sampling_frequency" in result and "n_samples" in result:
                if result["sampling_frequency"] > 0:
                    result["duration"] = (
                        result["n_samples"] / result["sampling_frequency"]
                    )

            # Extract channel names from chanlocs structure
            if hasattr(eeg, "chanlocs") and eeg.chanlocs is not None:
                try:
                    chanlocs = eeg.chanlocs
                    ch_names = []

                    # chanlocs can be a single struct or array of structs
                    if hasattr(chanlocs, "__iter__") and not isinstance(chanlocs, str):
                        for ch in chanlocs:
                            if hasattr(ch, "labels"):
                                label = ch.labels
                                if hasattr(label, "item"):
                                    label = label.item()
                                ch_names.append(str(label))
                    elif hasattr(chanlocs, "labels"):
                        # Single channel case
                        label = chanlocs.labels
                        if hasattr(label, "item"):
                            label = label.item()
                        ch_names.append(str(label))

                    if ch_names:
                        result["ch_names"] = ch_names
                        # Update nchans if not set
                        if "nchans" not in result:
                            result["nchans"] = len(ch_names)
                except (AttributeError, IndexError, TypeError, ValueError) as e:
                    # .chanlocs is an arbitrary MATLAB structure — any of
                    # these mean the labels field isn't shaped like we
                    # expect. Recoverable: we just skip channel-name
                    # extraction and let downstream callers fall back to
                    # synthesised names.
                    logger.debug("Could not extract channel names: %s", e)

            # Check for external data file reference
            if hasattr(eeg, "datfile") and eeg.datfile:
                datfile = eeg.datfile
                if hasattr(datfile, "item"):
                    datfile = datfile.item()
                result["external_datafile"] = str(datfile)

            # Epoched .set are 3-D (EEG.trials > 1); the flat .fdt divide is
            # only valid for continuous data. Capture trials to gate it below.
            trials: int | None = None
            if hasattr(eeg, "trials"):
                try:
                    t = eeg.trials
                    if hasattr(t, "item"):
                        t = t.item()
                    trials = int(t)
                except (TypeError, ValueError):
                    trials = None

            # External-.fdt zero-fetch n_times: only when n_times/n_samples is
            # still absent and the data is continuous (trials <= 1 or unknown).
            result["trials"] = trials
            _maybe_fill_ntimes_from_fdt(result, set_path, trials)

            if result:
                logger.debug(
                    "Extracted from .set: sfreq=%s, nchans=%s, has_fdt=%s",
                    result.get("sampling_frequency"),
                    result.get("nchans"),
                    result.get("has_fdt"),
                )
                return result

    except ImportError:
        logger.debug("scipy not available for .set parsing")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        NotImplementedError,
    ) as e:
        # `scipy.io.loadmat` raises NotImplementedError on MAT-v7.3 (HDF5)
        # files, which is the trigger for our h5py fallback below. The
        # other classes cover: OSError (file not found / unreadable),
        # ValueError (truncated or wrong magic), KeyError (missing 'EEG'
        # struct in the .set), TypeError (unexpected MATLAB field shape).
        logger.debug("scipy.io.loadmat failed: %s", e)
    except _MatReadError as e:
        # MatReadError is scipy's own class. Imported lazily below the
        # try so the import isn't required when scipy is missing.
        logger.debug("scipy.io.loadmat: corrupted MAT file: %s", e)

    # Try h5py for HDF5-based .set files (newer EEGLAB format)
    try:
        import h5py

        with h5py.File(str(set_path), "r") as f:
            fields = set_fields_from_h5(f)
            if fields:
                result.update(fields)
                trials_h5 = fields.get("trials")
                _maybe_fill_ntimes_from_fdt(result, set_path, trials_h5)

                if result:
                    return result

    except ImportError:
        logger.debug("h5py not available for .set parsing")
    except (OSError, ValueError, KeyError) as e:
        # h5py raises OSError for "not an HDF5 file" (which happens on
        # every MAT-v5 .set), ValueError for malformed datasets, and
        # KeyError when navigating an unexpected struct layout. None
        # of these are programmer bugs — log and fall through to the
        # 'return None or partial result' tail.
        logger.debug("h5py parsing failed: %s", e)

    return None if not result else result


def diagnose_set_issues(set_path: Path | str) -> dict[str, Any]:
    """Diagnose issues with an EEGLAB .set file and its companion .fdt.

    Parameters
    ----------
    set_path : Path | str
        Path to the .set file.

    Returns
    -------
    dict[str, Any]
        Diagnosis with keys:
        - status: "ok" | "missing_fdt" | "error"
        - issues: list of issue descriptions
        - metadata: extracted metadata (if available)
        - can_extract_metadata: bool

    """
    set_path = Path(set_path)
    result = {
        "status": "ok",
        "issues": [],
        "metadata": None,
        "can_extract_metadata": False,
    }

    if not set_path.exists():
        result["status"] = "error"
        result["issues"].append(f".set file not found: {set_path}")
        return result

    # Try to extract metadata
    metadata = parse_set_metadata(set_path)
    if metadata:
        result["metadata"] = metadata
        result["can_extract_metadata"] = True

        # Check for missing .fdt
        if not metadata.get("has_fdt", True):
            if metadata.get("external_datafile"):
                result["status"] = "missing_fdt"
                result["issues"].append(
                    f"Missing .fdt file: {metadata['external_datafile']}"
                )
            else:
                # Data might be embedded, which is fine
                pass
    else:
        result["status"] = "error"
        result["issues"].append("Could not parse .set file")

    return result
