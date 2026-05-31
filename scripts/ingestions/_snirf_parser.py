"""Parse metadata from SNIRF (Shared Near Infrared Spectroscopy Format) files.

SNIRF files are HDF5-based files for fNIRS data. This module uses MNE
to extract sampling_frequency, nchans, and ch_names from SNIRF files.

Reference: https://github.com/fNIRS/snirf
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency
    h5py = None  # type: ignore[assignment]

try:
    from mne.io import read_raw_snirf
except ImportError:  # pragma: no cover - optional dependency
    read_raw_snirf = None  # type: ignore[assignment]

from _parser_utils import validate_file_path

logger = logging.getLogger(__name__)


def parse_snirf_metadata(snirf_path: Path | str) -> dict[str, Any] | None:
    """Parse metadata from SNIRF file using MNE.

    Extracts sampling frequency, number of channels, channel names,
    and recording length from a SNIRF file.

    Parameters
    ----------
    snirf_path : Path | str
        Path to the SNIRF file.

    Returns
    -------
    dict[str, Any] | None
        Dictionary with keys:
        - sampling_frequency: float (in Hz)
        - nchans: int
        - ch_names: list[str]
        - n_times: int (sample count along the time axis)
        Returns None if file cannot be parsed.

    Notes
    -----
     (C5.1 pattern): ``n_times`` was added after
    a real OpenNeuro fixture (ds007554) surfaced the gap. The synthetic
    h5py fixture validated the parser against itself; the real one
    revealed that ``raw.n_times`` (MNE) / ``len(time)`` (h5py fallback)
    were never read.
    """
    snirf_path = Path(snirf_path)

    # Validate file path (handles broken symlinks from git-annex)
    if not validate_file_path(snirf_path):
        return None

    if read_raw_snirf is None:
        return _parse_snirf_with_h5py(snirf_path)

    try:
        # Read SNIRF file with MNE (preload=False to avoid loading data)
        raw = read_raw_snirf(str(snirf_path), preload=False, verbose=False)
        try:
            result: dict[str, Any] = {}

            # Extract sampling frequency
            sfreq = raw.info.get("sfreq")
            if sfreq:
                result["sampling_frequency"] = float(sfreq)

            # Extract channel info
            ch_names = raw.info.get("ch_names")
            if ch_names:
                result["ch_names"] = list(ch_names)
                result["nchans"] = len(ch_names)

            # Extract recording length.
            # ``raw.n_times`` is always populated on a successfully-read
            # MNE Raw — but defensive-guard against subclasses where it
            # could be 0 or missing (e.g. truncated SNIRF stubs).
            n_times = getattr(raw, "n_times", None)
            if n_times and n_times > 0:
                result["n_times"] = int(n_times)

            if not result:
                return None

            return result
        finally:
            try:
                raw.close()
            except (OSError, AttributeError):
                # OSError from already-closed file; AttributeError if `raw`
                # wasn't a real MNE object (unlikely but defensive).
                pass

    except (OSError, ValueError, KeyError, RuntimeError) as e:
        # MNE's SNIRF reader raises RuntimeError on unsupported variants,
        # OSError on file-system issues, ValueError on schema mismatch,
        # KeyError on missing fields. All recoverable; the h5py fallback
        # catches a different subset, so retrying via the fallback is
        # cheap and worth doing.
        logger.debug("MNE SNIRF parse failed for %s: %s — trying h5py", snirf_path, e)
        return _parse_snirf_with_h5py(snirf_path)


def _parse_snirf_with_h5py(snirf_path: Path) -> dict[str, Any] | None:
    """Fallback SNIRF parser using h5py directly.

    Parameters
    ----------
    snirf_path : Path
        Path to the SNIRF file.

    Returns
    -------
    dict[str, Any] | None
        Parsed metadata or None.

    """
    if h5py is None:
        return None

    result: dict[str, Any] = {}

    try:
        with h5py.File(snirf_path, "r") as f:
            # Find the nirs group
            nirs_group = None
            for key in f.keys():
                if key.startswith("nirs"):
                    nirs_group = f[key]
                    break

            if nirs_group is None:
                return None

            # Extract sampling frequency from time vector. Also record
            # ``n_times`` — the
            # time vector length IS the sample count along the time axis.
            for data_key in nirs_group.keys():
                if data_key.startswith("data"):
                    data_group = nirs_group[data_key]
                    if "time" in data_group:
                        time_ds = data_group["time"]
                        # ``.shape`` is HDF5 metadata — reads ZERO elements,
                        # unlike the previous ``[:]`` which loaded the whole vector.
                        n_time_points = int(time_ds.shape[0]) if time_ds.shape else 0
                        if n_time_points > 0:
                            result["n_times"] = n_time_points
                        if n_time_points > 1:
                            # Only the first two samples are needed for the rate.
                            first_two = time_ds[:2]
                            dt = float(first_two[1] - first_two[0])
                            if dt > 0:
                                result["sampling_frequency"] = float(1.0 / dt)
                    break

            # Count channels from measurementList
            nchans = 0
            ch_names = []

            # Get source and detector labels if available
            source_labels = []
            detector_labels = []

            if "probe" in nirs_group:
                probe = nirs_group["probe"]
                if "sourceLabels" in probe:
                    source_labels = [
                        s.decode("utf-8") if isinstance(s, bytes) else str(s)
                        for s in probe["sourceLabels"][:]
                    ]
                if "detectorLabels" in probe:
                    detector_labels = [
                        d.decode("utf-8") if isinstance(d, bytes) else str(d)
                        for d in probe["detectorLabels"][:]
                    ]

            # Count measurement lists (channels)
            for data_key in nirs_group.keys():
                if data_key.startswith("data"):
                    data_group = nirs_group[data_key]
                    for ml_key in data_group.keys():
                        if ml_key.startswith("measurementList"):
                            nchans += 1
                            ml = data_group[ml_key]

                            src_idx = int(ml["sourceIndex"][()]) - 1
                            det_idx = int(ml["detectorIndex"][()]) - 1

                            if source_labels and detector_labels:
                                src_label = (
                                    source_labels[src_idx]
                                    if src_idx < len(source_labels)
                                    else f"S{src_idx + 1}"
                                )
                                det_label = (
                                    detector_labels[det_idx]
                                    if det_idx < len(detector_labels)
                                    else f"D{det_idx + 1}"
                                )
                                ch_names.append(f"{src_label}-{det_label}")
                            else:
                                ch_names.append(f"S{src_idx + 1}-D{det_idx + 1}")

            if nchans > 0:
                result["nchans"] = int(nchans)
            if ch_names:
                result["ch_names"] = ch_names

    except (OSError, ValueError, KeyError, AttributeError) as e:
        # OSError = not a valid HDF5; KeyError = expected dataset path
        # missing (different SNIRF version); ValueError/AttributeError =
        # unexpected element shape. All are recoverable parse failures.
        logger.debug("h5py SNIRF parse failed for %s: %s", snirf_path, e)
        return None

    if not result:
        return None

    return result
