"""A hand-written WFDB record as I-CARE ships it, for the to_bids tests.

A text header ``<name>.hea`` and a MATLAB v4 ``<name>.mat`` holding one int16
matrix ``val`` (signals as rows), written byte by byte, so no WFDB writer is
needed: the signal lines name the ``.mat`` file in format ``16+24`` (format 16
after the MATLAB header's 24 bytes), as PhysioNet's I-CARE 2.1 headers do.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Sequence

import numpy as np

MAT_INT16 = 30  # MATLAB v4 type: little-endian, int16, full numeric matrix


def write_record(
    folder: Path,
    name: str,
    labels: Sequence[str],
    data: np.ndarray,
    *,
    sfreq: float,
    gains: Sequence[float],
    baselines: Sequence[int],
    units: str | None = "nu",
    comments: Sequence[str] = (),
) -> Path:
    """Write *data* (samples x signals, int16) as ``<name>.hea`` + ``<name>.mat``; returns the header.

    *units* ``None`` writes no unit (WFDB then reads ``mV``); *comments* are
    header lines after ``#``.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data, dtype="<i2")
    n_times, ns = data.shape
    var = b"val\x00"
    mat = struct.pack("<5i", MAT_INT16, ns, n_times, 0, len(var)) + var
    # column-major (signals x samples) is the samples-x-signals array row by row
    (folder / f"{name}.mat").write_bytes(mat + data.tobytes())
    lines = [f"{name} {ns} {sfreq} {n_times}"]
    for k, label in enumerate(labels):
        unit = f"/{units}" if units else ""
        checksum = int(np.sum(data[:, k], dtype=np.int64)) & 0xFFFF
        checksum = checksum - 0x10000 if checksum >= 0x8000 else checksum
        lines.append(
            f"{name}.mat 16+24 {gains[k]}({baselines[k]}){unit} 16 0 "
            f"{int(data[0, k])} {checksum} 0 {label}"
        )
    lines += [f"#{c}" for c in comments]
    header = folder / f"{name}.hea"
    header.write_text("\n".join(lines) + "\n")
    return header


def write_patient(folder: Path, patient: str, fields: dict[str, str]) -> Path:
    """The patient's clinical file ``<patient>.txt``: ``Patient`` then ``Key: value`` lines."""
    path = Path(folder) / f"{patient}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"Patient: {patient}", *(f"{k}: {v}" for k, v in fields.items())]
    path.write_text("\n".join(lines) + "\n")
    return path
