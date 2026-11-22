"""A hand-written EDF (or EDF+ with an annotation signal), any label list, for the to_bids tests.

Header and int16 records are written byte by byte, as in ``tests/unit_tests/io/test_edf.py``,
so no EDF writer (edfio, MNE export) is needed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

ANNOTATION_SAMPLES = 16  # int16 slots of the EDF+ annotation signal per record


def _field(value, width: int) -> bytes:
    return str(value).ljust(width).encode("latin-1")[:width]


def write_edf(
    path: Path,
    labels: list[str],
    *,
    sfreq: int = 256,
    seconds: int = 12,
    start: datetime = datetime(2020, 1, 1, 10, 0, 0),
    unit: str = "uV",
    annotations: bool = False,
    seed: int = 0,
) -> Path:
    """Write *labels* as 1-s records of *sfreq* int16 samples (±3200 *unit*).

    With *annotations*, an EDF+C file whose last signal is ``EDF Annotations``
    (the time-keeping annotation of each record), a signal MNE leaves out of
    its channel list.
    """
    signals = [*labels, "EDF Annotations"] if annotations else list(labels)
    ns = len(signals)
    date = start.strftime("%d-%b-%Y").upper()
    recording = f"Startdate {date} X X X" if annotations else "X"
    header = b"".join(
        [
            _field("0", 8),
            _field("X X X X", 80),
            _field(recording, 80),
            _field(f"{start:%d.%m.%y}", 8),
            _field(f"{start:%H.%M.%S}", 8),
            _field(256 * (ns + 1), 8),
            _field("EDF+C" if annotations else "", 44),
            _field(seconds, 8),
            _field(1, 8),
            _field(ns, 4),
        ]
    )
    header += b"".join(_field(s, 16) for s in signals)
    n_eeg = len(labels)
    per_signal = [
        (80, [""] * ns),  # transducer
        (8, [unit] * n_eeg + [""] * (ns - n_eeg)),
        (8, [-3200] * n_eeg + [-1] * (ns - n_eeg)),
        (8, [3200] * n_eeg + [1] * (ns - n_eeg)),
        (8, [-32768] * ns),
        (8, [32767] * ns),
        (80, [""] * ns),  # prefiltering
        (8, [sfreq] * n_eeg + [ANNOTATION_SAMPLES] * (ns - n_eeg)),
        (32, [""] * ns),
    ]
    for width, values in per_signal:
        header += b"".join(_field(v, width) for v in values)
    rng = np.random.default_rng(seed)
    body = b""
    for k in range(seconds):
        body += rng.integers(-1000, 1000, (n_eeg, sfreq)).astype("<i2").tobytes()
        if annotations:
            tal = f"+{k}\x14\x14\x00".encode("latin-1")
            body += tal.ljust(2 * ANNOTATION_SAMPLES, b"\x00")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + body)
    return path
