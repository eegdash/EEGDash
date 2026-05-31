"""Pure EDF/BDF 256-byte main-header math — annotation-safe, no network.

The EDF/BDF main header is a fixed 256-byte block. ``n_times`` is derived from
the **record count**, not a flat ``size / (nchans x dtype)`` divide::

    n_times = number_of_data_records x round(sfreq x duration_of_data_record)

Using the record count is what makes this **annotation-safe**: an EDF+ file
carries an extra *EDF Annotations* signal whose ``samples_per_record`` differs
from the data channels, so a flat size divide is corrupted by it — but the
record count is independent of how many signals (data or annotation) a record
holds.

Field layout in the 256-byte main header (each field is left-justified ASCII):

==========================  ========  ======
field                       offset    width
==========================  ========  ======
number_of_data_records      236       8
duration_of_data_record     244       8
n_signals                   252       4
==========================  ========  ======

``sfreq`` and ``nchans`` come from the BIDS sidecar — the EDF itself contributes
**0 bytes** for the size-arithmetic path, and only the 256-byte main header for
the record-count path.

Every helper obeys the ``FormatParser`` contract: **never raise on a recoverable
failure — return ``None``** (unclean-stop ``-1``, missing sfreq, bad ASCII,
truncated buffer, …). Callers fall back to the next resolution tier.
"""

from __future__ import annotations

# ─── EDF main-header geometry (bytes) ────────────────────────────────────────

#: Total length of the EDF/BDF main header.
EDF_HEADER_LEN = 256

#: ``number_of_data_records`` — 8 ASCII chars at byte offset 236.
_OFF_N_RECORDS = 236
_LEN_N_RECORDS = 8

#: ``duration_of_data_record`` (seconds) — 8 ASCII chars at byte offset 244.
_OFF_REC_DURATION = 244
_LEN_REC_DURATION = 8

#: ``n_signals`` — 4 ASCII chars at byte offset 252.
_OFF_N_SIGNALS = 252
_LEN_N_SIGNALS = 4

#: Bytes per sample by extension. EDF stores 16-bit, BDF 24-bit integers.
_BYTES_PER_SAMPLE = {".edf": 2, ".bdf": 3}


def _read_ascii_int(buf: bytes, offset: int, length: int) -> int | None:
    """Parse a fixed-width ASCII integer field; return ``None`` on any failure.

    EDF header fields are ASCII, left-justified, space-padded. A blank field,
    a non-ASCII byte, or a truncated buffer all yield ``None`` rather than
    raising — upholding the ``FormatParser`` recoverable-failure contract.
    """
    if offset + length > len(buf):
        return None
    try:
        text = buf[offset : offset + length].decode("ascii").strip()
    except (UnicodeDecodeError, ValueError):
        return None
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def edf_n_times_from_main_header(buf: bytes, sfreq: float | None) -> int | None:
    """Compute ``n_times`` from the 256-byte EDF/BDF main header (annotation-safe).

    Parses ``number_of_data_records`` (@236), ``duration_of_data_record`` (@244)
    and ``n_signals`` (@252) from ``buf``. When the record count is positive and
    both ``sfreq`` and the record duration are positive::

        n_times = round(sfreq x duration_of_data_record) x number_of_data_records

    Returns ``None`` (→ caller falls back to size-arithmetic / unresolved) on:
    unclean-stop (``number_of_data_records == -1``), zero records, missing or
    non-positive ``sfreq``, non-positive record duration, bad ASCII, or a
    truncated buffer. **Never raises.**

    This uses the RECORD COUNT, not a flat size divide, so an EDF+ annotations
    signal cannot inflate the result.
    """
    try:
        n_records = _read_ascii_int(buf, _OFF_N_RECORDS, _LEN_N_RECORDS)
        record_duration = _read_ascii_int(buf, _OFF_REC_DURATION, _LEN_REC_DURATION)
        # n_signals is parsed for completeness/validation; not used in the
        # annotation-safe record-count formula.
        _ = _read_ascii_int(buf, _OFF_N_SIGNALS, _LEN_N_SIGNALS)

        if n_records is None or n_records <= 0:
            return None  # unclean-stop (-1), zero, or unparsable
        if record_duration is None or record_duration <= 0:
            return None
        if sfreq is None or sfreq <= 0:
            return None

        samples_per_record = round(sfreq * record_duration)
        if samples_per_record <= 0:
            return None
        return samples_per_record * n_records
    except Exception:  # noqa: BLE001
        # Defensive catch-all: the contract is "never raise on recoverable
        # failure". Any unexpected error degrades to the next tier.
        return None


def edf_bytes_per_sample(ext: str) -> int:
    """Bytes per sample for an EDF/BDF extension (``.edf`` → 2, ``.bdf`` → 3).

    Case-insensitive. Unknown extensions default to 2 (EDF) — callers that need
    strictness should check the extension before calling.
    """
    return _BYTES_PER_SAMPLE.get(ext.lower(), 2)


def edf_size_arithmetic_n_times(file_size: int, nchans: int, ext: str) -> int | None:
    """``n_times`` from file size alone — UNSAFE for EDF+ (annotations channel).

    Computes::

        n_times = (file_size - 256 x (nchans + 1)) / (nchans x bytes_per_sample)

    only when the division is exact; otherwise returns ``None`` (never a guessed
    value). Verified on ``ds004577``::

        (2,254,720 - 256 x 20) / (19 x 2) = 59,200.

    .. warning::
        This is **UNSAFE for EDF+/BDF+** files: an *EDF Annotations* signal is
        counted in ``nchans`` but its ``samples_per_record`` differs, so the
        flat divide is wrong. Callers must use this **only** when no annotations
        channel is present (uniform sampling). It is provided for
        completeness/tests; the annotation-safe path is
        :func:`edf_n_times_from_main_header`.

    Returns ``None`` on non-positive ``nchans``, a size smaller than the header,
    or a non-even divide. **Never raises.**
    """
    try:
        if nchans <= 0:
            return None
        header_bytes = EDF_HEADER_LEN * (nchans + 1)
        data_bytes = file_size - header_bytes
        if data_bytes <= 0:
            return None
        divisor = nchans * edf_bytes_per_sample(ext)
        if divisor <= 0 or data_bytes % divisor != 0:
            return None
        return data_bytes // divisor
    except Exception:  # noqa: BLE001
        return None
