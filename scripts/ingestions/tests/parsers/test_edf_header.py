"""Unit tests for the pure EDF/BDF 256-byte main-header math (``_edf_header``).

These tests are annotation-safe and network-free: they hand-build a 256-byte EDF
main header and assert that ``n_times`` is derived from the RECORD COUNT (not a
flat size divide), so an EDF+ annotations channel cannot corrupt the result.

The golden value (59,200) is the verified ``ds004577`` recording:
``37 records × round(200 Hz × 8 s) = 37 × 1600 = 59,200``.
"""

from __future__ import annotations

import pytest

from _edf_header import (
    edf_bytes_per_sample,
    edf_n_times_from_main_header,
    edf_size_arithmetic_n_times,
)

# ─── Module-level builders / constants (pre-commit forbids nested defs) ──────

#: EDF main-header field offsets (each header field is fixed-width ASCII).
_OFF_N_RECORDS = 236  # number_of_data_records  @236:244  (8 ASCII)
_OFF_REC_DUR = 244  # duration_of_data_record @244:252  (8 ASCII)
_OFF_N_SIGNALS = 252  # n_signals               @252:256  (4 ASCII)
_HEADER_LEN = 256


def _put_ascii_field(buf: bytearray, offset: int, width: int, value: int | str) -> None:
    """Write a left-justified, space-padded ASCII field into ``buf`` in place."""
    field = str(value).ljust(width)[:width].encode("ascii")
    buf[offset : offset + width] = field


def build_edf_main_header(
    n_records: int | str,
    record_duration: int | str,
    n_signals: int | str,
) -> bytes:
    """Assemble a 256-byte EDF main header with the three count fields set.

    Fields are left-justified ASCII padded with spaces, exactly as the EDF
    spec mandates. Bytes outside the three count fields are spaces (their
    content is irrelevant to ``n_times`` math).
    """
    buf = bytearray(b" " * _HEADER_LEN)
    _put_ascii_field(buf, _OFF_N_RECORDS, 8, n_records)
    _put_ascii_field(buf, _OFF_REC_DUR, 8, record_duration)
    _put_ascii_field(buf, _OFF_N_SIGNALS, 4, n_signals)
    return bytes(buf)


# The verified ds004577 header: 37 records, 8 s/record, 19 signals.
DS004577_HEADER = build_edf_main_header(37, 8, 19)
DS004577_SFREQ = 200.0
DS004577_N_TIMES = 59200  # 37 × round(200 × 8)


# ─── edf_n_times_from_main_header — golden + record-count semantics ──────────


def test_n_times_matches_verified_ds004577_value():
    assert (
        edf_n_times_from_main_header(DS004577_HEADER, sfreq=DS004577_SFREQ)
        == DS004577_N_TIMES
    )


def test_n_times_is_record_count_times_samples_per_record():
    # 10 records × round(256 × 1) = 2560 — independent of n_signals (annotation-safe).
    hdr = build_edf_main_header(n_records=10, record_duration=1, n_signals=33)
    assert edf_n_times_from_main_header(hdr, sfreq=256.0) == 2560


def test_n_times_rounds_fractional_samples_per_record():
    # round(199.6 × 8) = round(1596.8) = 1597 samples/record × 5 records.
    hdr = build_edf_main_header(n_records=5, record_duration=8, n_signals=4)
    assert edf_n_times_from_main_header(hdr, sfreq=199.6) == 1597 * 5


# ─── edf_n_times_from_main_header — every failure returns None (never raises) ─


def test_unclean_stop_record_count_minus_one_returns_none():
    hdr = build_edf_main_header(n_records=-1, record_duration=8, n_signals=19)
    assert edf_n_times_from_main_header(hdr, sfreq=DS004577_SFREQ) is None


def test_zero_records_returns_none():
    hdr = build_edf_main_header(n_records=0, record_duration=8, n_signals=19)
    assert edf_n_times_from_main_header(hdr, sfreq=DS004577_SFREQ) is None


def test_missing_sfreq_returns_none():
    assert edf_n_times_from_main_header(DS004577_HEADER, sfreq=None) is None


def test_zero_sfreq_returns_none():
    assert edf_n_times_from_main_header(DS004577_HEADER, sfreq=0.0) is None


def test_zero_record_duration_returns_none():
    hdr = build_edf_main_header(n_records=37, record_duration=0, n_signals=19)
    assert edf_n_times_from_main_header(hdr, sfreq=DS004577_SFREQ) is None


def test_truncated_buffer_returns_none():
    truncated = DS004577_HEADER[:240]  # cut off before the count fields end
    assert edf_n_times_from_main_header(truncated, sfreq=DS004577_SFREQ) is None


def test_empty_buffer_returns_none():
    assert edf_n_times_from_main_header(b"", sfreq=DS004577_SFREQ) is None


def test_non_ascii_count_field_returns_none():
    hdr = bytearray(DS004577_HEADER)
    hdr[_OFF_N_RECORDS : _OFF_N_RECORDS + 8] = b"\xff\xff\xff\xff    "
    assert edf_n_times_from_main_header(bytes(hdr), sfreq=DS004577_SFREQ) is None


def test_blank_count_field_returns_none():
    hdr = build_edf_main_header(n_records="", record_duration=8, n_signals=19)
    assert edf_n_times_from_main_header(hdr, sfreq=DS004577_SFREQ) is None


# ─── edf_bytes_per_sample ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("ext", "expected"),
    [
        (".edf", 2),
        (".EDF", 2),
        (".bdf", 3),
        (".BDF", 3),
    ],
)
def test_bytes_per_sample(ext: str, expected: int):
    assert edf_bytes_per_sample(ext) == expected


# ─── edf_size_arithmetic_n_times (UNSAFE for EDF+ — completeness/tests only) ──


def test_size_arithmetic_matches_verified_ds004577_value():
    # (2,254,720 − 256×(19+1)) / (19 × 2) = 59,200.
    assert edf_size_arithmetic_n_times(2254720, 19, ".edf") == DS004577_N_TIMES


def test_size_arithmetic_bdf_uses_three_bytes_per_sample():
    nchans = 8
    n_times = 1000
    size = 256 * (nchans + 1) + nchans * 3 * n_times
    assert edf_size_arithmetic_n_times(size, nchans, ".bdf") == n_times


def test_size_arithmetic_non_even_divide_returns_none():
    # One trailing byte breaks the even divide → None (never a guessed value).
    assert edf_size_arithmetic_n_times(2254721, 19, ".edf") is None


def test_size_arithmetic_zero_or_negative_nchans_returns_none():
    assert edf_size_arithmetic_n_times(2254720, 0, ".edf") is None
    assert edf_size_arithmetic_n_times(2254720, -1, ".edf") is None


def test_size_arithmetic_size_smaller_than_header_returns_none():
    assert edf_size_arithmetic_n_times(10, 19, ".edf") is None
