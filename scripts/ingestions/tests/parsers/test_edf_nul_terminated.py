"""EDF headers whose ASCII number fields end in a NUL byte.

Some clinical exporters (the ``POL ...`` labelled iEEG of on004551, on006233,
on006914) write C-style terminators into the fixed-width fields:
``number_of_data_records`` is ``b"551\\x00    "`` and every signal's
``samples_per_record`` is ``b"614\\x00    "``. MNE reads these files; the header
math returned ``None``, so those recordings were left without ``ntimes`` and
their datasets without a duration (825 recordings across the three).

Golden values: on006914 ``sub-..._ieeg.edf`` (33,844,456 B), read with
``mne.io.read_raw_edf``: sfreq 1000.0, n_times 338,314 (= 551 x 614), 49 channels
(50 signals minus one ``EDF Annotations``).
"""

from __future__ import annotations

from _edf_header import edf_n_times_from_main_header, edf_raw_shape

_HEADER_LEN = 256


def _field(value: str, width: int, *, nul: bool) -> bytes:
    text = f"{value}\x00" if nul else value
    return text.ljust(width)[:width].encode("ascii")


def _main(n_records: int, duration: str, n_signals: int, *, nul: bool) -> bytes:
    buf = bytearray(b" " * _HEADER_LEN)
    buf[236:244] = _field(str(n_records), 8, nul=nul)
    buf[244:252] = _field(duration, 8, nul=False)
    buf[252:256] = _field(str(n_signals), 4, nul=False)
    return bytes(buf)


def _signals(labels: list[str], spr: list[int], *, nul: bool) -> bytes:
    n = len(labels)
    buf = bytearray(b" " * (256 * n))
    for i, label in enumerate(labels):
        buf[16 * i : 16 * i + 16] = label.ljust(16).encode("ascii")
        off = 216 * n + 8 * i
        buf[off : off + 8] = _field(str(spr[i]), 8, nul=nul)
    return bytes(buf)


_LABELS = [f"POL B{i}-B{i + 8}" for i in range(49)] + ["EDF Annotations"]
_SPR = [614] * 49 + [60]


def test_raw_shape_reads_nul_terminated_fields_like_mne():
    main = _main(551, "0.614000", 50, nul=True)
    sig = _signals(_LABELS, _SPR, nul=True)
    assert edf_raw_shape(main, sig) == (1000.0, 338314, 49)


def test_raw_shape_space_padded_control_unchanged():
    main = _main(551, "0.614000", 50, nul=False)
    sig = _signals(_LABELS, _SPR, nul=False)
    assert edf_raw_shape(main, sig) == (1000.0, 338314, 49)


def test_main_formula_reads_nul_terminated_record_count():
    main = _main(394, "1", 19, nul=True)
    assert edf_n_times_from_main_header(main, 1000.0) == 394000


def test_nul_inside_digits_is_still_rejected():
    # Only a trailing terminator is tolerated; "5\x0051" is not a number.
    buf = bytearray(_main(551, "0.614000", 50, nul=False))
    buf[236:244] = b"5\x0051    "
    assert edf_n_times_from_main_header(bytes(buf), 1000.0) is None
