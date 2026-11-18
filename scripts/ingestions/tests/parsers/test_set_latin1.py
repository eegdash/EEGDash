"""``.set`` whose char fields hold non-ASCII uint16 code units (on004166).

MATLAB stores char arrays as uint16 code units. scipy's default UTF-8 codec
can decode them to fewer characters than stored and ``loadmat`` raises
``TypeError: buffer is too small for requested array``; ``uint16_codec=
'latin-1'`` reads them. ``scipy.io.savemat`` always writes chars as UTF-8,
so the fixture is written by hand.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

scipy_io = pytest.importorskip("scipy.io")

from _set_parser import parse_set_metadata

_MI_INT8, _MI_UINT16, _MI_INT32, _MI_UINT32, _MI_DOUBLE, _MI_MATRIX = 1, 4, 5, 6, 9, 14
_MX_STRUCT, _MX_CHAR, _MX_DOUBLE = 2, 4, 6

# Latin-1 path; its UTF-8 reading of "Ã©" is one character, not two.
_LATIN1_PATH = "D:\\donnÃ©es\\sub16"


def _element(mdtype: int, payload: bytes) -> bytes:
    """A MAT v5 data element (tag + payload padded to 8 bytes)."""
    pad = -len(payload) % 8
    return struct.pack("<II", mdtype, len(payload)) + payload + b"\0" * pad


def _matrix(mx_class: int, dims: tuple[int, int], name: str, body: bytes) -> bytes:
    flags = _element(_MI_UINT32, struct.pack("<II", mx_class, 0))
    shape = _element(_MI_INT32, struct.pack("<ii", *dims))
    return _element(
        _MI_MATRIX, flags + shape + _element(_MI_INT8, name.encode()) + body
    )


def _double(value: float) -> bytes:
    return _matrix(
        _MX_DOUBLE, (1, 1), "", _element(_MI_DOUBLE, struct.pack("<d", value))
    )


def _char_uint16(text: str) -> bytes:
    units = struct.pack(f"<{len(text)}H", *(ord(c) for c in text))
    return _matrix(_MX_CHAR, (1, len(text)), "", _element(_MI_UINT16, units))


def _write_set(path: Path, fields: dict[str, bytes]) -> Path:
    """Write a MAT v5 file holding one struct ``EEG`` with *fields*."""
    width = 32
    names = b"".join(n.encode().ljust(width, b"\0") for n in fields)
    body = (
        _element(_MI_INT32, struct.pack("<i", width))
        + _element(_MI_INT8, names)
        + b"".join(fields.values())
    )
    header = b"MATLAB 5.0 MAT-file, Platform: PCWIN64".ljust(116, b" ")
    header += b"\0" * 8 + struct.pack("<H", 0x0100) + b"IM"
    path.write_bytes(header + _matrix(_MX_STRUCT, (1, 1), "EEG", body))
    return path


def _latin1_set(path: Path) -> Path:
    return _write_set(
        path,
        {
            "filepath": _char_uint16(_LATIN1_PATH),
            "srate": _double(500.0),
            "nbchan": _double(66.0),
            "pnts": _double(999450.0),
            "trials": _double(1.0),
        },
    )


def test_fixture_fails_plain_loadmat(tmp_path):
    """The hand-written file reproduces on004166's loadmat failure."""
    path = _latin1_set(tmp_path / "sub-16_eeg.set")
    with pytest.raises(TypeError, match="buffer is too small"):
        scipy_io.loadmat(str(path), struct_as_record=False, squeeze_me=True)
    eeg = scipy_io.loadmat(
        str(path), struct_as_record=False, squeeze_me=True, uint16_codec="latin-1"
    )["EEG"]
    assert eeg.filepath == _LATIN1_PATH


def test_parse_set_metadata_retries_with_latin1(tmp_path):
    """The digest parser sizes the file like the runtime reader does."""
    result = parse_set_metadata(_latin1_set(tmp_path / "sub-16_eeg.set"))
    assert result is not None
    assert result["sampling_frequency"] == 500.0
    assert result["nchans"] == 66
    assert result["n_samples"] == 999450
    assert result["duration"] == pytest.approx(1998.9)


def test_ascii_set_unchanged(tmp_path):
    """Plain-ASCII char fields still read on the first, default-codec pass."""
    path = _write_set(
        tmp_path / "a_eeg.set",
        {
            "filepath": _char_uint16("D:\\data\\sub16"),
            "srate": _double(250.0),
            "nbchan": _double(4.0),
            "pnts": _double(1000.0),
        },
    )
    result = parse_set_metadata(path)
    assert result is not None
    assert (result["sampling_frequency"], result["nchans"], result["n_samples"]) == (
        250.0,
        4,
        1000,
    )
