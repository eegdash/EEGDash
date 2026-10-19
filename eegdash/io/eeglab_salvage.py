"""Read a truncated, uncompressed MAT-v5 EEGLAB data matrix, never pad it.

MAT-file Level 5 format: typed, 8-byte-aligned elements; miMATRIX contains
flags, dimensions, name, then real data (column-major). Only a scalar EEG
struct or named top-level fields are supported. No signature scanning,
compressed streams, complex arrays, epochs, or guessed channel metadata.
"""

import csv
import json
import struct
from pathlib import Path

import mne
import numpy as np
from mne.io import BaseRaw
from mne.utils import warn

from .bids import tsv_encoding

_TYPES = {
    1: "i1",
    2: "u1",
    3: "<i2",
    4: "<u2",
    5: "<i4",
    6: "<u4",
    7: "<f4",
    9: "<f8",
    12: "<i8",
    13: "<u8",
}
_CHANNEL_TYPES = {
    "EEG": "eeg",
    "EOG": "eog",
    "ECG": "ecg",
    "EMG": "emg",
    "MISC": "misc",
    "TRIG": "stim",
    "STIM": "stim",
    "REF": "eeg",
}
_UNITS = {"V": 1.0, "mV": 1e-3, "uV": 1e-6, "µV": 1e-6, "μV": 1e-6, "nV": 1e-9}


def _read(fid, pos, n, end):
    if n < 0 or pos + n > end:
        raise ValueError("Incomplete MAT header")
    fid.seek(pos)
    data = fid.read(n)
    if len(data) != n:
        raise ValueError("Incomplete MAT header")
    return data


def _tag(fid, pos, end):
    first, second = struct.unpack("<II", _read(fid, pos, 8, end))
    if first >> 16:
        kind, n = first & 65535, first >> 16
        if n > 4:
            raise ValueError("Invalid small MAT element")
        return kind, n, pos + 4, pos + 8
    return first, second, pos + 8, pos + 8 + ((second + 7) // 8) * 8


def _element(fid, pos, end, kind):
    t, n, start, next_pos = _tag(fid, pos, end)
    if t != kind or n > 65536:
        raise ValueError("Unsupported MAT header element")
    return _read(fid, start, n, end), next_pos


def _matrix(fid, pos, end):
    t, n, start, next_pos = _tag(fid, pos, end)
    if t != 14 or start + n > end:
        raise ValueError("Unsupported MAT matrix/container")
    end = start + n
    flags, p = _element(fid, start, end, 6)
    if len(flags) != 8:
        raise ValueError("Invalid MAT flags")
    flags = struct.unpack("<II", flags)[0]
    dims, p = _element(fid, p, end, 5)
    if len(dims) != 8:
        raise ValueError("Only 2-D continuous data can be salvaged")
    dims = struct.unpack("<ii", dims)
    name, p = _element(fid, p, end, 1)
    if flags & 0x0E00:  # complex, global or logical: no reinterpretation
        raise ValueError("Unsupported MAT flags")
    return name.decode("ascii"), flags & 255, dims, p, end, next_pos


def _data_location(path):
    """Follow actual MAT tags to data; return its offset/dtype/shape and scalars."""
    size = path.stat().st_size
    fields = {}
    with path.open("rb") as fid:
        header = _read(fid, 0, 128, size)
        if header[124:] != b"\x00\x01IM":
            raise ValueError("Only little-endian MAT-v5 salvage is supported")
        p, names, container_end = 128, None, 2**63 - 1
        while p < size:
            name, cls, dims, q, end, next_pos = _matrix(fid, p, container_end)
            if names is not None:
                if not names:
                    raise ValueError("Extra EEG struct field")
                name = names.pop(0)
            elif name == "EEG" and cls == 2 and dims == (1, 1):
                length, q = _element(fid, q, end, 5)
                if len(length) != 4:
                    raise ValueError("Invalid EEG field-name width")
                width = struct.unpack("<i", length)[0]
                labels, q = _element(fid, q, end, 1)
                if width <= 0 or len(labels) % width:
                    raise ValueError("Invalid EEG field names")
                names = [
                    labels[i : i + width].split(b"\0")[0].decode("ascii")
                    for i in range(0, len(labels), width)
                ]
                if len(set(names)) != len(names):
                    raise ValueError("Duplicate EEG fields")
                p, container_end = q, end
                continue
            if name == "data":
                t, n, offset, _ = _tag(fid, q, end)
                if cls not in (6, 7) or t not in (7, 9) or min(dims) <= 0:
                    raise ValueError("No supported embedded float matrix")
                dtype = np.dtype(_TYPES[t])
                if n != dims[0] * dims[1] * dtype.itemsize or offset + n > end:
                    raise ValueError("Inconsistent MAT data size")
                keep = (size - offset) // (dims[0] * dtype.itemsize)
                if not 0 < keep < dims[1]:
                    raise ValueError("No truncated sample tail to salvage")
                for key, expected in (
                    ("nbchan", dims[0]),
                    ("pnts", dims[1]),
                    ("trials", 1),
                ):
                    if key in fields and fields[key] != expected:
                        raise ValueError(f"Inconsistent EEG {key}")
                return offset, dtype, dims, keep, fields
            if name in ("nbchan", "pnts", "trials", "srate"):
                t, n, start, _ = _tag(fid, q, end)
                if cls not in range(6, 16) or dims != (1, 1) or t not in _TYPES:
                    raise ValueError("Invalid EEG scalar")
                dtype = np.dtype(_TYPES[t])
                if n != dtype.itemsize or name in fields:
                    raise ValueError("Invalid or duplicate EEG scalar")
                fields[name] = np.frombuffer(_read(fid, start, n, end), dtype)[0].item()
            if next_pos > size:
                break
            p = next_pos
    raise ValueError("File ends before embedded EEG data")


class _RawSalvagedEEGLAB(BaseRaw):
    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        extra = self._raw_extras[fi]
        nch = extra["nch"]
        with open(self.filenames[fi], "rb") as fid:
            fid.seek(extra["offset"] + start * nch * extra["dtype"].itemsize)
            block = np.fromfile(fid, extra["dtype"], (stop - start) * nch)
        block = block.reshape(-1, nch).T.astype(float)
        block *= extra["scales"][:, None]
        if mult is not None:
            data[:] = mult @ block
        else:
            data[:] = block[idx] * cals


def read_raw_truncated_eeglab(path, *, preload=False):
    """Salvage complete columns using this recording's deposited BIDS metadata."""
    path = Path(path)
    offset, dtype, (nch, declared), keep, fields = _data_location(path)
    # ponytail: exact recording sidecars only; inherited metadata needs a BIDS resolver.
    channels = path.with_name(path.stem.rsplit("_", 1)[0] + "_channels.tsv")
    with channels.open(encoding=tsv_encoding(channels), newline="") as fid:
        rows = list(csv.DictReader(fid, delimiter="\t"))
    if len(rows) != nch:
        raise ValueError("Deposited channels.tsv does not match data dimensions")
    names = [r["name"].strip() for r in rows]
    if any(not n or n == "n/a" for n in names) or len(set(names)) != nch:
        raise ValueError("Missing or duplicate channel names")
    types, scales = [], []
    for r in rows:
        source_type = r["type"].strip().upper()
        if source_type not in _CHANNEL_TYPES:
            raise ValueError(f"Unknown deposited channel type: {source_type}")
        kind = _CHANNEL_TYPES[source_type]
        unit = r["units"].strip()
        if unit not in _UNITS and not (
            kind in ("stim", "misc") and unit in ("n/a", "AU", "a.u.")
        ):
            raise ValueError(f"Unknown deposited channel units: {unit}")
        scales.append(
            1.0
            if kind in ("stim", "misc") and unit in ("n/a", "AU", "a.u.")
            else _UNITS[unit]
        )
        if kind not in ("stim", "misc") and _UNITS.get(unit) != 1e-6:
            raise ValueError(
                "Deposited units conflict with EEGLAB microvolt storage convention"
            )
        types.append(kind)
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8-sig"))
    sfreq = float(sidecar["SamplingFrequency"])
    if (
        not np.isfinite(sfreq)
        or sfreq <= 0
        or ("srate" in fields and fields["srate"] != sfreq)
    ):
        raise ValueError("Missing or inconsistent deposited sampling rate")
    for row in rows:
        rate = row.get("sampling_frequency", "n/a")
        if rate not in ("", "n/a") and float(rate) != sfreq:
            raise ValueError("Inconsistent channel sampling frequency")
    if sidecar.get("RecordingType", "continuous") != "continuous":
        raise ValueError("Cannot salvage epoched data")
    issue = (
        f"DataIntegrity: truncated EEGLAB file; reading {keep} of {declared} samples; "
        f"lost_fraction={(declared - keep) / declared:.9g} "
        f"({(declared - keep) / sfreq:.3f} s missing). "
        "Channel metadata from deposited channels.tsv; events/positions and other "
        "metadata after the cut are unavailable. No samples were padded."
    )
    info = mne.create_info(names, sfreq, types)
    info["description"] = issue
    info["bads"] = [
        name
        for name, row in zip(names, rows, strict=True)
        if row.get("status") == "bad"
    ]
    raw = _RawSalvagedEEGLAB(
        info,
        preload=preload,
        filenames=[str(path.resolve())],
        last_samps=[keep - 1],
        raw_extras=[dict(offset=offset, dtype=dtype, nch=nch, scales=np.array(scales))],
        verbose="error",
    )
    warn(f"{path.name}: {issue}")
    return raw
