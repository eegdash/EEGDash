"""Tests for the EDF/BDF header consistency checks."""

from __future__ import annotations

from eegdash.header_check import (
    Channel,
    check,
    header_size,
    parse_header,
    read_local,
    sidecar_units,
)


def make_header(channels):
    """A BDF header for ``(name, unit, physical_min, physical_max)`` channels."""
    n = len(channels)
    main = bytearray(b" " * 256)
    main[184:192] = str(256 * (n + 1)).ljust(8).encode()
    main[252:256] = str(n).ljust(4).encode()

    def column(values, width):
        return b"".join(str(v).ljust(width)[:width].encode("latin-1") for v in values)

    names, units, pmin, pmax = zip(*channels)
    signals = (
        column(names, 16)
        + column([""] * n, 80)
        + column(units, 8)
        + column(pmin, 8)
        + column(pmax, 8)
        + column([-8388607] * n, 8)
        + column([8388607] * n, 8)
        + column([""] * n, 80)
        + column([2000] * n, 8)
        + column([""] * n, 32)
    )
    return bytes(main) + signals


def test_parse_header_reads_units_and_ranges():
    buf = make_header([("emg0", "uV", -4770.6, 3129.4), ("joint0", "rad", -0.8, 2.0)])
    assert header_size(buf[:256]) == len(buf)
    assert parse_header(buf) == [
        Channel("emg0", "uV", -4770.6, 3129.4),
        Channel("joint0", "rad", -0.8, 2.0),
    ]


def test_consistent_dataset_has_no_finding():
    channels = [Channel("emg0", "uV", -5000, 5000), Channel("joint0", "rad", -1, 2)]
    sidecar = {"emg0": "µV", "joint0": "rad"}
    assert check({"a": (channels, sidecar), "b": (channels, sidecar)}) == []


def test_emg2pose_relabelled_joints_are_flagged():
    # nm000281: joints stored as rad x 1e6 under "uV", relabelled "rad".
    sidecar = {"joint0": "rad"}
    old = ([Channel("joint0", "uV", -845946, 1997237)], sidecar)
    new = ([Channel("joint0", "rad", -652619, 1998119)], sidecar)
    findings = check({"sub-01/a.bdf": old, "sub-03/b.bdf": new})
    assert {f.check for f in findings} == {"unit", "sidecar", "range"}
    (sidecar_finding,) = [f for f in findings if f.check == "sidecar"]
    assert sidecar_finding.files == ("sub-01/a.bdf",)
    (range_finding,) = [f for f in findings if f.check == "range"]
    assert range_finding.files == ("sub-03/b.bdf",)


def test_read_local_pairs_headers_with_sidecars(tmp_path):
    emg = tmp_path / "sub-01" / "emg"
    emg.mkdir(parents=True)
    (emg / "sub-01_task-x_emg.bdf").write_bytes(
        make_header([("joint0", "uV", -1e6, 2e6)]) + b"\0" * 64
    )
    (emg / "sub-01_task-x_channels.tsv").write_text(
        "name\ttype\tunits\njoint0\tMISC\trad\n"
    )
    recordings = read_local(tmp_path)
    assert list(recordings) == ["sub-01/emg/sub-01_task-x_emg.bdf"]
    assert {f.check for f in check(recordings)} == {"sidecar"}


def test_sidecar_units():
    assert sidecar_units("name\ttype\tunits\nemg0\tEMG\tV\n") == {"emg0": "V"}


def test_prefix_only_mismatch_is_minor():
    channels = [Channel("emg0", "uV", -5000, 5000)]
    findings = check({"a": (channels, {"emg0": "V"})})
    assert [f.check for f in findings] == ["prefix"]
