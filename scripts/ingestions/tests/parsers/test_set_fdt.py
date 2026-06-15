"""RH1 — SET external `.fdt` zero-fetch n_times arithmetic.

EEGLAB external `.set` stores continuous data in a companion `.fdt` as raw
float32, channels × points, no header: ``n_times = fdt_size / (nchans × 4)``.
The `.fdt` size comes from its git-annex key on a shallow clone — **zero fetch**,
exactly like VHDR's `.eeg`.

Every helper/stub here is module-level (the repo's no-nested-functions
pre-commit hook forbids `def`/class-methods nested inside a `def`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _set_parser import _fdt_n_times, parse_set_metadata

# ─── unit tests of the pure helper ─────────────────────────────────────────


def test_fdt_n_times_from_real_file_size(tmp_path: Path):
    # 4 channels × 1000 samples × float32 (4 bytes) -> n_times = 1000.
    fdt = tmp_path / "e.fdt"
    fdt.write_bytes(b"\x00" * (4 * 1000 * 4))
    assert _fdt_n_times(fdt, 4) == 1000


def test_fdt_n_times_from_broken_annex_symlink(tmp_path: Path):
    # A broken git-annex symlink whose key encodes the size -> still 1000,
    # zero bytes read (the target never exists on a shallow clone).
    size = 4 * 1000 * 4  # 16000 bytes
    target = f".git/annex/objects/aB/Cd/MD5E-s{size}--deadbeef.fdt"
    fdt = tmp_path / "e.fdt"
    fdt.symlink_to(target)
    assert not fdt.exists()  # broken symlink, no content present
    assert _fdt_n_times(fdt, 4) == 1000


def test_fdt_n_times_nchans_none_returns_none(tmp_path: Path):
    fdt = tmp_path / "e.fdt"
    fdt.write_bytes(b"\x00" * (4 * 1000 * 4))
    assert _fdt_n_times(fdt, None) is None


def test_fdt_n_times_nchans_zero_returns_none(tmp_path: Path):
    fdt = tmp_path / "e.fdt"
    fdt.write_bytes(b"\x00" * (4 * 1000 * 4))
    assert _fdt_n_times(fdt, 0) is None


def test_fdt_n_times_non_divisible_size_returns_none(tmp_path: Path):
    # 16001 bytes is not a whole multiple of nchans*4 -> None (never guess).
    fdt = tmp_path / "e.fdt"
    fdt.write_bytes(b"\x00" * (4 * 1000 * 4 + 1))
    assert _fdt_n_times(fdt, 4) is None


def test_fdt_n_times_missing_file_returns_none(tmp_path: Path):
    fdt = tmp_path / "nope.fdt"
    assert _fdt_n_times(fdt, 4) is None


# ─── integration: parse_set_metadata wires the .fdt size in ────────────────


def test_set_external_ntimes_from_fdt_size(tmp_path: Path):
    pytest.importorskip("scipy.io")
    from _helpers.builders import build_synthetic_set_v5  # noqa: PLC0415

    # External, continuous .set: builder writes srate/nbchan/chanlocs but
    # pnts=0 so n_samples/n_times is absent after extraction. The companion
    # .fdt SIZE encodes n_times = 1000.
    set_path = build_synthetic_set_v5(tmp_path / "e.set", srate=250.0, nbchan=4, pnts=0)
    (tmp_path / "e.fdt").write_bytes(b"\x00" * (4 * 1000 * 4))

    out = parse_set_metadata(set_path)
    assert out is not None
    assert out["has_fdt"] is True
    assert out.get("n_times") == 1000 or out.get("n_samples") == 1000


def test_set_external_ntimes_not_overwritten_when_present(tmp_path: Path):
    pytest.importorskip("scipy.io")
    from _helpers.builders import build_synthetic_set_v5  # noqa: PLC0415

    # pnts=512 is already known -> the .fdt arithmetic must NOT clobber it,
    # even though the .fdt size would imply a different value.
    set_path = build_synthetic_set_v5(
        tmp_path / "e.set", srate=250.0, nbchan=4, pnts=512
    )
    (tmp_path / "e.fdt").write_bytes(b"\x00" * (4 * 1000 * 4))

    out = parse_set_metadata(set_path)
    assert out is not None
    assert out["n_samples"] == 512
    # n_times, if set at all, must agree with the already-known sample count.
    assert out.get("n_times", 512) == 512


def test_set_external_no_fdt_no_ntimes(tmp_path: Path):
    pytest.importorskip("scipy.io")
    from _helpers.builders import build_synthetic_set_v5  # noqa: PLC0415

    # No .fdt on disk -> has_fdt False -> the arithmetic does not fire.
    set_path = build_synthetic_set_v5(tmp_path / "e.set", srate=250.0, nbchan=4, pnts=0)
    out = parse_set_metadata(set_path)
    assert out is not None
    assert out["has_fdt"] is False
    assert out.get("n_times") is None
