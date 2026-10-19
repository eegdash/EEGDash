"""eegdash.io.brainvision: stale DataFile=/MarkerFile= read through MNE 1.13 ``overrides``.

EEGDash used to rewrite these headers in the download cache (and write stub
.vmrk files). The header must now stay byte-identical while the same files are
read.
"""

from pathlib import Path

import mne
import numpy as np
import pytest

from eegdash.io import get_reader
from eegdash.io.brainvision import (
    _find_best_matching_file,
    overrides,
    read_raw_brainvision,
)

STEM = "sub-01_task-rest_eeg"
ANNEX_EEG = "SHA256E-s9999--" + "ab" * 32 + ".eeg"
ANNEX_VMRK = "MD5E-s11657--7a519e74754041a678931b7b7d72f0ab.vmrk"


def _vhdr(folder, datafile, markerfile=None, stem=STEM):
    lines = ["Brain Vision Data Exchange Header File Version 1.0", "[Common Infos]"]
    lines.append(f"DataFile={datafile}")
    if markerfile is not None:
        lines.append(f"MarkerFile={markerfile}")
    path = folder / f"{stem}.vhdr"
    path.write_text("\n".join(lines) + "\n")
    return path


def _touch(folder, *names):
    for name in names:
        (folder / name).touch()


# overrides(): which files the header should have named (the cases the old
# _repair_vhdr_pointers rewrote), with the header left as it is.


@pytest.mark.parametrize(
    "files,datafile,markerfile,expected",
    [
        # internal pre-BIDS names, BIDS files present
        (
            [f"{STEM}.eeg", f"{STEM}.vmrk"],
            "INTERNAL_NAME.eeg",
            "INTERNAL_NAME.vmrk",
            {"data_fname": f"{STEM}.eeg", "marker_fname": f"{STEM}.vmrk"},
        ),
        # annex-key MarkerFile
        (
            [f"{STEM}.eeg", f"{STEM}.vmrk"],
            f"{STEM}.eeg",
            ANNEX_VMRK,
            {"marker_fname": f"{STEM}.vmrk"},
        ),
        # internal DataFile + annex-key MarkerFile
        (
            [f"{STEM}.eeg", f"{STEM}.vmrk"],
            "s2_run1_08062017.eeg",
            ANNEX_VMRK,
            {"data_fname": f"{STEM}.eeg", "marker_fname": f"{STEM}.vmrk"},
        ),
        # annex keys and no BIDS file yet: the data file is named, the
        # missing marker file is left to MNE (no annotations, with a warning)
        ([], ANNEX_EEG, ANNEX_VMRK, {"data_fname": f"{STEM}.eeg"}),
        # an annex key that resolves (the key-named file exists)
        (
            [ANNEX_EEG, f"{STEM}.vmrk"],
            ANNEX_EEG,
            f"{STEM}.vmrk",
            {"data_fname": f"{STEM}.eeg"},
        ),
        # typos (ds "sternbeg", "rsub-")
        (
            [f"{STEM}.eeg", f"{STEM}.vmrk"],
            "sub-01_task-rset_eeg.eeg",
            "rsub-01_task-rest_eeg.vmrk",
            {"data_fname": f"{STEM}.eeg", "marker_fname": f"{STEM}.vmrk"},
        ),
        # valid pointers: nothing to override
        ([f"{STEM}.eeg", f"{STEM}.vmrk"], f"{STEM}.eeg", f"{STEM}.vmrk", {}),
        # no MarkerFile key, sibling .vmrk present: keep its markers
        (
            [f"{STEM}.eeg", f"{STEM}.vmrk"],
            f"{STEM}.eeg",
            None,
            {"marker_fname": f"{STEM}.vmrk"},
        ),
        # no MarkerFile key and no .vmrk: MNE reads no annotations itself
        ([f"{STEM}.eeg"], f"{STEM}.eeg", None, {}),
        # a data file under another, non-BIDS name, with no header of its own
        (
            ["recording_0815.eeg"],
            "rec_0815.eeg",
            None,
            {"data_fname": "recording_0815.eeg"},
        ),
    ],
    ids=[
        "internal",
        "annex_marker",
        "internal_and_annex",
        "annex_no_bids_file",
        "annex_resolved",
        "typos",
        "valid",
        "no_markerfile_key",
        "no_markerfile_no_vmrk",
        "non_bids_data_name",
    ],
)
def test_overrides_leave_header_unchanged(
    tmp_path, files, datafile, markerfile, expected
):
    _touch(tmp_path, *files)
    vhdr = _vhdr(tmp_path, datafile, markerfile)
    before = vhdr.read_bytes()
    assert overrides(vhdr) == expected
    assert vhdr.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([*files, vhdr.name])


def test_overrides_never_take_another_recordings_data(tmp_path):
    """run-02's .eeg is missing: run-01's (it has its own .vhdr) is not read instead."""
    _touch(tmp_path, "sub-01_run-01_eeg.eeg", "sub-01_run-01_eeg.vhdr")
    vhdr = _vhdr(tmp_path, "s1_run2.eeg", stem="sub-01_run-02_eeg")
    assert overrides(vhdr) == {}


def test_overrides_unreadable_header(tmp_path):
    assert overrides(tmp_path / "missing.vhdr") == {}


def test_find_best_matching_file(tmp_path):
    assert _find_best_matching_file(tmp_path, "x.eeg", ".eeg") is None
    _touch(tmp_path, "actual_file.eeg")
    assert (
        _find_best_matching_file(tmp_path, "different.eeg", ".eeg") == "actual_file.eeg"
    )
    _touch(tmp_path, "sub-01_task-rest_eeg.eeg", "sub-02_task-rest_eeg.eeg")
    match = _find_best_matching_file(tmp_path, "rsub-01_task-rest_eeg.eeg", ".eeg")
    assert match == "sub-01_task-rest_eeg.eeg"


# Reading real BrainVision files written by pybv.


def _write(folder, n_times=200, events=True):
    pybv = pytest.importorskip("pybv")
    data = np.random.default_rng(0).standard_normal((2, n_times)) * 1e-5
    kwargs = {}
    if events:
        kwargs["events"] = np.array([[50, 1], [120, 2]])  # onset, code
    pybv.write_brainvision(
        data=data,
        sfreq=100.0,
        ch_names=["Fz", "Cz"],
        fname_base=STEM,
        folder_out=str(folder),
        overwrite=True,
        **kwargs,
    )
    return Path(folder) / f"{STEM}.vhdr", data


def _set_pointers(vhdr, datafile, markerfile):
    lines = []
    for line in vhdr.read_text().splitlines():
        if line.startswith("DataFile="):
            line = f"DataFile={datafile}"
        elif line.startswith("MarkerFile="):
            if markerfile is None:
                continue
            line = f"MarkerFile={markerfile}"
        lines.append(line)
    vhdr.write_text("\n".join(lines) + "\n")


def _snapshot(folder):
    return {p.name: p.read_bytes() for p in sorted(Path(folder).iterdir())}


def test_read_annex_key_pointers(tmp_path):
    vhdr, data = _write(tmp_path)
    _set_pointers(vhdr, ANNEX_EEG, ANNEX_VMRK)
    before = _snapshot(tmp_path)
    with pytest.raises(FileNotFoundError):
        mne.io.read_raw_brainvision(vhdr, verbose="ERROR")
    raw = read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    np.testing.assert_allclose(raw.get_data(), data, rtol=1e-6, atol=1e-12)
    assert len(raw.annotations) >= 2  # the markers of the BIDS .vmrk
    assert _snapshot(tmp_path) == before


def test_read_missing_markerfile_key_keeps_markers(tmp_path):
    vhdr, _ = _write(tmp_path)
    reference = read_raw_brainvision(vhdr, verbose="ERROR").annotations
    _set_pointers(vhdr, f"{STEM}.eeg", None)
    before = _snapshot(tmp_path)
    assert len(mne.io.read_raw_brainvision(vhdr, verbose="ERROR").annotations) == 0
    raw = read_raw_brainvision(vhdr, verbose="ERROR")
    assert list(raw.annotations.description) == list(reference.description)
    np.testing.assert_array_equal(raw.annotations.onset, reference.onset)
    assert _snapshot(tmp_path) == before


def test_read_missing_vmrk_writes_no_stub(tmp_path):
    vhdr, data = _write(tmp_path)
    (tmp_path / f"{STEM}.vmrk").unlink()
    before = _snapshot(tmp_path)
    raw = read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    assert len(raw.annotations) == 0
    np.testing.assert_allclose(raw.get_data(), data, rtol=1e-6, atol=1e-12)
    assert _snapshot(tmp_path) == before  # no stub .vmrk


def test_caller_overrides_win(tmp_path):
    vhdr, _ = _write(tmp_path)
    _set_pointers(vhdr, ANNEX_EEG, f"{STEM}.vmrk")
    raw = read_raw_brainvision(vhdr, overrides={"marker_fname": False}, verbose="ERROR")
    assert len(raw.annotations) == 0


def test_direct_reader_is_ours():
    assert get_reader(".vhdr") is read_raw_brainvision
