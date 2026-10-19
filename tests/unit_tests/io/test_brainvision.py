"""eegdash.io.brainvision.overrides: the files a stale .vhdr should name, header untouched.

Reading through MNE with these overrides is tested end to end in
tests/unit_tests/dataset/test_no_rewrite.py.
"""

import pytest

from eegdash.io.brainvision import overrides

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
