"""BrainVision ``n_times`` when ``DataFile=`` still names the pre-BIDS file.

Six NEMAR datasets (on002158, on003702, on004075, on005863, on007471,
on007813) were renamed to BIDS without updating the header: on007471's
``sub-01_task-jointaction_eeg.vhdr`` says ``DataFile=IBS_0001.eeg`` while the
binary next to it is ``sub-01_task-jointaction_eeg.eeg``. The size lookup
followed the stale name, found nothing, and every record stayed without
``ntimes``. The EEGDash reader already reads the BIDS sibling in this case
(``eegdash/io/brainvision.py``); the digest must size the same file.

Golden values: on007471 sub-01, 64 channels, INT_16, ``.eeg`` annex key
``SHA256E-s296783360--...`` -> 296,783,360 / (64 x 2) = 2,318,620 samples.
"""

from __future__ import annotations

import os
from pathlib import Path

from _vhdr_parser import _vhdr_n_times

_KEY = "SHA256E-s{size}--{h}.{ext}"


def _annexed(path: Path, size: int, ext: str) -> None:
    """A broken git-annex symlink, as in a shallow clone."""
    key = _KEY.format(size=size, h="ab" * 32, ext=ext)
    os.symlink(f"../../.git/annex/objects/xx/yy/{key}/{key}", path)


def _header(datafile: str) -> str:
    return (
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\n"
        f"DataFile={datafile}\n"
        "MarkerFile=IBS_0001.vmrk\n"
        "DataFormat=BINARY\n"
        "DataOrientation=MULTIPLEXED\n"
        "NumberOfChannels=64\n"
        "SamplingInterval=1000\n"
        "[Binary Infos]\n"
        "BinaryFormat=INT_16\n"
    )


def _layout(tmp_path: Path, datafile: str) -> Path:
    eeg_dir = tmp_path / "on007471" / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    vhdr = eeg_dir / "sub-01_task-jointaction_eeg.vhdr"
    vhdr.write_text(_header(datafile))
    _annexed(eeg_dir / "sub-01_task-jointaction_eeg.eeg", 296783360, "eeg")
    return vhdr


def test_stale_datafile_sizes_the_bids_sibling(tmp_path: Path):
    vhdr = _layout(tmp_path, "IBS_0001.eeg")
    assert _vhdr_n_times(vhdr.read_text(), vhdr, 64) == 2318620


def test_matching_datafile_unchanged(tmp_path: Path):
    vhdr = _layout(tmp_path, "sub-01_task-jointaction_eeg.eeg")
    assert _vhdr_n_times(vhdr.read_text(), vhdr, 64) == 2318620


def test_header_name_wins_when_it_has_a_size(tmp_path: Path):
    # The header's own file, when present, is what MNE reads: size it, not the sibling.
    vhdr = _layout(tmp_path, "IBS_0001.eeg")
    _annexed(vhdr.parent / "IBS_0001.eeg", 64 * 2 * 1000, "eeg")
    assert _vhdr_n_times(vhdr.read_text(), vhdr, 64) == 1000


def test_no_data_file_at_all_stays_none(tmp_path: Path):
    eeg_dir = tmp_path / "on000001" / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    vhdr = eeg_dir / "sub-01_task-x_eeg.vhdr"
    vhdr.write_text(_header("missing.eeg"))
    assert _vhdr_n_times(vhdr.read_text(), vhdr, 64) is None
