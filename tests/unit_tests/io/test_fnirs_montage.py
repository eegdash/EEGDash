"""eegdash.io.fnirs_montage: a recording's positions from derivatives/fnirs-montage.

The tree is laid out as the NEMAR fNIRS deposits carry it (montage_recordings.tsv,
montages.tsv, montages/<id>_channels.tsv and channels.json).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from eegdash.io.fnirs_montage import _key, read_fnirs_montage

REC = "sub-01/ses-01/nirs/sub-01_ses-01_task-rest_run-1_nirs.snirf"
COLS = (
    "name\tsource\tdetector\twavelength_nominal\tchromophore\tx\ty\tz\tprovenance_tier"
)
ROWS = [
    "S1_D1 760\tS1\tD1\t760\tn/a\t10.00\t20.00\t30.00\tT1",
    "S1_D1 850\tS1\tD1\t850\tn/a\t10.00\t20.00\t30.00\tT1",
    # a name rounded from 829.6 nm; MNE names the channel int(829.6) = 829
    "S2_D1 830\tS2\tD1\t829.6\tn/a\t-5.50\t0.25\t60.00\tT1",
    "S3_D1 760\tS3\tD1\t760\tn/a\tn/a\tn/a\tn/a\tT1",
]


def _tsv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows]) + "\n")


def _tree(root, *, written="yes", montages=True, rows=ROWS):
    tree = root / "derivatives" / "fnirs-montage"
    _tsv(
        tree / "montage_recordings.tsv",
        "recording\tmontage_id\ttier\twritten",
        [f"{REC}\tmab12\tT1\t{written}", "sub-02/nirs/x_nirs.snirf\tmcd34\tT8\tno"],
    )
    if montages:
        _tsv(
            tree / "montages.tsv",
            "montage_id\ttier\theuristic\tmethod\tchannels_table",
            ["mab12\tT1\tno\tMNI as reported\tmontages/mab12_channels.tsv"],
        )
    _tsv(tree / "montages" / "mab12_channels.tsv", COLS, rows)
    (tree / "montages" / "channels.json").write_text(
        json.dumps({"x": {"Units": "mm"}, "y": {"Units": "mm"}, "z": {"Units": "mm"}})
    )
    return tree


def test_positions_follow_ch_names(tmp_path):
    _tree(tmp_path)
    ch_names = ["S2_D1 829", "S1_D1 850", "S9_D9 760", "S1_D1 760", "S3_D1 760"]
    pos = read_fnirs_montage(tmp_path, REC, ch_names)
    assert list(pos.index) == ch_names
    np.testing.assert_array_equal(
        pos[["x", "y", "z"]].to_numpy(),
        [
            [-5.5, 0.25, 60.0],
            [10.0, 20.0, 30.0],
            [np.nan] * 3,  # not in the table
            [10.0, 20.0, 30.0],
            [np.nan] * 3,  # n/a in the table
        ],
    )
    assert pos.loc["S2_D1 829", "source"] == "S2"
    assert pos.loc["S1_D1 760", "provenance_tier"] == "T1"
    assert pos.attrs["montage_id"] == "mab12"
    assert pos.attrs["tier"] == "T1"
    assert pos.attrs["heuristic"] == "no"
    assert pos.attrs["units"] == "mm"


def test_absolute_recording_path(tmp_path):
    _tree(tmp_path)
    pos = read_fnirs_montage(tmp_path, tmp_path / REC, ["S1_D1 760"])
    assert pos.loc["S1_D1 760", "x"] == 10.0


def test_table_path_without_montages_tsv(tmp_path):
    _tree(tmp_path, montages=False)
    pos = read_fnirs_montage(tmp_path, REC, ["S1_D1 760"])
    assert pos.loc["S1_D1 760", "z"] == 30.0
    assert pos.attrs["tier"] is None


@pytest.mark.parametrize(
    "setup",
    ["no-derivative", "not-written", "not-listed"],
)
def test_no_montage_returns_none(tmp_path, setup):
    rec = REC
    if setup == "not-written":
        _tree(tmp_path, written="no")
    elif setup == "not-listed":
        _tree(tmp_path)
        rec = "sub-03/nirs/sub-03_task-rest_nirs.snirf"
    assert read_fnirs_montage(tmp_path, rec, ["S1_D1 760"]) is None


def test_names_match_mne_spelling():
    assert _key("S1_D1 760.0") == _key("S1_D1 760") == "S1_D1 760"
    assert _key("s1_d1 HbO") == _key("S1_D1 hbo") == "S1_D1 hbo"
    assert _key("S01_D02 hbr") == "S1_D2 hbr"
    # the table's nominal wavelength wins over a rounded name, as MNE truncates
    assert _key("S2_D1 830", "829.6") == _key("S2_D1 829") == "S2_D1 829"
    assert _key("Fp1") == "Fp1"


def test_repeated_names_keep_first_row(tmp_path):
    _tree(tmp_path, rows=[ROWS[0], ROWS[0].replace("10.00", "99.00")])
    pos = read_fnirs_montage(tmp_path, REC, ["S1_D1 760"])
    assert pos.loc["S1_D1 760", "x"] == 10.0
