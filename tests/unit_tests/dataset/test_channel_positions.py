"""Positions of iEEG channels named after contacts without being a row of electrodes.tsv.

Survey of 2026-10-26 (live catalogue + deposit sidecars, ``tests/fixtures/ieeg_positions``):

- on004789 / on004809 ``acq-bipolar``: channels ``RAFA1-RAFA2``..; both contacts
  are positioned for 99 % of the pairs, yet the records read no position.
- on006910 / on006914 / on005545: the EDF holds bipolar ``POL B10-B18``..
  channels (an EDF type word, then a pair).
- on004551: ``EEG A1-Ref`` (contact against a reference that is not a contact)
  and ``POL A3`` (type word) for contacts ``A1``, ``A3``.
- nm000358: channels named exactly like the contacts; nothing may change.

Approved rule (Bruno, 2026-10-27): exact name first; lookup-only aliases (case,
white space, a leading ``POL``/``EEG``), an alias that matches two contacts is
ambiguous; ``<contact>-Ref`` is the contact when ``Ref`` is not one; a pair is
placed only when exactly one ``-`` split names two positioned contacts, at their
midpoint, with the contacts and their distance as provenance.
"""

import json
import math
import warnings
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd
import pytest

from eegdash.dataset.channel_positions import (
    BIPOLAR_MIDPOINT,
    ELECTRODE_ALIAS,
    MEASURED,
    derived_positions,
    fill_missing_positions,
    read_electrode_table,
    resolve_channel_positions,
)

DATA = Path(__file__).resolve().parents[2] / "fixtures" / "ieeg_positions"
ON006910 = DATA / "on006910_sub-098_ses-3_electrodes.tsv"


def _names(channels_tsv):
    return pd.read_csv(DATA / channels_tsv, sep="\t", dtype=str)["name"].tolist()


def _header_labels(dataset):
    labels = json.loads((DATA / "edf_header_labels.json").read_text())[dataset]
    return [n for n in labels["labels"] if n != "EDF Annotations"]


def _midpoint(table, a, b):
    return tuple((u + v) / 2 for u, v in zip(table[a], table[b]))


# ── the shared rule, on the deposits' own sidecars ──


def test_exact_monopolar_names_are_unchanged():
    """nm000358: every channel named like a contact keeps that contact's position."""
    table = read_electrode_table(DATA / "nm000358_sub-03_ses-5_electrodes.tsv")
    names = _names("nm000358_sub-03_ses-5_task-naturalistic_channels.tsv")
    resolved = resolve_channel_positions(names, table)

    assert {p.source for p in resolved.values()} == {MEASURED}
    assert all(resolved[n].xyz == table[n] for n in resolved)
    assert all(table[n] is not None for n in resolved)
    assert derived_positions(names, table) == {}
    # contacts the deposit leaves at n/a stay unpositioned
    assert table["LAD_080"] is None
    assert "LAD_080" not in resolved


def test_edf_type_word_pair_is_placed_at_the_contacts_midpoint():
    """on006910: 49 EDF channels ``POL B10-B18``..; 47 pairs, plus ``POL DC01/02``."""
    table = read_electrode_table(ON006910)
    labels = _header_labels("on006910")
    derived = derived_positions(labels, table)

    assert len(labels) == 49
    assert len(derived) == 47
    assert set(labels) - set(derived) == {"POL DC01", "POL DC02"}
    pos = derived["POL B10-B18"]
    assert pos.source == BIPOLAR_MIDPOINT
    assert pos.contacts == ("B10", "B18")
    assert pos.xyz == pytest.approx(_midpoint(table, "B10", "B18"))
    assert pos.provenance() == {
        "position_source": "bipolar_midpoint",
        "contacts": ["B10", "B18"],
        "contact_distance": pytest.approx(math.dist(table["B10"], table["B18"])),
    }


@pytest.mark.parametrize(
    ("electrodes", "channels"),
    [
        pytest.param(
            "on004789_sub-R1148P_ses-0_task-FR1_space-MNI152NLin6ASym_electrodes.tsv",
            "on004789_sub-R1148P_ses-0_task-FR1_acq-bipolar_channels.tsv",
            id="on004789",
        ),
        pytest.param(
            "on004809_sub-R1106M_ses-0_task-catFR1_space-MNI152NLin6ASym_electrodes.tsv",
            "on004809_sub-R1106M_ses-0_task-catFR1_acq-bipolar_channels.tsv",
            id="on004809",
        ),
    ],
)
def test_pair_without_type_word_is_placed(electrodes, channels):
    """``acq-bipolar`` records: ``RAFA1-RAFA2``.. all name two positioned contacts."""
    table = read_electrode_table(DATA / electrodes)
    names = _names(channels)
    derived = derived_positions(names, table)

    assert set(derived) == set(names)
    for name, pos in derived.items():
        a, b = name.split("-")
        assert pos.source == BIPOLAR_MIDPOINT
        assert pos.contacts == (a, b)
        assert pos.xyz == pytest.approx(_midpoint(table, a, b))
        assert pos.contact_distance == pytest.approx(math.dist(table[a], table[b]))


def test_eeg_type_word_and_ref_name_one_contact():
    """on004551: ``EEG A1-Ref`` is A1, ``POL A3`` is A3; ``POL $A11``/``POL E`` none."""
    table = read_electrode_table(DATA / "on004551_sub-098_ses-01_electrodes.tsv")
    labels = _header_labels("on004551")
    derived = derived_positions(labels, table)

    for channel, contact in (("EEG A1-Ref", "A1"), ("POL A3", "A3")):
        assert derived[channel].source == ELECTRODE_ALIAS
        assert derived[channel].contacts == (contact,)
        assert derived[channel].xyz == table[contact]
        assert derived[channel].contact_distance is None
        assert "contact_distance" not in derived[channel].provenance()
    for unplaced in ("POL $A11", "POL E", "POL DC01"):
        assert unplaced not in derived
    # every placed channel is one contact of the deposit, no pair, nothing invented
    assert {p.source for p in derived.values()} == {ELECTRODE_ALIAS}
    assert all(p.contacts[0] in table for p in derived.values())


A, B, C = (1.0, 2.0, 3.0), (3.0, 2.0, 1.0), (10.0, 0.0, 0.0)


def test_ref_is_a_contact_only_when_the_table_has_one():
    """``A1-Ref`` is A1; with a contact named ``Ref`` it is the pair A1-Ref."""
    alone = resolve_channel_positions(["A1-Ref", "a1-REF"], {"A1": A})
    assert alone["A1-Ref"].source == ELECTRODE_ALIAS
    assert alone["A1-Ref"].contacts == ("A1",)
    assert alone["a1-REF"].contacts == ("A1",)

    pair = resolve_channel_positions(["A1-Ref"], {"A1": A, "Ref": B})["A1-Ref"]
    assert pair.source == BIPOLAR_MIDPOINT
    assert pair.contacts == ("A1", "Ref")
    assert pair.xyz == pytest.approx((2.0, 2.0, 2.0))


@pytest.mark.parametrize(
    ("channel", "electrodes"),
    [
        pytest.param("POL B10-X99", {"B10": A}, id="missing-endpoint"),
        pytest.param("POL B10-B18", {"B10": A, "B18": None}, id="na-endpoint"),
        pytest.param("POL A1", {"A1": A, "a1": B}, id="alias-collision"),
        pytest.param("POL A1-B1", {"A1": A, "a1": B, "B1": C}, id="ambiguous-endpoint"),
        pytest.param("A-B-C", {"A": A, "B-C": B, "A-B": C, "C": A}, id="two-splits"),
        pytest.param("B1-B1", {"B1": A}, id="same-contact"),
        pytest.param("POL A1", {"A1": None}, id="na-alias"),
        pytest.param("E", {"A1": A}, id="unknown"),
    ],
)
def test_unresolved_or_ambiguous_names_fail_closed(channel, electrodes):
    assert resolve_channel_positions([channel], electrodes) == {}


def test_na_endpoint_on_a_real_deposit():
    """nm000358 LAD_080 is n/a: a pair with it gets no position."""
    table = read_electrode_table(DATA / "nm000358_sub-03_ses-5_electrodes.tsv")
    assert table["GRID_000"] is not None
    assert resolve_channel_positions(["GRID_000-LAD_080"], table) == {}
    assert resolve_channel_positions(["LAD_080-GRID_000"], table) == {}


def test_exact_name_wins_over_a_colliding_alias():
    resolved = resolve_channel_positions(["A1", "a1"], {"A1": A, "a1": B})
    assert resolved["A1"].xyz == A
    assert resolved["a1"].xyz == B
    assert {p.source for p in resolved.values()} == {MEASURED}


def test_distant_pair_keeps_its_distance():
    """No distance cut-off: contacts 7 cm apart are placed, their distance kept."""
    table = read_electrode_table(ON006910)
    near = derived_positions(["POL B10-B18"], table)["POL B10-B18"]
    far = derived_positions(["POL A1-B64"], table)["POL A1-B64"]
    assert far.xyz == pytest.approx(_midpoint(table, "A1", "B64"))
    assert far.contact_distance == pytest.approx(math.dist(table["A1"], table["B64"]))
    assert far.contact_distance > 70  # mm
    assert 0 < near.contact_distance < 5


def test_duplicate_electrode_rows_have_no_position(tmp_path):
    tsv = tmp_path / "e.tsv"
    tsv.write_text("name\tx\ty\tz\nA1\t1\t2\t3\nA1\t4\t5\t6\nA2\tn/a\tn/a\tn/a\n")
    assert read_electrode_table(tsv) == {"A1": None, "A2": None}


# ── runtime: after MNE-BIDS, only channels without a position are filled ──


def _write_bids(root, ch_names, electrodes_tsv, coordsystem):
    """A BrainVision iEEG recording with the deposit's electrodes.tsv and coordsystem."""
    from mne_bids import BIDSPath, write_raw_bids

    pytest.importorskip("pybv")
    info = mne.create_info(ch_names, 100.0, "ecog")
    raw = mne.io.RawArray(np.zeros((len(ch_names), 500)), info)
    raw.set_meas_date(datetime(2020, 1, 1, tzinfo=UTC))
    bp = BIDSPath(root=root, subject="01", task="sleep", datatype="ieeg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_raw_bids(
            raw, bp, format="BrainVision", allow_preload=True, verbose="ERROR"
        )
    ieeg = root / "sub-01" / "ieeg"
    for pattern in ("*_electrodes.*", "*_coordsystem.json"):  # n/a placeholders
        for p in root.rglob(pattern):
            p.unlink()
    (ieeg / "sub-01_electrodes.tsv").write_bytes(electrodes_tsv.read_bytes())
    (ieeg / "sub-01_coordsystem.json").write_bytes(coordsystem.read_bytes())
    return bp


def _record(bp, root):
    relpath = bp.fpath.relative_to(root).as_posix()
    entities = {"subject": "01", "session": None, "task": "sleep", "run": None}
    return {
        "dataset": "fakeds",
        "bids_relpath": relpath,
        "datatype": "ieeg",
        "suffix": "ieeg",
        "extension": ".vhdr",
        "storage": {
            "backend": "local",
            "base": str(root),
            "raw_key": relpath,
            "dep_keys": [],
        },
        "entities": entities,
        "entities_mne": entities,
    }


ON006910_CHANNELS = ["B10", "B18", "POL B10-B18", "POL B11-B19", "POL DC01"]


def test_eegdashraw_places_pairs_and_keeps_measured_positions(tmp_path):
    """on006910 sidecars: ``B10``/``B18`` measured, the pair at their midpoint."""
    from eegdash.dataset.base import EEGDashRaw

    root = tmp_path / "fakeds"
    bp = _write_bids(
        root,
        ON006910_CHANNELS,
        ON006910,
        DATA / "on006910_sub-098_ses-3_coordsystem.json",
    )
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        raw = EEGDashRaw(_record(bp, root), cache_dir=str(tmp_path)).raw

    table = read_electrode_table(ON006910)
    loc = {ch["ch_name"]: ch["loc"][:3] for ch in raw.info["chs"]}
    # measured, as MNE-BIDS reads them (mm -> m)
    assert loc["B10"] == pytest.approx(np.array(table["B10"]) / 1000)
    assert loc["B18"] == pytest.approx(np.array(table["B18"]) / 1000)
    # derived, in the same frame as the measured channels
    assert loc["POL B10-B18"] == pytest.approx((loc["B10"] + loc["B18"]) / 2)
    assert loc["POL B11-B19"] == pytest.approx(
        np.array(_midpoint(table, "B11", "B19")) / 1000
    )
    frames = {
        ch["coord_frame"] for ch in raw.info["chs"] if ch["ch_name"] != "POL DC01"
    }
    assert len(frames) == 1
    assert np.isnan(loc["POL DC01"]).all()
    # provenance, distance in the layout's frame (metres)
    prov = raw.info["temp"]["derived_positions"]
    assert set(prov) == {"POL B10-B18", "POL B11-B19"}
    assert prov["POL B10-B18"]["position_source"] == "bipolar_midpoint"
    assert prov["POL B10-B18"]["contacts"] == ["B10", "B18"]
    assert prov["POL B10-B18"]["contact_distance"] == pytest.approx(
        math.dist(table["B10"], table["B18"]) / 1000
    )
    # the montage shows them; names and signal untouched
    montage_pos = raw.get_montage().get_positions()["ch_pos"]
    assert montage_pos["POL B10-B18"] == pytest.approx(loc["POL B10-B18"])
    assert raw.ch_names == ON006910_CHANNELS
    assert not raw.get_data().any()


def test_fill_does_not_overwrite_a_measured_position():
    """A channel that already has a position keeps it, whatever its name."""
    info = mne.create_info(["POL B10", "POL B10-B18"], 100.0, "ecog")
    raw = mne.io.RawArray(np.zeros((2, 10)), info, verbose="ERROR")
    measured = np.array([0.01, 0.02, 0.03])
    raw.info["chs"][0]["loc"][:3] = measured
    filled = fill_missing_positions(
        raw, ON006910, DATA / "on006910_sub-098_ses-3_coordsystem.json", "ieeg"
    )
    assert set(filled) == {"POL B10-B18"}
    assert raw.info["chs"][0]["loc"][:3] == pytest.approx(measured)
    table = read_electrode_table(ON006910)
    assert raw.info["chs"][1]["loc"][:3] == pytest.approx(
        np.array(_midpoint(table, "B10", "B18")) / 1000
    )


def test_fill_without_derivable_channels_reads_nothing():
    info = mne.create_info(["EKG", "DC01"], 100.0, "ecog")
    raw = mne.io.RawArray(np.zeros((2, 10)), info, verbose="ERROR")
    with patch(
        "eegdash.dataset.channel_positions._contacts_as_mne_places_them"
    ) as read:
        assert fill_missing_positions(raw, ON006910, Path("unused.json")) == {}
    read.assert_not_called()


def test_private_mne_bids_reader_is_still_there():
    """The runtime places contacts with MNE-BIDS's own reader (as base.py does for NWB)."""
    import inspect

    from mne_bids.read import _read_dig_bids

    assert list(inspect.signature(_read_dig_bids).parameters) == [
        "electrodes_fpath",
        "coordsystem_fpath",
        "datatype",
        "raw",
    ]
