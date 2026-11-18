"""Electrode positions of iEEG records (catalogue survey, 2026-10-19).

- ``dep_keys`` list only sidecars named like the recording, so the electrodes.tsv
  and coordsystem.json BIDS inheritance applies were never downloaded:
  ``space-`` files (nm000362 ``sub-031_acq-10min_space-MNI152NLin2009aSym_*``),
  ``task``/``acq`` level ones (on007118 ``*_task-sleep_acq-A_coordsystem.json``)
  and subject-level ones (on004624). 6,389 records with deposited positions read
  none; on007118 got a generated ``Other / m`` coordsystem and its millimetres
  were read as metres.
- Two ``space-`` electrodes files per recording (nm000473 ACPC and Other) tie in
  MNE-BIDS, which then reads no positions.
- The generated coordsystem.json said ``"m"`` (a guess), and was written even
  when every coordinate is n/a (ds004944).
- Positions without a coordsystem.json (ds004770, on005007, on006107): any
  generated file guesses the system and the units, so positions are skipped
  with one warning, as MNE-BIDS does (Bruno, 2026-10-19).
"""

import json
import shutil
import warnings
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import call, patch

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, get_entities_from_fname, write_raw_bids

from eegdash.dataset.base import EEGDashRaw

BASE = "s3://fake/fakeds"
CHS = ["A1", "A2"]


def _electrodes(xyz):
    rows = [f"{ch}\t{x}\t{y}\t{z}" for ch, (x, y, z) in zip(CHS, xyz)]
    return "name\tx\ty\tz\n" + "\n".join(rows) + "\n"


MM = _electrodes([(40.0, -12.5, 102.0), (-30.0, 8.0, 55.0)])
NA = _electrodes([("n/a",) * 3] * 2)


def _coordsystem(system, units="mm"):
    return json.dumps({"iEEGCoordinateSystem": system, "iEEGCoordinateUnits": units})


def _write_ieeg(root, **entities):
    pytest.importorskip("pybv")
    info = mne.create_info(CHS, 100.0, "seeg")
    raw = mne.io.RawArray(np.zeros((2, 500)), info)
    raw.set_meas_date(datetime(2020, 1, 1, tzinfo=UTC))
    bp = BIDSPath(root=root, subject="01", task="sleep", datatype="ieeg", **entities)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_raw_bids(
            raw, bp, format="BrainVision", allow_preload=True, verbose="ERROR"
        )
    for pattern in ("*_electrodes.*", "*_coordsystem.json"):  # n/a placeholders
        for p in root.rglob(pattern):
            p.unlink()
    return bp.fpath.relative_to(root).as_posix()


def _record(relpath, base, backend, dep_keys):
    from_name = get_entities_from_fname(Path(relpath).name)
    entities = {k: from_name[k] for k in ("subject", "session", "task", "run")}
    return {
        "dataset": "fakeds",
        "bids_relpath": relpath,
        "datatype": "ieeg",
        "suffix": "ieeg",
        "extension": ".vhdr",
        "storage": {
            "backend": backend,
            "base": base,
            "raw_key": relpath,
            "dep_keys": dep_keys,
        },
        "entities": entities,
        "entities_mne": entities,
    }


def _digest_dep_keys(relpath):
    """What the digest listed: only sidecars named like the recording."""
    stem = relpath.rsplit("_", 1)[0]
    keys = [f"{stem}_channels.tsv", f"{stem}_ieeg.json"]
    return keys + [relpath.replace(".vhdr", ext) for ext in (".eeg", ".vmrk")]


class _FakeS3:
    """An anonymous bucket ``s3://fake`` holding the deposit under ``fakeds/``."""

    def __init__(self, remote):
        self.remote = remote

    def _iter_objects(self, path, delimiter=None):
        prefix = path.split("s3://fake/", 1)[1]
        for p in sorted(self.remote.rglob("*")):
            key = "fakeds/" + p.relative_to(self.remote).as_posix()
            if p.is_file() and key.startswith(prefix):
                yield {"Key": key}


def _read_remote(remote, cache, relpath, dep_keys):
    def get(uri, local_path, *, filesystem=None):
        src = remote / uri.split(f"{BASE}/", 1)[1]
        if not src.is_file():
            raise FileNotFoundError(uri)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, local_path)
        return local_path

    def get_many(pairs, *, filesystem=None, skip_existing=True, skip_missing=False):
        for uri, dest in pairs:
            try:
                get(uri, dest)
            except FileNotFoundError:
                if not skip_missing:
                    raise

    with (
        patch("eegdash.dataset.base.validate_record", return_value=None),
        patch("eegdash.downloader.get_s3_filesystem", return_value=_FakeS3(remote)),
        patch("eegdash.downloader.download_s3_file", side_effect=get),
        patch("eegdash.downloader.download_files", side_effect=get_many),
    ):
        ds = EEGDashRaw(_record(relpath, BASE, "s3", dep_keys), cache_dir=str(cache))
        return ds.raw, ds


def _positions(raw):
    return {ch["ch_name"]: ch["loc"][:3].round(6).tolist() for ch in raw.info["chs"]}


# ── 1. inherited electrodes / coordsystem sidecars are downloaded ──


@pytest.mark.parametrize(
    "layout",
    ["space-acq (nm000362)", "task-acq coordsystem (on007118)", "subject (on004624)"],
)
def test_remote_record_reads_the_positions_bids_inheritance_applies(tmp_path, layout):
    remote = tmp_path / "remote"
    if layout.startswith("space"):
        relpath = _write_ieeg(remote, acquisition="10min")
        ieeg = remote / "sub-01" / "ieeg"
        for acq, xyz in (("10min", MM), ("1min", _electrodes([(1, 2, 3)] * 2))):
            name = f"sub-01_acq-{acq}_space-MNI152NLin2009aSym"
            (ieeg / f"{name}_electrodes.tsv").write_text(xyz)
            (ieeg / f"{name}_coordsystem.json").write_text(
                _coordsystem("MNI152NLin2009aSym")
            )
        dep_keys = _digest_dep_keys(relpath)
    elif layout.startswith("task"):
        relpath = _write_ieeg(remote, session="01", acquisition="A", run="01")
        ieeg = remote / "sub-01" / "ses-01" / "ieeg"
        electrodes = relpath.replace("_ieeg.vhdr", "_electrodes.tsv")
        (remote / electrodes).write_text(MM)
        (ieeg / "sub-01_ses-01_task-sleep_acq-A_coordsystem.json").write_text(
            _coordsystem("fsaverage")
        )
        dep_keys = [*_digest_dep_keys(relpath), electrodes]
    else:
        relpath = _write_ieeg(remote)
        (remote / "sub-01" / "sub-01_electrodes.tsv").write_text(MM)
        (remote / "sub-01" / "sub-01_coordsystem.json").write_text(_coordsystem("ACPC"))
        dep_keys = _digest_dep_keys(relpath)

    raw, ds = _read_remote(remote, tmp_path / "cache", relpath, dep_keys)

    # the deposit's millimetres, in metres; nothing generated
    assert _positions(raw) == {
        "A1": [0.04, -0.0125, 0.102],
        "A2": [-0.03, 0.008, 0.055],
    }
    generated = [
        p
        for p in ds.bids_root.rglob("*_coordsystem.json")
        if not (remote / p.relative_to(ds.bids_root)).exists()
    ]
    assert generated == []


def test_position_sidecars_follow_bids_inheritance():
    from eegdash.dataset.bids_dataset import position_sidecars

    files = [
        "sub-01/ieeg/sub-01_acq-10min_space-MNI_electrodes.tsv",
        "sub-01/ieeg/sub-01_acq-1min_space-MNI_electrodes.tsv",
        "sub-01/ieeg/sub-01_acq-10min_space-MNI_coordsystem.json",
        "sub-01/ses-02/ieeg/sub-01_ses-02_electrodes.tsv",
        "sub-01/eeg/sub-01_space-CapTrak_electrodes.tsv",
        "sub-01/ieeg/sub-01_task-x_acq-10min_channels.tsv",
    ]
    assert position_sidecars(
        "sub-01/ieeg/sub-01_task-x_acq-10min_ieeg.vhdr", files
    ) == [
        "sub-01/ieeg/sub-01_acq-10min_space-MNI_electrodes.tsv",
        "sub-01/ieeg/sub-01_acq-10min_space-MNI_coordsystem.json",
    ]
    two_spaces = [
        "sub-01/ieeg/sub-01_space-ACPC_electrodes.tsv",
        "sub-01/ieeg/sub-01_space-Other_electrodes.tsv",
    ]
    assert position_sidecars("sub-01/ieeg/sub-01_task-x_ieeg.vhdr", two_spaces) == (
        two_spaces
    )
    assert position_sidecars("README", files) == []


# ── 2. several space- electrodes files: one space, chosen by a fixed rule ──


def test_two_spaces_read_the_named_one_not_other(tmp_path):
    """nm000473: ``space-ACPC`` and ``space-Other`` for each recording."""
    root = tmp_path / "fakeds"
    relpath = _write_ieeg(root)
    ieeg = root / "sub-01" / "ieeg"
    for space, xyz in (("ACPC", MM), ("Other", _electrodes([(1, 2, 3)] * 2))):
        (ieeg / f"sub-01_space-{space}_electrodes.tsv").write_text(xyz)
        (ieeg / f"sub-01_space-{space}_coordsystem.json").write_text(
            _coordsystem(space)
        )
    with (
        patch("eegdash.dataset.base.validate_record", return_value=None),
        patch("eegdash.io.bids.logger") as log,
    ):
        ds = EEGDashRaw(
            _record(relpath, str(root), "local", []), cache_dir=str(tmp_path)
        )
        raw = ds.raw

    assert _positions(raw) == {
        "A1": [0.04, -0.0125, 0.102],
        "A2": [-0.03, 0.008, 0.055],
    }
    assert log.warning.call_args.args[1:] == (
        "sub-01_task-sleep_ieeg.vhdr",
        "ACPC, Other",
        "ACPC",
    )


# ── 3. no coordsystem.json in the deposit: nothing generated ──


def test_no_coordsystem_is_generated_when_every_coordinate_is_na(tmp_path):
    """ds004944: electrodes.tsv with only n/a, no coordsystem.json."""
    root = tmp_path / "fakeds"
    relpath = _write_ieeg(root)
    (root / "sub-01" / "ieeg" / "sub-01_task-sleep_electrodes.tsv").write_text(NA)
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        raw = EEGDashRaw(
            _record(relpath, str(root), "local", []), cache_dir=str(tmp_path)
        ).raw

    assert list(root.rglob("*_coordsystem.json")) == []
    assert np.isnan([ch["loc"][0] for ch in raw.info["chs"]]).all()


def test_positions_without_a_coordsystem_are_not_applied(tmp_path):
    """ds004770: positions, no coordsystem.json: nothing generated, one warning."""
    root = tmp_path / "fakeds"
    relpath = _write_ieeg(root)
    (root / "sub-01" / "ieeg" / "sub-01_task-sleep_electrodes.tsv").write_text(MM)
    with (
        patch("eegdash.dataset.base.validate_record", return_value=None),
        patch("eegdash.dataset.io.logger") as log,
    ):
        raw = EEGDashRaw(
            _record(relpath, str(root), "local", []), cache_dir=str(tmp_path)
        ).raw

    assert list(root.rglob("*_coordsystem.json")) == []
    assert np.isnan([ch["loc"][0] for ch in raw.info["chs"]]).all()
    assert log.warning.call_args_list == [
        call(
            "%s %s: no coordsystem.json: positions not applied.",
            "fakeds",
            "sub-01_task-sleep_ieeg.vhdr",
        )
    ]


# ── 4. labels-only: anatomical labels surfaced, positions stay absent ──


def test_labels_only_electrodes_tsv_is_surfaced_without_positions(tmp_path):
    """nm000410 ``region``, nm000477 ``target``: label, no invented xyz."""
    from eegdash.dataset.base import electrode_labels

    root = tmp_path / "fakeds"
    relpath = _write_ieeg(root)
    ieeg = root / "sub-01" / "ieeg"
    (ieeg / "sub-01_task-sleep_electrodes.tsv").write_text(
        "name\tx\ty\tz\tregion\nA1\tn/a\tn/a\tn/a\thippocampus\nA2\tn/a\tn/a\tn/a\tamygdala\n"
    )
    rec = root / relpath
    labels = electrode_labels(rec)
    assert labels.loc["A1", "region"] == "hippocampus"
    assert labels.loc["A2", "region"] == "amygdala"

    with patch("eegdash.dataset.base.validate_record", return_value=None):
        raw = EEGDashRaw(
            _record(relpath, str(root), "local", []), cache_dir=str(tmp_path)
        ).raw
    # no download needed: electrodes.tsv is already fetched with the recording
    assert raw.info["temp"]["electrode_labels"].loc["A1", "region"] == "hippocampus"
    assert np.isnan([ch["loc"][0] for ch in raw.info["chs"]]).all()


def test_labels_in_a_different_column_channels_tsv(tmp_path):
    """channels.tsv ``anatomy`` column (a different column than electrodes.tsv)."""
    from eegdash.dataset.base import electrode_labels

    root = tmp_path / "fakeds"
    relpath = _write_ieeg(root)
    (root / "sub-01" / "ieeg" / "sub-01_task-sleep_channels.tsv").write_text(
        "name\ttype\tanatomy\nA1\tSEEG\tsuperior temporal gyrus\nA2\tSEEG\tn/a\n"
    )
    labels = electrode_labels(root / relpath)
    assert labels.loc["A1", "anatomy"] == "superior temporal gyrus"
    assert "A2" not in labels.index  # all-n/a row dropped
