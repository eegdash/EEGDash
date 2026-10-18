"""Three read bugs of EEGDashRaw found in the batch4 read-check (2026-10-18).

- The BIDSPath dropped the entities a record does not carry (``desc``, ``split``,
  ``proc``, ``space``...). For on006923 ``*_desc-preprocessed_eeg.set`` MNE-BIDS
  looked for ``*_eeg.set``, and the direct reader then read the file without
  its sidecars (channel types, bad channels, events).
- The retry that reads a recording without its events.tsv renamed every
  ``*_events.tsv`` of the folder, so runs read at the same time by other threads
  lost their events (on003343 run-02 failed on run-01's retry).
- A ``.fdt`` shorter than its ``.set`` (ds006554 sub-11/16/29/43: 78-80 % of the
  samples, cut inside a sample) was hidden by rewriting ``pnts`` in the cached
  ``.set``: the recording read as a shorter one, with an INFO log line only.
"""

import json
import threading
import warnings
from datetime import UTC, datetime
from unittest.mock import patch

import mne
import mne_bids.read as mne_bids_read
import numpy as np
import pytest
from mne_bids import BIDSPath, write_raw_bids
from scipy import io

from eegdash.dataset.base import EEGDashRaw

SFREQ = 100.0
MEAS_DATE = datetime(2020, 1, 1, 10, 0, 0, tzinfo=UTC)


def _eegdashraw(root, cache_dir, relpath, entities):
    record = {
        "dataset": root.name,
        "bids_relpath": relpath,
        "datatype": relpath.split("/")[-2],
        "suffix": relpath.rsplit("_", 1)[1].split(".")[0],
        "extension": "." + relpath.split(".", 1)[1],
        "storage": {"backend": "local", "base": str(root), "raw_key": relpath},
        "entities": entities,
        "entities_mne": entities,
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        return EEGDashRaw(record, cache_dir=str(cache_dir))


def _write_bids(root, run=None):
    pytest.importorskip("pybv")
    info = mne.create_info(["Fz", "Cz", "Pz"], SFREQ, "eeg")
    data = np.random.default_rng(0).standard_normal((3, int(10 * SFREQ))) * 1e-5
    raw = mne.io.RawArray(data, info)
    raw.set_meas_date(MEAS_DATE)
    bp = BIDSPath(subject="01", task="rest", run=run, datatype="eeg", root=root)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_raw_bids(
            raw, bp, format="BrainVision", allow_preload=True, verbose="ERROR"
        )
    return root / "sub-01" / "eeg"


# ── 1. entities of the file name reach the BIDSPath ──


def test_bidspath_keeps_every_entity_of_the_file_name(tmp_path):
    name = (
        "sub-01_ses-a_task-x_acq-b_run-02_proc-sss_space-CapTrak_split-01"
        "_desc-clean_meg.fif"
    )
    ds = _eegdashraw(
        tmp_path / "ds",
        tmp_path,
        f"sub-01/ses-a/meg/{name}",
        {"subject": "01", "session": "a", "task": "x", "run": "02"},
    )
    bp = ds.bidspath
    assert (bp.acquisition, bp.processing, bp.space) == ("b", "sss", "CapTrak")
    assert (bp.split, bp.description) == ("01", "clean")
    assert bp.fpath == ds.filecache


def test_bidspath_run_stays_the_records(tmp_path):
    """A non-numeric run (``run-5H``) is ``None`` in entities_mne and stays so."""
    ds = _eegdashraw(
        tmp_path / "ds",
        tmp_path,
        "sub-01/eeg/sub-01_task-x_run-5H_desc-x_eeg.vhdr",
        {"subject": "01", "task": "x", "run": None},
    )
    assert ds.bidspath.run is None
    assert ds.bidspath.description == "x"


def test_desc_recording_is_read_with_its_sidecars(tmp_path):
    """on006923: ``*_desc-preprocessed_eeg.set``, channels.tsv without ``desc``."""
    root = tmp_path / "ds-desc"
    eeg = _write_bids(root)
    old, new = "sub-01_task-rest_eeg", "sub-01_task-rest_desc-preprocessed_eeg"
    for ext in (".vhdr", ".vmrk"):
        (eeg / f"{new}{ext}").write_text(
            (eeg / f"{old}{ext}").read_text().replace(old, new)
        )
        (eeg / f"{old}{ext}").unlink()
    (eeg / f"{old}.eeg").rename(eeg / f"{new}.eeg")
    scans = root / "sub-01" / "sub-01_scans.tsv"
    scans.write_text(scans.read_text().replace(old, new))
    # Sidecar content the direct reader cannot know: Cz bad, Pz an EOG channel.
    channels = eeg / "sub-01_task-rest_channels.tsv"
    rows = [line.split("\t") for line in channels.read_text().splitlines()]
    col = {k: i for i, k in enumerate(rows[0])}
    for row in rows[1:]:
        if row[col["name"]] == "Cz":
            row[col["status"]] = "bad"
        if row[col["name"]] == "Pz":
            row[col["type"]] = "EOG"
    channels.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    (eeg / "sub-01_task-rest_events.tsv").write_text(
        "onset\tduration\ttrial_type\n1.0\t0\tgo\n"
    )

    ds = _eegdashraw(
        root, tmp_path, f"sub-01/eeg/{new}.vhdr", {"subject": "01", "task": "rest"}
    )
    assert ds.bidspath.fpath == ds.filecache
    with patch(
        "eegdash.dataset.base._load_raw_direct",
        side_effect=AssertionError("fell back to the direct reader"),
    ):
        raw = ds.raw
    assert raw.info["bads"] == ["Cz"]
    assert raw.get_channel_types(picks=["Pz"]) == ["eog"]
    assert list(raw.annotations.description) == ["go"]


# ── 2. reading without events.tsv touches no file ──


def test_events_retry_does_not_hide_other_runs_events(tmp_path):
    """run-01's events.tsv cannot be read; run-02, read meanwhile, keeps its own."""
    root = tmp_path / "ds-runs"
    for run in ("01", "02"):
        eeg = _write_bids(root, run=run)
        (eeg / f"sub-01_task-rest_run-{run}_events.tsv").write_text(
            "onset\tduration\ttrial_type\n1.0\t0\tgo\n2.0\t0\tstop\n"
        )
    listing = sorted(p.name for p in eeg.iterdir())

    def relpath(run):
        return f"sub-01/eeg/sub-01_task-rest_run-{run}_eeg.vhdr"

    def entities(run):
        return {"subject": "01", "task": "rest", "run": run}

    orig_events = mne_bids_read._handle_events_reading
    orig_read = mne_bids_read._read_raw
    seen = []

    def events(events_fname, raw, *args, **kwargs):
        if "run-01" in str(events_fname):
            raise AssertionError("malformed events.tsv")
        return orig_events(events_fname, raw, *args, **kwargs)

    def read_run02():
        try:
            raw = _eegdashraw(root, tmp_path, relpath("02"), entities("02")).raw
            seen.append(list(raw.annotations.description))
        except Exception as error:  # noqa: BLE001 - reported below
            seen.append(error)

    def read(raw_path, *args, **kwargs):
        if "run-01" in str(raw_path):  # each attempt on run-01, retries included
            thread = threading.Thread(target=read_run02)
            thread.start()
            thread.join(60)
        return orig_read(raw_path, *args, **kwargs)

    with (
        patch.object(mne_bids_read, "_handle_events_reading", events),
        patch.object(mne_bids_read, "_read_raw", read),
    ):
        raw1 = _eegdashraw(root, tmp_path, relpath("01"), entities("01")).raw

    assert raw1.n_times == int(10 * SFREQ)
    assert "go" not in raw1.annotations.description  # run-01 read without events
    assert len(seen) >= 2
    assert seen == [["go", "stop"]] * len(seen)  # run-02 had its events every time
    assert sorted(p.name for p in eeg.iterdir()) == listing


# ── 3. a truncated .fdt reads its complete samples, loudly ──


def test_truncated_fdt_reads_complete_samples_with_a_warning(tmp_path):
    """ds006554 sub-11: the .fdt ends inside a sample, 80 % of pnts present."""
    root = tmp_path / "ds-trunc"
    eeg = root / "sub-01" / "eeg"
    eeg.mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.8.0"})
    )
    set_path = eeg / "sub-01_task-rest_eeg.set"
    fdt_path = set_path.with_suffix(".fdt")
    n_ch, pnts, keep = 3, 20, 13
    data = np.arange(n_ch * pnts, dtype="<f4").reshape(n_ch, pnts)
    blob = data.T.tobytes()  # all channels of one sample together
    fdt_path.write_bytes(blob[: keep * n_ch * 4 + 6])  # cut inside sample 14
    chanlocs = np.array(
        [(c.encode(),) for c in ("Fz", "Cz", "Pz")], [("labels", "S10")]
    )
    eeg_struct = dict(
        nbchan=n_ch,
        pnts=pnts,
        trials=1,
        srate=SFREQ,
        xmin=0.0,
        data=fdt_path.name,
        chanlocs=chanlocs,
        event=[],
        epoch=[],
    )
    io.savemat(set_path, {"EEG": eeg_struct}, appendmat=False, oned_as="row")
    before = set_path.read_bytes()

    ds = _eegdashraw(
        root,
        tmp_path,
        "sub-01/eeg/sub-01_task-rest_eeg.set",
        {"subject": "01", "task": "rest"},
    )
    with pytest.warns(RuntimeWarning, match="truncated EEGLAB data file"):
        raw = ds.raw

    assert set_path.read_bytes() == before  # the cached .set is not rewritten
    assert raw.n_times == keep
    np.testing.assert_allclose(raw.get_data(), data[:, :keep] * 1e-6, rtol=1e-6)
    assert f"Truncated data file: {keep} of {pnts} samples" in raw.info["description"]


def test_complete_fdt_is_left_alone(tmp_path):
    from eegdash.io.eeglab import crop_truncated_fdt

    set_path = tmp_path / "ok.set"
    data = np.zeros((2, 10), "<f4")
    data.T.tofile(set_path.with_suffix(".fdt"))
    chanlocs = np.array([(b"A",), (b"B",)], [("labels", "S10")])
    eeg_struct = dict(
        nbchan=2,
        pnts=10,
        trials=1,
        srate=SFREQ,
        xmin=0.0,
        data="ok.fdt",
        chanlocs=chanlocs,
        event=[],
        epoch=[],
    )
    io.savemat(set_path, {"EEG": eeg_struct}, appendmat=False, oned_as="row")
    assert crop_truncated_fdt(mne.io.read_raw_eeglab(set_path)) is None
    info = mne.create_info(["a"], SFREQ, "eeg")
    assert crop_truncated_fdt(mne.io.RawArray(np.zeros((1, 10)), info)) is None
