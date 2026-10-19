import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest
from scipy.io import savemat
from scipy.io.matlab import MatWriteWarning

from eegdash.dataset.io import (
    _convert_time_with_numeric_dash,
    _deduplicate_channel_names,
    _eeglab_ch_names_from_eeg,
    _eeglab_fdt_path,
    _eeglab_get_first_eeg,
    _eeglab_load_first_eeg,
    _ensure_coordsystem,
    _fix_negative_annotation_durations,
    _generate_coordsystem_json,
    _generate_vhdr_from_metadata,
    _load_raw_eeglab_alleeg,
    _load_raw_from_eeglab_epochs,
    _parse_set_metadata,
    _read_bids_channels_tsv,
    _repair_channels_tsv,
)


def _savemat_without_matwritewarning(path: Path, data: dict) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MatWriteWarning)
        savemat(str(path), data, do_compression=False)


def test_convert_time_with_numeric_dash_dd_mm_yyyy():
    """Numeric dash date 14-10-1925 (DD-MM-YYYY) is normalized to 14/10/1925 and delegated."""
    orig = MagicMock(return_value=12345.0)
    out = _convert_time_with_numeric_dash("14-10-1925", "12:00:00", orig=orig)
    assert out == 12345.0
    orig.assert_called_once_with("14/10/1925", "12:00:00")


def test_convert_time_with_numeric_dash_mm_dd_yyyy():
    """Numeric dash date 10-14-1925 (MM-DD-YYYY) is normalized and delegated."""
    orig = MagicMock(return_value=0.0)
    _convert_time_with_numeric_dash("10-14-1925", "00:00:00", orig=orig)
    orig.assert_called_once_with("14/10/1925", "00:00:00")


def test_convert_time_with_numeric_dash_iso():
    """ISO-style YYYY-MM-DD is normalized to dd/mm/yyyy and delegated."""
    orig = MagicMock(return_value=1.0)
    _convert_time_with_numeric_dash("1925-10-14", "01:00:00", orig=orig)
    orig.assert_called_once_with("14/10/1925", "01:00:00")


def test_convert_time_with_numeric_dash_fallback():
    """Completely unparsable date is passed through to orig unchanged."""
    orig = MagicMock(return_value=999.0)
    out = _convert_time_with_numeric_dash("not-a-date", "12:00:00", orig=orig)
    assert out == 999.0
    orig.assert_called_once_with("not-a-date", "12:00:00")


def test_convert_time_locale_german_okt():
    """German month abbreviation 'Okt' is normalised to numeric date."""
    orig = MagicMock(return_value=0.0)
    _convert_time_with_numeric_dash("14-Okt-1925", "12:00:00", orig=orig)
    orig.assert_called_once_with("14/10/1925", "12:00:00")


def test_convert_time_locale_italian_ott():
    """Italian month abbreviation 'Ott' is normalised to numeric date."""
    orig = MagicMock(return_value=0.0)
    _convert_time_with_numeric_dash("14-Ott-1925", "12:00:00", orig=orig)
    orig.assert_called_once_with("14/10/1925", "12:00:00")


def test_convert_time_locale_german_maerz():
    """German month abbreviation 'Mär' is normalised to numeric date."""
    orig = MagicMock(return_value=0.0)
    _convert_time_with_numeric_dash("05-Mär-2000", "09:00:00", orig=orig)
    orig.assert_called_once_with("05/03/2000", "09:00:00")


def test_convert_time_locale_already_english():
    """English month abbreviation passes through dateutil unchanged."""
    orig = MagicMock(return_value=0.0)
    _convert_time_with_numeric_dash("14-Oct-1925", "12:00:00", orig=orig)
    orig.assert_called_once_with("14/10/1925", "12:00:00")


def test_convert_time_with_numeric_dash_strips_whitespace():
    """Date string is stripped before parsing."""
    orig = MagicMock(return_value=0.0)
    _convert_time_with_numeric_dash("  14-10-1925  ", "00:00:00", orig=orig)
    orig.assert_called_once_with("14/10/1925", "00:00:00")


def test_ensure_coordsystem_uses_the_subject_level_one(tmp_path):
    """A subject-level coordsystem.json applies as is: no link, no generated file."""
    eeg_dir = tmp_path / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "sub-01_task-rest_electrodes.tsv").write_text(_ELECTRODES)
    (tmp_path / "sub-01" / "sub-01_coordsystem.json").write_text(
        json.dumps({"EEGCoordinateSystem": "CapTrak", "EEGCoordinateUnits": "m"})
    )

    _ensure_coordsystem(_bp(tmp_path, "eeg"))

    assert sorted(p.name for p in eeg_dir.iterdir()) == [
        "sub-01_task-rest_electrodes.tsv"
    ]


def test_io_error_handling(tmp_path):
    """Test error handling branches in IO module."""
    # ensure_coordsystem on a missing dataset or without electrodes
    _ensure_coordsystem(_bp(tmp_path / "missing_dir", "eeg"))  # Should not raise
    assert not (tmp_path / "missing_dir").exists()


@pytest.mark.parametrize(
    "chanlocs,expected_names",
    [
        (
            [SimpleNamespace(labels="Fp1"), SimpleNamespace(labels="Fp2")],
            ["Fp1", "Fp2"],
        ),
        (SimpleNamespace(labels="Cz"), ["Cz"]),
    ],
    ids=["iterable_chanlocs", "single_chanloc"],
)
def test_parse_set_metadata_chanloc_variants(tmp_path, chanlocs, expected_names):
    """Extract channel names from iterable and scalar chanloc variants."""
    eeg = SimpleNamespace(
        srate=np.array([256.0]),
        nbchan=np.array([len(expected_names)]),
        pnts=np.array([4]),
        chanlocs=chanlocs,
        data=np.ones((len(expected_names), 4), dtype=np.float64),
    )
    with patch("scipy.io.loadmat", return_value={"EEG": eeg}):
        meta = _parse_set_metadata(tmp_path / "dummy.set")

    assert meta["srate"] == 256.0
    assert meta["nbchan"] == len(expected_names)
    assert meta["pnts"] == 4
    assert meta["ch_names"] == expected_names
    assert meta["data"].dtype == np.float32


@pytest.mark.parametrize("missing_field", ["srate", "nbchan", "pnts"])
def test_parse_set_metadata_missing_required_fields(tmp_path, missing_field):
    """Missing required EEG fields raise ValueError with a clear message."""
    eeg = SimpleNamespace(
        srate=np.array([256.0]),
        nbchan=np.array([2]),
        pnts=np.array([10]),
        chanlocs=None,
    )
    setattr(eeg, missing_field, None)

    with patch("scipy.io.loadmat", return_value={"EEG": eeg}):
        with pytest.raises(
            ValueError, match=f"Missing required field '{missing_field}'"
        ):
            _parse_set_metadata(tmp_path / "dummy.set")


@pytest.mark.parametrize(
    "rows,expected",
    [
        ("Fz\tEEG\tuV\nEOG1\tEOG\tuV\n", (["Fz", "EOG1"], ["eeg", "eog"])),
        ("A1\tXYZ\tuV\nA2\t\t\n", (["A1", "A2"], ["eeg", "eeg"])),
        (" \tEEG\tuV\n", None),
    ],
    ids=["known_types", "unknown_or_empty_type_fallback", "blank_names_dropped"],
)
def test_read_bids_channels_tsv_parametrized(tmp_path, rows, expected):
    """BIDS channel parsing maps types and drops empty-name rows."""
    channels_tsv = tmp_path / "sub-01_task-rest_channels.tsv"
    channels_tsv.write_text("name\ttype\tunits\n" + rows)

    assert _read_bids_channels_tsv(tmp_path) == expected


# Tests for VHDR/VMRK generation from metadata


def test_generate_vhdr_from_metadata_success(tmp_path):
    """Test successful VHDR generation from complete metadata."""
    vhdr_path = tmp_path / "sub-01_task-rest_eeg.vhdr"
    record = {
        "ch_names": ["Fp1", "Fp2", "F3", "F4"],
        "sampling_frequency": 500,
        "nchans": 4,
    }

    assert _generate_vhdr_from_metadata(vhdr_path, record) is True

    content = vhdr_path.read_text()
    assert "NumberOfChannels=4" in content
    assert "SamplingInterval=2000" in content  # 1_000_000 / 500
    assert "Ch1=Fp1,,0.1,µV" in content
    # no stub .vmrk: MNE 1.13 reads a missing marker file as no annotations
    assert not (tmp_path / "sub-01_task-rest_eeg.vmrk").exists()


@pytest.mark.parametrize(
    "record,reason",
    [
        ({"sampling_frequency": 500, "nchans": 4}, "missing ch_names"),
        ({"ch_names": ["Fp1"], "nchans": 1}, "missing sampling_frequency"),
        ({"ch_names": ["Fp1"], "sampling_frequency": 500}, "missing nchans"),
        (
            {"ch_names": ["Fp1"], "sampling_frequency": 500, "nchans": 4},
            "ch_names/nchans mismatch",
        ),
    ],
    ids=["no_ch_names", "no_sfreq", "no_nchans", "mismatch"],
)
def test_generate_vhdr_invalid_metadata(tmp_path, record, reason):
    """Test VHDR generation fails gracefully with invalid metadata."""
    vhdr_path = tmp_path / "test.vhdr"
    assert _generate_vhdr_from_metadata(vhdr_path, record) is False, reason
    assert not vhdr_path.exists()


def test_generate_vhdr_creates_parent_dirs(tmp_path):
    """Test VHDR generation creates parent directories if needed."""
    vhdr_path = tmp_path / "sub-01" / "eeg" / "test.vhdr"
    record = {"ch_names": ["Fp1", "Fp2"], "sampling_frequency": 256, "nchans": 2}

    assert _generate_vhdr_from_metadata(vhdr_path, record) is True
    assert vhdr_path.exists()


def test_generate_vhdr_does_not_overwrite_vmrk(tmp_path):
    """Test VHDR generation doesn't overwrite existing VMRK file."""
    vmrk_path = tmp_path / "test.vmrk"
    vmrk_path.write_text("Custom VMRK content")

    record = {"ch_names": ["Fp1"], "sampling_frequency": 500, "nchans": 1}
    _generate_vhdr_from_metadata(tmp_path / "test.vhdr", record)

    assert vmrk_path.read_text() == "Custom VMRK content"


# Tests for coordsystem.json generation with datatype-specific keys


@pytest.mark.parametrize(
    "datatype,expected_prefix",
    [("eeg", "EEG"), ("ieeg", "iEEG"), ("meg", "MEG"), ("emg", "EMG")],
    ids=["eeg", "ieeg", "meg", "emg"],
)
def test_generate_coordsystem_json_datatype_keys(tmp_path, datatype, expected_prefix):
    """Generated coordsystem.json uses correct keys per BIDS datatype."""
    electrodes = tmp_path / "sub-01_space-MNI_electrodes.tsv"
    electrodes.write_text("name\tx\ty\tz\nCh1\t0\t0\t0\n")

    assert _generate_coordsystem_json(electrodes, datatype=datatype) is True

    coordsystem = tmp_path / "sub-01_space-MNI_coordsystem.json"
    assert coordsystem.exists()
    data = json.loads(coordsystem.read_text())
    assert f"{expected_prefix}CoordinateSystem" in data
    assert data[f"{expected_prefix}CoordinateSystem"] == "MNI"
    assert f"{expected_prefix}CoordinateUnits" in data
    assert data[f"{expected_prefix}CoordinateUnits"] == "n/a"


def test_generate_coordsystem_json_no_space_entity(tmp_path):
    """Coordsystem defaults to 'Other' when electrodes filename has no space entity."""
    electrodes = tmp_path / "sub-01_electrodes.tsv"
    electrodes.write_text("name\tx\ty\tz\nCh1\t0\t0\t0\n")

    assert _generate_coordsystem_json(electrodes, datatype="ieeg") is True

    coordsystem = tmp_path / "sub-01_coordsystem.json"
    data = json.loads(coordsystem.read_text())
    assert data["iEEGCoordinateSystem"] == "Other"


def test_generate_coordsystem_json_default_datatype(tmp_path):
    """Default datatype is 'eeg' when not specified."""
    electrodes = tmp_path / "sub-01_electrodes.tsv"
    electrodes.write_text("name\tx\ty\tz\nCh1\t0\t0\t0\n")

    assert _generate_coordsystem_json(electrodes) is True

    coordsystem = tmp_path / "sub-01_coordsystem.json"
    data = json.loads(coordsystem.read_text())
    assert "EEGCoordinateSystem" in data


_ELECTRODES = "name\tx\ty\tz\nCh1\t40.0\t-12.5\t102.0\nCh2\tn/a\tn/a\tn/a\n"


def _bp(root, datatype):
    from mne_bids import BIDSPath

    return BIDSPath(
        root=root,
        subject="01",
        task="rest",
        datatype=datatype,
        suffix=datatype,
        extension=".vhdr",
        check=False,
    )


def test_ensure_coordsystem_generates_ieeg_keys(tmp_path):
    """No coordsystem.json anywhere: iEEG keys, units n/a, system Other, warned."""
    ieeg_dir = tmp_path / "sub-01" / "ieeg"
    ieeg_dir.mkdir(parents=True)
    (ieeg_dir / "sub-01_electrodes.tsv").write_text(_ELECTRODES)

    with patch("eegdash.dataset.io.logger") as log:
        _ensure_coordsystem(_bp(tmp_path, "ieeg"))

    data = json.loads((ieeg_dir / "sub-01_coordsystem.json").read_text())
    assert data == {"iEEGCoordinateSystem": "Other", "iEEGCoordinateUnits": "n/a"}
    assert "the deposit gives none" in log.warning.call_args.args


def test_ensure_coordsystem_keeps_correct_keys_ieeg(tmp_path):
    """coordsystem.json with correct iEEG keys in ieeg dir is left unchanged."""
    ieeg_dir = tmp_path / "sub-01" / "ieeg"
    ieeg_dir.mkdir(parents=True)

    (ieeg_dir / "sub-01_electrodes.tsv").write_text(_ELECTRODES)
    coordsystem = ieeg_dir / "sub-01_coordsystem.json"
    original_content = json.dumps(
        {"iEEGCoordinateSystem": "MNI305", "iEEGCoordinateUnits": "mm"}
    )
    coordsystem.write_text(original_content)

    _ensure_coordsystem(_bp(tmp_path, "ieeg"))

    assert coordsystem.read_text() == original_content
    assert len(list(ieeg_dir.iterdir())) == 2


# Tests for TSV decimal separator repair


# ── EEGLAB fallback loader tests ──────────────────────────────────────


def _create_minimal_set_fdt(tmp_path, n_channels=3, n_samples=100, sfreq=256.0):
    """Helper: create a minimal .set + .fdt pair for testing."""
    import numpy as np

    set_path = tmp_path / "test.set"
    fdt_path = tmp_path / "test.fdt"

    # Create data and write .fdt
    data = np.random.randn(n_channels, n_samples).astype(np.float32)
    data.tofile(str(fdt_path))

    # Build a minimal EEG struct as a dict of numpy arrays (scipy-compatible)
    eeg = {
        "srate": np.array([[sfreq]]),
        "nbchan": np.array([[n_channels]]),
        "pnts": np.array([[n_samples]]),
        "datfile": np.array(["test.fdt"]),
        "data": np.array(["test.fdt"]),
    }

    _savemat_without_matwritewarning(set_path, {"EEG": eeg})
    return set_path, fdt_path, data


def test_load_raw_eeglab_fallback_with_fdt(tmp_path):
    """Fallback loader reads .set metadata + .fdt binary and returns RawArray."""
    from eegdash.dataset.io import _load_raw_eeglab_fallback

    set_path, _, data = _create_minimal_set_fdt(
        tmp_path, n_channels=3, n_samples=100, sfreq=256.0
    )

    raw = _load_raw_eeglab_fallback(set_path)

    assert raw.info["sfreq"] == 256.0
    assert len(raw.ch_names) == 3
    assert raw.n_times == 100
    # Default channel names when no chanlocs
    assert raw.ch_names == ["EEG001", "EEG002", "EEG003"]


def test_load_raw_eeglab_fallback_with_bids_channels(tmp_path):
    """Channel names/types from BIDS channels.tsv take priority."""
    from eegdash.dataset.io import _load_raw_eeglab_fallback

    set_path, _, _ = _create_minimal_set_fdt(
        tmp_path, n_channels=3, n_samples=100, sfreq=256.0
    )

    # Write a channels.tsv sidecar
    channels_tsv = tmp_path / "sub-01_task-rest_channels.tsv"
    channels_tsv.write_text(
        "name\ttype\tunits\nFz\tEEG\tuV\nCz\tEEG\tuV\nEOG1\tEOG\tuV\n"
    )

    raw = _load_raw_eeglab_fallback(set_path)

    assert raw.ch_names == ["Fz", "Cz", "EOG1"]
    assert raw.get_channel_types() == ["eeg", "eeg", "eog"]


def test_load_raw_eeglab_fallback_no_data_raises(tmp_path):
    """Raises ValueError when no .fdt and no embedded data."""
    import numpy as np

    from eegdash.dataset.io import _load_raw_eeglab_fallback

    set_path = tmp_path / "test.set"

    eeg = {
        "srate": np.array([[256.0]]),
        "nbchan": np.array([[3]]),
        "pnts": np.array([[100]]),
        "datfile": np.array([""]),
        "data": np.array(["test.fdt"]),
    }
    _savemat_without_matwritewarning(set_path, {"EEG": eeg})

    with pytest.raises(ValueError, match="No data source"):
        _load_raw_eeglab_fallback(set_path)


def test_load_raw_eeglab_fallback_size_mismatch_raises(tmp_path):
    """Raises ValueError when .fdt size doesn't match metadata."""
    import numpy as np

    from eegdash.dataset.io import _load_raw_eeglab_fallback

    set_path = tmp_path / "test.set"
    fdt_path = tmp_path / "test.fdt"

    # Write wrong-sized .fdt (too small)
    np.zeros(10, dtype=np.float32).tofile(str(fdt_path))

    eeg = {
        "srate": np.array([[256.0]]),
        "nbchan": np.array([[3]]),
        "pnts": np.array([[100]]),
        "datfile": np.array(["test.fdt"]),
        "data": np.array(["test.fdt"]),
    }
    _savemat_without_matwritewarning(set_path, {"EEG": eeg})

    with pytest.raises(ValueError, match="FDT size mismatch"):
        _load_raw_eeglab_fallback(set_path)


@pytest.mark.parametrize(
    "embedded_data,expected_shape,raises",
    [
        (np.arange(6, dtype=np.float32).reshape(2, 3), (2, 3), False),
        (np.arange(6, dtype=np.float32), (2, 3), False),
        (np.arange(5, dtype=np.float32), None, True),
    ],
    ids=["already_2d", "reshape_1d", "shape_mismatch_raises"],
)
def test_load_raw_eeglab_fallback_embedded_data_paths(
    tmp_path, embedded_data, expected_shape, raises
):
    """Embedded .set data path handles reshape and mismatch cases."""
    from eegdash.dataset.io import _load_raw_eeglab_fallback

    set_path = tmp_path / "embedded.set"
    set_path.write_text("placeholder")

    meta = {
        "nbchan": 2,
        "pnts": 3,
        "srate": 128.0,
        "ch_names": ["Fz", "Cz"],
        "data": embedded_data,
    }

    with patch("eegdash.dataset.io._parse_set_metadata", return_value=meta):
        if raises:
            with pytest.raises(ValueError, match="Embedded data shape"):
                _load_raw_eeglab_fallback(set_path)
        else:
            raw = _load_raw_eeglab_fallback(set_path)
            assert raw.get_data().shape == expected_shape
            assert raw.ch_names == ["Fz", "Cz"]


# ---------- _repair_channels_tsv ----------


def test_repair_channels_tsv_empty_file(tmp_path):
    """Empty channels.tsv is removed so MNE-BIDS skips gracefully."""
    (tmp_path / "sub-01_channels.tsv").write_text("")
    assert _repair_channels_tsv(tmp_path) is True
    assert not (tmp_path / "sub-01_channels.tsv").exists()


def test_repair_channels_tsv_whitespace_only(tmp_path):
    """Whitespace-only channels.tsv is removed."""
    (tmp_path / "sub-01_channels.tsv").write_text("   \n\n  \t  \n")
    assert _repair_channels_tsv(tmp_path) is True
    assert not (tmp_path / "sub-01_channels.tsv").exists()


def test_repair_channels_tsv_missing_name_column(tmp_path):
    """First column is renamed to 'name' when header lacks it."""
    content = "foo\ttype\tunits\nEEG1\teeg\tuV\nEEG2\teeg\tuV\n"
    (tmp_path / "sub-01_channels.tsv").write_text(content)
    assert _repair_channels_tsv(tmp_path) is True
    lines = (tmp_path / "sub-01_channels.tsv").read_text().splitlines()
    assert lines[0] == "name\ttype\tunits"
    assert lines[1] == "EEG1\teeg\tuV"
    assert lines[2] == "EEG2\teeg\tuV"


def test_repair_channels_tsv_valid_file(tmp_path):
    """Valid channels.tsv with 'name' column is left unchanged."""
    content = "name\ttype\tunits\nEEG1\teeg\tuV\n"
    (tmp_path / "sub-01_channels.tsv").write_text(content)
    assert _repair_channels_tsv(tmp_path) is False
    assert (tmp_path / "sub-01_channels.tsv").read_text() == content


def test_repair_channels_tsv_header_only(tmp_path):
    """Header-only channels.tsv with 'name' is left unchanged."""
    content = "name\ttype\tunits\n"
    (tmp_path / "sub-01_channels.tsv").write_text(content)
    assert _repair_channels_tsv(tmp_path) is False


def test_repair_channels_tsv_nonexistent_dir(tmp_path):
    """Returns False for nonexistent directory."""
    assert _repair_channels_tsv(tmp_path / "missing") is False


def test_repair_channels_tsv_no_channels_files(tmp_path):
    """Returns False when no *_channels.tsv files exist."""
    (tmp_path / "sub-01_events.tsv").write_text("onset\tduration\n")
    assert _repair_channels_tsv(tmp_path) is False


def test_repair_channels_tsv_bom_valid(tmp_path):
    """BOM + valid header is left unchanged."""
    content = "name\ttype\tunits\nEEG1\teeg\tuV\n"
    (tmp_path / "sub-01_channels.tsv").write_text(content, encoding="utf-8-sig")
    assert _repair_channels_tsv(tmp_path) is False


def test_repair_channels_tsv_bom_empty(tmp_path):
    """BOM-only file (no real content) is removed."""
    (tmp_path / "sub-01_channels.tsv").write_bytes(b"\xef\xbb\xbf")
    assert _repair_channels_tsv(tmp_path) is True
    assert not (tmp_path / "sub-01_channels.tsv").exists()


def test_repair_channels_tsv_multiple_files(tmp_path):
    """Only bad files are repaired; good files are unchanged."""
    good = "name\ttype\tunits\nEEG1\teeg\tuV\n"
    (tmp_path / "sub-01_channels.tsv").write_text(good)
    (tmp_path / "sub-02_channels.tsv").write_text("")
    assert _repair_channels_tsv(tmp_path) is True
    assert (tmp_path / "sub-01_channels.tsv").read_text() == good
    assert not (tmp_path / "sub-02_channels.tsv").exists()


# ── EEGLAB helpers ──


def test_eeglab_get_first_eeg_flat():
    """Flat dict with nbchan/pnts returns the same dict."""
    mat = {"nbchan": 2, "pnts": 100, "srate": 250.0}
    out = _eeglab_get_first_eeg(mat)
    assert out is mat


def test_eeglab_get_first_eeg_eeg_key():
    """Dict with EEG key returns nested struct as dict."""
    inner = {"nbchan": 2, "pnts": 50}
    mat = {"EEG": inner}
    out = _eeglab_get_first_eeg(mat)
    assert out == inner


def test_eeglab_get_first_eeg_no_eeg_returns_none():
    """Dict without EEG/ALLEEG/nbchan returns None."""
    assert _eeglab_get_first_eeg({}) is None
    assert _eeglab_get_first_eeg({"other": 1}) is None


def test_eeglab_fdt_path_external(tmp_path):
    """External data string resolves to .fdt path."""
    set_path = tmp_path / "sub-01_task-rest_eeg.set"
    (tmp_path / "sub-01_task-rest_eeg.fdt").touch()
    eeg = {"data": "sub-01_task-rest_eeg.fdt"}
    out = _eeglab_fdt_path(set_path, eeg)
    assert out is not None
    assert out.name == "sub-01_task-rest_eeg.fdt"


def test_eeglab_fdt_path_fallback_suffix(tmp_path):
    """When data ref file missing, fallback to .set.with_suffix(.fdt)."""
    set_path = tmp_path / "file.set"
    fdt_path = tmp_path / "file.fdt"
    fdt_path.touch()
    eeg = {"data": "missing.fdt"}
    out = _eeglab_fdt_path(set_path, eeg)
    assert out == fdt_path


def test_eeglab_fdt_path_prefers_own_fdt(tmp_path):
    """on004348: EEG.data names another recording's .fdt that exists; take the own one."""
    set_path = tmp_path / "sub-01_task-wake_eeg.set"
    (tmp_path / "sub-01_task-wake_eeg.fdt").touch()
    (tmp_path / "sub-01_task-sleep_eeg.fdt").touch()
    out = _eeglab_fdt_path(set_path, {"data": "sub-01_task-sleep_eeg.fdt"})
    assert out == tmp_path / "sub-01_task-wake_eeg.fdt"


def test_eeglab_fdt_path_inline_returns_none():
    """Inline data (non-string) returns None."""
    set_path = Path("/tmp/file.set")
    eeg = {"data": [1.0, 2.0]}
    assert _eeglab_fdt_path(set_path, eeg) is None


def test_eeglab_ch_names_from_eeg_fallback():
    """No chanlocs yields CH1, CH2, ..."""
    eeg = {}
    assert _eeglab_ch_names_from_eeg(eeg, 3) == ["CH1", "CH2", "CH3"]


def test_eeglab_ch_names_from_eeg_labels():
    """Chanlocs with labels yield those names."""
    eeg = {"chanlocs": [{"labels": "Fp1"}, {"labels": "Fp2"}]}
    assert _eeglab_ch_names_from_eeg(eeg, 2) == ["Fp1", "Fp2"]


def test_eeglab_get_first_eeg_alleeg():
    """Dict with ALLEEG key returns first element as dict."""
    first_eeg = {"nbchan": 1, "pnts": 10}
    mat = {"ALLEEG": [first_eeg]}
    out = _eeglab_get_first_eeg(mat)
    assert out == first_eeg


def test_eeglab_get_first_eeg_alleeg_empty_returns_none():
    """ALLEEG empty array returns None."""
    mat = {"ALLEEG": np.array([], dtype=object)}
    out = _eeglab_get_first_eeg(mat)
    assert out is None


def test_eeglab_load_first_eeg_returns_eeg_when_valid_set(tmp_path):
    """Valid .set file returns first EEG dict."""
    pytest.importorskip("scipy")

    set_path = tmp_path / "valid.set"
    eeg = {"nbchan": 2, "pnts": 10, "srate": 250.0, "data": "valid.fdt"}
    _savemat_without_matwritewarning(set_path, eeg)
    out = _eeglab_load_first_eeg(set_path)
    assert out is not None
    assert int(out["nbchan"]) == 2 and int(out["pnts"]) == 10


def test_eeglab_load_first_eeg_nonexistent_returns_none(tmp_path):
    """Nonexistent .set returns None."""
    assert _eeglab_load_first_eeg(tmp_path / "missing.set") is None


def test_load_raw_eeglab_alleeg_no_eeg_raises(tmp_path):
    """Nonexistent or invalid .set raises ValueError."""
    with pytest.raises(ValueError, match="No EEG or ALLEEG"):
        _load_raw_eeglab_alleeg(tmp_path / "missing.set")


def test_load_raw_eeglab_alleeg_success(tmp_path):
    """Minimal .set + .fdt loads to MNE Raw."""
    pytest.importorskip("scipy")

    set_path = tmp_path / "raw.set"
    fdt_path = tmp_path / "raw.fdt"
    nbchan, pnts = 2, 4
    fdt_path.write_bytes(b"\x00" * (nbchan * pnts * 4))
    eeg = {
        "nbchan": nbchan,
        "pnts": pnts,
        "srate": 250.0,
        "data": "raw.fdt",
        "chanlocs": [{"labels": "Fp1"}, {"labels": "Fp2"}],
    }
    _savemat_without_matwritewarning(set_path, eeg)
    raw = _load_raw_eeglab_alleeg(set_path)
    assert raw is not None
    assert raw.info["nchan"] == nbchan
    assert raw.times.size == pnts


def test_load_raw_eeglab_alleeg_truncated_fdt_keeps_complete_samples(tmp_path):
    """External .fdt smaller than header: complete samples only, no zero padding."""
    pytest.importorskip("scipy")

    set_path = tmp_path / "trunc.set"
    fdt_path = tmp_path / "trunc.fdt"
    nbchan, pnts = 2, 8
    # 3 complete samples and half of a 4th
    np.arange(7, dtype="<f4").tofile(fdt_path)
    eeg = {
        "nbchan": nbchan,
        "pnts": pnts,
        "srate": 250.0,
        "data": "trunc.fdt",
        "chanlocs": [{"labels": "A"}, {"labels": "B"}],
    }
    _savemat_without_matwritewarning(set_path, eeg)
    raw = _load_raw_eeglab_alleeg(set_path)
    assert raw.info["nchan"] == nbchan
    assert raw.times.size == 3
    np.testing.assert_allclose(raw.get_data() * 1e6, [[0, 2, 4], [1, 3, 5]])


def test_load_raw_eeglab_alleeg_inline_data(tmp_path):
    """Inline data (array in .set) uses else branch."""
    pytest.importorskip("scipy")

    set_path = tmp_path / "inline.set"
    nbchan, pnts = 2, 4
    data = np.zeros((nbchan, pnts), order="F", dtype=np.float64)
    eeg = {
        "nbchan": nbchan,
        "pnts": pnts,
        "srate": 250.0,
        "data": data,
        "chanlocs": [{"labels": "Fp1"}, {"labels": "Fp2"}],
    }
    _savemat_without_matwritewarning(set_path, eeg)
    raw = _load_raw_eeglab_alleeg(set_path)
    assert raw is not None
    assert raw.info["nchan"] == nbchan
    assert raw.times.size == pnts


# ── _load_raw_from_eeglab_epochs ──────────────────────────────────────


def test_load_raw_from_eeglab_epochs_concatenates(tmp_path):
    """Epoched EEGLAB data is concatenated into continuous RawArray with annotations."""
    import mne

    n_epochs, n_channels, n_times = 3, 2, 200
    sfreq = 256.0
    data = np.random.randn(n_epochs, n_channels, n_times) * 1e-6
    tmin = -0.2  # 0.2s before trigger

    info = mne.create_info(ch_names=["Fz", "Cz"], sfreq=sfreq, ch_types="eeg")
    # Simulate original-timeline latencies (far beyond concatenated data)
    events = np.array([[10000, 0, 1], [50000, 0, 2], [90000, 0, 1]])
    event_id = {"target": 1, "standard": 2}
    epochs = mne.EpochsArray(data, info, events=events, tmin=tmin, event_id=event_id)

    set_path = tmp_path / "epoched.set"

    with patch("mne.io.read_epochs_eeglab", return_value=epochs) as mock_read:
        raw = _load_raw_from_eeglab_epochs(set_path)

    mock_read.assert_called_once_with(str(set_path), verbose="ERROR")
    assert raw.info["nchan"] == n_channels
    assert raw.n_times == n_epochs * n_times
    assert "3 epochs" in raw.info["description"]
    assert raw.ch_names == ["Fz", "Cz"]

    # Verify annotations are recalculated for concatenated timeline
    assert len(raw.annotations) == n_epochs
    descriptions = list(raw.annotations.description)
    assert descriptions == ["target", "standard", "target"]

    # Annotations must be within the raw's time range
    offset = int(round(-tmin * sfreq))  # 51 samples
    expected_onsets = (np.arange(n_epochs) * n_times + offset) / sfreq
    np.testing.assert_allclose(raw.annotations.onset, expected_onsets, atol=1e-6)

    # Verify all annotations fall within the raw duration
    raw_duration = raw.n_times / sfreq
    assert all(onset < raw_duration for onset in raw.annotations.onset)


# ── _deduplicate_channel_names ────────────────────────────────────────


def test_deduplicate_channel_names_no_duplicates():
    """No-op when all names are already unique."""
    names = ["Fp1", "Fp2", "Cz"]
    assert _deduplicate_channel_names(names) == names


def test_deduplicate_channel_names_all_same():
    """All-same names get sequential suffixes."""
    names = ["EEG", "EEG", "EEG"]
    result = _deduplicate_channel_names(names)
    assert result == ["EEG-0", "EEG-1", "EEG-2"]
    assert len(set(result)) == 3


def test_deduplicate_channel_names_mixed():
    """Only duplicated names get suffixes; unique names are unchanged."""
    names = ["Fp1", "Cz", "Fp1", "Oz", "Cz"]
    result = _deduplicate_channel_names(names)
    assert result == ["Fp1-0", "Cz-0", "Fp1-1", "Oz", "Cz-1"]
    assert len(set(result)) == 5


def test_deduplicate_channel_names_empty():
    """Empty list returns empty list."""
    assert _deduplicate_channel_names([]) == []


def test_read_raw_mef_has_pymef(tmp_path):
    """MEF3 (.mefd) needs pymef, which mne does not install (on004774)."""
    # Without pymef this is RuntimeError("... pymef module is needed").
    with pytest.raises(FileNotFoundError):
        mne.io.read_raw_mef(tmp_path / "missing.mefd")


def test_fix_negative_annotation_durations_clamps():
    """on006923: the patched crop must bind ``self`` (was "'int' has no len()")."""
    annot = mne.Annotations([0.0, 1.0], [1.0, -0.5], ["a", "b"])
    orig = mne.Annotations.crop
    with pytest.raises(AssertionError):
        annot.copy().crop(0, 10)
    with _fix_negative_annotation_durations():
        cropped = annot.crop(0, 10)
    assert list(cropped.description) == ["a", "b"]
    np.testing.assert_array_equal(cropped.duration, [1.0, 0.0])
    assert mne.Annotations.crop is orig  # restored
