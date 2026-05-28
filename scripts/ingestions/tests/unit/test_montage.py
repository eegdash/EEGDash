"""Tests for ``_montage.py`` — montage detection and caching."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

import _montage
from _montage import (
    _companion_coords_for,
    _hash_sensors,
    _parse_channels_tsv_for_eeg,
    _parse_coordsystem_json,
    _parse_sensor_tsv,
    _resolve_fif_total_size,
    _round_mm,
    _score_template_match,
    _walk_up_find,
    extract_layout,
)

# ─── 1. Pure helpers ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("metres", "expected_mm"),
    [
        pytest.param(0.001, 1, id="one_mm"),
        pytest.param(0.0015, 2, id="bankers_rounding_to_even"),
        pytest.param(0.05, 50, id="fifty_mm"),
        pytest.param(-0.0123, -12, id="negative_value"),
        pytest.param(0.0, 0, id="zero"),
    ],
)
def test_round_mm_scales_metres_to_integer_mm(metres: float, expected_mm: int):
    """Converts metres to integer mm; banker's rounding; sign-preserving."""
    assert _round_mm(metres) == expected_mm


def test_hash_sensors_deterministic_for_same_input():
    """Same modality + same sensors → same 16-char hash."""
    sensors = [
        {"name": "Cz", "x": 0.0, "y": 0.0, "z": 0.1},
        {"name": "Fz", "x": 0.0, "y": 0.05, "z": 0.08},
    ]
    h1 = _hash_sensors("eeg", sensors)
    h2 = _hash_sensors("eeg", sensors)
    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) == 16


def test_hash_sensors_order_invariant():
    """Different sensor order → same hash (canonical sort)."""
    s1 = [
        {"name": "Cz", "x": 0.0, "y": 0.0, "z": 0.1},
        {"name": "Fz", "x": 0.0, "y": 0.05, "z": 0.08},
    ]
    s2 = list(reversed(s1))
    assert _hash_sensors("eeg", s1) == _hash_sensors("eeg", s2)


def test_hash_sensors_modality_in_input():
    """Different modality → different hash; prevents EEG/MEG hash collision."""
    sensors = [{"name": "X", "x": 0.0, "y": 0.0, "z": 0.0}]
    assert _hash_sensors("eeg", sensors) != _hash_sensors("meg", sensors)


def test_hash_sensors_sensitive_to_position_change():
    """A 1 mm position shift produces a different hash."""
    s1 = [{"name": "Cz", "x": 0.0, "y": 0.0, "z": 0.100}]
    s2 = [{"name": "Cz", "x": 0.0, "y": 0.0, "z": 0.101}]
    assert _hash_sensors("eeg", s1) != _hash_sensors("eeg", s2)


def test_hash_sensors_insensitive_to_sub_mm_jitter():
    """Sub-mm float jitter (< 0.5 mm) → same hash; real BIDS datasets have this."""
    s1 = [{"name": "Cz", "x": 0.0, "y": 0.0, "z": 0.10000001}]
    s2 = [{"name": "Cz", "x": 0.0, "y": 0.0, "z": 0.10000005}]
    assert _hash_sensors("eeg", s1) == _hash_sensors("eeg", s2)


def _write_coordsystem_file(tmp_path: Path, filename: str, content: str | None) -> Path:
    p = tmp_path / filename
    if content is not None:
        p.write_text(content)
    return p


@pytest.mark.parametrize(
    ("filename", "content", "expected_space", "expected_units"),
    [
        pytest.param(
            "missing.json",
            None,
            None,
            None,
            id="missing_file",
        ),
        pytest.param(
            "coords.json",
            json.dumps({"EEGCoordinateSystem": "CTF", "EEGCoordinateUnits": "mm"}),
            "CTF",
            "mm",
            id="eeg_block",
        ),
        pytest.param(
            "coords.json",
            json.dumps({"MEGCoordinateSystem": "Neuromag", "MEGCoordinateUnits": "M"}),
            "Neuromag",
            "m",  # lowercased
            id="meg_block_when_no_eeg",
        ),
        pytest.param(
            "coords.json",
            "{ this is not json",
            None,
            None,
            id="malformed_json",
        ),
        pytest.param(
            "coords.json",
            "{}",
            None,
            None,
            id="empty_doc",
        ),
    ],
)
def test_parse_coordsystem_json(
    tmp_path: Path,
    filename: str,
    content: str | None,
    expected_space: str | None,
    expected_units: str | None,
):
    """Handles missing / EEG / MEG / malformed / empty inputs without raising."""
    path = _write_coordsystem_file(tmp_path, filename, content)
    space, units = _parse_coordsystem_json(path)
    assert space == expected_space
    assert units == expected_units


@pytest.mark.parametrize(
    ("description", "setup", "expected_rel"),
    [
        pytest.param(
            "file_in_data_dir",
            {
                "dirs": ["sub-01/eeg"],
                "data_rel": "sub-01/eeg/sub-01_eeg.edf",
                "sidecar_rel": "sub-01/eeg/sub-01_coordsystem.json",
            },
            "sub-01/eeg/sub-01_coordsystem.json",
            id="finds_file_in_data_directory",
        ),
        pytest.param(
            "session_level",
            {
                "dirs": ["sub-01/ses-01/eeg"],
                "data_rel": "sub-01/ses-01/eeg/sub-01_ses-01_eeg.edf",
                "sidecar_rel": "sub-01/ses-01/sub-01_ses-01_coordsystem.json",
            },
            "sub-01/ses-01/sub-01_ses-01_coordsystem.json",
            id="walks_up_to_session",
        ),
        pytest.param(
            "not_found",
            {
                "dirs": ["sub-01/eeg"],
                "data_rel": "sub-01/eeg/sub-01_eeg.edf",
                "sidecar_rel": None,
            },
            None,
            id="stops_at_root",
        ),
    ],
)
def test_walk_up_find(
    tmp_path: Path,
    description: str,
    setup: dict,
    expected_rel: str | None,
):
    """Follows BIDS inheritance toward root; returns None when absent."""
    for d in setup["dirs"]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    data = tmp_path / setup["data_rel"]
    data.touch()
    if setup["sidecar_rel"] is not None:
        (tmp_path / setup["sidecar_rel"]).touch()

    found = _walk_up_find(data, tmp_path, "*_coordsystem.json")

    if expected_rel is None:
        assert found is None
    else:
        assert found == tmp_path / expected_rel


def test_walk_up_find_rejects_data_outside_root(tmp_path: Path):
    """Data file outside the BIDS root → None (security boundary)."""
    outside = tmp_path / "outside.edf"
    outside.touch()
    bids_root = tmp_path / "bids"
    bids_root.mkdir()
    assert _walk_up_find(outside, bids_root, "*_coordsystem.json") is None


def test_companion_coords_for_electrodes_tsv():
    """``sub-01_electrodes.tsv`` → ``sub-01_coordsystem.json`` in same dir."""
    tsv = Path("/data/sub-01_eeg_electrodes.tsv")
    out = _companion_coords_for(tsv)
    assert out.name == "sub-01_eeg_coordsystem.json"
    assert out.parent == tsv.parent


def test_companion_coords_for_optodes_tsv():
    """Custom ``bids_suffix='_optodes.tsv'`` works the same way."""
    tsv = Path("/data/sub-01_fnirs_optodes.tsv")
    out = _companion_coords_for(tsv, bids_suffix="_optodes.tsv")
    assert out.name == "sub-01_fnirs_coordsystem.json"


def test_companion_coords_for_handles_no_matching_suffix():
    """Non-matching suffix → regex substitution still stays in the same parent."""
    tsv = Path("/data/random_file.tsv")
    out = _companion_coords_for(tsv)
    assert out.parent == tsv.parent


@pytest.mark.parametrize(
    ("tsv_content", "check"),
    [
        pytest.param(
            "name\tx\ty\tz\nCz\t0.0\t0.0\t0.1\nFz\t0.0\t0.05\t0.08\nPz\t0.0\t-0.05\t0.08\n",
            lambda rows: (
                len(rows) == 3
                and rows[0]["name"] == "Cz"
                and rows[0]["x"] == 0.0
                and rows[0]["z"] == 0.1
                and isinstance(rows[0]["z"], float)
            ),
            id="basic_x_y_z",
        ),
        pytest.param(
            "name\tx\ty\tz\nCz\t0.0\t0.0\t0.1\nBad\tn/a\t0.0\t0.0\nFz\t0.0\t0.05\t0.08\n",
            lambda rows: len(rows) == 2 and {r["name"] for r in rows} == {"Cz", "Fz"},
            id="drops_n_a_rows",
        ),
        pytest.param(
            "name\tx\ty\tz\ttype\tmaterial\nCz\t0.0\t0.0\t0.1\tEEG\tAg/AgCl\n",
            lambda rows: rows[0]["type"] == "EEG" and rows[0]["material"] == "Ag/AgCl",
            id="preserves_extras",
        ),
        pytest.param(
            "name\tx\ty\tz\n\t0\t0\t0\nCz\t0\t0\t0.1\n",
            lambda rows: "" not in [r["name"] for r in rows],
            id="drops_empty_name_rows",
        ),
    ],
)
def test_parse_sensor_tsv(tmp_path: Path, tsv_content: str, check):
    """Parses TSV rows; n/a and empty-name rows are dropped; extra columns preserved."""
    tsv = tmp_path / "electrodes.tsv"
    tsv.write_text(tsv_content)
    rows = _parse_sensor_tsv(tsv)
    assert check(rows)


def test_parse_sensor_tsv_raises_on_missing_required_column(tmp_path: Path):
    """TSV without the required ``z`` column → ValueError."""
    tsv = tmp_path / "electrodes.tsv"
    tsv.write_text("name\tx\ty\nCz\t0\t0\n")
    with pytest.raises(ValueError, match="missing required columns"):
        _parse_sensor_tsv(tsv)


@pytest.mark.parametrize(
    ("tsv_content", "expected"),
    [
        pytest.param(
            "name\ttype\nCz\tEEG\nFz\tEEG\nHEOG\tEOG\nTrigger\tTRIG\nPz\tEEG\n",
            ["Cz", "Fz", "Pz"],
            id="keeps_eeg_rows",
        ),
        pytest.param(
            "name\nCz\nFz\nHEOG\n",
            ["Cz", "Fz", "HEOG"],
            id="no_type_column_keeps_all",
        ),
    ],
)
def test_parse_channels_tsv_for_eeg_returns_list(
    tmp_path: Path, tsv_content: str, expected: list[str]
):
    """EEG rows are kept; non-EEG (EOG/TRIG) are dropped; missing type keeps all."""
    tsv = tmp_path / "channels.tsv"
    tsv.write_text(tsv_content)
    out = _parse_channels_tsv_for_eeg(tsv)
    assert sorted(out) == sorted(expected)


def test_parse_channels_tsv_pandas_nan_handled_for_missing_type(tmp_path: Path):
    """Pandas reads empty TSV cells as NaN, treated as non-EEG → row dropped.

    Pinned to surface any refactor that changes this behaviour — if truly-empty
    types should be accepted, that requires a follow-up fix.
    """
    tsv = tmp_path / "channels.tsv"
    tsv.write_text("name\ttype\nCz\tEEG\nFz\t\n")
    out = _parse_channels_tsv_for_eeg(tsv)
    assert "Cz" in out


@pytest.mark.parametrize(
    "tsv_content",
    [
        pytest.param(None, id="missing_file"),
        pytest.param("type\tdescription\nEEG\tfoo\n", id="no_name_column"),
    ],
)
def test_parse_channels_tsv_returns_empty(tmp_path: Path, tsv_content: str | None):
    """Missing file or absent 'name' column → []."""
    if tsv_content is None:
        path = tmp_path / "missing.tsv"
    else:
        path = tmp_path / "channels.tsv"
        path.write_text(tsv_content)
    assert _parse_channels_tsv_for_eeg(path) == []


def test_score_template_match_picks_best_overlap():
    """Among two templates, picks the one with most channel overlap."""
    templates = {
        "tiny": {"CZ": (0, 0, 0)},
        "medium": {"CZ": (0, 0, 0), "FZ": (0, 0, 0), "PZ": (0, 0, 0), "OZ": (0, 0, 0)},
    }
    _montage._MNE_TEMPLATE_KEYSETS = None
    out = _score_template_match(["Cz", "Fz", "Pz", "Oz"], templates, min_hits=2)
    assert out is not None
    name, positions = out
    assert name == "medium"
    assert len(positions) == 4


@pytest.mark.parametrize(
    ("channels", "templates", "kwargs"),
    [
        pytest.param(
            [],
            {"x": {"CZ": (0, 0, 0)}},
            {},
            id="empty_channels",
        ),
        pytest.param(
            ["Cz", "X1", "X2", "X3"],
            {"big": {"CZ": (0, 0, 0)}},
            {"min_hits": 4},
            id="below_min_hits",
        ),
        pytest.param(
            ["Cz", "Fz", "Pz", "Oz"] + [f"X{i}" for i in range(96)],
            {
                "big": {f"CH{i}": (0, 0, 0) for i in range(100)}
                | {"CZ": (0, 0, 0), "FZ": (0, 0, 0), "PZ": (0, 0, 0), "OZ": (0, 0, 0)}
            },
            # 4 hits but dataset has 100 channels → ratio 4/100 = 0.04 < min_ratio=0.8
            {"min_ratio": 0.8},
            id="below_min_ratio",
        ),
    ],
)
def test_score_template_match_returns_none(channels, templates, kwargs):
    """Returns None when channels are empty, below min_hits, or below min_ratio."""
    _montage._MNE_TEMPLATE_KEYSETS = None
    out = _score_template_match(channels, templates, **kwargs)
    assert out is None


def test_score_template_match_tiebreaker_prefers_smaller_template():
    """Equal hit count → smaller template wins (more specific fit).

    A 64-channel dataset should map to biosemi64 (64 positions) not
    standard_1005 (343 positions).
    """
    _montage._MNE_TEMPLATE_KEYSETS = None
    channels = ["CZ", "FZ", "PZ", "OZ"]
    templates = {
        "small": {"CZ": (0, 0, 0), "FZ": (0, 0, 0), "PZ": (0, 0, 0), "OZ": (0, 0, 0)},
        "big": {
            "CZ": (0, 0, 0),
            "FZ": (0, 0, 0),
            "PZ": (0, 0, 0),
            "OZ": (0, 0, 0),
            "EXTRA1": (0, 0, 0),
            "EXTRA2": (0, 0, 0),
        },
    }
    out = _score_template_match(channels, templates, min_hits=4)
    assert out is not None
    name, _ = out
    assert name == "small"


def test_extract_layout_returns_none_for_unsupported_datatype(tmp_path: Path):
    """Non-EEG/iEEG/MEG/EMG/nirs datatypes (e.g. 'anat') → None."""
    eeg_dir = tmp_path / "sub-01" / "anat"
    eeg_dir.mkdir(parents=True)
    data = eeg_dir / "sub-01_T1w.nii.gz"
    data.touch()
    out = extract_layout(data, tmp_path, datatype="anat")
    assert out is None


def test_extract_layout_returns_none_for_empty_datatype(tmp_path: Path):
    """Empty datatype string → None."""
    eeg_dir = tmp_path / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    data = eeg_dir / "sub-01_eeg.edf"
    data.touch()
    assert extract_layout(data, tmp_path, datatype="") is None


def test_extract_layout_fnirs_alias_to_nirs(tmp_path: Path):
    """'fnirs' datatype is aliased to 'nirs'; must not raise on the alias step."""
    nirs_dir = tmp_path / "sub-01" / "nirs"
    nirs_dir.mkdir(parents=True)
    data = nirs_dir / "sub-01_nirs.snirf"
    data.touch()
    # No optodes.tsv → None, but the alias is handled before that check.
    out = extract_layout(data, tmp_path, datatype="fnirs")
    assert out is None


# ─── 2. Cache layer ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def digest() -> ModuleType:
    """The montage-cache logic (``_attach_montage_to_record`` + its ``extract_layout``
    dependency) lives in ``_bids_digest`` after the BIDS-path extraction."""
    import _bids_digest  # noqa: PLC0415

    return _bids_digest


def _meg_record(nchans: int, name: str = "sub-01_run-01_meg.fif") -> dict:
    return {
        "datatype": "meg",
        "nchans": nchans,
        "bids_relpath": name,
    }


def test_first_meg_record_calls_extract_layout(
    digest: ModuleType, tmp_path: Path
) -> None:
    """First MEG record with a given nchans triggers extract_layout."""
    record = _meg_record(306)
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        return_value=("hash-306-A", {"system": "neuromag306"}),
    ) as mocked:
        errors = digest._attach_montage_to_record(
            record,
            tmp_path / "sub-01_meg.fif",
            tmp_path,
            montages,
            "ds-meg-001",
            "2026-05-22T00:00:00+00:00",
            montage_cache=cache,
        )

    assert errors == []
    assert mocked.call_count == 1
    assert record["montage_hash"] == "hash-306-A"
    assert "hash-306-A" in montages
    assert cache[("ds-meg-001", 306)] == (
        "hash-306-A",
        {
            "system": "neuromag306",
            "first_seen": "2026-05-22T00:00:00+00:00",
            "representative_dataset": "ds-meg-001",
        },
    )


def test_second_meg_record_same_nchans_reuses_cache(
    digest: ModuleType, tmp_path: Path
) -> None:
    """Second record with same (dataset, nchans) MUST NOT call extract_layout."""
    record_a = _meg_record(306, "sub-01_run-01_meg.fif")
    record_b = _meg_record(306, "sub-01_run-02_meg.fif")
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        return_value=("hash-306-A", {"system": "neuromag306"}),
    ) as mocked:
        digest._attach_montage_to_record(
            record_a,
            tmp_path / "a.fif",
            tmp_path,
            montages,
            "ds-meg-001",
            "2026-05-22T00:00:00+00:00",
            montage_cache=cache,
        )
        digest._attach_montage_to_record(
            record_b,
            tmp_path / "b.fif",
            tmp_path,
            montages,
            "ds-meg-001",
            "2026-05-22T00:00:00+00:00",
            montage_cache=cache,
        )

    assert mocked.call_count == 1
    assert record_a["montage_hash"] == "hash-306-A"
    assert record_b["montage_hash"] == "hash-306-A"


def test_different_nchans_skips_cache(digest: ModuleType, tmp_path: Path) -> None:
    """Two records with different nchans → two extract_layout calls."""
    record_a = _meg_record(306)
    record_b = _meg_record(204)
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        side_effect=[
            ("hash-306-A", {"system": "neuromag306"}),
            ("hash-204-B", {"system": "ctf204"}),
        ],
    ) as mocked:
        digest._attach_montage_to_record(
            record_a,
            tmp_path / "a.fif",
            tmp_path,
            montages,
            "ds-meg-001",
            "now",
            montage_cache=cache,
        )
        digest._attach_montage_to_record(
            record_b,
            tmp_path / "b.fif",
            tmp_path,
            montages,
            "ds-meg-001",
            "now",
            montage_cache=cache,
        )

    assert mocked.call_count == 2
    assert record_a["montage_hash"] == "hash-306-A"
    assert record_b["montage_hash"] == "hash-204-B"


def test_cache_does_not_leak_across_datasets(
    digest: ModuleType, tmp_path: Path
) -> None:
    """Same nchans, different dataset_id → cache miss (different cache keys)."""
    record_a = _meg_record(306)
    record_b = _meg_record(306)
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        side_effect=[
            ("hash-306-A", {"system": "neuromag306"}),
            (
                "hash-306-A",
                {"system": "neuromag306"},
            ),
        ],
    ) as mocked:
        digest._attach_montage_to_record(
            record_a,
            tmp_path / "a.fif",
            tmp_path,
            montages,
            "ds-meg-001",
            "now",
            montage_cache=cache,
        )
        digest._attach_montage_to_record(
            record_b,
            tmp_path / "b.fif",
            tmp_path,
            montages,
            "ds-meg-002",
            "now",
            montage_cache=cache,
        )

    assert mocked.call_count == 2


def test_non_meg_record_bypasses_cache(digest: ModuleType, tmp_path: Path) -> None:
    """EEG records call extract_layout per-file — the cache only applies to MEG."""
    record_a = {"datatype": "eeg", "nchans": 64, "bids_relpath": "a.vhdr"}
    record_b = {"datatype": "eeg", "nchans": 64, "bids_relpath": "b.vhdr"}
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        side_effect=[
            ("hash-eeg-a", {"layout": "ten-twenty"}),
            ("hash-eeg-b", {"layout": "ten-twenty"}),
        ],
    ) as mocked:
        digest._attach_montage_to_record(
            record_a,
            tmp_path / "a.vhdr",
            tmp_path,
            montages,
            "ds-eeg-001",
            "now",
            montage_cache=cache,
        )
        digest._attach_montage_to_record(
            record_b,
            tmp_path / "b.vhdr",
            tmp_path,
            montages,
            "ds-eeg-001",
            "now",
            montage_cache=cache,
        )

    assert mocked.call_count == 2
    assert cache == {}


def test_missing_nchans_skips_cache(digest: ModuleType, tmp_path: Path) -> None:
    """MEG record with no nchans → not cached (no safe key without channel count)."""
    record = {"datatype": "meg", "bids_relpath": "broken.fif"}
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        return_value=("hash-x", {"layout": "x"}),
    ) as mocked:
        digest._attach_montage_to_record(
            record,
            tmp_path / "broken.fif",
            tmp_path,
            montages,
            "ds-x",
            "now",
            montage_cache=cache,
        )

    assert mocked.call_count == 1
    assert cache == {}
    assert record["montage_hash"] == "hash-x"


def test_extract_layout_returning_none_is_not_cached(
    digest: ModuleType, tmp_path: Path
) -> None:
    """extract_layout returning None is not cached; the next record gets another attempt."""
    record_a = _meg_record(306)
    record_b = _meg_record(306)
    montages: dict = {}
    cache: dict = {}

    with patch.object(
        digest,
        "extract_layout",
        side_effect=[None, ("hash-306-A", {"system": "neuromag306"})],
    ) as mocked:
        digest._attach_montage_to_record(
            record_a,
            tmp_path / "a.fif",
            tmp_path,
            montages,
            "ds-meg-001",
            "now",
            montage_cache=cache,
        )
        digest._attach_montage_to_record(
            record_b,
            tmp_path / "b.fif",
            tmp_path,
            montages,
            "ds-meg-001",
            "now",
            montage_cache=cache,
        )

    assert mocked.call_count == 2
    assert record_a["montage_hash"] is None
    assert record_b["montage_hash"] == "hash-306-A"


# ─── 3. git-annex key shortcut ──────────────────────────────────────────────


def test_returns_annex_size_when_symlink_present(tmp_path: Path) -> None:
    """Broken git-annex symlink → size parsed from symlink target."""
    fif = tmp_path / "sub-01_run-01_meg.fif"
    target = "../.git/annex/objects/aa/bb/MD5E-s4194304--abc123def.fif/MD5E-s4194304--abc123def.fif"
    fif.symlink_to(target)
    assert not fif.exists()  # broken — annex content not fetched

    with patch("_montage.head_content_length") as mock_head:
        size = _resolve_fif_total_size(fif, "https://s3.example.com/sub-01.fif")

    assert size == 4_194_304
    mock_head.assert_not_called()


def test_falls_back_to_head_when_no_annex_symlink(tmp_path: Path) -> None:
    """Plain file (no annex) → HEAD round-trip."""
    fif = tmp_path / "sub-01_run-01_meg.fif"
    fif.write_bytes(b"\x00" * 1024)

    with patch("_montage.head_content_length", return_value=8_192) as mock_head:
        size = _resolve_fif_total_size(fif, "https://s3.example.com/sub-01.fif")

    assert size == 8_192
    mock_head.assert_called_once()


def test_falls_back_to_head_on_malformed_annex_key(tmp_path: Path) -> None:
    """Symlink target doesn't match MD5E-s{size}-- pattern → HEAD fallback."""
    fif = tmp_path / "sub-01_run-01_meg.fif"
    fif.symlink_to("not-an-annex-key.fif")

    with patch("_montage.head_content_length", return_value=2_048) as mock_head:
        size = _resolve_fif_total_size(fif, "https://s3.example.com/sub-01.fif")

    assert size == 2_048
    mock_head.assert_called_once()


def test_returns_none_when_neither_annex_nor_head_succeeds(
    tmp_path: Path,
) -> None:
    """All paths exhausted → None (caller treats as transient failure)."""
    fif = tmp_path / "sub-01_run-01_meg.fif"
    fif.write_bytes(b"")

    with patch("_montage.head_content_length", return_value=None):
        size = _resolve_fif_total_size(fif, "https://s3.example.com/sub-01.fif")

    assert size is None


def test_zero_byte_annex_key_falls_back_to_head(tmp_path: Path) -> None:
    """Annex key reporting size=0 is nonsensical for FIF — fall back to HEAD."""
    fif = tmp_path / "sub-01_run-01_meg.fif"
    fif.symlink_to("../.git/annex/objects/aa/bb/MD5E-s0--abc.fif/MD5E-s0--abc.fif")

    with patch("_montage.head_content_length", return_value=5_000) as mock_head:
        size = _resolve_fif_total_size(fif, "https://s3.example.com/sub-01.fif")

    assert size == 5_000
    mock_head.assert_called_once()
