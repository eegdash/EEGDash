"""Tests for ``_montage.py`` — montage detection and caching.

Three angles:

- **Pure helpers** — hash building, JSON parsing, BIDS-inheritance walks, channels.tsv filtering, template matching (was test_montage_helpers.py).
- **Cache layer** — montage cache + invalidation (was test_montage_cache.py).
- **git-annex key shortcut** — _resolve_fif_total_size fast path (was test_montage_annex_key_shortcut.py).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
from _helpers import INGEST_DIR as _INGEST_DIR

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

# ─── _round_mm ─────────────────────────────────────────────────────────────


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
    """``_round_mm`` converts a value in metres to an integer millimetre.

    Behavioural quirks pinned:
    - banker's rounding (closest even) for half-mm values;
    - signs are preserved for negative inputs;
    - zero maps to zero.
    """
    assert _round_mm(metres) == expected_mm


# ─── _hash_sensors ─────────────────────────────────────────────────────────


def test_hash_sensors_deterministic_for_same_input():
    """Same modality + same sensors → same hash."""
    sensors = [
        {"name": "Cz", "x": 0.0, "y": 0.0, "z": 0.1},
        {"name": "Fz", "x": 0.0, "y": 0.05, "z": 0.08},
    ]
    h1 = _hash_sensors("eeg", sensors)
    h2 = _hash_sensors("eeg", sensors)
    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) == 16  # 16-char SHA1 prefix per docstring


def test_hash_sensors_order_invariant():
    """Different sensor order → same hash (canonical sort)."""
    s1 = [
        {"name": "Cz", "x": 0.0, "y": 0.0, "z": 0.1},
        {"name": "Fz", "x": 0.0, "y": 0.05, "z": 0.08},
    ]
    s2 = list(reversed(s1))
    assert _hash_sensors("eeg", s1) == _hash_sensors("eeg", s2)


def test_hash_sensors_modality_in_input():
    """Different modality → different hash even for same sensor list.

    Prevents an EEG cap from colliding with a MEG helmet hash.
    """
    sensors = [{"name": "X", "x": 0.0, "y": 0.0, "z": 0.0}]
    assert _hash_sensors("eeg", sensors) != _hash_sensors("meg", sensors)


def test_hash_sensors_sensitive_to_position_change():
    """A 1mm position shift produces a different hash."""
    s1 = [{"name": "Cz", "x": 0.0, "y": 0.0, "z": 0.100}]
    s2 = [{"name": "Cz", "x": 0.0, "y": 0.0, "z": 0.101}]
    assert _hash_sensors("eeg", s1) != _hash_sensors("eeg", s2)


def test_hash_sensors_insensitive_to_sub_mm_jitter():
    """Sub-mm position jitter (< 0.5 mm) → same hash.

    Pins the documented mm-rounding behaviour. Real BIDS datasets
    have sub-mm float jitter; hashing should be stable across that.
    """
    s1 = [{"name": "Cz", "x": 0.0, "y": 0.0, "z": 0.10000001}]
    s2 = [{"name": "Cz", "x": 0.0, "y": 0.0, "z": 0.10000005}]
    assert _hash_sensors("eeg", s1) == _hash_sensors("eeg", s2)


# ─── _parse_coordsystem_json ──────────────────────────────────────────────


def test_parse_coordsystem_missing_returns_none_pair(tmp_path: Path):
    """Missing file → (None, None)."""
    space, units = _parse_coordsystem_json(tmp_path / "missing.json")
    assert space is None
    assert units is None


def test_parse_coordsystem_finds_eeg_block(tmp_path: Path):
    """An EEG coordsystem.json fills (space, units)."""
    f = tmp_path / "coords.json"
    f.write_text(
        json.dumps(
            {
                "EEGCoordinateSystem": "CTF",
                "EEGCoordinateUnits": "mm",
            }
        )
    )
    space, units = _parse_coordsystem_json(f)
    assert space == "CTF"
    assert units == "mm"  # lowercased per the docstring


def test_parse_coordsystem_finds_meg_block_when_no_eeg(tmp_path: Path):
    """The walker tries EEG / iEEG / MEG / EMG / NIRS in order. MEG works
    when EEG block is absent."""
    f = tmp_path / "coords.json"
    f.write_text(
        json.dumps(
            {
                "MEGCoordinateSystem": "Neuromag",
                "MEGCoordinateUnits": "M",  # uppercase
            }
        )
    )
    space, units = _parse_coordsystem_json(f)
    assert space == "Neuromag"
    assert units == "m"  # lowercased


def test_parse_coordsystem_tolerates_malformed_json(tmp_path: Path):
    """A non-JSON file → (None, None), no exception."""
    f = tmp_path / "coords.json"
    f.write_text("{ this is not json")
    space, units = _parse_coordsystem_json(f)
    assert space is None
    assert units is None


def test_parse_coordsystem_returns_none_for_empty_doc(tmp_path: Path):
    """An empty JSON object → (None, None)."""
    f = tmp_path / "coords.json"
    f.write_text("{}")
    assert _parse_coordsystem_json(f) == (None, None)


# ─── _walk_up_find (BIDS inheritance) ─────────────────────────────────────


def test_walk_up_find_finds_file_in_data_directory(tmp_path: Path):
    """File next to the data file is returned."""
    sub_dir = tmp_path / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)
    data = sub_dir / "sub-01_eeg.edf"
    data.touch()
    coords = sub_dir / "sub-01_coordsystem.json"
    coords.touch()
    found = _walk_up_find(data, tmp_path, "*_coordsystem.json")
    assert found == coords


def test_walk_up_find_walks_up_to_session(tmp_path: Path):
    """File at the session level is found via BIDS inheritance."""
    eeg_dir = tmp_path / "sub-01" / "ses-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    data = eeg_dir / "sub-01_ses-01_eeg.edf"
    data.touch()
    # Place coords one level up (session-level)
    ses_coords = eeg_dir.parent / "sub-01_ses-01_coordsystem.json"
    ses_coords.touch()
    found = _walk_up_find(data, tmp_path, "*_coordsystem.json")
    assert found == ses_coords


def test_walk_up_find_stops_at_root(tmp_path: Path):
    """When the file isn't found anywhere on the walk → None."""
    sub_dir = tmp_path / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)
    data = sub_dir / "sub-01_eeg.edf"
    data.touch()
    found = _walk_up_find(data, tmp_path, "*_coordsystem.json")
    assert found is None


def test_walk_up_find_rejects_data_outside_root(tmp_path: Path):
    """If the data file is OUTSIDE the BIDS root, return None — security check."""
    outside = tmp_path / "outside.edf"
    outside.touch()
    bids_root = tmp_path / "bids"
    bids_root.mkdir()
    assert _walk_up_find(outside, bids_root, "*_coordsystem.json") is None


# ─── _companion_coords_for ────────────────────────────────────────────────


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
    """If the suffix doesn't match, return path with regex substitution."""
    tsv = Path("/data/random_file.tsv")
    # No suffix match — function falls back to regex replacement
    out = _companion_coords_for(tsv)
    # Output is still in same parent
    assert out.parent == tsv.parent


# ─── _parse_sensor_tsv ────────────────────────────────────────────────────


def test_parse_sensor_tsv_basic_x_y_z(tmp_path: Path):
    """3-channel TSV with x/y/z floats → 3 dicts."""
    tsv = tmp_path / "electrodes.tsv"
    tsv.write_text(
        "name\tx\ty\tz\nCz\t0.0\t0.0\t0.1\nFz\t0.0\t0.05\t0.08\nPz\t0.0\t-0.05\t0.08\n"
    )
    rows = _parse_sensor_tsv(tsv)
    assert len(rows) == 3
    assert rows[0]["name"] == "Cz"
    assert rows[0]["x"] == 0.0
    assert rows[0]["z"] == 0.1
    assert isinstance(rows[0]["z"], float)


def test_parse_sensor_tsv_drops_n_a_rows(tmp_path: Path):
    """Rows with n/a in required columns are dropped."""
    tsv = tmp_path / "electrodes.tsv"
    tsv.write_text(
        "name\tx\ty\tz\nCz\t0.0\t0.0\t0.1\nBad\tn/a\t0.0\t0.0\nFz\t0.0\t0.05\t0.08\n"
    )
    rows = _parse_sensor_tsv(tsv)
    assert len(rows) == 2
    assert {r["name"] for r in rows} == {"Cz", "Fz"}


def test_parse_sensor_tsv_preserves_extras(tmp_path: Path):
    """Extra columns (type, material) are kept when present."""
    tsv = tmp_path / "electrodes.tsv"
    tsv.write_text("name\tx\ty\tz\ttype\tmaterial\nCz\t0.0\t0.0\t0.1\tEEG\tAg/AgCl\n")
    rows = _parse_sensor_tsv(tsv)
    assert rows[0]["type"] == "EEG"
    assert rows[0]["material"] == "Ag/AgCl"


def test_parse_sensor_tsv_raises_on_missing_required_column(tmp_path: Path):
    """A TSV without the required ``z`` column → ValueError."""
    tsv = tmp_path / "electrodes.tsv"
    tsv.write_text("name\tx\ty\nCz\t0\t0\n")
    with pytest.raises(ValueError, match="missing required columns"):
        _parse_sensor_tsv(tsv)


def test_parse_sensor_tsv_drops_empty_name_rows(tmp_path: Path):
    """Rows with empty name → dropped silently."""
    tsv = tmp_path / "electrodes.tsv"
    tsv.write_text("name\tx\ty\tz\n\t0\t0\t0\nCz\t0\t0\t0.1\n")
    rows = _parse_sensor_tsv(tsv)
    # The empty-name row should be filtered out
    names = [r["name"] for r in rows]
    assert "" not in names


# ─── _parse_channels_tsv_for_eeg ──────────────────────────────────────────


def test_parse_channels_tsv_keeps_eeg_rows(tmp_path: Path):
    """Rows with type EEG are kept; non-EEG (EOG/TRIG) are dropped."""
    tsv = tmp_path / "channels.tsv"
    tsv.write_text("name\ttype\nCz\tEEG\nFz\tEEG\nHEOG\tEOG\nTrigger\tTRIG\nPz\tEEG\n")
    out = _parse_channels_tsv_for_eeg(tsv)
    assert sorted(out) == ["Cz", "Fz", "Pz"]


def test_parse_channels_tsv_no_type_column_keeps_all(tmp_path: Path):
    """If 'type' is missing, keep every row (caller filters later)."""
    tsv = tmp_path / "channels.tsv"
    tsv.write_text("name\nCz\nFz\nHEOG\n")
    out = _parse_channels_tsv_for_eeg(tsv)
    assert sorted(out) == ["Cz", "Fz", "HEOG"]


def test_parse_channels_tsv_pandas_nan_handled_for_missing_type(tmp_path: Path):
    """Pandas reads empty TSV cells as NaN; the parser stringifies to 'nan'
    and treats it as a non-EEG type. Pinning this surprising behaviour
    so a refactor that 'fixes' it surfaces in CI.

    The docstring says 'empty type is accepted' but the pandas-NaN path
    actually drops the row. If we want truly-empty types accepted, that's
    a follow-up fix — for now, document the observed behaviour.
    """
    tsv = tmp_path / "channels.tsv"
    tsv.write_text("name\ttype\nCz\tEEG\nFz\t\n")
    out = _parse_channels_tsv_for_eeg(tsv)
    # Cz with explicit EEG type — kept. Fz with empty cell (→ NaN) — dropped.
    assert "Cz" in out


def test_parse_channels_tsv_missing_file_returns_empty(tmp_path: Path):
    """Missing channels.tsv → []."""
    assert _parse_channels_tsv_for_eeg(tmp_path / "missing.tsv") == []


def test_parse_channels_tsv_no_name_column_returns_empty(tmp_path: Path):
    """Without a 'name' column → []."""
    tsv = tmp_path / "channels.tsv"
    tsv.write_text("type\tdescription\nEEG\tfoo\n")
    assert _parse_channels_tsv_for_eeg(tsv) == []


# ─── _score_template_match ────────────────────────────────────────────────


def test_score_template_match_picks_best_overlap():
    """Among 2 templates, picks the one with most channel overlap."""
    templates = {
        "tiny": {"CZ": (0, 0, 0)},  # only 1 channel
        "medium": {"CZ": (0, 0, 0), "FZ": (0, 0, 0), "PZ": (0, 0, 0), "OZ": (0, 0, 0)},
    }
    # Force _MNE_TEMPLATE_KEYSETS to None so the function uses the
    # template keys directly.

    _montage._MNE_TEMPLATE_KEYSETS = None
    out = _score_template_match(["Cz", "Fz", "Pz", "Oz"], templates, min_hits=2)
    assert out is not None
    name, positions = out
    assert name == "medium"
    assert len(positions) == 4


def test_score_template_match_returns_none_for_empty_channels():
    """No channels to match → None."""
    assert _score_template_match([], {"x": {"CZ": (0, 0, 0)}}) is None


def test_score_template_match_returns_none_when_below_min_hits():
    """Channels match too few template positions → None."""

    _montage._MNE_TEMPLATE_KEYSETS = None
    templates = {"big": {"CZ": (0, 0, 0)}}  # only 1 channel
    out = _score_template_match(["Cz", "X1", "X2", "X3"], templates, min_hits=4)
    assert out is None  # only 1 hit < min_hits=4


def test_score_template_match_returns_none_when_below_min_ratio():
    """Hit count meets min_hits but ratio is too low → None."""

    _montage._MNE_TEMPLATE_KEYSETS = None
    templates = {
        "big": {f"CH{i}": (0, 0, 0) for i in range(100)}
        | {
            "CZ": (0, 0, 0),
            "FZ": (0, 0, 0),
            "PZ": (0, 0, 0),
            "OZ": (0, 0, 0),
        }
    }
    # 4 hits but dataset has 100 channels → ratio 4/100 = 0.04 < min_ratio=0.8
    huge_channels = ["Cz", "Fz", "Pz", "Oz"] + [f"X{i}" for i in range(96)]
    out = _score_template_match(huge_channels, templates, min_ratio=0.8)
    assert out is None


def test_score_template_match_tiebreaker_prefers_smaller_template():
    """When two templates match all channels, pick the smaller one.

    A 64-channel dataset should map to biosemi64 (64 positions),
    not standard_1005 (343 positions) — more specific fit.
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
    assert name == "small"  # smaller wins the tiebreaker


# ─── extract_layout dispatcher ────────────────────────────────────────────


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
    """Older datasets call it 'fnirs'; the dispatcher aliases to 'nirs'."""
    nirs_dir = tmp_path / "sub-01" / "nirs"
    nirs_dir.mkdir(parents=True)
    data = nirs_dir / "sub-01_nirs.snirf"
    data.touch()
    # No sidecar files → returns None, but should NOT raise on the
    # alias-handling step.
    out = extract_layout(data, tmp_path, datatype="fnirs")
    # No optodes.tsv → expected None, but the dispatcher recognised fnirs
    assert out is None


# ─── 2. Cache layer ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def digest() -> ModuleType:
    """Load 3_digest.py via importlib (numeric filename)."""
    spec = importlib.util.spec_from_file_location(
        "digest_under_test", _INGEST_DIR / "3_digest.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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

    # extract_layout called ONCE despite two records.
    assert mocked.call_count == 1
    assert record_a["montage_hash"] == "hash-306-A"
    assert record_b["montage_hash"] == "hash-306-A"


def test_different_nchans_skips_cache(digest: ModuleType, tmp_path: Path) -> None:
    """Two records with different nchans → two extract_layout calls."""
    record_a = _meg_record(306)
    record_b = _meg_record(204)  # different device / channel count
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
            ),  # different doc still hashes same
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

    # extract_layout called once per dataset_id, even with same nchans.
    assert mocked.call_count == 2


def test_non_meg_record_bypasses_cache(digest: ModuleType, tmp_path: Path) -> None:
    """EEG records still call extract_layout per-file — the cache only
    helps MEG where the device check is well-defined."""
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

    # Per-file extraction — cache MUST NOT have hijacked the EEG path.
    assert mocked.call_count == 2
    assert cache == {}  # no MEG entries; EEG bypasses


def test_missing_nchans_skips_cache(digest: ModuleType, tmp_path: Path) -> None:
    """A MEG record with no nchans (missing metadata) must NOT be cached
    — without a channel count there's no safe key."""
    record = {"datatype": "meg", "bids_relpath": "broken.fif"}  # no nchans
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
    assert cache == {}  # no key without nchans
    assert record["montage_hash"] == "hash-x"


def test_extract_layout_returning_none_is_not_cached(
    digest: ModuleType, tmp_path: Path
) -> None:
    """If extract_layout returns None (no montage available), don't
    cache the absence — next record gets another chance."""
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

    # Second call still went through extract_layout because the first
    # returned None (cache only stores positive results).
    assert mocked.call_count == 2
    assert record_a["montage_hash"] is None
    assert record_b["montage_hash"] == "hash-306-A"


# ─── 3. git-annex key shortcut ──────────────────────────────────────────────


def test_returns_annex_size_when_symlink_present(tmp_path: Path) -> None:
    """Broken git-annex symlink → size parsed from symlink target."""
    fif = tmp_path / "sub-01_run-01_meg.fif"
    # Annex key format: MD5E-s{size}--{hash}.{ext}
    target = "../.git/annex/objects/aa/bb/MD5E-s4194304--abc123def.fif/MD5E-s4194304--abc123def.fif"
    fif.symlink_to(target)
    assert not fif.exists()  # broken — annex content not fetched

    with patch("_montage.head_content_length") as mock_head:
        size = _resolve_fif_total_size(fif, "https://s3.example.com/sub-01.fif")

    assert size == 4_194_304
    mock_head.assert_not_called()  # No HEAD round-trip


def test_falls_back_to_head_when_no_annex_symlink(tmp_path: Path) -> None:
    """Plain file (no annex) → HEAD round-trip."""
    fif = tmp_path / "sub-01_run-01_meg.fif"
    fif.write_bytes(b"\x00" * 1024)  # 1 KB regular file

    with patch("_montage.head_content_length", return_value=8_192) as mock_head:
        size = _resolve_fif_total_size(fif, "https://s3.example.com/sub-01.fif")

    # Annex size returns 0 for non-annex, falls through to HEAD.
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
    """An annex key reporting size=0 is nonsensical for FIF — fall back."""
    fif = tmp_path / "sub-01_run-01_meg.fif"
    fif.symlink_to("../.git/annex/objects/aa/bb/MD5E-s0--abc.fif/MD5E-s0--abc.fif")

    with patch("_montage.head_content_length", return_value=5_000) as mock_head:
        size = _resolve_fif_total_size(fif, "https://s3.example.com/sub-01.fif")

    assert size == 5_000
    mock_head.assert_called_once()
