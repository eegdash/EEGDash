"""Unit tests for the SourceAdapter Module — Phase 8 S1.thick.

Two production Adapters (OpenNeuroAdapter, NEMARAdapter) + one
DefaultAdapter for everything else. The tests pin every Interface
method against the divergent behaviour the if-ladders in
``3_digest.py`` used to express.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

from source_adapter import (
    DefaultAdapter,
    NEMARAdapter,
    OpenNeuroAdapter,
    SourceAdapter,
    get_source_adapter,
)

# ─── Factory dispatch ──────────────────────────────────────────────────────


def test_factory_dispatches_openneuro_to_dedicated_class():
    adapter = get_source_adapter("openneuro", "ds002893")
    assert isinstance(adapter, OpenNeuroAdapter)


def test_factory_dispatches_nemar_to_dedicated_class():
    adapter = get_source_adapter("nemar", "nm000176")
    assert isinstance(adapter, NEMARAdapter)


@pytest.mark.parametrize(
    "source",
    ["gin", "figshare", "zenodo", "osf", "scidb", "datarn"],
)
def test_factory_dispatches_secondary_sources_to_default_adapter(source: str):
    """Per ADR 0001 the 5 secondary Sources + GIN share DefaultAdapter."""
    adapter = get_source_adapter(source, f"{source}-ds-001")
    assert isinstance(adapter, DefaultAdapter)
    assert adapter.source_name == source


def test_factory_unknown_source_falls_back_to_default_adapter():
    """An unknown source still gets a DefaultAdapter (with DEFAULT_STORAGE_CONFIG)."""
    adapter = get_source_adapter("brand-new-source", "abc-123")
    assert isinstance(adapter, DefaultAdapter)
    assert adapter.source_name == "brand-new-source"
    # The base falls back to "https://unknown" (DEFAULT_STORAGE_CONFIG).
    assert "unknown" in adapter.storage_base


# ─── OpenNeuro Adapter — storage + URL ────────────────────────────────────


def test_openneuro_storage_base_is_s3_canonical():
    adapter = OpenNeuroAdapter("ds002893")
    assert adapter.storage_base == "s3://openneuro.org/ds002893"


def test_openneuro_storage_backend_is_s3():
    adapter = OpenNeuroAdapter("ds002893")
    assert adapter.storage_backend == "s3"


def test_openneuro_dataset_url():
    adapter = OpenNeuroAdapter("ds002893")
    assert adapter.dataset_url() == "https://openneuro.org/datasets/ds002893"


def test_openneuro_address_file_is_direct_join():
    """OpenNeuro: no annex resolution; address is base + relpath."""
    adapter = OpenNeuroAdapter("ds002893")
    addr = adapter.address_file("sub-001/eeg/sub-001_task-rest_eeg.edf")
    assert addr == "s3://openneuro.org/ds002893/sub-001/eeg/sub-001_task-rest_eeg.edf"


def test_openneuro_resolve_storage_extensions_returns_empty():
    """OpenNeuro doesn't use annex keys or inline sidecars."""
    adapter = OpenNeuroAdapter("ds002893")
    annex, inline = adapter.resolve_storage_extensions(
        Path("/tmp/recording.edf"), [Path("/tmp/channels.tsv")]
    )
    assert annex == {}
    assert inline == {}


# ─── NEMAR Adapter — storage + URL + behaviour overrides ──────────────────


def test_nemar_storage_backend_is_marker_string():
    """The 'nemar' backend marker means 'do not attempt public S3 fetch'."""
    adapter = NEMARAdapter("nm000176")
    assert adapter.storage_backend == "nemar"


def test_nemar_storage_base_is_s3_canonical():
    adapter = NEMARAdapter("nm000176")
    assert adapter.storage_base == "s3://nemar/nm000176"


def test_nemar_dataset_url_points_at_dataexplorer():
    adapter = NEMARAdapter("nm000176")
    assert adapter.dataset_url() == "https://nemar.org/dataexplorer/detail/nm000176"


def test_nemar_apex_cache_built_lazily_from_bids_root(tmp_path: Path):
    """First call to resolve_storage_extensions reads NEMAR_ROOT_METADATA_FILES."""
    # Create a small NEMAR-shaped tree with one apex sidecar
    bids_root = tmp_path / "nm000176"
    bids_root.mkdir()
    participants = bids_root / "participants.tsv"
    participants.write_text("participant_id\tage\nsub-01\t30\n")
    dataset_desc = bids_root / "dataset_description.json"
    dataset_desc.write_text('{"Name": "Test Dataset", "BIDSVersion": "1.7.0"}\n')

    adapter = NEMARAdapter("nm000176", bids_root=bids_root)
    # Trigger cache build with a no-op call
    _annex, inline = adapter.resolve_storage_extensions(
        bids_root / "sub-01/eeg/sub-01_eeg.edf", []
    )
    # Apex files should be inlined
    assert "participants.tsv" in inline
    assert "sub-01\t30" in inline["participants.tsv"]
    assert "dataset_description.json" in inline


def test_nemar_apex_cache_returns_empty_when_bids_root_is_none():
    """No bids_root means no filesystem reads; cache stays empty."""
    adapter = NEMARAdapter("nm000176", bids_root=None)
    annex, inline = adapter.resolve_storage_extensions(Path("/tmp/x.edf"), [])
    assert annex == {}
    assert inline == {}


def test_nemar_resolve_dep_inlines_small_sidecars(tmp_path: Path):
    """Sidecar files that aren't annex-tracked get inlined."""
    bids_root = tmp_path / "nm000176"
    eeg_dir = bids_root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    channels_tsv = eeg_dir / "sub-01_task-rest_channels.tsv"
    channels_tsv.write_text("name\ttype\nFp1\tEEG\nFp2\tEEG\n")

    adapter = NEMARAdapter("nm000176", bids_root=bids_root)
    record_path = eeg_dir / "sub-01_task-rest_eeg.edf"
    record_path.touch()

    _, inline = adapter.resolve_storage_extensions(record_path, [channels_tsv])
    assert "sub-01/eeg/sub-01_task-rest_channels.tsv" in inline
    assert "Fp1\tEEG" in inline["sub-01/eeg/sub-01_task-rest_channels.tsv"]


# ─── DefaultAdapter — table-driven fallback ───────────────────────────────


def test_default_adapter_uses_table_for_backend():
    adapter = DefaultAdapter("zenodo-123", "zenodo")
    assert adapter.storage_backend == "https"


def test_default_adapter_uses_table_for_storage_base():
    adapter = DefaultAdapter("zenodo-123", "zenodo")
    assert adapter.storage_base == "https://zenodo.org/records/zenodo-123"


def test_default_adapter_has_no_dataset_url():
    """Secondary sources don't have a canonical landing page from dataset_id."""
    adapter = DefaultAdapter("zenodo-123", "zenodo")
    assert adapter.dataset_url() is None


def test_default_adapter_returns_empty_storage_extensions():
    """No annex / no inline for secondaries."""
    adapter = DefaultAdapter("zenodo-123", "zenodo")
    annex, inline = adapter.resolve_storage_extensions(Path("/tmp/x.edf"), [])
    assert annex == {}
    assert inline == {}


# ─── Abstract class invariants ────────────────────────────────────────────


def test_sourceadapter_is_abstract():
    """The base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        SourceAdapter("ds002893")  # type: ignore[abstract]
