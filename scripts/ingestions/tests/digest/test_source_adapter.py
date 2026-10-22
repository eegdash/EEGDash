"""Unit tests for the SourceAdapter Module — Phase 8 S1.thick.

Two production Adapters (OpenNeuroAdapter, NEMARAdapter) + one
DefaultAdapter for everything else. The tests pin every Interface
method against the divergent behaviour the if-ladders in
``3_digest.py`` used to express.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_openneuro_resolve_storage_extensions_no_annex_files():
    """Files that are not git-annex pointers give no key and nothing inlined."""
    adapter = OpenNeuroAdapter("ds002893")
    annex, inline = adapter.resolve_storage_extensions(
        Path("/tmp/recording.edf"), [Path("/tmp/channels.tsv")]
    )
    assert annex == {}
    assert inline == {}


_MD5_KEY = "MD5E-s36869376--0123456789abcdef0123456789abcdef.edf"
_SHA_KEY = "SHA256E-s4096--" + "a" * 64 + ".fdt"


def _openneuro_clone(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Shallow OpenNeuroDatasets clone: annex symlink, pointer file, plain sidecar."""
    root = tmp_path / "ds002893"
    eeg = root / "sub-01" / "eeg"
    eeg.mkdir(parents=True)
    raw = eeg / "sub-01_task-rest_eeg.edf"
    raw.symlink_to(f"../../.git/annex/objects/Pq/Wx/{_MD5_KEY}/{_MD5_KEY}")
    fdt = eeg / "sub-01_task-rest_eeg.fdt"
    fdt.write_text(f"/annex/objects/{_SHA_KEY}\n")
    channels = eeg / "sub-01_task-rest_channels.tsv"
    channels.write_text("name\ttype\nFp1\tEEG\n")
    return root, raw, fdt, channels


def test_openneuro_resolves_annex_keys_from_git_annex_clone(tmp_path: Path):
    """s3://openneuro.org records carry the key of github.com/OpenNeuroDatasets/<id>.

    The bucket listing has no key; the dataset's git-annex repository does
    (symlink target or pointer file), as for the NEMAR on<id> mirrors.
    """
    root, raw, fdt, channels = _openneuro_clone(tmp_path)
    adapter = get_source_adapter("openneuro", "ds002893", root)
    annex, inline = adapter.resolve_storage_extensions(raw, [fdt, channels])
    assert annex == {
        "sub-01/eeg/sub-01_task-rest_eeg.edf": _MD5_KEY,
        "sub-01/eeg/sub-01_task-rest_eeg.fdt": _SHA_KEY,
    }
    assert inline == {}


def test_openneuro_and_nemar_resolve_the_same_keys(tmp_path: Path):
    """Same clone, same keys: the two git-annex sources share one resolution."""
    root, raw, fdt, channels = _openneuro_clone(tmp_path)
    on = OpenNeuroAdapter("ds002893", bids_root=root)
    nm = NEMARAdapter("ds002893", bids_root=root)
    assert on.resolve_storage_extensions(
        raw, [fdt, channels]
    ) == nm.resolve_storage_extensions(raw, [fdt, channels])


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


def test_nemar_does_not_inline_apex_sidecars(tmp_path: Path):
    """Apex sidecars (dataset_description, participants, ...) are NOT inlined.

    They bloated every record and drifted as datasets evolved; the runtime
    now fetches them from the NEMAR GitHub mirror on demand.
    """
    bids_root = tmp_path / "nm000176"
    bids_root.mkdir()
    (bids_root / "participants.tsv").write_text("participant_id\tage\nsub-01\t30\n")
    (bids_root / "dataset_description.json").write_text(
        '{"Name": "Test Dataset", "BIDSVersion": "1.7.0"}\n'
    )

    adapter = NEMARAdapter("nm000176", bids_root=bids_root)
    _annex, inline = adapter.resolve_storage_extensions(
        bids_root / "sub-01/eeg/sub-01_eeg.edf", []
    )
    assert inline == {}


def test_nemar_resolve_returns_empty_inline_when_bids_root_is_none():
    """No bids_root, no inlining; annex resolution yields an empty map."""
    adapter = NEMARAdapter("nm000176", bids_root=None)
    annex, inline = adapter.resolve_storage_extensions(Path("/tmp/x.edf"), [])
    assert annex == {}
    assert inline == {}


def test_nemar_does_not_inline_dep_sidecars(tmp_path: Path):
    """Non-annex dep sidecars are not inlined (fetched from GitHub at runtime)."""
    bids_root = tmp_path / "nm000176"
    eeg_dir = bids_root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    channels_tsv = eeg_dir / "sub-01_task-rest_channels.tsv"
    channels_tsv.write_text("name\ttype\nFp1\tEEG\nFp2\tEEG\n")

    adapter = NEMARAdapter("nm000176", bids_root=bids_root)
    record_path = eeg_dir / "sub-01_task-rest_eeg.edf"
    record_path.touch()

    _annex, inline = adapter.resolve_storage_extensions(record_path, [channels_tsv])
    assert inline == {}


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


def test_directory_recording_records_member_keys(tmp_path: Path):
    """A CTF ``.ds`` / MEF ``.mefd`` recording is a directory: record its files' keys.

    The directory itself has no key; its annexed members do (on002885
    ``sub-02_task-EmptyRoom_meg.ds/…meg4``, extension-less ``BadChannels``).
    """
    root = tmp_path / "on002885"
    ds = root / "sub-02" / "meg" / "sub-02_task-EmptyRoom_meg.ds"
    (ds / "hz.ds").mkdir(parents=True)
    meg4 = "MD5E-s2048--" + "1" * 32 + ".meg4"
    bad = "MD5E-s0--d41d8cd98f00b204e9800998ecf8427e"
    hz = "SHA256E-s64--" + "2" * 64 + ".res4"
    (ds / "sub-02_task-EmptyRoom_meg.meg4").symlink_to(
        f"../../../.git/annex/objects/Aa/Bb/{meg4}/{meg4}"
    )
    (ds / "BadChannels").symlink_to(f"../../../.git/annex/objects/Cc/Dd/{bad}/{bad}")
    (ds / "hz.ds" / "hz.res4").write_text(f"/annex/objects/{hz}\n")
    (ds / "notes.txt").write_text("in git, not annexed\n")

    for adapter in (NEMARAdapter("on002885", root), OpenNeuroAdapter("ds002885", root)):
        annex, _ = adapter.resolve_storage_extensions(ds, [])
        rel = "sub-02/meg/sub-02_task-EmptyRoom_meg.ds"
        assert annex == {
            f"{rel}/BadChannels": bad,
            f"{rel}/hz.ds/hz.res4": hz,
            f"{rel}/sub-02_task-EmptyRoom_meg.meg4": meg4,
        }
