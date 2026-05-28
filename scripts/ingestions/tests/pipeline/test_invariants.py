"""Property-based invariant tests for Record schema, provenance enum, and parser registry."""

from __future__ import annotations

import json
from pathlib import Path

from _helpers import load_digest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from _file_utils import list_local_bids_files
from _format_parser_registry import get_parser_for_extension, registered_extensions
from eegdash.testing import data_file
from record_enumerator import EnumerationResult


def test_provenance_sources_are_a_closed_set():
    digest = load_digest()
    valid = {
        digest._PROV_MNE_BIDS,
        digest._PROV_MODALITY_SIDECAR,
        digest._PROV_CHANNELS_TSV,
        digest._PROV_BINARY_PARSER,
        digest._PROV_MNE_FALLBACK,
    }
    assert valid == {
        "mne_bids",
        "modality_sidecar",
        "channels_tsv",
        "binary_parser",
        "mne_fallback",
    }


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    sfreq=st.one_of(
        st.none(),
        st.floats(
            min_value=-1e6,
            max_value=2e6,
            allow_nan=False,
            allow_infinity=False,
        ),
    ),
    nchans=st.one_of(st.none(), st.integers(min_value=-100, max_value=20000)),
)
def test_clamp_preserves_value_provenance_invariant(
    sfreq: float | None, nchans: int | None
):
    # _clamp_metadata_extremes invariant: provenance is None iff value is None
    digest = load_digest()
    prov = {
        "sampling_frequency": "mne_bids" if sfreq is not None else None,
        "nchans": "channels_tsv" if nchans is not None else None,
        "ntimes": None,
        "ch_names": None,
    }
    out_sf, out_n = digest._clamp_metadata_extremes(
        sampling_frequency=sfreq,
        nchans=nchans,
        ch_names=None,
        bids_relpath="hypothesis/test",
        provenance=prov,
    )
    assert (out_sf is None) == (prov["sampling_frequency"] is None)
    assert (out_n is None) == (prov["nchans"] is None)


def test_secondary_source_adapters_return_lists():
    # ADR 0001: adapters return [] on failure, never None
    assert isinstance(list_local_bids_files("/no/such/path"), list)


def test_all_registered_parsers_return_none_or_dict(tmp_path: Path):
    for ext in registered_extensions():
        parser = get_parser_for_extension(ext)
        result = parser(tmp_path / f"does_not_exist{ext}")
        assert result is None or isinstance(result, dict), (
            f"parser for {ext} returned {type(result).__name__}, expected None or dict"
        )


@given(
    sources=st.lists(
        st.sampled_from(
            ["mne_bids", "modality_sidecar", "channels_tsv", "binary_parser"]
        ),
        min_size=2,
        max_size=5,
    )
)
def test_stamp_provenance_idempotent_under_first_writer_wins(sources):
    digest = load_digest()
    prov = digest._empty_provenance()
    field = "sampling_frequency"
    digest._stamp_provenance(
        prov, sources[0], field=field, old_value=None, new_value=250.0
    )
    expected = sources[0]
    for s in sources[1:]:
        digest._stamp_provenance(prov, s, field=field, old_value=250.0, new_value=250.0)
    assert prov[field] == expected


def test_every_snapshot_record_has_canonical_fields():
    snapshot_root = data_file("digest_snapshots/outputs")
    for ds_dir in snapshot_root.iterdir():
        records_path = ds_dir / f"{ds_dir.name}_records.json"
        if not records_path.exists():
            continue
        payload = json.loads(records_path.read_text())
        for rec in payload["records"]:
            assert "dataset" in rec
            assert "bids_relpath" in rec
            assert "digested_at" in rec
            assert "storage" in rec
            assert isinstance(rec["storage"], dict)
            assert "base" in rec["storage"]
            assert "backend" in rec["storage"]


def test_every_snapshot_dataset_carries_ingestion_fingerprint():
    snapshot_root = data_file("digest_snapshots/outputs")
    for ds_dir in snapshot_root.iterdir():
        ds_path = ds_dir / f"{ds_dir.name}_dataset.json"
        if not ds_path.exists():
            continue
        ds = json.loads(ds_path.read_text())
        assert "ingestion_fingerprint" in ds
        assert isinstance(ds["ingestion_fingerprint"], str)
        assert len(ds["ingestion_fingerprint"]) >= 16


def test_records_dataset_id_matches_container_directory():
    snapshot_root = data_file("digest_snapshots/outputs")
    for ds_dir in snapshot_root.iterdir():
        records_path = ds_dir / f"{ds_dir.name}_records.json"
        if not records_path.exists():
            continue
        payload = json.loads(records_path.read_text())
        for rec in payload["records"]:
            assert rec["dataset"] == ds_dir.name


def test_provenance_values_only_in_documented_enum():
    snapshot_root = data_file("digest_snapshots/outputs")
    valid = {
        "mne_bids",
        "modality_sidecar",
        "channels_tsv",
        "binary_parser",
        "mne_fallback",
        None,
    }
    for ds_dir in snapshot_root.iterdir():
        records_path = ds_dir / f"{ds_dir.name}_records.json"
        if not records_path.exists():
            continue
        payload = json.loads(records_path.read_text())
        for rec in payload["records"]:
            prov = rec.get("_metadata_provenance")
            if prov is None:  # manifest path Records bypass the cascade
                continue
            for field, source in prov.items():
                assert source in valid, (
                    f"unexpected provenance source {source!r} for "
                    f"{field} in {ds_dir.name}"
                )


def test_enumeration_result_with_empty_records_has_no_montages():
    result = EnumerationResult(
        dataset_meta={"dataset_id": "ds_test"},
        records=[],
        errors=[],
        montages={},
    )
    assert len(result.records) == 0
    assert len(result.montages) == 0
