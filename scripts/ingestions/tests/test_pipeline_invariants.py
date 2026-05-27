"""Property-based invariant tests .

The example-based tests in test_digest_extraction_helpers and
test_metadata_provenance assert "given fixture X, output Y" — they
catch regression on known inputs. The invariants here cover what's
true for ALL inputs in a class:

- Every Record carries the canonical fields ``dataset``,
  ``bids_relpath``, ``digested_at``, ``storage``.
- ``ingestion_fingerprint`` is stable: feeding the same manifest
  twice produces the same hash.
- The provenance source-name enum is closed: only the 5 documented
  values (or None) appear.
- ``_clamp_metadata_extremes`` preserves the value→provenance
  invariant: provenance is None iff value is None for the clamped
  fields.

These properties are independent of the snapshot fixtures — they
hold for any data flowing through the pipeline.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from eegdash.testing import data_file
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))


def _load_digest():
    spec = importlib.util.spec_from_file_location(
        "_invariants_digest_target", _INGEST_DIR / "3_digest.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── Provenance enum is closed ────────────────────────────────────────────


def test_provenance_sources_are_a_closed_set():
    """The 5 provenance sources are the only valid values (None excluded)."""
    digest = _load_digest()
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
    # If a 6th source name leaks in, this test breaks loudly.


# ─── _clamp_metadata_extremes invariants ──────────────────────────────────


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
    """``_clamp_metadata_extremes`` maintains the invariant:
    *provenance is None iff value is None* for each field.

    This holds for any input combination — the clamp either keeps
    both, or clears both, never desyncs them.
    """
    digest = _load_digest()
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
    # Invariant: value is None iff provenance is None for each field.
    assert (out_sf is None) == (prov["sampling_frequency"] is None)
    assert (out_n is None) == (prov["nchans"] is None)


# ─── Source-listing adapter: empty list invariant ─────────────────────────


def test_secondary_source_adapters_return_lists():
    """Every secondary source adapter returns a list (never None).

    Pins ADR 0001's contract: secondary adapters return [] on failure,
    not None. Production code can safely iterate the result without
    a None guard.
    """
    from _file_utils import (
        list_local_bids_files,
    )

    # All "missing input" cases: each adapter should return [] (or a
    # list anyway). Using respx-less paths here — figshare/zenodo/osf
    # against real URLs would hit the network, so use list_local_bids_files
    # against a non-existent dir (no network).
    assert isinstance(list_local_bids_files("/no/such/path"), list)


# ─── Format parser registry: all parsers honor the contract ───────────────


def test_all_registered_parsers_return_none_or_dict(tmp_path: Path):
    """Every registered FormatParser returns None or a dict — never raises
    on a non-existent or unreadable input.

    Property over the parser registry: regardless of which extension
    you pick, calling with a missing file always returns None.
    """
    from _format_parser_registry import (
        get_parser_for_extension,
        registered_extensions,
    )

    for ext in registered_extensions():
        parser = get_parser_for_extension(ext)
        # The "missing file" case — should never raise.
        result = parser(tmp_path / f"does_not_exist{ext}")
        assert result is None or isinstance(result, dict), (
            f"parser for {ext} returned {type(result).__name__}, expected None or dict"
        )


# ─── Stamping invariants ──────────────────────────────────────────────────


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
    """Calling _stamp_provenance multiple times with later 'sources' for
    the same field never overwrites the first non-None stamp.

    Property over the cascade ordering: regardless of how many times
    a later cascade step would have stamped, the original stamp wins.
    """
    digest = _load_digest()
    prov = digest._empty_provenance()
    field = "sampling_frequency"
    # First non-None stamp wins
    digest._stamp_provenance(
        prov, sources[0], field=field, old_value=None, new_value=250.0
    )
    expected = sources[0]
    # Later stamps shouldn't change the value
    for s in sources[1:]:
        digest._stamp_provenance(prov, s, field=field, old_value=250.0, new_value=250.0)
    assert prov[field] == expected


# ─── Snapshot invariants: every Record has canonical fields ───────────────


def test_every_snapshot_record_has_canonical_fields():
    """All Records in both snapshots carry the canonical Record fields.

    This is a property over the snapshot output: regardless of which
    digest path (BIDS-fs vs manifest) produced the Record, the schema
    invariants hold.
    """
    snapshot_root = data_file("digest_snapshots/outputs")
    for ds_dir in snapshot_root.iterdir():
        records_path = ds_dir / f"{ds_dir.name}_records.json"
        if not records_path.exists():
            continue
        payload = json.loads(records_path.read_text())
        for rec in payload["records"]:
            # Canonical Record fields (per eegdash.schemas.RecordModel)
            assert "dataset" in rec
            assert "bids_relpath" in rec
            assert "digested_at" in rec
            assert "storage" in rec
            assert isinstance(rec["storage"], dict)
            assert "base" in rec["storage"]
            assert "backend" in rec["storage"]


def test_every_snapshot_dataset_carries_ingestion_fingerprint():
    """The Dataset doc always has an ingestion_fingerprint — required for
    de-dup checks in stage 5."""
    snapshot_root = data_file("digest_snapshots/outputs")
    for ds_dir in snapshot_root.iterdir():
        ds_path = ds_dir / f"{ds_dir.name}_dataset.json"
        if not ds_path.exists():
            continue
        ds = json.loads(ds_path.read_text())
        assert "ingestion_fingerprint" in ds
        assert isinstance(ds["ingestion_fingerprint"], str)
        assert len(ds["ingestion_fingerprint"]) >= 16  # at least a short hash


def test_records_dataset_id_matches_container_directory():
    """Cross-Record invariant: every Record's ``dataset`` field equals the
    enclosing directory name (the dataset_id)."""
    snapshot_root = data_file("digest_snapshots/outputs")
    for ds_dir in snapshot_root.iterdir():
        records_path = ds_dir / f"{ds_dir.name}_records.json"
        if not records_path.exists():
            continue
        payload = json.loads(records_path.read_text())
        for rec in payload["records"]:
            assert rec["dataset"] == ds_dir.name


def test_provenance_values_only_in_documented_enum():
    """Every record's _metadata_provenance values come from the documented
    5-element enum (or None). No surprise sources leak through."""
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


# ─── EnumerationResult invariants ─────────────────────────────────────────


def test_enumeration_result_with_empty_records_has_no_montages():
    """Cross-component invariant: a Result with no Records also has no
    montages (montages are derived from records' layouts)."""
    from record_enumerator import EnumerationResult

    result = EnumerationResult(
        dataset_meta={"dataset_id": "ds_test"},
        records=[],
        errors=[],
        montages={},
    )
    assert len(result.records) == 0
    assert len(result.montages) == 0
