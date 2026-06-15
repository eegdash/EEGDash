"""Unit tests for :class:`eegdash.dataset.snapshot.DatasetSnapshot`.

Coverage targets the contract documented in
``docs_pipeline_architecture_review.md`` §3 B1 and B2 and verified by
``docs_pipeline_validation_plan.md`` §3 step 5:

- Happy path (``source == "live"``, dataset_count > 0) — when the live
  API is reachable from the test environment.
- ``source == "cached"`` after a successful build primes the disk cache
  and a subsequent build hits a forced-failure network.
- ``source == "package-csv"`` when neither the live API nor a disk
  cache resolves.
- :attr:`api_errors` is populated whenever a fallback fires.
- Cache key includes ``api_base`` *and* ``database`` so two consumers
  pointed at different shards never see each other's data — the
  explicit bug ``_DATASET_SUMMARY_CACHE`` had in ``conf.py``.

Every test stubs the network so the suite stays offline-safe.
``test_build_live`` is also marked ``@pytest.mark.network`` so a
CI/local run with the marker enabled can hit the real production API
when that visibility is desired.
"""

from __future__ import annotations

import json
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from eegdash.dataset import snapshot as snapshot_mod
from eegdash.dataset.snapshot import DatasetSnapshot

# ---------------------------------------------------------------------------
# Required cases per the task spec (≥ 5)
# ---------------------------------------------------------------------------


def test_build_live_via_stubbed_http(
    chart_data_payload, server_manifest, routing_urlopen
):
    """Happy path: chart-data returns success → source == "live".

    Offline-safe equivalent of the ``test_build_live`` case in the
    task spec. The real-network version is at the bottom of this file
    behind ``@pytest.mark.network``.

    Stubs both the chart-data and build-manifest endpoints; the
    snapshot's :attr:`manifest` carries the server's response verbatim
    when reachable (see A2).
    """
    side_effect = routing_urlopen(
        {
            "datasets/chart-data": chart_data_payload("ds_live_1"),
            "build-manifest": server_manifest(dataset_count=1),
        }
    )

    with patch("urllib.request.urlopen", side_effect=side_effect):
        snapshot = DatasetSnapshot.build(
            api_base="https://stub.example",
            database="liveshard",
            limit=10,
        )

    assert snapshot.source == "live"
    assert snapshot.dataset_count == 1
    assert snapshot.rows().iloc[0]["dataset"] == "ds_live_1"
    assert snapshot.aggregations()["modality_counts"] == {"eeg": 1}
    assert snapshot.api_errors == []
    assert snapshot.fetched_at.tzinfo is not None
    # Manifest now carries the server's ``/build-manifest`` response
    # verbatim — including schema_version and dataset_count.
    assert snapshot.manifest["schema_version"] == "2.1.0"
    assert snapshot.manifest["dataset_count"] == 1
    assert snapshot.schema_version == "2.1.0"


def test_cached_on_api_failure_after_priming(
    chart_data_payload, server_manifest, routing_urlopen
):
    """First live call primes the disk cache; second call (forced
    failure) reads the cache and tags source == "cached".
    """
    side_effect = routing_urlopen(
        {
            "datasets/chart-data": chart_data_payload("ds_cached_1"),
            "build-manifest": server_manifest(dataset_count=1),
        }
    )
    api_base = "https://stub.example"
    database = "cacheshard"

    # 1. Prime: one successful live build writes the disk cache.
    with patch("urllib.request.urlopen", side_effect=side_effect):
        primed = DatasetSnapshot.build(api_base=api_base, database=database)
    assert primed.source == "live"
    # Reset in-memory instance cache so the next call exercises the
    # *disk* cache rather than the process cache.
    snapshot_mod._reset_instance_cache_for_testing()

    # 2. Force-fail network: chart-data and summary both raise; snapshot
    # must read the disk cache and tag source == "cached".
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("network unreachable"),
    ):
        cached = DatasetSnapshot.build(api_base=api_base, database=database)

    assert cached.source == "cached"
    assert cached.dataset_count == 1
    assert cached.rows().iloc[0]["dataset"] == "ds_cached_1"
    assert cached.api_errors, "fallback paths must record the API error text"
    # mtime of the JSON rows file is the snapshot's fetched_at.
    cache_path = snapshot_mod._disk_cache_path(database)
    assert cache_path.exists()
    expected = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
    assert cached.fetched_at == expected


@pytest.mark.parametrize(
    "package_csv_missing",
    [False, True],
    ids=["package-csv-present", "package-csv-missing"],
)
def test_csv_fallback(monkeypatch, package_csv_missing):
    """No disk cache + dead API falls back to the package CSV.

    Collapses ``test_csv_fallback_on_no_cache`` (CSV present → rows wired
    through, no disk-cache write) and
    ``test_csv_fallback_when_package_csv_missing`` (CSV missing → empty,
    source still ``package-csv``). Both never raise and record
    ``api_errors``.
    """
    if package_csv_missing:
        # When neither network nor disk cache nor package CSV resolves,
        # the snapshot still returns a value (source=package-csv, empty
        # DataFrame) with the failure recorded — never raise.
        monkeypatch.setattr(
            snapshot_mod, "PACKAGE_CSV_PATH", Path("/does/not/exist.csv")
        )

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("network unreachable"),
        ):
            snapshot = DatasetSnapshot.build(
                api_base="https://stub.example", database="empty_shard"
            )

        assert snapshot.source == "package-csv"
        assert snapshot.dataset_count == 0
        assert snapshot.api_errors, "every fallback path must populate api_errors"
    else:
        # No disk cache + dead API + present package CSV → "package-csv".
        api_base = "https://stub.example"
        database = "no_cache_shard"

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("network unreachable"),
        ):
            snapshot = DatasetSnapshot.build(api_base=api_base, database=database)

        assert snapshot.source == "package-csv"
        # The shipped CSV has hundreds of rows; we don't assert an exact
        # count (it changes whenever someone runs ``update_dataset_summary``)
        # but we do assert the fallback actually wired data through.
        assert snapshot.dataset_count > 0
        assert snapshot.api_errors, "fallback must record API errors"
        # No accidental disk cache write on the failure path.
        assert not snapshot_mod._disk_cache_path(database).exists()


def test_api_errors_populated_with_exception_text():
    """The ``api_errors`` list carries the actual exception strings, not
    just a count — that's what the B2 CI gate inspects.
    """
    err = urllib.error.URLError("simulated DNS failure")
    monkey_csv = pd.DataFrame([{"dataset": "ds_csv"}])
    with (
        patch("urllib.request.urlopen", side_effect=err),
        patch.object(snapshot_mod, "_read_package_csv", return_value=monkey_csv),
    ):
        snapshot = DatasetSnapshot.build(
            api_base="https://stub.example", database="errshard"
        )

    assert snapshot.source == "package-csv"
    assert any("simulated DNS failure" in e for e in snapshot.api_errors)


def test_cache_keyed_by_api_base_and_database(
    chart_data_payload, make_urlopen_response
):
    """Two builds with different ``api_base`` must NOT share an instance.

    Reproduces the bug the unkeyed ``_DATASET_SUMMARY_CACHE`` global in
    ``conf.py`` had: any caller that pointed at a different shard
    silently got stale data from the previous caller.
    """
    payload_a = chart_data_payload("ds_from_a")
    payload_b = chart_data_payload("ds_from_b")

    def urlopen_side_effect(url, *_, **__):
        if "shard_a" in url:
            return make_urlopen_response(payload_a)
        if "shard_b" in url:
            return make_urlopen_response(payload_b)
        raise AssertionError(f"unexpected URL: {url}")

    with patch("urllib.request.urlopen", side_effect=urlopen_side_effect):
        a = DatasetSnapshot.build(api_base="https://shard_a.example", database="x")
        b = DatasetSnapshot.build(api_base="https://shard_b.example", database="x")

    assert a is not b, "cache leaked across api_base"
    assert a.rows().iloc[0]["dataset"] == "ds_from_a"
    assert b.rows().iloc[0]["dataset"] == "ds_from_b"

    # Second call with the same key returns the SAME instance — the
    # idempotency contract from the task spec.
    with patch("urllib.request.urlopen", side_effect=urlopen_side_effect):
        a2 = DatasetSnapshot.build(api_base="https://shard_a.example", database="x")
    assert a is a2, "same key should return the cached instance"


def test_snapshot_is_immutable():
    """Attempting to reassign provenance attributes must raise."""
    snapshot = DatasetSnapshot(
        rows=pd.DataFrame([{"dataset": "ds_immutable"}]),
        aggregations={},
        montages={},
        source="live",
        fetched_at=datetime.now(timezone.utc),
    )
    with pytest.raises(AttributeError):
        snapshot.source = "cached"  # type: ignore[misc]


def test_provenance_logged_once(
    caplog, chart_data_payload, server_manifest, routing_urlopen
):
    """:meth:`DatasetSnapshot.build` emits exactly one I8-formatted
    INFO line per build — the invariant the validation plan §2 grep's
    for.
    """
    side_effect = routing_urlopen(
        {
            "datasets/chart-data": chart_data_payload("ds_log_1"),
            "build-manifest": server_manifest(dataset_count=1),
        }
    )
    caplog.set_level("INFO", logger="eegdash.dataset.snapshot")
    with patch("urllib.request.urlopen", side_effect=side_effect):
        DatasetSnapshot.build(api_base="https://stub.example", database="logshard")

    matching = [
        rec
        for rec in caplog.records
        if rec.name == "eegdash.dataset.snapshot"
        and rec.getMessage().startswith("DatasetSnapshot source=")
    ]
    assert len(matching) == 1, matching
    msg = matching[0].getMessage()
    assert "source=live" in msg
    assert "dataset_count=1" in msg
    assert "fetched_at=" in msg


def test_load_roundtrip(tmp_path, chart_data_payload, server_manifest, routing_urlopen):
    """A snapshot written by build can be re-hydrated via load."""
    side_effect = routing_urlopen(
        {
            "datasets/chart-data": chart_data_payload("ds_roundtrip"),
            "build-manifest": server_manifest(dataset_count=1),
        }
    )
    api_base = "https://stub.example"
    database = "loadshard"
    with patch("urllib.request.urlopen", side_effect=side_effect):
        built = DatasetSnapshot.build(api_base=api_base, database=database)

    cache_path = snapshot_mod._disk_cache_path(database)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source": built.source,
                "dataset_count": built.dataset_count,
                "fetched_at": built.fetched_at.isoformat(),
                "aggregations": built.aggregations(),
            }
        )
    )
    rows_dst = manifest_path.with_suffix(".rows.json")
    rows_dst.write_bytes(cache_path.read_bytes())

    loaded = DatasetSnapshot.load(manifest_path)
    assert loaded.source == built.source
    assert loaded.dataset_count == built.dataset_count
    assert loaded.rows().iloc[0]["dataset"] == "ds_roundtrip"


@pytest.mark.parametrize(
    "lookup_key, expect_present",
    [
        ("ds_montage", True),  # exact-case key present → not None
        ("DS_MONTAGE", True),  # different-case key present → case-insensitive
        ("ds_absent", False),  # absent key → None
        ("", False),  # empty id → None
    ],
    ids=[
        "exact-case-present",
        "different-case-present",
        "absent-key",
        "empty-key",
    ],
)
def test_montage_lookup(lookup_key, expect_present):
    """``montage()`` returns the projected dict for a present key (in any
    case) and ``None`` for an unknown or empty id rather than raising.

    Collapses the former ``test_montage_returns_none_when_absent``,
    ``test_montage_returns_data_when_present`` and
    ``test_montage_lookup_is_case_insensitive``. Snapshot construction is
    transcribed from ``test_montage_returns_data_when_present``.
    """
    snapshot = DatasetSnapshot(
        rows=pd.DataFrame([{"dataset": "ds_montage"}]),
        aggregations={},
        montages={"ds_montage": {"name": "standard_1020", "n_channels": 64}},
        source="live",
        fetched_at=datetime.now(timezone.utc),
    )
    montage = snapshot.montage(lookup_key)
    if expect_present:
        assert montage is not None
        assert montage["name"] == "standard_1020"
        # Returned value is a copy — mutations don't leak back to the cache.
        montage["mutated"] = True
        assert "mutated" not in snapshot.montage(lookup_key)
    else:
        assert montage is None


def test_chart_data_request_includes_montages_param(
    chart_data_payload, server_manifest, make_urlopen_response
):
    """The chart-data URL must carry ``?include=montages`` so the server
    joins the top per-dataset montage onto every row in a single
    round-trip (arch #5).
    """
    seen_urls: list[str] = []

    def capture_urlopen(url, *_, **__):
        # ``urlopen`` can take a string OR a Request; normalize both.
        url_str = url if isinstance(url, str) else url.get_full_url()
        seen_urls.append(url_str)
        if "datasets/chart-data" in url_str:
            return make_urlopen_response(chart_data_payload("ds_url_check"))
        if "build-manifest" in url_str:
            return make_urlopen_response(server_manifest(dataset_count=1))
        raise AssertionError(f"unexpected URL: {url_str}")

    with patch("urllib.request.urlopen", side_effect=capture_urlopen):
        DatasetSnapshot.build(
            api_base="https://stub.example", database="urlshard", limit=42
        )

    chart_urls = [u for u in seen_urls if "datasets/chart-data" in u]
    assert chart_urls, f"chart-data was never requested; saw {seen_urls}"
    assert "include=montages" in chart_urls[0], (
        f"chart-data request missing include=montages: {chart_urls[0]}"
    )
    assert "limit=42" in chart_urls[0]


def test_snapshot_populates_montages_when_included(
    chart_data_payload, server_manifest, routing_urlopen
):
    """Stub chart-data with montage objects on a couple of datasets;
    ``snapshot.montage(dataset_id)`` must return the projected dict.

    The projection layers viewer-friendly aliases (``label``,
    ``n_channels``, ``montage_id``) on top of the registry doc — kept
    backward-compatible with the retired build_electrode_layouts.py
    output so the consumer is a no-op rename.
    """
    payload = chart_data_payload("ds001785")
    payload["datasets"][0]["montage"] = {
        "hash": "42b9e8daf4ff0e6d",
        "subject_count": 54,
        "modality": "eeg",
        "n_sensors": 63,
        "space_declared": "CapTrak",
        "units_declared": "mm",
        "channel_names": ["AF3", "AF4", "Cz"],
    }
    # Second dataset omits the montage — it must NOT crash the parser
    # and must NOT appear in the resulting map (the "no scalp layout
    # indexed" placeholder is the correct downstream rendering).
    payload["datasets"].append(
        {
            "dataset_id": "ds_no_montage",
            "name": "no-montage",
            "demographics": {"subjects_count": 1},
            "total_files": 1,
            "tasks": ["t"],
            "recording_modality": ["eeg"],
            "tags": {},
            "montage": None,
        }
    )

    side_effect = routing_urlopen(
        {
            "datasets/chart-data": payload,
            "build-manifest": server_manifest(dataset_count=2),
        }
    )

    with patch("urllib.request.urlopen", side_effect=side_effect):
        snapshot = DatasetSnapshot.build(
            api_base="https://stub.example", database="montageshard"
        )

    assert snapshot.source == "live"

    mont = snapshot.montage("ds001785")
    assert mont is not None
    assert mont["modality"] == "eeg"
    assert mont["n_sensors"] == 63
    assert mont["n_channels"] == 63  # viewer-friendly alias
    assert mont["hash"] == "42b9e8daf4ff0e6d"
    assert mont["montage_id"] == "42b9e8daf4ff0e6d"  # alias
    assert mont["label"] == "EEG · 63 sensors"
    assert mont["space_declared"] == "CapTrak"
    assert mont["channel_names"] == ["AF3", "AF4", "Cz"]

    # Dataset without a montage must not appear in the map.
    assert snapshot.montage("ds_no_montage") is None


def test_snapshot_montages_persist_through_disk_cache(
    chart_data_payload, server_manifest, routing_urlopen
):
    """A live build writes a montages sidecar next to the JSON rows
    cache; a subsequent ``cached`` resolution rehydrates it so the
    docs build doesn't lose montage data the moment the API blips.
    """
    payload = chart_data_payload("ds_persisted")
    payload["datasets"][0]["montage"] = {
        "hash": "deadbeef00000000",
        "modality": "eeg",
        "n_sensors": 32,
        "channel_names": ["A1"],
    }

    side_effect = routing_urlopen(
        {
            "datasets/chart-data": payload,
            "build-manifest": server_manifest(dataset_count=1),
        }
    )

    with patch("urllib.request.urlopen", side_effect=side_effect):
        primed = DatasetSnapshot.build(
            api_base="https://stub.example", database="persistshard"
        )
    assert primed.source == "live"
    assert primed.montage("ds_persisted") is not None

    # Sidecar is on disk.
    sidecar = snapshot_mod._montages_sidecar_path("persistshard")
    assert sidecar.exists()

    # Force the cached path: drop the in-memory cache and fail the
    # network. The cached snapshot must still serve the montage.
    snapshot_mod._reset_instance_cache_for_testing()
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("network unreachable"),
    ):
        cached = DatasetSnapshot.build(
            api_base="https://stub.example", database="persistshard"
        )

    assert cached.source == "cached"
    rehydrated = cached.montage("ds_persisted")
    assert rehydrated is not None
    assert rehydrated["hash"] == "deadbeef00000000"
    assert rehydrated["n_channels"] == 32


def test_metadata_from_chart_data(chart_data_payload, server_manifest, routing_urlopen):
    """``include=metadata`` rows parse into NemarMetadata; lookup is case-insensitive."""
    payload = chart_data_payload(
        "ds_meta_1",
        metadata={
            "description": "A test dataset.",
            "bids_version": "1.8.0",
            "authors": [{"name": "Jane Doe", "orcid": "0000-0002-1825-0097"}],
            "keywords": [
                {
                    "term": "Face Perception",
                    "scheme": "MeSH",
                    "value_uri": "https://id.nlm.nih.gov/mesh/x",
                }
            ],
            "versions": [
                {
                    "version": "1.0.0",
                    "doi": "10.18112/x",
                    "created_at": "2021-06-03T00:00:00Z",
                }
            ],
        },
    )
    side_effect = routing_urlopen(
        {
            "datasets/chart-data": payload,
            "build-manifest": server_manifest(dataset_count=1),
        }
    )
    with patch("urllib.request.urlopen", side_effect=side_effect):
        snapshot = DatasetSnapshot.build(
            api_base="https://stub.example", database="metashard"
        )

    meta = snapshot.metadata("ds_meta_1")
    assert meta is not None
    assert meta.description == "A test dataset."
    assert meta.authors[0].name == "Jane Doe"
    assert meta.authors[0].orcid == "0000-0002-1825-0097"
    assert meta.keywords[0].term == "Face Perception"
    assert meta.versions[0].version == "1.0.0"
    # Case-insensitive hit; unknown dataset → None.
    assert snapshot.metadata("DS_META_1") is not None
    assert snapshot.metadata("nonexistent") is None


def test_metadata_absent_when_not_served(
    chart_data_payload, server_manifest, routing_urlopen
):
    """A chart-data row without a ``metadata`` block → ``metadata()`` returns None."""
    side_effect = routing_urlopen(
        {
            "datasets/chart-data": chart_data_payload("ds_no_meta"),
            "build-manifest": server_manifest(dataset_count=1),
        }
    )
    with patch("urllib.request.urlopen", side_effect=side_effect):
        snapshot = DatasetSnapshot.build(
            api_base="https://stub.example", database="nometashard"
        )
    assert snapshot.metadata("ds_no_meta") is None


def test_package_csv_fallback_skips_disk_cache_write():
    """When fetch fails, the snapshot must NOT pollute the disk cache
    with a package-csv fallback — that would mask a future "API is
    back" run as still-broken.
    """
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("dead"),
    ):
        DatasetSnapshot.build(api_base="https://stub.example", database="no_pollute")

    assert not snapshot_mod._disk_cache_path("no_pollute").exists()


# ---------------------------------------------------------------------------
# /build-manifest server integration (arch #4)
# ---------------------------------------------------------------------------


def test_manifest_uses_server_when_available(
    chart_data_payload, server_manifest, routing_urlopen
):
    """``snapshot.manifest`` carries the server's ``/build-manifest``
    response verbatim — including the keys the local projection never
    populated (``schema_version``, ``last_ingested_at``, ``git_sha``,
    ``name_coverage``).
    """
    server_payload = server_manifest(
        dataset_count=1, schema_version="2.1.0", git_sha="abc1234"
    )

    def fake_fetch_manifest(api_base, database, *, errors=None, timeout=5.0):
        return dict(server_payload)

    side_effect = routing_urlopen(
        {"datasets/chart-data": chart_data_payload("ds_server_manifest")}
    )

    with (
        patch("urllib.request.urlopen", side_effect=side_effect),
        patch.object(snapshot_mod, "_try_fetch_build_manifest", fake_fetch_manifest),
    ):
        snapshot = DatasetSnapshot.build(
            api_base="https://stub.example", database="manifestshard"
        )

    assert snapshot.source == "live"
    assert snapshot.manifest["schema_version"] == "2.1.0"
    assert snapshot.manifest["dataset_count"] == snapshot.dataset_count == 1
    assert snapshot.manifest["git_sha"] == "abc1234"
    assert snapshot.manifest["last_ingested_at"] == "2026-04-18T16:10:52.827000Z"
    # Convenience property mirrors the manifest field.
    assert snapshot.schema_version == "2.1.0"
    # Count agreed → no mismatch warning leaked through.
    assert not any("build-manifest dataset_count" in e for e in snapshot.api_errors)


def test_manifest_falls_back_on_manifest_error(chart_data_payload, routing_urlopen):
    """When ``/build-manifest`` raises, the snapshot still builds and
    ``manifest`` reverts to the locally-projected dict — no exception
    leaks to the caller.
    """

    def fake_fetch_manifest(api_base, database, *, errors=None, timeout=5.0):
        # Mimic the helper's contract: on failure, append to errors and
        # return ``None``. The caller (``_live_snapshot``) must still
        # produce a valid snapshot.
        if errors is not None:
            errors.append("build-manifest error at ...: simulated outage")
        return None

    side_effect = routing_urlopen(
        {"datasets/chart-data": chart_data_payload("ds_manifest_down")}
    )

    with (
        patch("urllib.request.urlopen", side_effect=side_effect),
        patch.object(snapshot_mod, "_try_fetch_build_manifest", fake_fetch_manifest),
    ):
        snapshot = DatasetSnapshot.build(
            api_base="https://stub.example", database="manifestdown"
        )

    assert snapshot.source == "live"
    assert snapshot.dataset_count == 1
    # Local projection format: {"source", "dataset_count", "fetched_at"}.
    assert snapshot.manifest["source"] == "live"
    assert snapshot.manifest["dataset_count"] == 1
    assert "fetched_at" in snapshot.manifest
    # No server-only fields leaked through.
    assert "schema_version" not in snapshot.manifest
    assert snapshot.schema_version is None
    # The fetch failure is surfaced on api_errors per the helper contract.
    assert any("build-manifest" in e for e in snapshot.api_errors)


def test_manifest_count_mismatch_tagged(
    chart_data_payload, server_manifest, routing_urlopen
):
    """Server's ``dataset_count`` disagreeing with the snapshot's row
    count is surfaced on ``api_errors`` (never raised) so the B2 CI gate
    can refuse to publish.
    """
    server_payload = server_manifest(dataset_count=9999, schema_version="2.1.0")

    def fake_fetch_manifest(api_base, database, *, errors=None, timeout=5.0):
        return dict(server_payload)

    side_effect = routing_urlopen(
        {"datasets/chart-data": chart_data_payload("ds_skew")}
    )

    with (
        patch("urllib.request.urlopen", side_effect=side_effect),
        patch.object(snapshot_mod, "_try_fetch_build_manifest", fake_fetch_manifest),
    ):
        snapshot = DatasetSnapshot.build(
            api_base="https://stub.example", database="skewshard"
        )

    # Snapshot still builds — divergence is a warning, not a failure.
    assert snapshot.source == "live"
    assert snapshot.dataset_count == 1  # the rows we actually loaded
    # Server manifest still surfaced verbatim.
    assert snapshot.manifest["dataset_count"] == 9999
    # Mismatch tagged on api_errors with the exact format from the spec.
    assert any(
        "build-manifest dataset_count=9999" in e and "snapshot.rows()=1" in e
        for e in snapshot.api_errors
    ), snapshot.api_errors


def test_manifest_not_fetched_in_csv_fallback_path():
    """CSV-fallback paths must NOT hit ``/build-manifest``.

    When the live API is unreachable we fall back to the disk cache
    (or package CSV). In that state the server is by definition not
    contributing data to this snapshot, so calling
    ``/build-manifest`` would be both wasteful and potentially
    misleading (it would attach server provenance to non-server
    data). Verify the helper is never called on the fallback path.
    """
    fake_fetch_manifest = MagicMock(return_value=None)

    with (
        patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("network unreachable"),
        ),
        patch.object(snapshot_mod, "_try_fetch_build_manifest", fake_fetch_manifest),
    ):
        snapshot = DatasetSnapshot.build(
            api_base="https://stub.example", database="csvfallback"
        )

    assert snapshot.source == "package-csv"
    fake_fetch_manifest.assert_not_called()
    # The fallback's local projection is what surfaces on ``manifest``.
    assert snapshot.manifest["source"] == "package-csv"
    assert snapshot.manifest["dataset_count"] == snapshot.dataset_count


# ---------------------------------------------------------------------------
# pin(date) — reproducible builds via /snapshots/{date} (arch #6)
# ---------------------------------------------------------------------------


def test_pin_returns_frozen_snapshot(snapshot_payload_pin, make_urlopen_response):
    """``pin(date)`` surfaces the frozen ``/snapshots/{date}`` payload's
    provenance verbatim: ``source == "live"``, ``pinned_at``, the
    ``build_manifest``, the payload's ``created_at`` as ``fetched_at``
    (NOT wall-clock — the point of pinning), and the same montage parse
    as the live path.
    """
    created_at = "2026-04-01T08:15:30.123456Z"
    payload = snapshot_payload_pin(
        date="2026-04-01", dataset_count=1, include_montage=True, created_at=created_at
    )

    def fake_urlopen(url, *_, **__):
        url_str = url if isinstance(url, str) else url.get_full_url()
        assert "/snapshots/2026-04-01" in url_str, f"unexpected URL: {url_str}"
        return make_urlopen_response(payload)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        snapshot = DatasetSnapshot.pin("2026-04-01")

    assert snapshot.source == "live"
    assert snapshot.pinned_at == "2026-04-01"
    assert snapshot.dataset_count == 1
    assert snapshot.manifest == payload["build_manifest"]
    assert snapshot.api_errors == []
    # fetched_at is the frozen created_at, normalised off the trailing 'Z'.
    assert snapshot.fetched_at == datetime.fromisoformat(
        created_at.replace("Z", "+00:00")
    )
    assert snapshot.fetched_at.tzinfo is not None
    # Frozen chart_data montages run the same parser as the live path.
    mont = snapshot.montage("ds_pin_1")
    assert mont is not None
    assert mont["hash"] == "42b9e8daf4ff0e6d"
    assert mont["modality"] == "eeg"
    assert mont["n_sensors"] == mont["n_channels"] == 63
    assert mont["label"] == "EEG · 63 sensors"
    assert mont["channel_names"] == ["AF3", "AF4", "Cz"]


@pytest.mark.parametrize(
    "is_404_case",
    [False, True],
    ids=["invalid-date-value-error", "404-lookup-error"],
)
def test_pin_rejects_bad_input(is_404_case):
    """``pin`` rejects malformed dates with ``ValueError`` (before any
    network call) and a server 404 with ``LookupError``.

    Collapses ``test_pin_invalid_date_raises_value_error`` and
    ``test_pin_404_raises_lookup_error``.
    """
    if not is_404_case:
        # Anything not matching ``YYYY-MM-DD`` must raise before any
        # network call. Catching the typo at the boundary keeps a wildcard
        # or path-injection from reaching the server route.
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            DatasetSnapshot.pin("not-a-date")
        with pytest.raises(ValueError):
            DatasetSnapshot.pin("2026/05/18")
        with pytest.raises(ValueError):
            DatasetSnapshot.pin("2026-5-18")  # missing zero pad
        with pytest.raises(ValueError):
            DatasetSnapshot.pin("")
    else:
        # A 404 from the server means "no snapshot frozen for that date".
        # Surface as ``LookupError`` so the caller can decide whether to
        # retry against a different date or fall back to :meth:`build`.
        err = urllib.error.HTTPError(
            "https://stub.example/eegdash/snapshots/2026-04-01",
            404,
            "not found",
            {},
            None,
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(LookupError, match="2026-04-01"):
                DatasetSnapshot.pin("2026-04-01")


def test_pin_does_not_share_cache_with_build(
    snapshot_payload_pin, chart_data_payload, server_manifest, make_urlopen_response
):
    """A pinned snapshot and a live :meth:`build` must be different
    objects. The in-process ``_INSTANCE_CACHE`` keys :meth:`build`
    results by ``(api_base, database, limit)``; if :meth:`pin` ever
    leaked into the same cache, a pin for one date could shadow the
    live build (or vice-versa).
    """
    snapshot_payload = snapshot_payload_pin(date="2026-05-18", dataset_count=1)
    build_chart = chart_data_payload("ds_live_after_pin")
    build_manifest = server_manifest(dataset_count=1)

    def fake_urlopen(url, *_, **__):
        url_str = url if isinstance(url, str) else url.get_full_url()
        if "/snapshots/2026-05-18" in url_str:
            return make_urlopen_response(snapshot_payload)
        if "datasets/chart-data" in url_str:
            return make_urlopen_response(build_chart)
        if "build-manifest" in url_str:
            return make_urlopen_response(build_manifest)
        raise AssertionError(f"unexpected URL: {url_str}")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        s1 = DatasetSnapshot.pin("2026-05-18")
        s2 = DatasetSnapshot.build(
            api_base="https://data.eegdash.org/api", database="eegdash"
        )

    assert s1 is not s2
    # Distinguishing provenance flags must also differ.
    assert s1.pinned_at == "2026-05-18"
    assert s2.pinned_at is None
    # And the underlying row identities are different.
    assert s1.rows().iloc[0]["dataset"] == "ds_pin_1"
    assert s2.rows().iloc[0]["dataset"] == "ds_live_after_pin"

    # Calling pin() twice for the same date returns a fresh object;
    # we don't dedupe pinned snapshots in-process (the immutable HTTP
    # cache is what makes the repeat cheap).
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        s1b = DatasetSnapshot.pin("2026-05-18")
    assert s1b is not s1


# ---------------------------------------------------------------------------
# Optional: real-network smoke test (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_build_live_against_real_api():
    """Production-API smoke test corresponding to the
    ``test_build_live`` case in the task spec.

    Marked ``@pytest.mark.network`` so the offline default suite skips
    it. Run with ``pytest -m network`` (or via the CI gate script
    ``scripts/validation/check_snapshot_health.py``) to verify
    end-to-end connectivity.
    """
    snapshot = DatasetSnapshot.build()
    assert snapshot.source == "live"
    assert snapshot.dataset_count > 0
    # A2 contract: a live build surfaces the server manifest verbatim,
    # including a schema_version string.
    assert snapshot.schema_version is not None
    assert "dataset_count" in snapshot.manifest
