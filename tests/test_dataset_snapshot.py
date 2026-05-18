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
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_caches(tmp_path, monkeypatch):
    """Redirect both the in-memory and on-disk snapshot caches.

    Each test gets a private temp directory for ``snapshot_*.parquet``
    files and a fresh in-memory instance cache so the order of test
    execution can never leak state across cases.
    """
    snapshot_mod._reset_instance_cache_for_testing()
    monkeypatch.setattr(snapshot_mod, "get_default_cache_dir", lambda: tmp_path)
    yield
    snapshot_mod._reset_instance_cache_for_testing()


def _chart_data_payload(dataset_id: str = "ds_live_1") -> dict:
    """A minimal but well-formed ``/datasets/chart-data`` response."""
    return {
        "success": True,
        "datasets": [
            {
                "dataset_id": dataset_id,
                "name": f"{dataset_id} dataset",
                "demographics": {"subjects_count": 12},
                "total_files": 30,
                "tasks": ["rest"],
                "sessions": ["s1"],
                "recording_modality": ["eeg"],
                "tags": {"modality": ["visual"]},
                "size_bytes": 1024,
                "source": "openneuro",
            }
        ],
        "aggregations": {
            "totals": {"datasets": 1, "subjects": 12},
            "modality_counts": {"eeg": 1},
            "source_counts": {"openneuro": 1},
        },
    }


def _summary_payload(dataset_id: str = "ds_summary_1") -> dict:
    """A minimal ``/datasets/summary`` response (legacy shape)."""
    return {
        "success": True,
        "data": [
            {
                "dataset_id": dataset_id,
                "name": f"{dataset_id} dataset",
                "demographics": {"subjects_count": 5},
                "total_files": 10,
                "tasks": ["task"],
                "recording_modality": ["eeg"],
                "tags": {},
            }
        ],
    }


def _make_urlopen_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    return response


def _server_manifest(
    dataset_count: int = 1,
    schema_version: str = "2.1.0",
    **overrides,
) -> dict:
    """A minimal but well-formed ``/build-manifest`` server response.

    Mirrors the production payload shape:
    ``{dataset_count, last_ingested_at, last_stats_computed_at,
    schema_version, git_sha, name_coverage}``.
    """
    base = {
        "dataset_count": dataset_count,
        "last_ingested_at": "2026-04-18T16:10:52.827000Z",
        "last_stats_computed_at": "2026-05-10T19:09:03.501782Z",
        "schema_version": schema_version,
        "git_sha": "unknown",
        "name_coverage": 0.03,
    }
    base.update(overrides)
    return base


def _routing_urlopen(routes: dict[str, dict]):
    """Build a urlopen side_effect that picks payloads by URL substring.

    The earliest-matching key in iteration order wins (Python 3.7+
    dicts preserve insertion order), so put more-specific paths first.
    """

    def urlopen_side_effect(url, *_, **__):
        for needle, payload in routes.items():
            if needle in url:
                return _make_urlopen_response(payload)
        raise AssertionError(f"unexpected URL: {url}")

    return urlopen_side_effect


# ---------------------------------------------------------------------------
# Required cases per the task spec (≥ 5)
# ---------------------------------------------------------------------------


def test_build_live_via_stubbed_http():
    """Happy path: chart-data returns success → source == "live".

    Offline-safe equivalent of the ``test_build_live`` case in the
    task spec. The real-network version is at the bottom of this file
    behind ``@pytest.mark.network``.

    Stubs both the chart-data and build-manifest endpoints; the
    snapshot's :attr:`manifest` carries the server's response verbatim
    when reachable (see A2).
    """
    side_effect = _routing_urlopen(
        {
            "datasets/chart-data": _chart_data_payload("ds_live_1"),
            "build-manifest": _server_manifest(dataset_count=1),
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


def test_cached_on_api_failure_after_priming(tmp_path):
    """First live call primes the disk cache; second call (forced
    failure) reads the cache and tags source == "cached".
    """
    side_effect = _routing_urlopen(
        {
            "datasets/chart-data": _chart_data_payload("ds_cached_1"),
            "build-manifest": _server_manifest(dataset_count=1),
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
    # mtime of the parquet file is the snapshot's fetched_at.
    parquet_path = snapshot_mod._disk_cache_path(database)
    assert parquet_path.exists()
    expected = datetime.fromtimestamp(parquet_path.stat().st_mtime, tz=timezone.utc)
    assert cached.fetched_at == expected


def test_csv_fallback_on_no_cache(tmp_path):
    """No disk cache + dead API + present package CSV → source == "package-csv"."""
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


def test_csv_fallback_when_package_csv_missing(monkeypatch):
    """When neither network nor disk cache nor package CSV resolves,
    the snapshot still returns a value (source=package-csv, empty
    DataFrame) with the failure recorded — never raise.
    """
    monkeypatch.setattr(snapshot_mod, "PACKAGE_CSV_PATH", Path("/does/not/exist.csv"))

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


def test_cache_keyed_by_api_base_and_database():
    """Two builds with different ``api_base`` must NOT share an instance.

    Reproduces the bug the unkeyed ``_DATASET_SUMMARY_CACHE`` global in
    ``conf.py`` had: any caller that pointed at a different shard
    silently got stale data from the previous caller.
    """
    payload_a = _chart_data_payload("ds_from_a")
    payload_b = _chart_data_payload("ds_from_b")

    def urlopen_side_effect(url, *_, **__):
        if "shard_a" in url:
            return _make_urlopen_response(payload_a)
        if "shard_b" in url:
            return _make_urlopen_response(payload_b)
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


# ---------------------------------------------------------------------------
# Surface checks for the public API
# ---------------------------------------------------------------------------


def test_public_surface_complete():
    """Sanity check: every member the validation plan §3 step 5 names
    is discoverable on the class.
    """
    required = {
        "build",
        "load",
        "rows",
        "aggregations",
        "montage",
        "source",
        "fetched_at",
        "dataset_count",
        "manifest",
        "api_errors",
    }
    missing = (
        required
        - set(dir(DatasetSnapshot))
        - {a for a in required if hasattr(DatasetSnapshot, a)}
    )
    assert not missing, f"missing required members: {missing}"


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


def test_provenance_logged_once(caplog):
    """:meth:`DatasetSnapshot.build` emits exactly one I8-formatted
    INFO line per build — the invariant the validation plan §2 grep's
    for.
    """
    side_effect = _routing_urlopen(
        {
            "datasets/chart-data": _chart_data_payload("ds_log_1"),
            "build-manifest": _server_manifest(dataset_count=1),
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


def test_load_roundtrip(tmp_path):
    """A snapshot written by build can be re-hydrated via load."""
    side_effect = _routing_urlopen(
        {
            "datasets/chart-data": _chart_data_payload("ds_roundtrip"),
            "build-manifest": _server_manifest(dataset_count=1),
        }
    )
    api_base = "https://stub.example"
    database = "loadshard"
    with patch("urllib.request.urlopen", side_effect=side_effect):
        built = DatasetSnapshot.build(api_base=api_base, database=database)

    parquet_path = snapshot_mod._disk_cache_path(database)
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
    parquet_dst = manifest_path.with_suffix(".parquet")
    parquet_dst.write_bytes(parquet_path.read_bytes())

    loaded = DatasetSnapshot.load(manifest_path)
    assert loaded.source == built.source
    assert loaded.dataset_count == built.dataset_count
    assert loaded.rows().iloc[0]["dataset"] == "ds_roundtrip"


def test_montage_returns_none_when_absent():
    """``montage()`` must return ``None`` rather than raising for an
    unknown id, an empty id, or a snapshot that never received montages.
    """
    snapshot = DatasetSnapshot(
        rows=pd.DataFrame([{"dataset": "ds_montage"}]),
        aggregations={},
        montages={},
        source="live",
        fetched_at=datetime.now(timezone.utc),
    )
    assert snapshot.montage("ds_montage") is None
    assert snapshot.montage("") is None


def test_montage_returns_data_when_present():
    """When chart-data is fetched with ``?include=montages``, the
    montages land on the snapshot; surface them through ``montage()``
    keyed by dataset_id.
    """
    snapshot = DatasetSnapshot(
        rows=pd.DataFrame([{"dataset": "ds_montage"}]),
        aggregations={},
        montages={"ds_montage": {"name": "standard_1020", "n_channels": 64}},
        source="live",
        fetched_at=datetime.now(timezone.utc),
    )
    montage = snapshot.montage("ds_montage")
    assert montage is not None
    assert montage["name"] == "standard_1020"
    # Returned value is a copy — mutations don't leak back to the cache.
    montage["mutated"] = True
    assert "mutated" not in snapshot.montage("ds_montage")


def test_montage_lookup_is_case_insensitive():
    """Consumer paths sometimes lowercase the dataset id before
    lookup (the dataset_page section does); other consumers pass it
    through unchanged. Both must hit the same montage row.
    """
    snapshot = DatasetSnapshot(
        rows=pd.DataFrame([{"dataset": "DS001785"}]),
        aggregations={},
        # Snapshot stores keys lowercased — the standard convention from
        # ``_montages_from_chart_data``.
        montages={"ds001785": {"hash": "abc", "n_channels": 63}},
        source="live",
        fetched_at=datetime.now(timezone.utc),
    )
    assert snapshot.montage("ds001785") == {"hash": "abc", "n_channels": 63}
    assert snapshot.montage("DS001785") == {"hash": "abc", "n_channels": 63}


def test_chart_data_request_includes_montages_param():
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
            return _make_urlopen_response(_chart_data_payload("ds_url_check"))
        if "build-manifest" in url_str:
            return _make_urlopen_response(_server_manifest(dataset_count=1))
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


def test_snapshot_populates_montages_when_included():
    """Stub chart-data with montage objects on a couple of datasets;
    ``snapshot.montage(dataset_id)`` must return the projected dict.

    The projection layers viewer-friendly aliases (``label``,
    ``n_channels``, ``montage_id``) on top of the registry doc — kept
    backward-compatible with the retired build_electrode_layouts.py
    output so the consumer is a no-op rename.
    """
    payload = _chart_data_payload("ds001785")
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

    side_effect = _routing_urlopen(
        {
            "datasets/chart-data": payload,
            "build-manifest": _server_manifest(dataset_count=2),
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


def test_snapshot_montages_persist_through_disk_cache(tmp_path):
    """A live build writes a montages sidecar next to the parquet
    rows; a subsequent ``cached`` resolution rehydrates it so the
    docs build doesn't lose montage data the moment the API blips.
    """
    payload = _chart_data_payload("ds_persisted")
    payload["datasets"][0]["montage"] = {
        "hash": "deadbeef00000000",
        "modality": "eeg",
        "n_sensors": 32,
        "channel_names": ["A1"],
    }

    side_effect = _routing_urlopen(
        {
            "datasets/chart-data": payload,
            "build-manifest": _server_manifest(dataset_count=1),
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


def test_snapshot_montages_empty_when_summary_fallback_fires():
    """When chart-data is unavailable and the snapshot falls back to
    ``/datasets/summary`` (which has no montage data), ``montage()``
    must return ``None`` for every dataset rather than raise.
    """

    def urlopen_side_effect(url, *_, **__):
        url_str = url if isinstance(url, str) else url.get_full_url()
        if "datasets/chart-data" in url_str:
            raise urllib.error.HTTPError(url_str, 404, "no chart-data", {}, None)
        if "datasets/summary" in url_str:
            return _make_urlopen_response(_summary_payload("ds_summary_fallback"))
        if "build-manifest" in url_str:
            return _make_urlopen_response(_server_manifest(dataset_count=1))
        raise AssertionError(f"unexpected URL: {url_str}")

    with patch("urllib.request.urlopen", side_effect=urlopen_side_effect):
        snapshot = DatasetSnapshot.build(
            api_base="https://stub.example", database="summaryonlyshard"
        )

    assert snapshot.source == "live"
    assert snapshot.dataset_count == 1
    assert snapshot.montage("ds_summary_fallback") is None
    # The chart-data 404 is recorded as context, but the build
    # succeeded on the summary fallback.
    assert any("chart-data 404" in e for e in snapshot.api_errors)


def test_package_csv_fallback_skips_disk_cache_write(tmp_path):
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
# EXCLUDED_DATASETS — single source of truth across registry / snapshot
# ---------------------------------------------------------------------------


def test_excluded_datasets_is_single_source_of_truth():
    """``registry`` and ``snapshot`` must share the exact same set
    object — the B1 refactor had drifted a snapshot-local copy that
    silently became the only effective filter for the docs build.

    Regression test for the P1 code-review finding: snapshot's 21-entry
    subset was missing 20 entries from the registry's curated list AND
    contained 4 entries (``AGUS``, ``ALI``, ``ALYTUS``, ``AMERICO``)
    that the registry never excluded.
    """
    from eegdash.dataset._excluded import EXCLUDED_DATASETS as canonical
    from eegdash.dataset.registry import EXCLUDED_DATASETS as via_registry
    from eegdash.dataset.snapshot import EXCLUDED_DATASETS as via_snapshot

    # Identity, not just equality — the whole point is that there is
    # one set object in memory and both modules re-export it.
    assert via_registry is canonical
    assert via_snapshot is canonical
    assert via_registry is via_snapshot


def test_excluded_datasets_canonical_membership():
    """Canonical content check: the 37-entry curated registry list.

    Locks in:
    - The size matches the pre-refactor production filter.
    - Entries from each of the formerly-divergent groups (the long
      ``ABUDUKADI_n`` / ``AILIJIANG_n`` / ``BAIHETI_n`` / etc.
      families that the snapshot copy had dropped) are present.
    - The four entries that should NEVER have been excluded (the
      snapshot-only additions ``AGUS``, ``ALI``, ``ALYTUS``,
      ``AMERICO``) are absent.
    """
    from eegdash.dataset._excluded import EXCLUDED_DATASETS

    assert len(EXCLUDED_DATASETS) == 37, (
        f"canonical filter size changed: {len(EXCLUDED_DATASETS)}; "
        "if intentional, update this test alongside the change"
    )

    # Representative entries from each formerly-divergent family.
    must_be_present = {
        "BIAN_3",
        "BOJIN",
        "AISHENG",
        "ABUDUKADI_2",
        "AILIJIANG_3",
        "BAIHETI",
        "BLIX",
        "BOUSSAGOL",
        "ACHOLA",
        "ANASHKIN",
    }
    missing = must_be_present - set(EXCLUDED_DATASETS)
    assert not missing, f"canonical entries dropped: {sorted(missing)}"

    # Entries that the snapshot copy had wrongly added — NEVER include.
    must_be_absent = {"AGUS", "ALI", "ALYTUS", "AMERICO"}
    accidental = must_be_absent & set(EXCLUDED_DATASETS)
    assert not accidental, (
        f"snapshot-only additions leaked into the canonical filter: "
        f"{sorted(accidental)}"
    )


# ---------------------------------------------------------------------------
# CI gate: check_snapshot_health.evaluate_snapshot
# ---------------------------------------------------------------------------


def _load_check_snapshot_health():
    """Load ``scripts/validation/check_snapshot_health.py`` as a module.

    The script lives outside the importable package tree (``scripts/``
    has no ``__init__.py`` on purpose: it's a CLI directory, not a
    library). For tests we load the file directly via ``importlib``
    rather than mutating ``sys.path`` globally.
    """
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "validation" / "check_snapshot_health.py"
    spec = importlib.util.spec_from_file_location(
        "check_snapshot_health_under_test", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gate_snapshot(*, source="live", dataset_count=1000, api_errors=()):
    """Build a minimal snapshot with the provenance flags the gate
    reads, without hitting the build/fetch path.

    Only the four attributes ``evaluate_snapshot`` consults are
    populated; the rest of the object is left at its constructor
    defaults.
    """
    rows = pd.DataFrame([{"dataset": f"ds_{i}"} for i in range(dataset_count)])
    return DatasetSnapshot(
        rows=rows,
        aggregations={},
        montages={},
        source=source,
        fetched_at=datetime.now(timezone.utc),
        api_errors=list(api_errors),
    )


def test_ci_gate_passes_clean_live_snapshot():
    """Source=live, count over threshold, api_errors empty → OK."""
    module = _load_check_snapshot_health()

    snap = _gate_snapshot(source="live", dataset_count=800, api_errors=[])
    assert module.evaluate_snapshot(snap, min_count=700) == []


def test_ci_gate_fails_on_fallback_source():
    """A cached snapshot is degraded, even with a full row count."""
    module = _load_check_snapshot_health()

    snap = _gate_snapshot(source="cached", dataset_count=900, api_errors=[])
    failures = module.evaluate_snapshot(snap, min_count=700)
    assert any("source" in msg for msg in failures), failures


def test_ci_gate_fails_on_low_count():
    """Row count at or below the floor must fail the gate."""
    module = _load_check_snapshot_health()

    snap = _gate_snapshot(source="live", dataset_count=700, api_errors=[])
    failures = module.evaluate_snapshot(snap, min_count=700)
    assert any("dataset_count" in msg for msg in failures), failures


def test_ci_gate_fails_on_partial_degradation_api_errors():
    """The motivating regression: chart-data dead, summary alive.

    ``_build_uncached`` records the chart-data failure on
    ``api_errors`` and then tags the snapshot ``source="live"``
    because the summary endpoint saved the day. The pre-fix gate
    looked only at ``source`` and ``dataset_count`` and let this
    through — losing the aggregations block silently on the docs
    site. The fixed gate catches it.
    """
    module = _load_check_snapshot_health()

    snap = _gate_snapshot(
        source="live",
        dataset_count=900,
        api_errors=[
            "chart-data 404 at https://api.example/db/datasets/chart-data; "
            "trying summary"
        ],
    )
    failures = module.evaluate_snapshot(snap, min_count=700)
    assert any("api_errors" in msg for msg in failures), (
        f"partial degradation did not trip the gate; failures={failures}"
    )


def test_ci_gate_api_errors_ignored_on_fallback_source():
    """When the snapshot has already fallen back, ``api_errors`` will
    obviously be populated (every fallback path appends to it). The
    source-tag check has already failed, so we should not double-count
    the same degradation. This locks in that ``api_errors`` only
    contributes a *new* failure when ``source == "live"``.
    """
    module = _load_check_snapshot_health()

    snap = _gate_snapshot(
        source="package-csv",
        dataset_count=900,
        api_errors=["chart-data error at ...", "summary error at ..."],
    )
    failures = module.evaluate_snapshot(snap, min_count=700)
    # Exactly one failure: the source-tag failure. No api_errors line.
    assert len(failures) == 1, failures
    assert "source" in failures[0]


# ---------------------------------------------------------------------------
# /build-manifest server integration (arch #4)
# ---------------------------------------------------------------------------


def test_manifest_uses_server_when_available():
    """``snapshot.manifest`` carries the server's ``/build-manifest``
    response verbatim — including the keys the local projection never
    populated (``schema_version``, ``last_ingested_at``, ``git_sha``,
    ``name_coverage``).
    """
    server_payload = _server_manifest(
        dataset_count=1, schema_version="2.1.0", git_sha="abc1234"
    )

    def fake_fetch_manifest(api_base, database, *, errors=None, timeout=5.0):
        return dict(server_payload)

    side_effect = _routing_urlopen(
        {"datasets/chart-data": _chart_data_payload("ds_server_manifest")}
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


def test_manifest_falls_back_on_manifest_error():
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

    side_effect = _routing_urlopen(
        {"datasets/chart-data": _chart_data_payload("ds_manifest_down")}
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


def test_manifest_count_mismatch_tagged():
    """Server's ``dataset_count`` disagreeing with the snapshot's row
    count is surfaced on ``api_errors`` (never raised) so the B2 CI gate
    can refuse to publish.
    """
    server_payload = _server_manifest(dataset_count=9999, schema_version="2.1.0")

    def fake_fetch_manifest(api_base, database, *, errors=None, timeout=5.0):
        return dict(server_payload)

    side_effect = _routing_urlopen(
        {"datasets/chart-data": _chart_data_payload("ds_skew")}
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
