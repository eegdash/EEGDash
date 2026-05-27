"""Tests for the Pydantic-settings inject config (C6.5).

Replaces the would-be 460-line argparse subprocess test with direct
construction + validation of :class:`InjectConfig`. Same coverage,
faster, no monkey-patching.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from pydantic import ValidationError

_INGEST_DIR = Path(__file__).resolve().parent.parent
from _inject_config import (
    DEFAULT_API_URL,
    LOCAL_FALLBACK_DATABASES,
    InjectConfig,
    _valid_databases_cache,
    clear_valid_databases_cache,
    fetch_valid_databases_from_api,
    load_inject_config_from_argv,
)

# ─── Autouse fixture: prime the cache so non-respx tests don't hit network ─


@pytest.fixture(autouse=True)
def _prime_valid_databases_cache():
    """Most existing tests don't mock the network. Pre-seed the cache
    for DEFAULT_API_URL with the local fallback so the database
    field_validator doesn't fire a real HTTP call (and time out)
    against a domain we can't reach in CI.

    Tests that exercise the network path explicitly (respx-decorated)
    call clear_valid_databases_cache() themselves at the start.
    """
    clear_valid_databases_cache()
    _valid_databases_cache[DEFAULT_API_URL] = LOCAL_FALLBACK_DATABASES
    yield
    clear_valid_databases_cache()


# ─── Defaults + required fields ───────────────────────────────────────────


def test_config_requires_database():
    """``database`` has no default — must be supplied."""
    with pytest.raises(ValidationError) as exc:
        InjectConfig(dry_run=True)
    assert "database" in str(exc.value).lower()


def test_config_rejects_invalid_database():
    """``database`` must match the ValidDatabase Literal set."""
    with pytest.raises(ValidationError) as exc:
        InjectConfig(database="not_a_real_db", dry_run=True)
    msg = str(exc.value)
    assert "database" in msg.lower()


def test_config_accepts_each_valid_database():
    """All 5 documented database names accepted."""
    for db in (
        "eegdash",
        "eegdash_dev",
        "eegdash_archive",
        "eegdash_staging",
        "eegdash_v1",
    ):
        c = InjectConfig(database=db, dry_run=True)
        assert c.database == db


def test_config_defaults():
    """Field defaults match the documented behaviour."""
    c = InjectConfig(database="eegdash_dev", dry_run=True)
    assert c.api_url == DEFAULT_API_URL
    assert c.input == Path("digestion_output")
    assert c.batch_size == 1000
    assert c.dry_run is True
    assert c.only_datasets is False
    assert c.only_records is False
    assert c.only_montages is False
    assert c.skip_montages is False
    assert c.force is False
    assert c.skip_validation is False
    assert c.data_quality_threshold == 10.0
    assert c.compute_stats is False
    assert c.datasets is None


# ─── Bounds ───────────────────────────────────────────────────────────────


def test_batch_size_lower_bound():
    """``batch_size`` must be ≥ 1."""
    with pytest.raises(ValidationError):
        InjectConfig(database="eegdash_dev", batch_size=0, dry_run=True)


def test_batch_size_upper_bound():
    """``batch_size`` must be ≤ 10_000 (avoids accidentally trying to
    upload a 100k-record batch that the Gateway would reject)."""
    with pytest.raises(ValidationError):
        InjectConfig(database="eegdash_dev", batch_size=20_000, dry_run=True)


def test_data_quality_threshold_bounds():
    """``data_quality_threshold`` is a percentage 0-100."""
    InjectConfig(database="eegdash_dev", data_quality_threshold=0.0, dry_run=True)
    InjectConfig(database="eegdash_dev", data_quality_threshold=100.0, dry_run=True)
    with pytest.raises(ValidationError):
        InjectConfig(
            database="eegdash_dev",
            data_quality_threshold=-1.0,
            dry_run=True,
        )
    with pytest.raises(ValidationError):
        InjectConfig(
            database="eegdash_dev",
            data_quality_threshold=101.0,
            dry_run=True,
        )


# ─── Mutually-exclusive only-* flags ──────────────────────────────────────


def test_two_only_flags_rejected():
    """Any pair of --only-* flags raises."""
    with pytest.raises(ValidationError) as exc:
        InjectConfig(
            database="eegdash_dev",
            only_datasets=True,
            only_records=True,
            dry_run=True,
        )
    assert "mutually exclusive" in str(exc.value).lower()


def test_three_only_flags_rejected():
    """All 3 --only-* set is the worst case → rejected."""
    with pytest.raises(ValidationError):
        InjectConfig(
            database="eegdash_dev",
            only_datasets=True,
            only_records=True,
            only_montages=True,
            dry_run=True,
        )


def test_each_single_only_flag_accepted():
    """Exactly one --only-* flag at a time is fine."""
    for flag in ("only_datasets", "only_records", "only_montages"):
        kw = {"database": "eegdash_dev", "dry_run": True, flag: True}
        c = InjectConfig(**kw)
        assert getattr(c, flag) is True


# ─── only-montages + skip-montages contradict ─────────────────────────────


def test_only_and_skip_montages_contradict():
    with pytest.raises(ValidationError) as exc:
        InjectConfig(
            database="eegdash_dev",
            only_montages=True,
            skip_montages=True,
            dry_run=True,
        )
    assert "contradict" in str(exc.value).lower()


# ─── input must exist (unless dry-run) ────────────────────────────────────


def test_missing_input_dir_rejected_when_not_dry_run(tmp_path: Path):
    """A real run needs a real --input dir."""
    with pytest.raises(ValidationError) as exc:
        InjectConfig(
            database="eegdash_dev",
            input=tmp_path / "does_not_exist",
            dry_run=False,
        )
    assert "input" in str(exc.value).lower()


def test_missing_input_dir_allowed_in_dry_run(tmp_path: Path):
    """Dry-run skips the existence check — useful for --help-style runs."""
    InjectConfig(
        database="eegdash_dev",
        input=tmp_path / "does_not_exist",
        dry_run=True,
    )


def test_existing_input_dir_accepted(tmp_path: Path):
    """Normal real-run case."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    c = InjectConfig(database="eegdash_dev", input=real_dir, dry_run=False)
    assert c.input == real_dir


# ─── Env var fallback for token ───────────────────────────────────────────


def test_token_explicit_arg_used_when_provided():
    c = InjectConfig(database="eegdash_dev", token="explicit", dry_run=True)
    assert c.token == "explicit"


def test_token_reads_from_eegdash_admin_token_env(monkeypatch):
    """When --token is missing, read EEGDASH_ADMIN_TOKEN env var
    (matches the legacy ops-script convention)."""
    monkeypatch.setenv("EEGDASH_ADMIN_TOKEN", "from_env")
    c = InjectConfig(database="eegdash_dev", dry_run=True)
    assert c.token == "from_env"


def test_token_none_when_neither_arg_nor_env(monkeypatch):
    monkeypatch.delenv("EEGDASH_ADMIN_TOKEN", raising=False)
    c = InjectConfig(database="eegdash_dev", dry_run=True)
    assert c.token is None


# ─── Convenience accessors (want_datasets / want_records / want_montages) ─


def test_want_accessors_default_true():
    """Default config injects everything."""
    c = InjectConfig(database="eegdash_dev", dry_run=True)
    assert c.want_datasets is True
    assert c.want_records is True
    assert c.want_montages is True


def test_only_datasets_filters_other_paths():
    c = InjectConfig(database="eegdash_dev", only_datasets=True, dry_run=True)
    assert c.want_datasets is True
    assert c.want_records is False
    assert c.want_montages is False


def test_only_records_filters_other_paths():
    c = InjectConfig(database="eegdash_dev", only_records=True, dry_run=True)
    assert c.want_datasets is False
    assert c.want_records is True
    assert c.want_montages is False


def test_only_montages_filters_other_paths():
    c = InjectConfig(database="eegdash_dev", only_montages=True, dry_run=True)
    assert c.want_datasets is False
    assert c.want_records is False
    assert c.want_montages is True


def test_skip_montages_disables_montage_only():
    """``--skip-montages`` only affects the montage leg; datasets +
    records still inject."""
    c = InjectConfig(database="eegdash_dev", skip_montages=True, dry_run=True)
    assert c.want_datasets is True
    assert c.want_records is True
    assert c.want_montages is False


# ─── CLI parsing ──────────────────────────────────────────────────────────


def test_argv_parses_minimum_required():
    c = load_inject_config_from_argv(["--database", "eegdash_dev", "--dry-run"])
    assert c.database == "eegdash_dev"
    assert c.dry_run is True


def test_argv_parses_all_string_flags():
    c = load_inject_config_from_argv(
        [
            "--database",
            "eegdash_dev",
            "--api-url",
            "https://test.example.com",
            "--token",
            "my-token",
            "--batch-size",
            "500",
            "--dry-run",
        ]
    )
    assert c.api_url == "https://test.example.com"
    assert c.token == "my-token"
    assert c.batch_size == 500


def test_argv_parses_only_datasets_flag():
    c = load_inject_config_from_argv(
        ["--database", "eegdash_dev", "--only-datasets", "--dry-run"]
    )
    assert c.only_datasets is True
    assert c.want_records is False


def test_argv_parses_dataset_filter_list():
    c = load_inject_config_from_argv(
        [
            "--database",
            "eegdash_dev",
            "--datasets",
            "ds-001",
            "ds-002",
            "ds-003",
            "--dry-run",
        ]
    )
    assert c.datasets == ["ds-001", "ds-002", "ds-003"]


def test_argv_validation_surfaces_via_validation_error():
    """An argparse-parsed config that fails validation raises
    ``pydantic.ValidationError`` (cleaner than argparse's stderr noise)."""
    with pytest.raises(ValidationError):
        load_inject_config_from_argv(
            [
                "--database",
                "eegdash_dev",
                "--only-datasets",
                "--only-records",
                "--dry-run",
            ]
        )


def test_argv_unknown_flag_rejected_by_argparse():
    """Unknown flags still produce argparse's SystemExit (preserves the
    CLI ergonomics)."""
    with pytest.raises(SystemExit):
        load_inject_config_from_argv(
            ["--database", "eegdash_dev", "--bogus-flag", "--dry-run"]
        )


# ─── fetch_valid_databases_from_api ────────────────────────────────────────


@respx.mock
def test_fetch_valid_databases_returns_api_list_on_200():
    """Happy path: API returns {"databases": [...]} → frozenset of names."""
    api_url = "https://api.example.test"
    respx.get(f"{api_url}/admin/valid-databases").mock(
        return_value=httpx.Response(
            200, json={"databases": ["eegdash", "eegdash_dev", "eegdash_v2"]}
        )
    )

    result = fetch_valid_databases_from_api(api_url, token="dummy")

    assert result == frozenset({"eegdash", "eegdash_dev", "eegdash_v2"})


@respx.mock
def test_fetch_valid_databases_is_cached_per_api_url():
    """Second call to the same api_url should NOT re-hit the server."""
    from _inject_config import clear_valid_databases_cache

    clear_valid_databases_cache()
    api_url = "https://cache-test.example"
    route = respx.get(f"{api_url}/admin/valid-databases").mock(
        return_value=httpx.Response(200, json={"databases": ["eegdash"]})
    )

    fetch_valid_databases_from_api(api_url, token=None)
    fetch_valid_databases_from_api(api_url, token=None)
    fetch_valid_databases_from_api(api_url, token=None)

    assert route.call_count == 1


@respx.mock
def test_fetch_valid_databases_returns_none_on_404():
    """Endpoint doesn't exist on this server -> None (caller falls back)."""
    from _inject_config import clear_valid_databases_cache

    clear_valid_databases_cache()
    api_url = "https://no-endpoint.example"
    respx.get(f"{api_url}/admin/valid-databases").mock(
        return_value=httpx.Response(404, json={"detail": "not found"})
    )
    assert fetch_valid_databases_from_api(api_url, token="x") is None


@respx.mock
def test_fetch_valid_databases_returns_none_on_network_error():
    """Connection error -> None."""
    from _inject_config import clear_valid_databases_cache

    clear_valid_databases_cache()
    api_url = "https://network-fail.example"
    respx.get(f"{api_url}/admin/valid-databases").mock(
        side_effect=httpx.ConnectError("boom")
    )
    assert fetch_valid_databases_from_api(api_url, token="x") is None


@respx.mock
def test_fetch_valid_databases_returns_none_on_missing_key():
    """Server returns 200 but payload shape is wrong -> None."""
    from _inject_config import clear_valid_databases_cache

    clear_valid_databases_cache()
    api_url = "https://bad-shape.example"
    respx.get(f"{api_url}/admin/valid-databases").mock(
        return_value=httpx.Response(200, json={"unexpected": "shape"})
    )
    assert fetch_valid_databases_from_api(api_url, token="x") is None


# ─── InjectConfig integration with the API-fetch field validator ──────────


@respx.mock
def test_inject_config_rejects_unknown_database_via_local_fallback(tmp_path):
    """Without network access, an unknown database is rejected by
    LOCAL_FALLBACK_DATABASES."""
    from _inject_config import clear_valid_databases_cache

    clear_valid_databases_cache()
    # Simulate the API endpoint being absent (the planned follow-up state)
    # so the validator falls back to LOCAL_FALLBACK_DATABASES.
    respx.get(f"{DEFAULT_API_URL}/admin/valid-databases").mock(
        return_value=httpx.Response(404, json={"detail": "not found"})
    )
    with pytest.raises(ValidationError) as exc:
        InjectConfig(
            database="eegdash_does_not_exist",
            input=tmp_path,
            dry_run=True,
        )
    assert "valid set" in str(exc.value)


@respx.mock
def test_inject_config_accepts_database_only_in_api_set(tmp_path):
    """An API that knows about a new database lets us inject to it
    even when LOCAL_FALLBACK_DATABASES does not."""
    from _inject_config import clear_valid_databases_cache

    clear_valid_databases_cache()
    respx.get(f"{DEFAULT_API_URL}/admin/valid-databases").mock(
        return_value=httpx.Response(
            200,
            json={"databases": ["eegdash", "eegdash_dev", "eegdash_v99_future"]},
        )
    )

    c = InjectConfig(
        database="eegdash_v99_future",
        input=tmp_path,
        dry_run=True,
    )
    assert c.database == "eegdash_v99_future"


@respx.mock
def test_inject_config_accepts_local_name_when_api_set_omits_it(tmp_path):
    """Union semantics: a name in LOCAL_FALLBACK_DATABASES is accepted
    even when the API set doesn't list it (API-side deprecation does
    not break long-running scripts at config-construction time)."""
    from _inject_config import clear_valid_databases_cache

    clear_valid_databases_cache()
    # API only returns 2 databases; eegdash_archive is in LOCAL_FALLBACK
    # but NOT in the API response.
    respx.get(f"{DEFAULT_API_URL}/admin/valid-databases").mock(
        return_value=httpx.Response(200, json={"databases": ["eegdash", "eegdash_dev"]})
    )

    c = InjectConfig(
        database="eegdash_archive",  # in LOCAL_FALLBACK, not in api_set
        input=tmp_path,
        dry_run=True,
    )
    assert c.database == "eegdash_archive"


@respx.mock
def test_fetch_valid_databases_caches_failure_to_avoid_repeated_network_hits():
    """After a network failure, subsequent calls must return None
    WITHOUT re-hitting the network."""
    from _inject_config import clear_valid_databases_cache

    clear_valid_databases_cache()
    api_url = "https://failure-cache.example"
    route = respx.get(f"{api_url}/admin/valid-databases").mock(
        side_effect=httpx.ConnectError("boom")
    )

    # First call — hits the network, fails, caches the failure
    assert fetch_valid_databases_from_api(api_url, token=None) is None
    # Second + third calls — must be cached, route call_count stays 1
    assert fetch_valid_databases_from_api(api_url, token=None) is None
    assert fetch_valid_databases_from_api(api_url, token=None) is None

    assert route.call_count == 1
