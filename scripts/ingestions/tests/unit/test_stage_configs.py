"""Tests for the Pydantic-settings stage configs (clone / inject / cross-stage invariants)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from pydantic import ValidationError

from _clone_config import (
    KNOWN_SOURCES,
    CloneConfig,
    load_clone_config_from_argv,
)
from _digest_config import (
    DEFAULT_DATASET_TIMEOUT_SECONDS,
    DigestConfig,
    load_digest_config_from_argv,
)
from _inject_config import (
    DEFAULT_API_URL,
    LOCAL_FALLBACK_DATABASES,
    InjectConfig,
    _valid_databases_cache,
    clear_valid_databases_cache,
    fetch_valid_databases_from_api,
    load_inject_config_from_argv,
)
from _validate_config import (
    ValidateConfig,
    load_validate_config_from_argv,
)

# ─── 1. Clone config ──────────────────────────────────────────────


def test_clone_config_defaults(tmp_path: Path):
    c = CloneConfig(input=tmp_path)
    assert c.input == tmp_path
    assert c.output == Path("data/cloned")
    assert c.sources is None
    assert c.timeout == 300
    assert c.workers == 8
    assert c.limit is None
    assert c.limit_per_source is None
    assert c.datasets is None
    assert c.manifest_only is False


def test_clone_config_rejects_missing_input(tmp_path: Path):
    with pytest.raises(ValidationError):
        CloneConfig(input=tmp_path / "does_not_exist")


# ─── Bounds ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        pytest.param("workers", 0, id="workers_below_min"),
        pytest.param("workers", 9999, id="workers_above_max"),
        pytest.param("timeout", 0, id="timeout_below_min"),
        pytest.param("timeout", 60 * 60 * 24, id="timeout_above_6h"),
        pytest.param("limit", 0, id="limit_below_min"),
        pytest.param("limit_per_source", 0, id="limit_per_source_below_min"),
    ],
)
def test_clone_config_field_bounds_reject_invalid(
    tmp_path: Path, field: str, invalid_value: int
):
    with pytest.raises(ValidationError):
        CloneConfig(input=tmp_path, **{field: invalid_value})


def test_clone_config_workers_in_range_accepted(tmp_path: Path):
    c = CloneConfig(input=tmp_path, workers=64)
    assert c.workers == 64


# ─── Sources validation ───────────────────────────────────────────────────


def test_clone_config_accepts_known_sources(tmp_path: Path):
    c = CloneConfig(
        input=tmp_path,
        sources=["openneuro", "nemar", "zenodo"],
    )
    assert c.sources == ["openneuro", "nemar", "zenodo"]


def test_clone_config_rejects_unknown_source(tmp_path: Path):
    with pytest.raises(ValidationError) as exc:
        CloneConfig(input=tmp_path, sources=["openneuro", "made_up_source"])
    msg = str(exc.value)
    assert "made_up_source" in msg


@pytest.mark.parametrize("source", sorted(KNOWN_SOURCES))
def test_clone_config_accepts_known_source(tmp_path: Path, source: str):
    c = CloneConfig(input=tmp_path, sources=[source])
    assert c.sources == [source]


# ─── CLI parsing ──────────────────────────────────────────────────────────


def test_clone_argv_parses_all_flags(tmp_path: Path):
    out_dir = tmp_path / "out"
    c = load_clone_config_from_argv(
        [
            "--input",
            str(tmp_path),
            "--output",
            str(out_dir),
            "--sources",
            "openneuro",
            "nemar",
            "--timeout",
            "600",
            "--workers",
            "4",
            "--limit",
            "10",
            "--limit-per-source",
            "5",
            "--datasets",
            "ds-001",
            "--manifest-only",
        ]
    )
    assert c.input == tmp_path
    assert c.output == out_dir
    assert c.sources == ["openneuro", "nemar"]
    assert c.timeout == 600
    assert c.workers == 4
    assert c.limit == 10
    assert c.limit_per_source == 5
    assert c.datasets == ["ds-001"]
    assert c.manifest_only is True


def test_clone_argv_env_var_picked_up(tmp_path: Path, monkeypatch):
    """EEGDASH_CLONE_WORKERS=16 → workers via env."""
    monkeypatch.setenv("EEGDASH_CLONE_WORKERS", "16")
    c = load_clone_config_from_argv(["--input", str(tmp_path)])
    assert c.workers == 16


def test_clone_argv_validation_error_for_unknown_source_via_argparse(
    tmp_path: Path,
):
    """argparse choices=KNOWN_SOURCES rejects unknown sources via SystemExit before Pydantic."""
    with pytest.raises(SystemExit):
        load_clone_config_from_argv(
            [
                "--input",
                str(tmp_path),
                "--sources",
                "made_up_source",
            ]
        )


# ─── 2. Inject config ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _prime_valid_databases_cache():
    """Pre-seed the cache so the field_validator skips real HTTP calls in CI."""
    clear_valid_databases_cache()
    _valid_databases_cache[DEFAULT_API_URL] = LOCAL_FALLBACK_DATABASES
    yield
    clear_valid_databases_cache()


# ─── Defaults + required fields ───────────────────────────────────────────


def test_config_requires_database():
    with pytest.raises(ValidationError) as exc:
        InjectConfig(dry_run=True)
    assert "database" in str(exc.value).lower()


def test_config_rejects_invalid_database():
    with pytest.raises(ValidationError) as exc:
        InjectConfig(database="not_a_real_db", dry_run=True)
    msg = str(exc.value)
    assert "database" in msg.lower()


@pytest.mark.parametrize(
    "database",
    [
        "eegdash",
        "eegdash_dev",
        "eegdash_archive",
        "eegdash_staging",
        "eegdash_v1",
    ],
)
def test_config_accepts_each_valid_database(database: str):
    c = InjectConfig(database=database, dry_run=True)
    assert c.database == database


def test_config_defaults():
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


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        pytest.param("batch_size", 0, id="batch_size_lower"),
        pytest.param("batch_size", 20_000, id="batch_size_upper"),
        pytest.param("data_quality_threshold", -1.0, id="dqt_below_zero"),
        pytest.param("data_quality_threshold", 101.0, id="dqt_above_100"),
    ],
)
def test_inject_config_field_bounds_reject_invalid(field: str, invalid_value):
    with pytest.raises(ValidationError):
        InjectConfig(database="eegdash_dev", dry_run=True, **{field: invalid_value})


@pytest.mark.parametrize("threshold", [0.0, 50.0, 100.0])
def test_inject_data_quality_threshold_in_range_accepted(threshold: float):
    c = InjectConfig(
        database="eegdash_dev", data_quality_threshold=threshold, dry_run=True
    )
    assert c.data_quality_threshold == threshold


# ─── Mutually-exclusive only-* flags ──────────────────────────────────────


def test_two_only_flags_rejected():
    with pytest.raises(ValidationError) as exc:
        InjectConfig(
            database="eegdash_dev",
            only_datasets=True,
            only_records=True,
            dry_run=True,
        )
    assert "mutually exclusive" in str(exc.value).lower()


def test_three_only_flags_rejected():
    with pytest.raises(ValidationError):
        InjectConfig(
            database="eegdash_dev",
            only_datasets=True,
            only_records=True,
            only_montages=True,
            dry_run=True,
        )


def test_each_single_only_flag_accepted():
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
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    c = InjectConfig(database="eegdash_dev", input=real_dir, dry_run=False)
    assert c.input == real_dir


# ─── Env var fallback for token ───────────────────────────────────────────


def test_token_explicit_arg_used_when_provided():
    c = InjectConfig(database="eegdash_dev", token="explicit", dry_run=True)
    assert c.token == "explicit"


def test_token_reads_from_eegdash_admin_token_env(monkeypatch):
    """When --token is missing, read EEGDASH_ADMIN_TOKEN (legacy ops-script convention)."""
    monkeypatch.setenv("EEGDASH_ADMIN_TOKEN", "from_env")
    c = InjectConfig(database="eegdash_dev", dry_run=True)
    assert c.token == "from_env"


def test_token_none_when_neither_arg_nor_env(monkeypatch):
    monkeypatch.delenv("EEGDASH_ADMIN_TOKEN", raising=False)
    c = InjectConfig(database="eegdash_dev", dry_run=True)
    assert c.token is None


# ─── Convenience accessors (want_datasets / want_records / want_montages) ─


def test_want_accessors_default_true():
    c = InjectConfig(database="eegdash_dev", dry_run=True)
    assert c.want_datasets is True
    assert c.want_records is True
    assert c.want_montages is True


@pytest.mark.parametrize(
    ("flag", "want_datasets", "want_records", "want_montages"),
    [
        pytest.param("only_datasets", True, False, False, id="only_datasets"),
        pytest.param("only_records", False, True, False, id="only_records"),
        pytest.param("only_montages", False, False, True, id="only_montages"),
    ],
)
def test_only_flag_filters_other_paths(
    flag: str,
    want_datasets: bool,
    want_records: bool,
    want_montages: bool,
):
    c = InjectConfig(database="eegdash_dev", dry_run=True, **{flag: True})
    assert c.want_datasets is want_datasets
    assert c.want_records is want_records
    assert c.want_montages is want_montages


def test_skip_montages_disables_montage_only():
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
    with pytest.raises(SystemExit):
        load_inject_config_from_argv(
            ["--database", "eegdash_dev", "--bogus-flag", "--dry-run"]
        )


# ─── fetch_valid_databases_from_api ────────────────────────────────────────


@respx.mock
def test_fetch_valid_databases_returns_api_list_on_200():
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
    """Second call to the same api_url skips the network."""
    clear_valid_databases_cache()
    api_url = "https://cache-test.example"
    route = respx.get(f"{api_url}/admin/valid-databases").mock(
        return_value=httpx.Response(200, json={"databases": ["eegdash"]})
    )

    fetch_valid_databases_from_api(api_url, token=None)
    fetch_valid_databases_from_api(api_url, token=None)
    fetch_valid_databases_from_api(api_url, token=None)

    assert route.call_count == 1


@pytest.mark.parametrize(
    ("mock_kwargs", "expected"),
    [
        pytest.param(
            {"return_value": httpx.Response(404, json={"detail": "not found"})},
            None,
            id="returns_none_on_404",
        ),
        pytest.param(
            {"side_effect": httpx.ConnectError("boom")},
            None,
            id="returns_none_on_network_error",
        ),
        pytest.param(
            {"return_value": httpx.Response(200, json={"unexpected": "shape"})},
            None,
            id="returns_none_on_missing_key",
        ),
    ],
)
@respx.mock
def test_fetch_valid_databases_returns_none_on_error(mock_kwargs, expected):
    clear_valid_databases_cache()
    api_url = "https://error-case.example"
    respx.get(f"{api_url}/admin/valid-databases").mock(**mock_kwargs)
    assert fetch_valid_databases_from_api(api_url, token="x") is expected


# ─── InjectConfig integration with the API-fetch field validator ──────────


@respx.mock
def test_inject_config_rejects_unknown_database_via_local_fallback(tmp_path):
    clear_valid_databases_cache()
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
    """API response extends the valid set beyond LOCAL_FALLBACK_DATABASES."""
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
    """LOCAL_FALLBACK_DATABASES names are accepted even when absent from the API set."""
    clear_valid_databases_cache()
    respx.get(f"{DEFAULT_API_URL}/admin/valid-databases").mock(
        return_value=httpx.Response(200, json={"databases": ["eegdash", "eegdash_dev"]})
    )

    c = InjectConfig(
        database="eegdash_archive",
        input=tmp_path,
        dry_run=True,
    )
    assert c.database == "eegdash_archive"


@respx.mock
def test_fetch_valid_databases_caches_failure_to_avoid_repeated_network_hits():
    """Network failure is cached; subsequent calls return None without re-hitting the server."""
    clear_valid_databases_cache()
    api_url = "https://failure-cache.example"
    route = respx.get(f"{api_url}/admin/valid-databases").mock(
        side_effect=httpx.ConnectError("boom")
    )

    assert fetch_valid_databases_from_api(api_url, token=None) is None
    assert fetch_valid_databases_from_api(api_url, token=None) is None
    assert fetch_valid_databases_from_api(api_url, token=None) is None

    assert route.call_count == 1


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4 — ValidateConfig
# ═══════════════════════════════════════════════════════════════════════════


def test_validate_config_defaults(tmp_path: Path):
    c = ValidateConfig(input=tmp_path)
    assert c.input == tmp_path
    assert c.pre_check is False
    assert c.verbose is False
    assert c.strict is False
    assert c.json_output is False


def test_validate_config_rejects_missing_input(tmp_path: Path):
    with pytest.raises(ValidationError) as exc:
        ValidateConfig(input=tmp_path / "does_not_exist")
    assert "input" in str(exc.value).lower()


def test_validate_config_accepts_real_input(tmp_path: Path):
    c = ValidateConfig(input=tmp_path)
    assert c.input == tmp_path


def test_validate_argv_parses_all_flags(tmp_path: Path):
    c = load_validate_config_from_argv(
        [
            "--input",
            str(tmp_path),
            "--pre-check",
            "--verbose",
            "--strict",
            "--json",
        ]
    )
    assert c.input == tmp_path
    assert c.pre_check is True
    assert c.verbose is True
    assert c.strict is True
    assert c.json_output is True


def test_validate_argv_short_flag_for_input(tmp_path: Path):
    """-i is an alias for --input."""
    c = load_validate_config_from_argv(["-i", str(tmp_path)])
    assert c.input == tmp_path


def test_validate_argv_short_flag_for_verbose(tmp_path: Path):
    """-v is an alias for --verbose."""
    c = load_validate_config_from_argv(["-i", str(tmp_path), "-v"])
    assert c.verbose is True


def test_validate_argv_env_var_picked_up(tmp_path: Path, monkeypatch):
    """EEGDASH_VALIDATE_STRICT=1 → strict via env."""
    monkeypatch.setenv("EEGDASH_VALIDATE_STRICT", "1")
    c = load_validate_config_from_argv(["-i", str(tmp_path)])
    assert c.strict is True


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3 — DigestConfig
# ═══════════════════════════════════════════════════════════════════════════


def test_digest_config_defaults(tmp_path: Path):
    c = DigestConfig(input=tmp_path)
    assert c.input == tmp_path
    assert c.output == Path("digestion_output")
    assert c.datasets is None
    assert c.workers == 1
    assert c.limit is None
    assert c.dataset_timeout == float(DEFAULT_DATASET_TIMEOUT_SECONDS)


def test_digest_config_rejects_missing_input(tmp_path: Path):
    with pytest.raises(ValidationError):
        DigestConfig(input=tmp_path / "does_not_exist")


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        pytest.param("workers", 0, id="workers_lower_bound"),
        pytest.param("workers", 9999, id="workers_upper_bound"),
        pytest.param("limit", 0, id="limit_lower_bound"),
        pytest.param("dataset_timeout", 0.0, id="dataset_timeout_zero"),
        pytest.param("dataset_timeout", -1.0, id="dataset_timeout_negative"),
        pytest.param(
            "dataset_timeout",
            float(60 * 60 * 24),
            id="dataset_timeout_upper_bound",
        ),
    ],
)
def test_digest_config_field_bounds_reject_invalid(
    tmp_path: Path, field: str, invalid_value
):
    with pytest.raises(ValidationError):
        DigestConfig(input=tmp_path, **{field: invalid_value})


def test_digest_config_datasets_filter(tmp_path: Path):
    c = DigestConfig(input=tmp_path, datasets=["ds-001", "ds-002"])
    assert c.datasets == ["ds-001", "ds-002"]


def test_digest_argv_parses_all_flags(tmp_path: Path):
    out_dir = tmp_path / "out"
    c = load_digest_config_from_argv(
        [
            "--input",
            str(tmp_path),
            "--output",
            str(out_dir),
            "--datasets",
            "ds-001",
            "ds-002",
            "--workers",
            "4",
            "--limit",
            "10",
            "--dataset-timeout",
            "300",
        ]
    )
    assert c.input == tmp_path
    assert c.output == out_dir
    assert c.datasets == ["ds-001", "ds-002"]
    assert c.workers == 4
    assert c.limit == 10
    assert c.dataset_timeout == 300.0


def test_digest_argv_env_var_picked_up(tmp_path: Path, monkeypatch):
    """EEGDASH_DIGEST_WORKERS=8 → workers=8 via env."""
    monkeypatch.setenv("EEGDASH_DIGEST_WORKERS", "8")
    c = load_digest_config_from_argv(["--input", str(tmp_path)])
    assert c.workers == 8


def test_digest_argv_validation_error_surfaces(tmp_path: Path):
    """Bad CLI → ValidationError, not a vague argparse failure."""
    with pytest.raises(ValidationError):
        load_digest_config_from_argv(["--input", str(tmp_path), "--workers", "0"])
