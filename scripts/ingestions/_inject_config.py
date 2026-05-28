"""Pydantic-settings config for 5_inject.py CLI + env vars.

Replaces argparse boilerplate and ad-hoc post-parse validation with a
single Pydantic model. Wrong types and mutual-exclusion errors are caught
at config construction rather than deep in main(). Env vars (e.g.
``EEGDASH_ADMIN_TOKEN``) and CLI args are reconciled automatically.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import httpx
from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_API_URL = "https://data.eegdash.org"

# Source-of-truth fallback. Must stay aligned with the API Gateway's
# settings.valid_databases set; the bootstrap call below will detect
# drift at boot when the server exposes /admin/valid-databases. Until
# then, this list IS the contract.
LOCAL_FALLBACK_DATABASES: frozenset[str] = frozenset(
    {
        "eegdash",
        "eegdash_dev",
        "eegdash_archive",
        "eegdash_staging",
        "eegdash_v1",
    }
)

# Per-API-URL cache so validator calls don't re-hit the network on
# every InjectConfig construction. Cleared by tests via the
# clear_valid_databases_cache() helper.
_valid_databases_cache: dict[str, frozenset[str]] = {}


def fetch_valid_databases_from_api(
    api_url: str,
    token: str | None,
    *,
    timeout: float = 5.0,
) -> frozenset[str] | None:
    """Return the API Gateway's valid_databases set, or None on any failure.

    Cached per api_url; falls back to LOCAL_FALLBACK_DATABASES on network
    error, non-200 status, malformed JSON, or missing 'databases' key.
    """
    if api_url in _valid_databases_cache:
        cached = _valid_databases_cache[api_url]
        # Empty frozenset is the sentinel for "API previously failed";
        # return None so the validator falls back instead of treating
        # an empty set as "no valid databases".
        return cached if cached else None

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{api_url}/admin/valid-databases", headers=headers)
            resp.raise_for_status()
            data = resp.json()
            valid = frozenset(data["databases"])
    except (httpx.HTTPError, KeyError, ValueError, TypeError):
        _valid_databases_cache[api_url] = frozenset()  # sentinel "no api data"
        return None

    _valid_databases_cache[api_url] = valid
    return valid


def clear_valid_databases_cache() -> None:
    """Test helper. Resets the per-api_url cache."""
    _valid_databases_cache.clear()


# Replaces the Literal so we can dynamically extend the accepted set
# when the API exposes /admin/valid-databases. The frozenset above
# (LOCAL_FALLBACK_DATABASES) is the contract until the API is queried.
DatabaseName = str


class InjectConfig(BaseSettings):
    """Validated injection-pipeline config.

    Precedence: constructor kwargs > CLI args > env vars > Field defaults.
    Use :func:`load_inject_config_from_argv` in main(); construct directly
    in tests.
    """

    model_config = SettingsConfigDict(
        env_prefix="EEGDASH_INJECT_",
        extra="ignore",
    )

    # ─── Required: target database ────────────────────────────────────────
    database: DatabaseName = Field(
        description=(
            "Target MongoDB database. Valid names checked against the API's "
            "valid-databases endpoint at boot (falls back to "
            "LOCAL_FALLBACK_DATABASES on network failure)."
        ),
    )

    @field_validator("database")
    @classmethod
    def _database_must_be_valid(cls, value: str, info) -> str:
        """Reject databases not known to either the API or LOCAL_FALLBACK_DATABASES.

        Valid set is the union of the API response and the local fallback so
        that an API-side deprecation or network failure never locks out a
        running script at config-construction time.
        """
        # api_url / token may not yet be validated (declared after this field);
        # fall back to defaults so the probe is still best-effort.
        api_url = info.data.get("api_url") or DEFAULT_API_URL
        token = info.data.get("token")

        api_set = fetch_valid_databases_from_api(api_url, token)
        if api_set is None:
            logging.getLogger(__name__).warning(
                "Could not reach %s/admin/valid-databases; using "
                "LOCAL_FALLBACK_DATABASES alone. Database-list drift "
                "will not be detected this run.",
                api_url,
            )
        valid: frozenset[str] = (api_set or frozenset()) | LOCAL_FALLBACK_DATABASES

        if value not in valid:
            raise ValueError(
                f"database={value!r} is not in the valid set "
                f"({sorted(valid)}); update the API's valid_databases or "
                f"LOCAL_FALLBACK_DATABASES if you are adding a new one."
            )
        return value

    # ─── I/O ─────────────────────────────────────────────────────────────
    input: Path = Field(
        default=Path("digestion_output"),
        description="Directory containing digested datasets.",
    )
    api_url: str = Field(
        default=DEFAULT_API_URL,
        description="EEGDash API Gateway URL.",
    )

    # ─── Auth (env-var fallback: EEGDASH_ADMIN_TOKEN) ────────────────────
    # AliasChoices lets both ``token=`` constructor kwarg and the legacy
    # EEGDASH_ADMIN_TOKEN env var populate this field.
    token: str | None = Field(
        default=None,
        description="Admin Bearer token (env fallback: EEGDASH_ADMIN_TOKEN).",
        validation_alias=AliasChoices("token", "EEGDASH_ADMIN_TOKEN"),
    )

    # ─── Filters ─────────────────────────────────────────────────────────
    datasets: list[str] | None = Field(
        default=None,
        description=(
            "Specific dataset IDs to inject. Default: all datasets in --input."
        ),
    )

    # ─── Behaviour flags ─────────────────────────────────────────────────
    dry_run: bool = Field(default=False, description="Validate without uploading.")
    batch_size: int = Field(
        default=1000,
        ge=1,
        le=10_000,
        description="Max records per API request.",
    )
    only_datasets: bool = Field(default=False)
    only_records: bool = Field(default=False)
    only_montages: bool = Field(default=False)
    skip_montages: bool = Field(default=False)
    force: bool = Field(
        default=False,
        description="Inject even if ingestion_fingerprint matches existing.",
    )
    skip_validation: bool = Field(
        default=False,
        description="Skip validation before injection (not recommended).",
    )
    data_quality_threshold: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description=(
            "Max percentage of records with missing nchans/sampling_frequency "
            "before failing the pre-inject quality gate."
        ),
    )
    compute_stats: bool = Field(
        default=False,
        description="Recompute dataset stats after injection.",
    )

    # ─── Validators ──────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _only_flags_are_mutually_exclusive(self) -> InjectConfig:
        """At most one of --only-datasets / --only-records / --only-montages."""
        only_flags = sum((self.only_datasets, self.only_records, self.only_montages))
        if only_flags > 1:
            raise ValueError(
                "--only-datasets, --only-records, --only-montages are "
                "mutually exclusive (max one)"
            )
        return self

    @model_validator(mode="after")
    def _only_and_skip_montages_are_contradictory(self) -> InjectConfig:
        if self.only_montages and self.skip_montages:
            raise ValueError(
                "--only-montages and --skip-montages contradict each other"
            )
        return self

    @model_validator(mode="after")
    def _input_dir_must_exist_unless_dry_run(self) -> InjectConfig:
        """Require the input directory to exist, unless dry_run skips actual I/O."""
        if not self.dry_run and not self.input.exists():
            raise ValueError(f"--input directory does not exist: {self.input}")
        return self

    # ─── Convenience accessors ───────────────────────────────────────────

    @property
    def want_datasets(self) -> bool:
        return not self.only_records and not self.only_montages

    @property
    def want_records(self) -> bool:
        return not self.only_datasets and not self.only_montages

    @property
    def want_montages(self) -> bool:
        if self.only_datasets or self.only_records:
            return False
        if self.skip_montages:
            return False
        return True


def load_inject_config_from_argv(argv: list[str] | None = None) -> InjectConfig:
    """Parse argv (or sys.argv) into a validated :class:`InjectConfig`.

    Argparse handles ``--help`` and usage-message ergonomics; the parsed
    namespace becomes constructor kwargs for :class:`InjectConfig`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Inject digested datasets and records into MongoDB via "
            "the EEGDash API Gateway."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Directory containing digested datasets (default: digestion_output/).",
    )
    parser.add_argument(
        "--database",
        type=str,
        required=True,
        help=(
            "Target MongoDB database: eegdash | eegdash_dev | eegdash_archive | "
            "eegdash_staging | eegdash_v1."
        ),
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        dest="api_url",
        help=f"EEGDash API URL (default: {DEFAULT_API_URL}).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Admin token (env fallback: EEGDASH_ADMIN_TOKEN).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific dataset IDs to inject (default: all).",
    )
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    parser.add_argument("--only-datasets", action="store_true", dest="only_datasets")
    parser.add_argument("--only-records", action="store_true", dest="only_records")
    parser.add_argument("--only-montages", action="store_true", dest="only_montages")
    parser.add_argument("--skip-montages", action="store_true", dest="skip_montages")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-validation", action="store_true", dest="skip_validation"
    )
    parser.add_argument(
        "--data-quality-threshold",
        type=float,
        default=None,
        dest="data_quality_threshold",
    )
    parser.add_argument("--compute-stats", action="store_true", dest="compute_stats")

    ns = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # Drop None values so Field defaults take effect; argparse can't
    # express "no flag passed → use the model default" naturally.
    kwargs = {k: v for k, v in vars(ns).items() if v is not None}
    return InjectConfig(**kwargs)
