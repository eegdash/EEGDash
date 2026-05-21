"""Pydantic-settings config for 5_inject.py CLI + env vars.

C6.5 — replaces 95 lines of argparse boilerplate + 25 lines of ad-hoc
post-parse validation with a single Pydantic model. Benefits:

- **Type-safe at boot**. Wrong int / bool / path errors at config
  construction, not deep in main() where they cascade into stack
  traces.
- **Validators are first-class**. The 'mutually-exclusive --only-*
  flags' check that lived as a stray ``if sum(only_flags) > 1`` is
  now a ``@model_validator`` — close to where the fields live and
  exercised by tests.
- **Env vars are native**. ``EEGDASH_ADMIN_TOKEN`` fallback used to
  be a ``args.token or os.environ.get(...)`` line in main(); now it's
  the ``env`` argument on the Field.
- **Auto-tested**. Constructing ``InjectConfig(database=..., ...)``
  in a unit test is one line; the same args via argparse needed
  subprocess + monkeypatch.
- **Reusable**. The same pattern applies to stage 4 / stage 3
  configs — consolidating CLI layouts pays interest.

The argparse layer in ``5_inject.py:main()`` becomes a 30-line thin
wrapper that calls :func:`load_inject_config_from_argv` and hands the
typed config to the rest of the orchestration. The 200 lines of
injection logic don't change.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_API_URL = "https://data.eegdash.org"

# Allowed values for ``--database``. Mirrors the API Gateway's
# valid_databases set (settings.valid_databases in api/main.py).
# Keep aligned with the cluster's config.
ValidDatabase = Literal[
    "eegdash",
    "eegdash_dev",
    "eegdash_archive",
    "eegdash_staging",
    "eegdash_v1",
]


class InjectConfig(BaseSettings):
    """Validated injection-pipeline config.

    Sources, in precedence order:
    1. Constructor kwargs (e.g. tests: ``InjectConfig(database="eegdash_dev")``)
    2. CLI args (via :func:`load_inject_config_from_argv` — argparse-shaped)
    3. Env vars (e.g. ``EEGDASH_ADMIN_TOKEN``)
    4. Defaults declared on each Field

    Use :func:`load_inject_config_from_argv` in main(); construct
    directly in tests.
    """

    model_config = SettingsConfigDict(
        env_prefix="EEGDASH_INJECT_",
        extra="ignore",
    )

    # ─── Required: target database ────────────────────────────────────────
    database: ValidDatabase = Field(
        description=(
            "Target MongoDB database. eegdash=production, "
            "eegdash_dev=development, eegdash_archive=old data, "
            "eegdash_staging=staging, eegdash_v1=legacy."
        ),
    )

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
    # `token` reads from the legacy env var name (EEGDASH_ADMIN_TOKEN), so
    # ops scripts that already export it work without changes. Configure
    # validation_alias to make BOTH ``token=`` constructor + the env var
    # populate this field.
    token: str | None = Field(
        default=None,
        description="Admin Bearer token (env fallback: EEGDASH_ADMIN_TOKEN).",
        # Accept both ``token=`` (kwarg / CLI) AND the canonical legacy
        # env var ``EEGDASH_ADMIN_TOKEN``. ``AliasChoices`` lets us list
        # multiple input names that populate the same field.
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
        """A run that's actually going to inject needs a real input dir.

        Allowing missing-dir in dry-run mode helps with --help-style
        exploration (e.g. someone running with all flags to see the
        summary against an empty plan).
        """
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

    Thin wrapper around argparse. Argparse handles the ``--help`` /
    short-name / usage-message ergonomics that pydantic-settings can't
    do alone; the parsed namespace becomes constructor kwargs.

    Pydantic-settings reads env vars + applies validators. Errors are
    surfaced as :class:`pydantic.ValidationError` (caller can render
    them nicely) instead of vague argparse errors.
    """
    import argparse

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
