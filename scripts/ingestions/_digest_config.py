"""Pydantic-settings config for 3_digest.py (C7.2).

Same pattern as ``_inject_config.py`` (C6.5) and ``_validate_config.py``
(C7.1). Replaces the argparse + ``_positive_float`` custom-type
boilerplate with declarative bounds.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DATASET_TIMEOUT_SECONDS = 2 * 60  # mirrors 3_digest.py


class DigestConfig(BaseSettings):
    """Validated config for the BIDS digest stage.

    Sources, in precedence order: constructor kwargs → CLI args
    (via :func:`load_digest_config_from_argv`) → env vars → defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="EEGDASH_DIGEST_",
        extra="ignore",
    )

    input: Path = Field(
        default=Path("data/cloned"),
        description="Directory containing cloned datasets.",
    )
    output: Path = Field(
        default=Path("digestion_output"),
        description="Output directory for JSON files.",
    )
    datasets: list[str] | None = Field(
        default=None,
        description="Specific dataset IDs to digest (default: all).",
    )
    workers: int = Field(
        default=1,
        ge=1,
        le=128,
        description="Parallel worker processes (1 = sequential).",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Stop after digesting N datasets (test-mode short-circuit).",
    )
    dataset_timeout: float = Field(
        default=float(DEFAULT_DATASET_TIMEOUT_SECONDS),
        gt=0.0,
        le=60 * 60 * 4.0,  # 4-hour ceiling — anything bigger is a misconfig
        description=(
            "Seconds before killing a worker. Default: 120s. Ceiling: 4h "
            "(beyond that, fix the underlying slowness instead of waiting)."
        ),
    )

    @model_validator(mode="after")
    def _input_must_exist(self) -> DigestConfig:
        if not self.input.exists():
            raise ValueError(f"--input directory does not exist: {self.input}")
        return self


def load_digest_config_from_argv(
    argv: list[str] | None = None,
) -> DigestConfig:
    """Parse argv (or sys.argv) into a validated :class:`DigestConfig`."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Digest BIDS datasets and generate Dataset + Record JSON for MongoDB."
        )
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dataset-timeout",
        type=float,
        default=None,
        dest="dataset_timeout",
        help=(
            "Seconds to allow one dataset before killing its worker "
            f"(default: {DEFAULT_DATASET_TIMEOUT_SECONDS:g})"
        ),
    )

    ns = parser.parse_args(argv if argv is not None else sys.argv[1:])
    kwargs = {k: v for k, v in vars(ns).items() if v is not None}
    return DigestConfig(**kwargs)
