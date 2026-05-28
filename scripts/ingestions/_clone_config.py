"""Pydantic-settings config for 2_clone.py (C8).

Final stage in the C6.5-style config refactor — completes the pattern
across all 4 main()-style scripts (digest, validate, inject, clone).

The HANDLERS dict in 2_clone.py is treated as the source of truth for
``--sources`` choices, same as stage 4's _validate.py is for the
validators that consume it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Source names that 2_clone.py's HANDLERS dict knows how to clone.
# Kept aligned with HANDLERS — adding a new source there requires adding
# it here too (or refactoring to import HANDLERS lazily; that's a
# future round if the list grows).
KNOWN_SOURCES: frozenset[str] = frozenset(
    {
        "openneuro",
        "nemar",
        "gin",
        "figshare",
        "zenodo",
        "osf",
        "scidb",
        "datarn",
        "hbn",
        "neurips25",
        "unknown",
    }
)


class CloneConfig(BaseSettings):
    """Validated config for the clone stage.

    Same pattern as ``InjectConfig`` / ``DigestConfig`` / ``ValidateConfig``.
    """

    model_config = SettingsConfigDict(
        env_prefix="EEGDASH_CLONE_",
        extra="ignore",
    )

    input: Path = Field(
        default=Path("consolidated"),
        description=(
            "Input JSON file or directory containing the consolidated "
            "per-source listings (output of stage 1)."
        ),
    )
    output: Path = Field(
        default=Path("data/cloned"),
        description="Output directory for cloned datasets + manifests.",
    )
    sources: list[str] | None = Field(
        default=None,
        description=(
            "Process only specific sources. Default: all sources in "
            "HANDLERS. Must be a subset of KNOWN_SOURCES."
        ),
    )
    timeout: int = Field(
        default=300,
        ge=1,
        le=60 * 60 * 6,  # 6h ceiling — anything bigger means a hung handler
        description="Per-dataset handler timeout in seconds (max 6h).",
    )
    workers: int = Field(
        default=8,
        ge=1,
        le=128,
        description="Parallel worker threads (1 = sequential).",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum datasets to process (total, after limit-per-source).",
    )
    limit_per_source: int | None = Field(
        default=None,
        ge=1,
        description="Maximum datasets per source.",
    )
    datasets: list[str] | None = Field(
        default=None,
        description="Process only specific dataset IDs.",
    )
    manifest_only: bool = Field(
        default=False,
        description="Skip git clones; only emit manifests.",
    )

    @field_validator("sources")
    @classmethod
    def _sources_must_be_known(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        unknown = set(value) - KNOWN_SOURCES
        if unknown:
            raise ValueError(
                f"unknown source(s): {sorted(unknown)}; valid: {sorted(KNOWN_SOURCES)}"
            )
        return value

    @model_validator(mode="after")
    def _input_must_exist(self) -> CloneConfig:
        if not self.input.exists():
            raise ValueError(f"--input does not exist: {self.input}")
        return self


def load_clone_config_from_argv(
    argv: list[str] | None = None,
) -> CloneConfig:
    """Parse argv (or sys.argv) into a validated :class:`CloneConfig`."""
    parser = argparse.ArgumentParser(
        description=(
            "Clone or fetch datasets per consolidated source listings "
            "(stage 2 of the ingest pipeline)."
        )
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        choices=sorted(KNOWN_SOURCES),
    )
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--limit-per-source", type=int, default=None, dest="limit_per_source"
    )
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--manifest-only", action="store_true", dest="manifest_only")

    ns = parser.parse_args(argv if argv is not None else sys.argv[1:])
    # Drop False bools (action="store_true" default) AND None values so
    # the model defaults apply.
    kwargs = {k: v for k, v in vars(ns).items() if v is not None and v is not False}
    return CloneConfig(**kwargs)
