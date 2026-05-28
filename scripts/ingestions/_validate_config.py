"""Pydantic-settings config for 4_validate_output.py (C7.1).

Applies the C6.5 pattern (Pydantic-settings + thin argparse bridge)
to the validate stage. 5 CLI flags become declarative fields with
descriptions; existing exit-code semantics preserved.

The pattern from _inject_config.py:
1. ``BaseSettings`` model — fields + validators + accessors.
2. ``load_*_config_from_argv()`` — argparse → kwargs bridge that
   preserves --help / short-flag ergonomics.
3. ``main()`` in the stage calls the loader, gets a typed config.


"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ValidateConfig(BaseSettings):
    """Validated config for the digest-output validation stage.

    Sources, in precedence order: constructor kwargs → CLI args
    (via :func:`load_validate_config_from_argv`) → env vars → defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="EEGDASH_VALIDATE_",
        extra="ignore",
    )

    # I/O
    input: Path = Field(
        default=Path("digestion_output"),
        description=(
            "Input directory. Default: ``digestion_output``. For "
            "``--pre-check``, point at ``data/cloned`` instead."
        ),
    )

    # Mode toggles
    pre_check: bool = Field(
        default=False,
        description=(
            "Pre-digestion validation: check manifests have valid "
            "data files (run BEFORE digest, not after)."
        ),
    )
    verbose: bool = Field(default=False, description="Print per-dataset diagnostics.")
    strict: bool = Field(
        default=False,
        description="Treat warnings as errors (empty datasets fail too).",
    )
    # The CLI flag is --json; ``json`` clashes with the stdlib module
    # name inside main() so we expose it as ``json_output`` in the model
    # and map back at the CLI layer.
    json_output: bool = Field(
        default=False,
        description="Output the result as a JSON document on stdout.",
    )

    @model_validator(mode="after")
    def _input_must_exist(self) -> ValidateConfig:
        """The validator can't validate nothing — the input dir has to
        exist. (No dry-run escape hatch like 5_inject; this stage IS
        the dry-check.)
        """
        if not self.input.exists():
            raise ValueError(f"--input directory does not exist: {self.input}")
        return self


def load_validate_config_from_argv(
    argv: list[str] | None = None,
) -> ValidateConfig:
    """Parse argv (or sys.argv) into a validated :class:`ValidateConfig`."""
    parser = argparse.ArgumentParser(
        description="Validate digestion output for eegdash compatibility"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help=("Input directory (digestion_output or data/cloned for --pre-check)"),
    )
    parser.add_argument(
        "--pre-check",
        action="store_true",
        dest="pre_check",
        help=("Pre-digestion validation: check manifests have valid data files"),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print verbose output"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (empty datasets become errors)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )

    ns = parser.parse_args(argv if argv is not None else sys.argv[1:])
    kwargs = {k: v for k, v in vars(ns).items() if v is not None and v is not False}
    # Re-add False booleans that were explicitly set to False by argparse
    # (action="store_true" defaults to False — only forward when True).
    return ValidateConfig(**kwargs)
