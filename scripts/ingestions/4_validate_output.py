#!/usr/bin/env python3
"""CLI wrapper around :mod:`_validate` — exit-code semantics + JSON output.

The validation logic itself lives in :mod:`_validate`. This script is the
argv-bound entry point: it parses the config, calls either
:func:`validate_pre_digestion` or :func:`validate_digestion_output`, and
emits a human-readable summary (or JSON via ``--json``).

See ``python 4_validate_output.py --help``.
"""

import json
import sys

from pydantic import ValidationError

from _validate import (  # noqa: F401 — re-export for time_openneuro_pipeline.py subprocess invocation
    DATA_QUALITY_FIELDS,
    NEURO_EXTENSIONS,
    RECOMMENDED_DATASET_FIELDS,
    RECOMMENDED_RECORD_FIELDS,
    VALID_SOURCES,
    VALID_STORAGE_PATTERNS,
    ValidationResult,
    _add_pydantic_errors,
    validate_dataset,
    validate_digestion_output,
    validate_pre_digestion,
    validate_record,
    validate_storage_url,
)
from _validate_config import load_validate_config_from_argv


def main():
    try:
        cfg = load_validate_config_from_argv()
    except ValidationError as exc:
        print("Config error(s):", file=sys.stderr)
        for err in exc.errors():
            field = ".".join(str(p) for p in err.get("loc", []))
            print(f"  {field}: {err.get('msg')}", file=sys.stderr)
        return 1

    if cfg.pre_check:
        result = validate_pre_digestion(cfg.input, verbose=cfg.verbose)
    else:
        result = validate_digestion_output(
            cfg.input,
            verbose=cfg.verbose,
            strict=cfg.strict,
        )

    if cfg.json_output:
        output = {
            "valid": result.is_valid(),
            "stats": result.stats,
            "errors": result.errors,
            "warnings": result.warnings,
            "empty_datasets": result.empty_datasets,
            "source_distribution": result.source_distribution,
            "modality_distribution": result.modality_distribution,
        }
        print(json.dumps(output, indent=2))
    else:
        print(result.summary())

        if result.source_distribution:
            print("\nSource distribution:")
            for src, count in sorted(
                result.source_distribution.items(), key=lambda x: -x[1]
            ):
                print(f"  {src}: {count}")

        if result.modality_distribution:
            print("\nModality distribution:")
            for mod, count in sorted(
                result.modality_distribution.items(), key=lambda x: -x[1]
            ):
                print(f"  {mod}: {count}")

    if cfg.strict and result.warnings:
        sys.exit(1)
    sys.exit(0 if result.is_valid() else 1)


if __name__ == "__main__":
    main()
