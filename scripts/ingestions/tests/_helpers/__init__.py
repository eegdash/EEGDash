"""Shared test helpers — paths, synthetic fixture builders, factories."""

from __future__ import annotations

from pathlib import Path

# Stable anchor: ``_helpers/__init__.py`` lives at ``tests/_helpers/``,
# so ``parents[2]`` is ``scripts/ingestions/`` regardless of how deeply
# a consumer test file is nested under ``tests/``.
INGEST_DIR = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parents[1]

__all__ = ["INGEST_DIR", "TESTS_DIR"]
