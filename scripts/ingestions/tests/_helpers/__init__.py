"""Shared test helpers — paths, synthetic fixture builders, factories."""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path
from types import ModuleType

# Stable anchor: ``_helpers/__init__.py`` lives at ``tests/_helpers/``,
# so ``parents[2]`` is ``scripts/ingestions/`` regardless of how deeply
# a consumer test file is nested under ``tests/``.
INGEST_DIR = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parents[1]


def load_module(filename: str) -> ModuleType:
    """Load a digit-prefixed ingestion script via ``importlib``.

    Replaces the per-file ``_load_digest()`` / ``_load_inject()`` shims
    that every consumer used to define. Fresh exec per call (matches the
    original per-test behaviour — no shared module state).
    """
    alias = f"loaded_{Path(filename).stem}"
    spec = _ilu.spec_from_file_location(alias, INGEST_DIR / filename)
    assert spec is not None
    assert spec.loader is not None
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_digest() -> ModuleType:
    """Load ``3_digest.py``."""
    return load_module("3_digest.py")


def load_inject() -> ModuleType:
    """Load ``5_inject.py``."""
    return load_module("5_inject.py")


__all__ = ["INGEST_DIR", "TESTS_DIR", "load_digest", "load_inject", "load_module"]
