"""Import the docs ``dataset_page`` extension modules in the test matrix.

The package lives under ``docs/source/_extensions`` and touches Sphinx only
through ``sphinx.util.logging.getLogger``. The test matrix installs the
package without the doc dependencies, so a stdlib logger stands in for it when
Sphinx is missing (the same shim ``tests/test_dataset_page_stats.py`` uses).
The functions these tests call never log.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import logging as stdlib_logging
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXTENSIONS_ROOT = _REPO_ROOT / "docs" / "source" / "_extensions"


def _stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return module


def _ensure_sphinx_logging_shim() -> None:
    # Check sys.modules first: once the shim is installed, find_spec on it
    # raises because a hand-built module has no real spec.
    if "sphinx" in sys.modules or importlib.util.find_spec("sphinx") is not None:
        return
    sphinx = _stub_module("sphinx")
    sphinx_util = _stub_module("sphinx.util")
    sphinx_util_logging = _stub_module("sphinx.util.logging")
    sphinx_util_logging.getLogger = stdlib_logging.getLogger
    sphinx_util.logging = sphinx_util_logging
    sphinx.util = sphinx_util
    sys.modules["sphinx"] = sphinx
    sys.modules["sphinx.util"] = sphinx_util
    sys.modules["sphinx.util.logging"] = sphinx_util_logging


def load_dataset_page_module(name: str) -> types.ModuleType:
    """Return ``dataset_page.<name>`` (e.g. ``"data_loaders"``, ``"sections"``)."""
    _ensure_sphinx_logging_shim()
    if str(_EXTENSIONS_ROOT) not in sys.path:
        sys.path.insert(0, str(_EXTENSIONS_ROOT))
    return importlib.import_module(f"dataset_page.{name}")
