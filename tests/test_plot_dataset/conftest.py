"""Shared fixtures for the ``plot_dataset`` chart-render tests.

Each chart's tests inject a fake ``/aggregations/{endpoint}`` payload by
patching :func:`docs.plot_dataset._live.fetch_aggregation`. The
``docs/`` directory isn't a package, so we extend ``sys.path`` to make
``plot_dataset.*`` imports resolve.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _REPO_ROOT / "docs"
if str(_DOCS) not in sys.path:
    sys.path.insert(0, str(_DOCS))
