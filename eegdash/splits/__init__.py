# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Leakage-safe evaluation utilities (Workstream 3).

This sub-package wraps MOABB's evaluation splitters (or scikit-learn's
``GroupKFold``/``StratifiedGroupKFold`` when MOABB is unavailable) behind a
small, JSON-serializable manifest format. It is the foundation for tutorial
``plot_11_leakage_safe_split.py`` and for any benchmark submission that must
prove its split was free of subject/session/dataset leakage.

Public API (import as ``from eegdash.splits import ...``):

- :func:`to_split_metadata` -- normalize EEGDash/Braindecode/feature datasets
  into a tabular metadata frame with stable ``sample_id`` columns.
- :func:`to_moabb_split_inputs` -- return ``(y, metadata)`` aligned to MOABB's
  splitter API.
- :func:`get_splitter` -- friendly-named factory that returns either a MOABB
  splitter or a sklearn ``GroupKFold``/``StratifiedGroupKFold`` fallback.
- :func:`make_split_manifest` -- run the splitter and serialize folds + IDs +
  splitter config + library versions + random seed + target definition +
  metadata hash into a JSON-friendly dict.
- :func:`apply_split_manifest` -- materialise a fold/split into a dataset
  subset (or a boolean mask for DataFrames).
- :func:`assert_no_leakage` -- assert disjointness on a metadata column and
  emit the JSON ``leakage_report`` line consumed by the runtime validator.
- :func:`describe_split` -- print/return a one-screen summary suitable for the
  tutorial "split audit" cell.
- :func:`majority_baseline` / :func:`median_baseline` -- chance-level baselines
  for E5.43 reporting.

These helpers are not re-exported from ``eegdash`` itself; ``eegdash`` keeps a
deliberately small top-level surface (cf. ``eegdash/__init__.py``). Always
import from ``eegdash.splits``.
"""

from __future__ import annotations

from ._assertions import LeakageError, assert_no_leakage
from ._baselines import majority_baseline, median_baseline
from ._describe import describe_split
from ._manifest import (
    SCHEMA_VERSION,
    apply_split_manifest,
    make_split_manifest,
    manifest_to_json,
)
from ._metadata import to_moabb_split_inputs, to_split_metadata
from ._splitters import get_splitter

__all__ = [
    "LeakageError",
    "SCHEMA_VERSION",
    "apply_split_manifest",
    "assert_no_leakage",
    "describe_split",
    "get_splitter",
    "majority_baseline",
    "make_split_manifest",
    "manifest_to_json",
    "median_baseline",
    "to_moabb_split_inputs",
    "to_split_metadata",
]
