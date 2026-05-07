# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Leakage-safe evaluation utilities for EEGDash datasets.

This subpackage is **opt-in** and requires MOABB:

.. code-block:: bash

    pip install eegdash[moabb]

The HuggingFace-style entry points are :func:`train_test_split` and
:func:`k_fold`; the rest of the public API supports the underlying
manifest format and per-fold audits.
"""

from __future__ import annotations

try:
    import moabb.evaluations.splitters  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised on bare installs
    raise ImportError(
        "eegdash.splits requires MOABB. Install with: pip install eegdash[moabb]"
    ) from exc

from ._audit import (
    LeakageError,
    assert_no_leakage,
    describe_split,
    majority_baseline,
    median_baseline,
)
from ._hf import k_fold, train_test_split
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
    "k_fold",
    "majority_baseline",
    "make_split_manifest",
    "manifest_to_json",
    "median_baseline",
    "to_moabb_split_inputs",
    "to_split_metadata",
    "train_test_split",
]
