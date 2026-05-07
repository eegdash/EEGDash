# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Leakage-safe evaluation utilities."""

from __future__ import annotations

from ._audit import (
    LeakageError,
    assert_no_leakage,
    describe_split,
    majority_baseline,
    median_baseline,
)
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
