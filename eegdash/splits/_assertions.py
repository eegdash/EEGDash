# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Leakage assertions for cross-validation manifests.

The single public entry point :func:`assert_no_leakage` is what tutorials
and benchmarks call after building a split. It emits a single JSON line on
stdout of the form::

    {"leakage_report": {"overlap": <int>, "by": "<by>"}}
"""

from __future__ import annotations

import json
import sys
from typing import Iterable, Sequence, Union

import pandas as pd


class LeakageError(ValueError):
    """Raised when a split manifest leaks groups across train/test."""


# ``manifest`` may be either a manifest dict produced by
# :func:`make_split_manifest`, or an explicit ``[(train_ids, test_ids), ...]``
# sequence of fold ID pairs.
ManifestLike = Union[dict, Sequence[tuple]]


def _emit_report(overlap: int, by: str) -> None:
    """Print the leakage-report JSON line on stdout."""
    payload = {"leakage_report": {"overlap": int(overlap), "by": str(by)}}
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _iter_fold_pairs(
    manifest_or_splits: ManifestLike,
) -> Iterable[tuple[Sequence, Sequence]]:
    """Yield ``(train_ids, test_ids)`` for each fold."""
    if isinstance(manifest_or_splits, dict) and "folds" in manifest_or_splits:
        for fold in manifest_or_splits["folds"]:
            yield fold["train"], fold["test"]
        return
    for pair in manifest_or_splits:  # type: ignore[arg-type]
        if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
            raise TypeError("Each fold must be a (train_ids, test_ids) tuple/list.")
        yield pair[0], pair[1]


def _values_for_ids(
    metadata: pd.DataFrame,
    ids: Sequence,
    by: str,
) -> set:
    """Return the unique values of column ``by`` for rows whose sample_id is in ``ids``."""
    if by not in metadata.columns:
        raise ValueError(
            f"Metadata has no column '{by}'. "
            f"Available columns: {list(metadata.columns)}"
        )
    if "sample_id" in metadata.columns:
        mask = metadata["sample_id"].isin(set(ids))
        if mask.any():
            return set(metadata.loc[mask, by].dropna().astype(str).unique().tolist())
    # ids may be integer row indices.
    try:
        idx = [int(i) for i in ids]
    except (TypeError, ValueError):
        return set()
    if not idx:
        return set()
    selected = metadata.iloc[idx]
    return set(selected[by].dropna().astype(str).unique().tolist())


def assert_no_leakage(
    manifest_or_splits: ManifestLike,
    metadata: pd.DataFrame,
    by: str = "subject",
) -> int:
    """Assert no train/test overlap on the ``by`` column for any fold.

    The function always emits a single JSON line of the form
    ``{"leakage_report": {"overlap": <int>, "by": "<by>"}}`` on stdout, even
    when the overlap is 0. This is what the E5.42 runtime validator parses.

    Parameters
    ----------
    manifest_or_splits
        Either a manifest dict produced by :func:`make_split_manifest` or an
        iterable of ``(train_ids, test_ids)`` pairs.
    metadata
        Per-sample metadata with at least a ``sample_id`` column and a column
        named after ``by``.
    by
        Column name used to detect overlap. Most tutorials should call this
        with ``by="subject"``.

    Returns
    -------
    int
        Maximum number of groups that overlap across train/test in any fold.

    Raises
    ------
    LeakageError
        When at least one fold leaks a value of column ``by`` across splits.

    """
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame.")
    if not isinstance(by, str) or not by:
        raise ValueError("`by` must be a non-empty string column name.")

    max_overlap = 0
    fold_overlaps: list[dict] = []
    for fold_index, (train_ids, test_ids) in enumerate(
        _iter_fold_pairs(manifest_or_splits)
    ):
        train_values = _values_for_ids(metadata, train_ids, by)
        test_values = _values_for_ids(metadata, test_ids, by)
        overlap = train_values & test_values
        n_overlap = len(overlap)
        if n_overlap:
            fold_overlaps.append(
                {
                    "fold": fold_index,
                    "n": n_overlap,
                    "values": sorted(overlap)[:10],  # cap to avoid huge logs
                }
            )
        if n_overlap > max_overlap:
            max_overlap = n_overlap

    _emit_report(max_overlap, by)

    if max_overlap > 0:
        raise LeakageError(
            f"Detected train/test overlap on column '{by}' "
            f"in {len(fold_overlaps)} fold(s); max_overlap={max_overlap}. "
            f"Details (first 10 values per fold): {fold_overlaps}"
        )
    return max_overlap


__all__ = ["LeakageError", "assert_no_leakage"]
